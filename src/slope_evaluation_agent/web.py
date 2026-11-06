"""Local Web API and browser UI for the slope-evaluation agent.

The Web layer is intentionally thin: it never calls an MCP tool directly and
never advances workflow state itself.  Every conversational action goes
through :class:`SlopeEvaluationAgent`, preserving the same confirmation gates,
scientific audit and session persistence as the CLI.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import hashlib
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from . import __version__
from .agent import AgentReply, SlopeEvaluationAgent
from .artifacts import DownloadError
from .config import AgentSettings, load_settings
from .mcp_clients import AgentMCPError
from .session_store import SessionError, validate_session_id
from .state_machine import WorkflowError

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as exc:  # pragma: no cover - exercised by the entry point
    raise RuntimeError(
        'WebUI 依赖尚未安装；请运行 pip install -e ".[web]"'
    ) from exc


STATIC_ROOT = Path(__file__).with_name("web_static")
IMAGE_NAME = re.compile(r"^session-[0-9a-f]{12}-[0-9a-f]{16}\.(?:png|jpg|webp|gif|bin)$")
MIME_SUFFIX = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def content_disposition(filename: str) -> str:
    """``Content-Disposition`` for a download, safe for non-ASCII names.

    The plain ``filename`` is restricted to ASCII so no header can be broken by
    a newline or a quote; the real name travels in the RFC 5987 parameter, which
    is what a modern browser uses. Both forms come from a name the agent already
    reduced to a single path component.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename.encode("ascii", "ignore").decode("ascii"))
    safe = safe.strip("._")[:120] or "download"
    return f"attachment; filename=\"{safe}\"; filename*=UTF-8''{quote(filename, safe='')}"


class SessionCreateRequest(BaseModel):
    language: str = Field(default="en", pattern=r"^(?:zh|en)$")


class SessionBatchDeleteRequest(BaseModel):
    model_config = {"extra": "forbid"}
    session_ids: list[str] = Field(min_length=1, max_length=200)

    @field_validator("session_ids")
    @classmethod
    def valid_ids(cls, values: list[str]) -> list[str]:
        # Validate the entire selection before any deletion can begin.
        try:
            return list(dict.fromkeys(validate_session_id(value) for value in values))
        except SessionError as exc:
            raise ValueError(str(exc)) from exc


class MessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)

    @field_validator("text")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("消息不能为空")
        return value


class OutletCandidatesRequest(BaseModel):
    """A river name to lay out on the siting map (recommended outlets included)."""

    river_name: str = Field(min_length=1, max_length=64)
    max_candidates: int = Field(default=0, ge=0, le=10)

    @field_validator("river_name")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("河流名称不能为空")
        return value


class ExportRequest(BaseModel):
    """A request to derive a viewable file from a registered one.

    An enum, not a free string: what can be derived is the service's decision,
    and a caller may only name one of the formats this build offers.
    """

    format: Literal["xlsx"]


class StudyAreaMapRequest(BaseModel):
    administrative_area: str | None = Field(default=None,min_length=1,max_length=64)
    river_name: str | None = Field(default=None,min_length=1,max_length=64)
    longitude: float | None = Field(default=None,ge=-180,le=180,allow_inf_nan=False)
    latitude: float | None = Field(default=None,ge=-90,le=90,allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_location(self):
        if (self.longitude is None)!=(self.latitude is None):
            raise ValueError("请同时提供地图中心经度与纬度")
        if self.administrative_area and (self.river_name or self.longitude is not None):
            raise ValueError("请选择区县、河名或坐标中的一种定位方式")
        if not (self.administrative_area or self.river_name or self.longitude is not None):
            raise ValueError("请提供区县、河名或地图中心坐标")
        if (self.river_name is not None and not self.river_name.strip()) or (self.administrative_area is not None and not self.administrative_area.strip()):
            raise ValueError("定位名称不能为空")
        return self


class StudyAreaRequest(BaseModel):
    administrative_area: str | None = Field(default=None,min_length=1,max_length=64)
    map_id: str | None = Field(default=None,max_length=64)
    coordinates: list[list[float]] = Field(min_length=3,max_length=500)
    assessment_mode: Literal["regional_screening","catchment_check"] = "regional_screening"


class TaskEditRequest(BaseModel):
    scope: Literal["location","boundary","rainfall","parameters"]
    rainfall: dict[str,Any] | None = None
    parameter_preset: Literal["low","medium","high"] | None = None
    parameter_overrides: dict[str,float] | None = None


class TaskRestoreRequest(BaseModel):
    revision_id: str | None = Field(default=None,pattern=r"^revision-[0-9a-f]{12}$")


class TaskDraftRequest(BaseModel):
    map_id: str = Field(min_length=1,max_length=64)
    coordinates: list[list[float]] = Field(default_factory=list,max_length=500)
    closed: bool = False
    assessment_mode: Literal["regional_screening","catchment_check"] | None = None

    @field_validator("coordinates")
    @classmethod
    def valid_points(cls,value):
        import math
        if any(len(p)!=2 or not all(math.isfinite(v) for v in p) or not -180<=p[0]<=180 or not -90<=p[1]<=90 for p in value):
            raise ValueError("草稿坐标必须是有效经纬度")
        return value

    @model_validator(mode="after")
    def valid_closed(self):
        if self.closed and len(self.coordinates)<3:
            raise ValueError("闭合边界至少需要三个顶点")
        return self


class WebRuntime:
    """One agent process plus a lock for each persisted conversation."""

    def __init__(self, agent: SlopeEvaluationAgent, settings: AgentSettings) -> None:
        self.agent = agent
        self.settings = settings
        self.locks: dict[str, asyncio.Lock] = {}
        self.cleanup_lock = asyncio.Lock()
        self.image_root = settings.sessions.resolved_root().parent / "web_images"

    @asynccontextmanager
    async def lock(self, session_id: str, *, allow_cleanup: bool = False):
        validate_session_id(session_id)
        async with self.locks.setdefault(session_id, asyncio.Lock()):
            if not allow_cleanup and (self.agent.store.directory(session_id) / "cleanup_pending.json").exists():
                raise HTTPException(status_code=409,
                    detail="该对话的产物清理尚未完成，请重试删除；完成前不能继续运行或修改任务。")
            yield

    def session_view(self, session: Any, *, include_conversation: bool = False) -> dict[str, Any]:
        from .task_context import workspace_view
        workspace=workspace_view(session)
        location=workspace["location"]
        location["map"]=location.pop("map_view",{})
        name=location.get("map_image_name")
        if name and IMAGE_NAME.fullmatch(name) and name.startswith(session.session_id+"-"):
            location.setdefault("map",{})["url"]=f"/api/v1/sessions/{session.session_id}/images/{name}"
        payload: dict[str, Any] = {
            "workspace":workspace,
            "session_id": session.session_id,
            "state": session.state,
            "pending_action": session.pending_action(),
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "run_status": session.hydro_run_status,
            "qa_passed": session.qa_passed,
            "has_review": bool(session.review_id),
            "watershed_confirmed": session.watershed_confirmed,
            "watershed_qa_passed": session.watershed_qa_passed,
            "selection_mode": session.selection_mode,
            "administrative_area": session.administrative_area,
            "has_rainfall": bool(session.rainfall_package_id),
            "has_model_inputs": bool(session.model_input_package_id),
            "has_hydro_input": bool(session.hydro_input_id),
            "has_run": bool(session.hydro_run_id),
            # Conversation metadata. Read defensively: a session written before
            # this release has no such fields, and must still open.
            "last_route": getattr(session, "last_route", None),
            "last_clarification": getattr(session, "last_clarification", None),
            "conversation_summary": getattr(session, "conversation_summary", None),
        }
        if include_conversation:
            payload["conversation"] = [
                {"role": turn.role, "text": turn.text, "at": turn.at}
                | {"answer_source": turn.answer_source}
                for turn in session.conversation
            ]
        return payload

    def reply_view(self, reply: AgentReply) -> dict[str, Any]:
        session_id = reply.session.session_id
        images = self.reply_images(session_id, reply)
        if reply.outlet_map and images and reply.session.selection_context:
            reply.session.selection_context["map_image_name"]=images[0]["url"].rsplit("/",1)[-1]
            self.agent.store.save(reply.session)
        review_qa = (reply.facts.get("review") or {}).get("qa") or {}
        return {
            "text": reply.text,
            "kind": reply.kind,
            "ok": reply.ok,
            "next_step": reply.next_step or reply.session.pending_action(),
            "rewritten": reply.rewritten,
            "answer_source": reply.answer_source,
            "audit_scope": reply.audit_scope,
            "spatial_qa_passed": review_qa.get("passed"),
            "hydrologic_qa_passed": (review_qa.get("hydrologic_check") or {}).get("passed"),
            "modeling_mode": (review_qa.get("modeling_check") or {}).get("assessment_mode"),
            "modeling_eligible": (review_qa.get("modeling_check") or {}).get("passed"),
            "knowledge_fallback": reply.facts.get("knowledge_fallback"),
            "audit_passed": (None if reply.audit_scope in {"general_knowledge", "task_context"} else not bool(reply.audit)),
            "images": images,
            "outlet_map": ({**reply.outlet_map,
                            "url": images[0]["url"] if images else None}
                           if reply.outlet_map is not None else None),
            # Conversation metadata (V2). Added alongside the fields above
            # rather than replacing any of them, so a client written against
            # the previous shape keeps working unchanged.
            "route": getattr(reply.session, "last_route", None),
            # The intent of *this turn* when the turn was conversation; the
            # session's last workflow intent is a different thing and would be
            # empty for a chat turn.
            "intent": reply.facts.get("turn_intent") or reply.session.intent,
            "needs_user_input": self.needs_user_input(reply),
            "suggestions": list(reply.facts.get("clarification_examples") or []),
            # Figures of this turn with the real files behind them, and the
            # run's SWMM model section on result turns. Descriptors only: an
            # artifact is addressed by id, never by a location.
            "visualizations": self.figure_views(session_id, reply.visualizations, images),
            "prepared_artifact": self.artifact_view(session_id, reply.facts.get("artifact")),
            "process_artifacts": self.process_artifact_views(
                session_id, reply.session.process_artifacts)
            if reply.kind == "result" else [],
            "swmm": self.swmm_view(session_id, reply.session.swmm_model)
            if reply.kind == "result" else None,
        }

    # ------------------------------------------------------------- artifacts
    @staticmethod
    def size_text(size: int | None) -> str | None:
        """A human-readable size, for a button label."""
        if not isinstance(size, int) or size < 0:
            return None
        for unit, scale in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
            if size >= scale:
                return f"{size / scale:.1f} {unit}"
        return f"{size} B"

    def artifact_view(self, session_id: str, descriptor: Any) -> dict[str, Any] | None:
        """One download as the browser sees it: a name, a size and an opaque id.

        Built from the agent's descriptor, which is already a projection — there
        is no ``internal_path`` field in it to leak, and the URL carries nothing
        but the artifact id.
        """
        if descriptor is None:
            return None
        payload = descriptor.model_dump(mode="json") \
            if hasattr(descriptor, "model_dump") else dict(descriptor)
        artifact_id = str(payload.get("artifact_id") or "")
        if not artifact_id:
            return None
        sha = str(payload.get("sha256") or "")
        # The primary action always downloads the registered original bytes.
        # Optional conversions remain available to an explicit conversational
        # request, but the UI must not imply that CSV needs conversion to Excel.
        exports: list[str] = []
        return {
            "artifact_id": artifact_id,
            "title": payload.get("title"),
            "role": payload.get("role"),
            "format": payload.get("format"),
            "media_type": payload.get("media_type"),
            "size_bytes": payload.get("size_bytes"),
            "size_text": self.size_text(payload.get("size_bytes")),
            "sha256_short": sha[:12] or None,
            "units": payload.get("units"),
            "crs": payload.get("crs"),
            "qa_status": payload.get("qa_status"),
            "stage_id": payload.get("stage_id"),
            "time_column": payload.get("time_column"),
            "rows": payload.get("rows"),
            "column_count": payload.get("column_count"),
            "diagnostic_only": bool(payload.get("qa_status") in {"failed", "qa_failed"}),
            "ready": bool(payload.get("ready")),
            "export_formats": exports,
            "download_url": f"/api/v1/sessions/{session_id}/artifacts/{artifact_id}/download",
        }

    def process_artifact_views(self, session_id: str,
                               descriptors: list[Any]) -> list[dict[str, Any]]:
        """All original run files, de-duplicated by opaque artifact id."""
        found: list[dict[str, Any]] = []
        seen: set[str] = set()
        for descriptor in descriptors or []:
            view = self.artifact_view(session_id, descriptor)
            if view is None or view["artifact_id"] in seen:
                continue
            seen.add(view["artifact_id"])
            found.append(view)
        return found

    def figure_views(self, session_id: str, figures: list[Any],
                     images: list[dict[str, str]] | None) -> list[dict[str, Any]]:
        """Figures with a preview URL and their download lists.

        ``images`` are this turn's just-persisted previews, matched by the index
        the agent recorded. When it is ``None`` — a page being restored — the
        preview URL points at the figure endpoint, which fetches it from the
        service again; nothing about the image is stored in the session.
        """
        views: list[dict[str, Any]] = []
        for figure in figures or []:
            payload = figure.public_view() if hasattr(figure, "public_view") else dict(figure)
            index = payload.pop("image_index", None)
            if images is not None and isinstance(index, int) and 0 <= index < len(images):
                payload["preview"] = images[index]
            else:
                payload["preview"] = {
                    "mime_type": "image/png",
                    "url": f"/api/v1/sessions/{session_id}/visualizations/"
                           f"{payload.get('visualization_id')}/image",
                }
            payload["artifacts"] = [
                view for view in (self.artifact_view(session_id, item)
                                  for item in payload.get("artifacts") or [])
                if view is not None]
            views.append(payload)
        return views

    def swmm_view(self, session_id: str, model: dict | None) -> dict[str, Any] | None:
        """The run's SWMM model files and package, as a browser may see them."""
        if not model or not isinstance(model, dict):
            return None
        artifacts = [view for view in
                     (self.artifact_view(session_id, item)
                      for item in model.get("artifacts") or [])
                     if view is not None]
        bundle = self.artifact_view(session_id, model.get("bundle"))
        if not artifacts and bundle is None:
            return None
        return {
            "run_id": model.get("run_id"),
            "complete": bool(model.get("complete", True)),
            "missing_required": list(model.get("missing_required") or []),
            "warnings": list(model.get("warnings") or []),
            "artifacts": artifacts,
            "bundle": bundle,
            "bundle_complete": bool(model.get("bundle_complete", True))
            if bundle else None,
        }

    @staticmethod
    def needs_user_input(reply: AgentReply) -> bool:
        """Whether the agent is waiting for an answer rather than reporting.

        A question is obvious; the other case is a workflow request the state
        machine refused and turned into a question, which is stored as the
        turn's route rather than as its kind.
        """
        return reply.kind == "question" or getattr(
            reply.session, "last_route", None) == "clarification"

    def reply_images(self, session_id: str, reply: AgentReply) -> list[dict[str, str]]:
        """Persist this turn's images, figures under their figure's own name.

        A figure's preview is cached under a name derived from the figure id, so
        the live turn and a later reload serve the same file instead of every
        page load adding a copy. Everything else keeps a one-off name.
        """
        validate_session_id(session_id)
        by_index: dict[int, str] = {}
        for figure in reply.visualizations or []:
            index = getattr(figure, "image_index", None)
            if isinstance(index, int) and 0 <= index < len(reply.images):
                url = self.cache_image(session_id, reply.images[index],
                                       key=figure.visualization_id)
                if url:
                    by_index[index] = url
        others = [image for index, image in enumerate(reply.images)
                  if index not in by_index]
        persisted = list(self.persist_images(session_id, others))
        result: list[dict[str, str]] = []
        cursor = 0
        for index, image in enumerate(reply.images):
            if index in by_index:
                result.append({"mime_type": str(image.get("mime_type")
                                                or "application/octet-stream"),
                               "url": by_index[index]})
            elif cursor < len(persisted):
                result.append(persisted[cursor])
                cursor += 1
        return result

    def persist_images(self, session_id: str,
                       images: list[dict[str, Any]]) -> list[dict[str, str]]:
        validate_session_id(session_id)
        result: list[dict[str, str]] = []
        for image in images:
            mime = str(image.get("mime_type") or "application/octet-stream").lower()
            encoded = image.get("data") or ""
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error):
                continue
            if not raw or len(raw) > MAX_IMAGE_BYTES:
                continue
            suffix = MIME_SUFFIX.get(mime, ".bin")
            name = f"{session_id}-{uuid4().hex[:16]}{suffix}"
            self.image_root.mkdir(parents=True, exist_ok=True)
            (self.image_root / name).write_bytes(raw)
            result.append({
                "mime_type": mime,
                "url": f"/api/v1/sessions/{session_id}/images/{name}",
            })
        return result

    def cache_image(self, session_id: str, image: dict[str, Any], *, key: str) -> str | None:
        """Persist a figure under a name derived from what it shows.

        The digest includes the semantic key and actual image bytes. Reopening
        unchanged content reuses its file, while a changed map layer gets a new
        URL instead of serving a previous district image forever. The session
        prefix keeps authorization and per-session deletion unchanged.
        """
        validate_session_id(session_id)
        mime = str(image.get("mime_type") or "application/octet-stream").lower()
        encoded = image.get("data") or ""
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            return None
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            return None
        suffix = MIME_SUFFIX.get(mime, ".bin")
        digest = hashlib.sha256(key.encode('utf-8') + b'\0' + raw).hexdigest()[:16]
        name = f"{session_id}-{digest}{suffix}"
        self.image_root.mkdir(parents=True, exist_ok=True)
        path = self.image_root / name
        if not path.is_file():
            path.write_bytes(raw)
        return f"/api/v1/sessions/{session_id}/images/{name}"

    def delete_images(self, session_id: str) -> int:
        """Remove only Web images that belong to one validated session."""
        identifier = validate_session_id(session_id)
        if not self.image_root.is_dir():
            return 0
        deleted = 0
        for path in self.image_root.iterdir():
            if (path.is_file() and IMAGE_NAME.fullmatch(path.name)
                    and path.name.startswith(identifier + "-")):
                path.unlink(missing_ok=True)
                deleted += 1
        return deleted


def create_app(*, settings: AgentSettings | None = None,
               agent: SlopeEvaluationAgent | None = None) -> FastAPI:
    """Create an app; injection keeps the Web adapter independently testable."""
    resolved = settings or load_settings()
    runtime = WebRuntime(agent or SlopeEvaluationAgent(resolved), resolved)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime
        await runtime.agent.open()
        try:
            yield
        finally:
            await runtime.agent.close()

    app = FastAPI(
        title="Chongqing Watershed Agent",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        llm = runtime.agent.settings.llm
        response_model = getattr(getattr(runtime.agent, "llm", None), "last_response_model", None)
        return {
            "ok": True,
            "version": __version__,
            "session_product_cleanup": True,
            "batch_session_product_cleanup": True,
            "adaptive_rainfall_stations": True,
            'concise_conversation': True,
            'grounded_result_reasoning': True,
            'readonly_artifact_investigation': True,
            "scope": "Preliminary regional screening and diagnosis of rainfall-driven slope stability",
            "llm": {
                "provider": llm.provider,
                "model": llm.model,
                "response_model": response_model,
                "api_format": llm.api_format,
                "reasoning_effort": llm.reasoning_effort,
                "deepseek_thinking": llm.deepseek_thinking if llm.provider == "deepseek" else None,
                "anthropic_thinking": llm.anthropic_thinking if llm.api_format == "anthropic_messages" else None,
                "anthropic_thinking_budget_tokens": (llm.anthropic_thinking_budget_tokens
                    if llm.api_format == "anthropic_messages" and llm.anthropic_thinking == "enabled" else None),
                "chat_streaming": llm.chat_streaming if llm.api_format == "chat_completions" else None,
                "qwen_thinking": llm.qwen_thinking if llm.provider == "qwen" else None,
                "qwen_thinking_budget_tokens": llm.qwen_thinking_budget_tokens if llm.provider == "qwen" else None,
                "gemini_thinking_budget_tokens": llm.gemini_thinking_budget_tokens if llm.provider == "gemini" else None,
                "configured": llm.is_available(),
                "status": "ready" if llm.is_available() else "deterministic_fallback",
                "reason": llm.unavailable_reason(),
            },
            "mcp": {
                "spatial_transport": resolved.mcp.spatial.transport,
                "hydro_transport": resolved.mcp.hydro.transport,
            },
        }

    from .model_connection import install_model_connection
    install_model_connection(app, runtime)

    @app.get("/api/v1/scope")
    async def scope() -> dict[str, str]:
        return {"text": await runtime.agent.scope_statement()}

    @app.post("/api/v1/sessions", status_code=201)
    async def create_session(body: SessionCreateRequest) -> dict[str, Any]:
        session = runtime.agent.new_session(language=body.language)
        return runtime.session_view(session, include_conversation=True)

    @app.get("/api/v1/sessions")
    async def list_sessions(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        """List persisted conversations, newest first, without loading artefacts."""
        return {"sessions": runtime.agent.store.list_sessions(limit=limit)}

    @app.get("/api/v1/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        try:
            session = runtime.agent.store.load(validate_session_id(session_id))
        except SessionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return runtime.session_view(session, include_conversation=True)

    async def remove_session(session_id: str) -> dict[str, Any]:
        """Clean managed products before removing the conversation and previews."""
        try:
            identifier = validate_session_id(session_id)
            async with runtime.lock(identifier, allow_cleanup=True):
                session = runtime.agent.store.load(identifier)
                cleanup = await runtime.agent.cleanup_session_products(session)
                runtime.delete_images(identifier)
                runtime.agent.store.delete(identifier)
            runtime.locks.pop(identifier, None)
        except HTTPException:
            raise
        except SessionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except WorkflowError as exc:
            raise HTTPException(status_code=409, detail=exc.user_message()) from exc
        except AgentMCPError as exc:
            raise HTTPException(status_code=409, detail=(
                "产物清理未完成，对话已保留，可重试删除。\n" + exc.user_message())) from exc
        except OSError as exc:
            raise HTTPException(status_code=409, detail="删除未完成，对话记录仍可重试；请检查文件占用。") from exc
        return {"deleted": identifier, "cleanup": cleanup}

    @app.delete("/api/v1/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict[str, Any]:
        async with runtime.cleanup_lock:
            return await remove_session(session_id)

    @app.post("/api/v1/sessions/batch-delete")
    async def batch_delete_sessions(body: SessionBatchDeleteRequest) -> dict[str, Any]:
        """Reuse per-session cleanup; failed sessions remain available for retry."""
        deleted, failed = [], []
        deleted_products, retained_products = set(), set()
        # Serial deletion lets the last selected reference release shared files.
        # It also prevents simultaneous deletion requests from retaining each
        # other's references after both conversations have disappeared.
        async with runtime.cleanup_lock:
            for identifier in body.session_ids:
                try:
                    result = await remove_session(identifier)
                except HTTPException as exc:
                    failed.append({"session_id": identifier, "status_code": exc.status_code,
                                   "detail": exc.detail})
                    continue
                deleted.append(identifier)
                deleted_products.update(result["cleanup"].get("deleted_ids") or [])
                retained_products.update(result["cleanup"].get("retained_ids") or [])
        return {"ok": not failed, "deleted": deleted, "failed": failed,
                "cleanup": {"deleted_ids": sorted(deleted_products),
                            "retained_ids": sorted(retained_products - deleted_products)}}

    @app.post("/api/v1/sessions/{session_id}/messages")
    async def message(session_id: str, body: MessageRequest) -> dict[str, Any]:
        try:
            identifier = validate_session_id(session_id)
            async with runtime.lock(identifier):
                session = runtime.agent.store.load(identifier)
                reply = await runtime.agent.handle(session, body.text)
        except SessionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "reply": runtime.reply_view(reply),
            "session": runtime.session_view(reply.session),
        }

    @app.get("/api/v1/sessions/{session_id}/administrative-areas")
    async def administrative_areas(session_id: str) -> dict[str, Any]:
        try:
            identifier=validate_session_id(session_id)
            async with runtime.lock(identifier):
                session=runtime.agent.store.load(identifier)
                return await runtime.agent.administrative_areas(session)
        except SessionError as exc:
            raise HTTPException(status_code=404,detail=str(exc)) from exc
        except (AgentMCPError,WorkflowError,ValueError) as exc:
            raise HTTPException(status_code=409,detail=str(exc)) from exc

    @app.post("/api/v1/sessions/{session_id}/study-area-map")
    async def study_area_map(session_id: str, body: StudyAreaMapRequest) -> dict[str, Any]:
        try:
            identifier=validate_session_id(session_id)
            async with runtime.lock(identifier):
                session=runtime.agent.store.load(identifier)
                record=(await runtime.agent.study_area_map(session,body.administrative_area)
                        if body.administrative_area else await runtime.agent.location_map(session,
                            river_name=body.river_name,longitude=body.longitude,latitude=body.latitude))
                map_payload=dict(record["map"])
                image=map_payload.pop("image_base64",None)
                map_payload["url"]=(runtime.cache_image(identifier,{"mime_type":"image/png","data":image},
                    key=f"study-area:{identifier}:{record.get('location_label')}") if image else None)
                if map_payload["url"]:
                    session.selection_context["map_image_name"]=map_payload["url"].rsplit("/",1)[-1]
                    runtime.agent.store.save(session)
                return {**record,"map":map_payload,"session":runtime.session_view(session)}
        except SessionError as exc:
            raise HTTPException(status_code=404,detail=str(exc)) from exc
        except (AgentMCPError,WorkflowError,ValueError) as exc:
            raise HTTPException(status_code=409,detail=str(exc)) from exc

    @app.post("/api/v1/sessions/{session_id}/study-area")
    async def submit_study_area(session_id: str, body: StudyAreaRequest) -> dict[str, Any]:
        try:
            identifier=validate_session_id(session_id)
            async with runtime.lock(identifier):
                session=runtime.agent.store.load(identifier)
                reply=await runtime.agent.submit_study_area(session,body.administrative_area,body.coordinates,map_id=body.map_id,assessment_mode=body.assessment_mode)
                return {"reply":runtime.reply_view(reply),"session":runtime.session_view(reply.session)}
        except SessionError as exc:
            raise HTTPException(status_code=404,detail=str(exc)) from exc
        except (AgentMCPError,WorkflowError,ValueError) as exc:
            raise HTTPException(status_code=409,detail=str(exc)) from exc

    @app.get("/api/v1/sessions/{session_id}/task")
    async def task_workspace(session_id:str):
        from .task_workspace import public_revisions
        try:
            identifier=validate_session_id(session_id)
            async with runtime.lock(identifier):
                session=runtime.agent.store.load(identifier)
                return {"session":runtime.session_view(session),
                        "revisions":public_revisions(runtime.agent.store,identifier)}
        except SessionError as exc:
            raise HTTPException(status_code=404,detail=str(exc)) from exc

    @app.post("/api/v1/sessions/{session_id}/task/edit")
    async def task_edit(session_id:str,body:TaskEditRequest):
        from .models import RainfallRequestPayload
        try:
            identifier=validate_session_id(session_id)
            async with runtime.lock(identifier):
                session=runtime.agent.store.load(identifier)
                reply=None
                if body.scope=="rainfall":
                    if not session.watershed_confirmed:
                        raise WorkflowError("请先确认通过检查的边界，再修改降雨事件。")
                    allowed=set(RainfallRequestPayload.model_fields)-{"review_id"}
                    if not body.rainfall or set(body.rainfall)-allowed:
                        raise ValueError("请提供有效的降雨模式、日期或时段。")
                    rainfall_fields = dict(body.rainfall)
                    if set(rainfall_fields) == {"station_count"}:
                        rainfall_fields = {**session.rainfall_request, **rainfall_fields}
                    request=RainfallRequestPayload.model_validate({**rainfall_fields,"review_id":session.review_id})
                    await runtime.agent.prepare_task_edit(session,"rainfall")
                    session.rainfall_request=request.model_dump(mode="json",exclude_none=True)
                elif body.scope=="parameters":
                    reply=await runtime.agent.set_parameter_overrides(session,
                        body.parameter_overrides if body.parameter_overrides is not None else session.parameter_overrides)
                    if reply.ok and body.parameter_preset:
                        session.parameter_preset=body.parameter_preset
                        reply=await runtime.agent._finish(session,"参数已修改，请重新生成并核对运行计划。",kind="question")
                else:
                    await runtime.agent.prepare_task_edit(session,body.scope)
                session.record_turn("user",{"location":"修改本任务的位置","boundary":"继续修改本任务的边界",
                    "rainfall":"修改本任务的降雨事件","parameters":"修改本任务的参数"}[body.scope])
                session.refresh_summary();runtime.agent.store.save(session)
                if reply is None:
                    reply=await runtime.agent._finish(session,"The previous revision was saved. Dependent workflow steps have been reset.\n"+session.pending_action(),kind="question")
                session.record_turn("agent",reply.text);runtime.agent.store.save(session)
                return {"reply":runtime.reply_view(reply),"session":runtime.session_view(session)}
        except SessionError as exc:
            raise HTTPException(status_code=404,detail=str(exc)) from exc
        except (AgentMCPError,WorkflowError,ValueError) as exc:
            raise HTTPException(status_code=409,detail=str(exc)) from exc

    @app.post("/api/v1/sessions/{session_id}/task/prepare")
    async def task_prepare(session_id:str):
        from .intent import ParsedIntent
        try:
            identifier=validate_session_id(session_id)
            async with runtime.lock(identifier):
                session=runtime.agent.store.load(identifier)
                await runtime.agent.ensure_task_editable(session)
                if not session.watershed_confirmed or not session.rainfall_request:
                    raise WorkflowError("请先确认边界并指定降雨日期或时段。")
                # Resume only missing intake stages. This operation never starts a run.
                steps={"watershed_confirmed":runtime.agent._build_rainfall,
                       "rainfall_ready":runtime.agent.prepare_model_inputs,
                       "model_inputs_ready":runtime.agent.import_model_inputs,
                       "hydro_input_imported":runtime.agent.validate_hydro_input}
                if session.state not in steps:
                    raise WorkflowError("当前任务不需要重新准备输入；请核对待办或生成新计划。")
                replies=[]
                session.record_turn("user","准备当前任务的降雨与模型输入，生成运行计划")
                while session.state in steps:
                    step=steps[session.state]
                    reply=await step(session,ParsedIntent()) if session.state=="watershed_confirmed" else await step(session)
                    replies.append(reply)
                    if not reply.ok:
                        break
                if replies[-1].ok:
                    reply=await runtime.agent.plan_hydro_run(session)
                    replies.append(reply)
                for item in replies:
                    session.record_turn("agent",item.text)
                runtime.agent.store.save(session)
                return {"reply":runtime.reply_view(reply),"session":runtime.session_view(session)}
        except SessionError as exc:
            raise HTTPException(status_code=404,detail=str(exc)) from exc
        except (AgentMCPError,WorkflowError,ValueError) as exc:
            raise HTTPException(status_code=409,detail=str(exc)) from exc

    @app.post("/api/v1/sessions/{session_id}/task/plan")
    async def task_plan(session_id:str):
        try:
            identifier=validate_session_id(session_id)
            async with runtime.lock(identifier):
                session=runtime.agent.store.load(identifier)
                reply=await runtime.agent.plan_hydro_run(session)
                session.record_turn("user","生成修改后的运行计划")
                session.record_turn("agent",reply.text);runtime.agent.store.save(session)
                return {"reply":runtime.reply_view(reply),"session":runtime.session_view(session)}
        except SessionError as exc:
            raise HTTPException(status_code=404,detail=str(exc)) from exc
        except (AgentMCPError,WorkflowError,ValueError) as exc:
            raise HTTPException(status_code=409,detail=str(exc)) from exc

    @app.post("/api/v1/sessions/{session_id}/task/restore")
    async def task_restore(session_id:str,body:TaskRestoreRequest):
        try:
            identifier=validate_session_id(session_id)
            async with runtime.lock(identifier):
                session=runtime.agent.store.load(identifier)
                reply=await runtime.agent.restore_task_revision(session,body.revision_id)
                session.record_turn("user","恢复本任务已保存的版本")
                session.record_turn("agent",reply.text);runtime.agent.store.save(session)
                return {"reply":runtime.reply_view(reply),"session":runtime.session_view(session)}
        except SessionError as exc:
            raise HTTPException(status_code=404,detail=str(exc)) from exc
        except (AgentMCPError,WorkflowError,ValueError) as exc:
            raise HTTPException(status_code=409,detail=str(exc)) from exc

    @app.put("/api/v1/sessions/{session_id}/task/draft")
    async def task_draft(session_id:str,body:TaskDraftRequest):
        try:
            identifier=validate_session_id(session_id)
            async with runtime.lock(identifier):
                session=runtime.agent.store.load(identifier)
                if body.map_id!=session.selection_context.get("map_id"):
                    raise WorkflowError("底图已改变，未保存过期地图上的草稿。")
                await runtime.agent.ensure_task_editable(session)
                bounds=session.selection_context.get("bbox_wgs84")
                if not bounds:
                    raise WorkflowError("请先加载可绘制的 DEM 地图。")
                w,s,e,n=bounds
                if any(not w<=x<=e or not s<=y<=n for x,y in body.coordinates):
                    raise ValueError("草稿顶点超出当前图幅。")
                session.study_area_draft=body.model_dump(mode="json",exclude_none=True)
                runtime.agent.store.save(session)
                return {"saved":True,"session":runtime.session_view(session)}
        except SessionError as exc:
            raise HTTPException(status_code=404,detail=str(exc)) from exc
        except (AgentMCPError,WorkflowError,ValueError) as exc:
            raise HTTPException(status_code=409,detail=str(exc)) from exc

    @app.post("/api/v1/sessions/{session_id}/outlet-candidates")
    async def outlet_candidates(session_id: str,
                                body: OutletCandidatesRequest) -> dict[str, Any]:
        """Recommended outlets for one river, plus a map the user can click.

        Generation goes through the agent, so the same workflow gate and session
        record apply as when the river is named in chat. Choosing a point here
        does not delineate anything: the chosen coordinates must still be sent as
        a message, which is what runs (and can be rejected by) delineation.
        """
        try:
            identifier = validate_session_id(session_id)
            async with runtime.lock(identifier):
                session = runtime.agent.store.load(identifier)
                record = await runtime.agent.outlet_candidates_map(
                    session, body.river_name)
        except SessionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (AgentMCPError, WorkflowError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        map_payload = dict(record.get("map") or {})
        url = None
        if map_payload.get("image_base64"):
            url = runtime.cache_image(
                identifier,
                {"mime_type": str(map_payload.get("media_type") or "image/png"),
                 "data": map_payload["image_base64"]},
                key=f"outlet-map:manual:{identifier}:{body.river_name}",
            )
        map_payload.pop("image_base64", None)
        map_payload["url"] = url
        return {
            "selection_mode":record.get("selection_mode"),
            "map_id":record.get("map_id"),"location_mode":record.get("location_mode"),
            "location_label":record.get("location_label"),
            "river_name": record.get("river_name") or body.river_name,
            "matched_names": list(record.get("matched_names") or []),
            "candidates": list(record.get("candidates") or []),
            "confidence": record.get("confidence"),
            "message": record.get("message"),
            "location_hint": record.get("location_hint"),
            "map": map_payload,
            "session": runtime.session_view(session),
        }

    @app.post("/api/v1/sessions/{session_id}/reconcile")
    async def reconcile(session_id: str) -> dict[str, Any]:
        try:
            identifier = validate_session_id(session_id)
            async with runtime.lock(identifier):
                session = runtime.agent.store.load(identifier)
                reply = await runtime.agent.reconcile(session)
                session = reply.session if reply is not None else runtime.agent.store.load(identifier)
        except SessionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "reply": runtime.reply_view(reply) if reply is not None else None,
            "session": runtime.session_view(session),
        }

    @app.post("/api/v1/sessions/{session_id}/cancel")
    async def cancel(session_id: str) -> dict[str, Any]:
        try:
            identifier = validate_session_id(session_id)
            async with runtime.lock(identifier):
                session = runtime.agent.store.load(identifier)
                reply = await runtime.agent.handle(session, "取消当前任务")
        except SessionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "reply": runtime.reply_view(reply),
            "session": runtime.session_view(reply.session),
        }

    @app.get("/api/v1/sessions/{session_id}/images/{name}")
    async def image(session_id: str, name: str):
        try:
            identifier = validate_session_id(session_id)
        except SessionError as exc:
            raise HTTPException(status_code=404, detail="图件不存在") from exc
        if not IMAGE_NAME.fullmatch(name) or not name.startswith(identifier + "-"):
            raise HTTPException(status_code=404, detail="图件不存在")
        path = runtime.image_root / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="图件不存在")
        return FileResponse(path)

    # ------------------------------------------------- restorable downloads
    @app.get("/api/v1/sessions/{session_id}/artifacts")
    async def session_artifacts(session_id: str) -> dict[str, Any]:
        """Everything this conversation may download, for a restored page.

        Figures come from the session's own record (ids and public descriptors);
        their previews are fetched from the service on demand. The SWMM model
        section is asked for once if the session does not have it yet — a run
        that finished before this release should not have to be re-run to be
        downloadable.
        """
        try:
            identifier = validate_session_id(session_id)
            async with runtime.lock(identifier):
                session = runtime.agent.store.load(identifier)
                figures = runtime.agent.visualizations(session)
                if session.hydro_run_id and not session.process_artifacts:
                    try:
                        await runtime.agent.refresh_process_artifacts(session)
                    except AgentMCPError as exc:
                        runtime.agent.logger.info("过程源文件清单不可用：%s", exc)
                if session.hydro_run_id and not (session.swmm_model or {}).get("artifacts"):
                    try:
                        await runtime.agent.refresh_swmm_model(session)
                    except AgentMCPError as exc:
                        # The run may be gone from the service; the figures are
                        # still downloadable, so this is a note, not a failure.
                        runtime.agent.logger.info("SWMM 建模成果清单不可用：%s", exc)
                model = session.swmm_model
        except SessionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "session_id": identifier,
            "visualizations": runtime.figure_views(identifier, figures, None),
            "process_artifacts": runtime.process_artifact_views(
                identifier, session.process_artifacts),
            "swmm": runtime.swmm_view(identifier, model),
        }

    @app.get("/api/v1/sessions/{session_id}/visualizations/{visualization_id}/image")
    async def visualization_image(session_id: str, visualization_id: str):
        """Re-render a figure's preview for a reloaded conversation.

        The image is fetched from the service again and cached under a name
        derived from the figure, so reloading a page does not accumulate copies.
        """
        try:
            identifier = validate_session_id(session_id)
            async with runtime.lock(identifier):
                session = runtime.agent.store.load(identifier)
                payload = await runtime.agent.visualization_image(session, visualization_id)
        except DownloadError as exc:
            raise HTTPException(status_code=exc.http_status, detail=str(exc)) from exc
        except SessionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        url = runtime.cache_image(identifier, payload, key=visualization_id)
        if url is None:
            raise HTTPException(status_code=502, detail="图件读取失败")
        path = runtime.image_root / url.rsplit("/", 1)[-1]
        return FileResponse(path, media_type=str(payload.get("mime_type") or "image/png"))

    @app.get("/api/v1/sessions/{session_id}/artifacts/{artifact_id}/download")
    async def download_artifact(session_id: str, artifact_id: str):
        """Stream one artifact this session is entitled to.

        The bytes are streamed, never assembled into a JSON body or a session
        field. Everything the browser is told — name, type, length, digest —
        comes from the agent's descriptor, which has no field for a path.
        """
        try:
            identifier = validate_session_id(session_id)
            async with runtime.lock(identifier):
                session = runtime.agent.store.load(identifier)
                download = await runtime.agent.download_artifact(session, artifact_id)
        except DownloadError as exc:
            raise HTTPException(status_code=exc.http_status, detail=str(exc)) from exc
        except AgentMCPError as exc:
            raise HTTPException(status_code=502, detail=exc.user_message()) from exc
        except SessionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        headers = {
            "Content-Disposition": content_disposition(download.filename),
            "Content-Length": str(download.size_bytes),
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, no-store",
        }
        if download.sha256:
            headers["ETag"] = f'"{download.sha256}"'
        if download.diagnostic_only:
            headers["X-Artifact-QA"] = "failed"
        return StreamingResponse(download.chunks, media_type=download.media_type,
                                 headers=headers)

    @app.post("/api/v1/sessions/{session_id}/artifacts/{artifact_id}/exports")
    async def export_artifact(session_id: str, artifact_id: str, body: ExportRequest):
        """Derive a viewable file (Excel) from one registered artifact."""
        try:
            identifier = validate_session_id(session_id)
            async with runtime.lock(identifier):
                session = runtime.agent.store.load(identifier)
                descriptor = await runtime.agent.prepare_artifact_export(
                    session, artifact_id, body.format)
                view = runtime.artifact_view(identifier, descriptor)
        except DownloadError as exc:
            raise HTTPException(status_code=exc.http_status, detail=str(exc)) from exc
        except AgentMCPError as exc:
            raise HTTPException(status_code=502, detail=exc.user_message()) from exc
        except SessionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"artifact": view, "source_artifact_id": artifact_id}

    @app.post("/api/v1/sessions/{session_id}/runs/{run_id}/swmm-bundle")
    async def swmm_bundle(session_id: str, run_id: str) -> dict[str, Any]:
        """Create — or reuse — the SWMM model package of this session's run.

        The run is taken from the session, and the path's ``run_id`` must be the
        same one: a bundle is never built for a run this conversation has not
        been working on.
        """
        try:
            identifier = validate_session_id(session_id)
            async with runtime.lock(identifier):
                session = runtime.agent.store.load(identifier)
                if not session.hydro_run_id or run_id != session.hydro_run_id:
                    raise HTTPException(status_code=404, detail="该运行不属于当前会话")
                descriptor = await runtime.agent.prepare_swmm_bundle(session)
                view = runtime.artifact_view(identifier, descriptor)
                swmm = runtime.swmm_view(identifier, session.swmm_model)
        except DownloadError as exc:
            raise HTTPException(status_code=exc.http_status, detail=str(exc)) from exc
        except AgentMCPError as exc:
            raise HTTPException(status_code=502, detail=exc.user_message()) from exc
        except SessionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"artifact": view, "swmm": swmm}

    app.mount("/assets", StaticFiles(directory=STATIC_ROOT), name="assets")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_ROOT / "index.html")

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="启动坡面稳定性评价智能体 WebUI")
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"])
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--config", help="智能体 YAML 配置文件")
    args = parser.parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit('缺少 uvicorn；请运行 pip install -e ".[web]"') from exc
    settings = load_settings(args.config)
    uvicorn.run(create_app(settings=settings), host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
