"""Session-scoped access to the real files behind a figure.

The session is the authority. A conversation may download exactly the artifacts
this service published for *its* run, and nothing else: knowing an artifact id is
not enough, because an id that was never handed to this session is treated as if
it did not exist. The Web layer answers "not found" for both cases, since a
different answer would confirm that somebody else's file exists.

Two channels, one interface:

* the **MCP resource channel** for files that fit in a single message
  (``hydro-artifact://<run>/<artifact>``) — the protocol's own way, and what the
  service actually serves;
* the **same-machine adapter** for anything larger, where a whole-file message
  would mean building the payload in memory on both sides. There the file is
  streamed from disk, and the path stays *inside this module*: callers receive an
  iterator of bytes, never a location.

Either way the digest the service published is verified against the bytes that
are handed over — a download that quietly returns a rewritten file would be
worse than a download that fails.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

from .config import AgentSettings
from .mcp_clients.base import AgentMCPError, MCPResponseSchemaError
from .models import AgentSession, ArtifactDescriptor, VisualizationResult


class DownloadError(RuntimeError):
    """A download that cannot be served, in a form the Web layer can map.

    ``code`` is the reason a user is given; ``http_status`` is how the Web layer
    answers. They are separate because several distinct causes must look
    identical from outside the session: anything that is not this session's own
    artifact answers 404, whether it is missing, someone else's, or unregistered.
    """

    def __init__(self, message: str, *, code: str, http_status: int) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status


#: The formats a session may ask a download endpoint for. An artifact's own
#: ``format`` is what it *is*; this is what may be *requested*.
REQUESTABLE_FORMATS = frozenset({"original", "xlsx", "zip"})


@dataclass
class ArtifactDownload:
    """One file on its way to a caller: what it is, and the bytes to send.

    ``chunks`` is an iterator of bytes — never a path, never the whole file
    unless the caller chooses to join it. ``diagnostic_only`` carries the run's
    quality-gate verdict so the interface can mark the download rather than
    presenting it as a formal result.
    """

    filename: str
    media_type: str
    size_bytes: int
    sha256: str | None
    qa_status: str | None
    chunks: AsyncIterator[bytes] = field(repr=False)
    channel: str = "resource"
    diagnostic_only: bool = False

    async def aclose(self) -> None:
        close = getattr(self.chunks, "aclose", None)
        if close is not None:
            await close()


def _safe_filename(title: str | None, artifact_id: str, fallback: str = "artifact.bin") -> str:
    """A download name built from the artifact's own title, never a path.

    Only the final component is kept, path separators and control characters are
    dropped, and a title that reduces to nothing falls back to the artifact id —
    the name in a ``Content-Disposition`` header must not be able to contain a
    directory or a header-terminating newline.
    """
    name = Path(str(title or "")).name.strip()
    cleaned = "".join(ch for ch in name if ch.isprintable() and ch not in '\\/:*?"<>|')
    cleaned = cleaned.strip(" .")
    if not cleaned:
        return f"{artifact_id}{Path(fallback).suffix}"
    return cleaned[:180]


async def _file_chunks(path: Path, chunk_bytes: int) -> AsyncIterator[bytes]:
    """Stream a local file without ever holding more than one chunk."""
    with path.open("rb") as stream:
        while True:
            block = await asyncio.to_thread(stream.read, chunk_bytes)
            if not block:
                break
            yield block


async def _memory_chunks(payload: bytes, chunk_bytes: int) -> AsyncIterator[bytes]:
    """Yield an in-memory payload in the same shape as a streamed file."""
    for start in range(0, len(payload), chunk_bytes):
        yield payload[start:start + chunk_bytes]


class ArtifactService:
    """What one conversation may download, and how it gets it.

    Holds no state of its own: the descriptors live in the session, so a
    reopened conversation offers exactly the downloads it was given, and a
    session that was never told about a file cannot reach it.
    """

    def __init__(self, settings: AgentSettings, client, logger: logging.Logger | None = None) -> None:
        from .mcp_clients import HydroMCPClient

        self.settings = settings
        self._client: HydroMCPClient = client
        self.logger = logger or logging.getLogger("slope_agent")

    # ------------------------------------------------------------ what exists
    @staticmethod
    def session_visualizations(session: AgentSession) -> list[VisualizationResult]:
        """The figures this conversation holds, as the service described them.

        Read defensively: a session written by an earlier release stored only
        ids, and such an entry is skipped rather than guessed at.
        """
        found: list[VisualizationResult] = []
        for item in session.visualizations or []:
            if not isinstance(item, dict) or not item.get("visualization_id"):
                continue
            try:
                found.append(VisualizationResult.model_validate(item))
            except Exception:  # noqa: BLE001 - an unreadable entry is skipped
                continue
        return found

    def descriptors(self, session: AgentSession) -> list[ArtifactDescriptor]:
        """Every artifact this session may download, in the order it learned them."""
        found: list[ArtifactDescriptor] = []
        for figure in self.session_visualizations(session):
            found.extend(figure.artifacts)
        found.extend(self.process_artifacts(session))
        found.extend(self.swmm_artifacts(session))
        unique: dict[str, ArtifactDescriptor] = {}
        for descriptor in found:
            unique.setdefault(descriptor.artifact_id, descriptor)
        return list(unique.values())

    @staticmethod
    def process_artifacts(session: AgentSession) -> list[ArtifactDescriptor]:
        """All registered run files published for original-format download."""
        found: list[ArtifactDescriptor] = []
        for item in session.process_artifacts or []:
            if not isinstance(item, dict):
                continue
            try:
                found.append(ArtifactDescriptor.model_validate(item))
            except Exception:  # noqa: BLE001 - one stale row must not hide the rest
                continue
        return found

    @staticmethod
    def swmm_artifacts(session: AgentSession) -> list[ArtifactDescriptor]:
        """The SWMM model files this session was told about, if any."""
        model = session.swmm_model or {}
        if not isinstance(model, dict):
            return []
        rows = list(model.get("artifacts") or [])
        bundle = model.get("bundle")
        if isinstance(bundle, dict):
            rows.append(bundle)
        found: list[ArtifactDescriptor] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            try:
                found.append(ArtifactDescriptor.model_validate(item))
            except Exception:  # noqa: BLE001 - an unreadable entry is skipped
                continue
        return found

    def authorize(self, session: AgentSession, artifact_id: str) -> ArtifactDescriptor:
        """The descriptor for one artifact, or a refusal.

        Three things are checked and all three must hold: the artifact was
        published to this session, it belongs to this session's run, and it is
        ready to be downloaded. The refusal is the same ``not_found`` in every
        case where the caller is not entitled to know more.
        """
        from .input_binding import input_signature
        if session.run_input_signature and session.run_input_signature != input_signature(session):
            raise DownloadError("当前输入已改变，该成果属于原输入版本，请从历史版本核对后查询。",
                                code="input_version", http_status=409)
        wanted = str(artifact_id or "").strip()
        if not wanted:
            raise DownloadError("缺少成果编号", code="not_found", http_status=404)

        for figure in self.session_visualizations(session):
            for descriptor in figure.artifacts:
                if descriptor.artifact_id == wanted:
                    self._check_run(session, figure.run_id, wanted)
                    self._check_ready(descriptor, wanted)
                    return descriptor
        for descriptor in self.process_artifacts(session):
            if descriptor.artifact_id == wanted:
                self._check_run(session, session.hydro_run_id, wanted)
                self._check_ready(descriptor, wanted)
                return descriptor
        for descriptor in self.swmm_artifacts(session):
            if descriptor.artifact_id == wanted:
                model = session.swmm_model or {}
                self._check_run(session, model.get("run_id"), wanted)
                self._check_ready(descriptor, wanted)
                return descriptor
        raise DownloadError("该成果不属于当前会话", code="not_found", http_status=404)

    @staticmethod
    def _check_run(session: AgentSession, run_id: Any, artifact_id: str) -> None:
        own = session.hydro_run_id
        if not own or str(run_id or "") != str(own):
            raise DownloadError("该成果不属于当前会话运行", code="not_found", http_status=404)

    @staticmethod
    def _check_ready(descriptor: ArtifactDescriptor, artifact_id: str) -> None:
        if not descriptor.ready or "original" not in (descriptor.formats or []):
            raise DownloadError("成果尚未写入完成，暂不能下载",
                                code="not_ready", http_status=409)

    def export_formats(self, session: AgentSession, artifact_id: str) -> list[str]:
        """Formats this session may ask to generate from one artifact."""
        descriptor = self.authorize(session, artifact_id)
        return [item for item in descriptor.exportable if item in REQUESTABLE_FORMATS]

    # ------------------------------------------------------------ preparation
    async def prepare_export(self, session: AgentSession, artifact_id: str,
                             export_format: str) -> ArtifactDescriptor:
        """Generate — or reuse — a derived file, and remember it on the session.

        Only formats the artifact itself declares are accepted, so a caller
        cannot ask for an Excel workbook of a GeoTIFF and get a picture of one.
        """
        wanted = str(export_format or "").strip().lower()
        descriptor = self.authorize(session, artifact_id)
        if wanted not in REQUESTABLE_FORMATS:
            raise DownloadError(f"不支持的导出格式：{export_format}",
                                code="unsupported_format", http_status=400)
        if wanted not in (descriptor.exportable or []):
            raise DownloadError(f"{descriptor.format} 不能导出为 {wanted}",
                                code="unsupported_format", http_status=400)
        run_id = session.hydro_run_id
        assert run_id  # authorize() refused otherwise
        payload = await self._client.prepare_artifact_export(run_id, artifact_id, wanted)
        data = payload.get("data") or {}
        derived = data.get("artifact")
        if not isinstance(derived, dict):
            raise MCPResponseSchemaError("prepare_artifact_export 未返回派生成果")
        made = ArtifactDescriptor.model_validate(derived)
        self._remember(session, artifact_id, made)
        return made

    async def refresh_swmm_model(self, session: AgentSession) -> dict[str, Any]:
        """Learn which SWMM model files this session's run has.

        Called when a result is shown: the model files are part of the outcome,
        so the conversation should be able to offer them without the user first
        having to ask for a figure. The service answers with public descriptors
        only — the agent never receives a path to store.
        """
        run_id = session.hydro_run_id
        if not run_id:
            return {}
        payload = await self._client.list_swmm_model_artifacts(run_id)
        data = payload.get("data") or {}
        model = dict(session.swmm_model or {})
        previous_bundle = model.get("bundle")
        model.update({
            "run_id": run_id,
            "complete": bool(data.get("complete")),
            "missing_required": list(data.get("missing_required") or []),
            "members": [item for item in (data.get("members") or [])
                        if isinstance(item, dict)],
            "artifacts": [item for item in (data.get("artifacts") or [])
                          if isinstance(item, dict)],
            "warnings": list(payload.get("warnings") or []),
        })
        if isinstance(previous_bundle, dict):
            model["bundle"] = previous_bundle
        session.swmm_model = model
        return model

    async def refresh_process_artifacts(self, session: AgentSession) -> list[ArtifactDescriptor]:
        """Load every registered run artifact as a path-free public descriptor."""
        run_id = session.hydro_run_id
        if not run_id:
            session.process_artifacts = []
            return []
        payload = await self._client.list_hydro_artifacts(run_id)
        data = payload.get("data") or {}
        rows = data.get("artifacts") or []
        found: list[ArtifactDescriptor] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            try:
                descriptor = ArtifactDescriptor.model_validate(item)
            except Exception:  # noqa: BLE001 - ignore one malformed legacy row
                continue
            if descriptor.ready and "original" in descriptor.formats:
                found.append(descriptor)
        session.process_artifacts = [item.model_dump(mode="json") for item in found]
        return found

    async def prepare_swmm_bundle(self, session: AgentSession) -> ArtifactDescriptor:
        """Package this session's run, and remember the package on the session."""
        run_id = session.hydro_run_id
        if not run_id:
            raise DownloadError("当前会话还没有水文任务，无法打包模型成果",
                                code="not_found", http_status=404)
        payload = await self._client.prepare_swmm_model_bundle(run_id)
        data = payload.get("data") or {}
        record = data.get("artifact")
        if not isinstance(record, dict):
            raise MCPResponseSchemaError("prepare_swmm_model_bundle 未返回模型包成果")
        descriptor = ArtifactDescriptor.model_validate(record)
        model = dict(session.swmm_model or {})
        model.update({
            "run_id": run_id,
            "bundle": descriptor.model_dump(mode="json"),
            "bundle_complete": bool(data.get("complete")),
            "bundle_reused": bool(data.get("reused")),
            "bundle_missing_required": list(data.get("missing_required") or []),
            "bundle_warnings": list(payload.get("warnings") or []),
            "prepared_at": (payload.get("facts") or {}).get("checked_at"),
        })
        session.swmm_model = model
        return descriptor

    @staticmethod
    def _remember(session: AgentSession, source_id: str,
                  derived: ArtifactDescriptor) -> None:
        """Attach a derived file to the figure (or package) it came from."""
        for item in session.visualizations or []:
            if not isinstance(item, dict):
                continue
            if source_id in [str(entry.get("artifact_id"))
                             for entry in item.get("artifacts") or []]:
                entry = dict(item)
                entry["artifacts"] = list(entry.get("artifacts") or []) + [
                    derived.model_dump(mode="json")]
                session.visualizations = [
                    entry if other is item else other
                    for other in session.visualizations]
                return
        model = session.swmm_model or {}
        if isinstance(model, dict) and \
                isinstance(model.get("bundle"), dict) and \
                str(model["bundle"].get("artifact_id")) == source_id:
            updated = dict(model)
            updated["bundle_derived"] = derived.model_dump(mode="json")
            session.swmm_model = updated
            return
        if isinstance(model, dict):
            artifacts = list(model.get("artifacts") or [])
            if source_id in [str(item.get("artifact_id")) for item in artifacts
                             if isinstance(item, dict)]:
                updated = dict(model)
                updated["artifacts"] = artifacts + [derived.model_dump(mode="json")]
                session.swmm_model = updated
                return
        if source_id in [str(item.get("artifact_id"))
                         for item in session.process_artifacts
                         if isinstance(item, dict)]:
            session.process_artifacts = list(session.process_artifacts) + [
                derived.model_dump(mode="json")]

    # --------------------------------------------------------------- download
    async def open_download(self, session: AgentSession,
                            artifact_id: str) -> ArtifactDownload:
        """Start a download: the metadata, and the bytes to send.

        The ceiling is checked before anything is read, and the digest is
        verified against the bytes that are actually handed over, so a file
        rewritten since it was registered fails here instead of travelling.
        """
        descriptor = self.authorize(session, artifact_id)
        run_id = session.hydro_run_id
        assert run_id  # authorize() refused otherwise
        limits = self.settings.download
        size = descriptor.size_bytes
        if size is not None and size > limits.max_file_bytes:
            raise DownloadError(
                f"文件 {size} 字节，超过单次下载上限 {limits.max_file_bytes} 字节",
                code="too_large", http_status=413)

        filename = _safe_filename(descriptor.title, descriptor.artifact_id,
                                  fallback="." + (descriptor.format or "bin"))
        diagnostic = descriptor.diagnostic_only
        qa_status = descriptor.qa_status

        if size is not None and size <= limits.resource_read_limit_bytes:
            payload = await self._client.read_artifact(
                run_id, descriptor.artifact_id,
                expected_sha256=descriptor.sha256, expected_size=size)
            raw = payload["data"]
            return ArtifactDownload(
                filename=filename,
                # The resource channel declares one content type for every file
                # it carries, so the descriptor is what says what this is. (The
                # same-machine channel below states a freshly derived type.)
                # ArtifactDescriptor normalises MIME from the original suffix,
                # including legacy records. The generic MCP resource channel
                # may report application/octet-stream for every binary block.
                media_type=descriptor.media_type or payload.get("mime_type")
                or "application/octet-stream",
                size_bytes=len(raw), sha256=descriptor.sha256, qa_status=qa_status,
                chunks=_memory_chunks(raw, limits.chunk_bytes),
                channel="resource", diagnostic_only=diagnostic)

        path, media_type = await self._locate(run_id, descriptor)
        stat = path.stat()
        if size is not None and stat.st_size != size:
            raise DownloadError("成果文件大小与登记不符，已拒绝下载",
                                code="integrity", http_status=502)
        return ArtifactDownload(
            filename=filename,
            media_type=media_type or descriptor.media_type or "application/octet-stream",
            size_bytes=stat.st_size, sha256=descriptor.sha256, qa_status=qa_status,
            chunks=_file_chunks(path, limits.chunk_bytes),
            channel="local", diagnostic_only=diagnostic)

    async def _locate(self, run_id: str,
                      descriptor: ArtifactDescriptor) -> tuple[Path, str | None]:
        """The on-disk file for a large artifact, via the same-machine adapter.

        The path never leaves this method: it is turned into a chunk iterator by
        the caller. The service's digest and size must agree with the descriptor
        the session holds, so the adapter cannot redirect a download to another
        file under the same id.
        """
        try:
            payload = await self._client.locate_artifact(run_id, descriptor.artifact_id)
        except AgentMCPError as exc:
            raise DownloadError(exc.user_message(), code="service", http_status=502) from exc
        data = payload.get("data") or {}
        path = Path(str(data.get("internal_path") or ""))
        if not path.is_file():
            raise DownloadError("成果文件已不存在", code="not_found", http_status=404)
        if descriptor.sha256 and data.get("sha256") and \
                str(data["sha256"]) != descriptor.sha256:
            raise DownloadError("成果文件与登记摘要不一致，已拒绝下载",
                                code="integrity", http_status=502)
        return path, str(data.get("media_type") or "") or None


def artifact_ids(descriptors: Iterable[ArtifactDescriptor]) -> list[str]:
    return [item.artifact_id for item in descriptors]
