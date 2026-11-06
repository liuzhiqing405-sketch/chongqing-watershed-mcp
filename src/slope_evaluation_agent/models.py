"""Structured models exchanged with the MCP servers.

Everything the agent sends to a server is built from one of these models, never
from user text directly. That is what makes "user text cannot specify an
arbitrary tool or argument" true by construction: a field that is not declared
here cannot reach a server, and a value outside its bounds fails validation
before any call is made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


#: Stability parameters a caller may override, with the unit each carries.
#: The bounds themselves come from the server's ``get_hydro_capabilities``; the
#: agent checks against those and never invents its own limits.
OVERRIDABLE_PARAMETERS: dict[str, str] = {
    "deposit_thickness_m": "m",
    "natural_density_g_cm3": "g/cm³",
    "saturated_density_g_cm3": "g/cm³",
    "saturated_cohesion_kpa": "kPa",
    "saturated_friction_angle_deg": "°",
    "initial_saturation": "-",
    "friction_velocity_coefficient_a": "-",
    "runoff_density_g_cm3": "g/cm³",
    "calculation_resolution_m": "m",
}

#: The fixed pipeline the hydro service runs. A client may name one of these to
#: stop after; it can never name a script, an interpreter or a command.
WHITELIST_STAGES: tuple[str, ...] = (
    "sync", "validate", "delineate", "rainfall", "build",
    "rainfall_plot", "swmm", "safety", "analysis", "export", "qa",
)

ParameterPreset = Literal["low", "medium", "high"]
RunMode = Literal["preflight_only", "rainfall_only", "build_run_qa"]


class RainfallRequestPayload(BaseModel):
    """The ``build_virtual_rainfall`` request.

    Two modes:

    * ``historical_max`` — the wettest complete calendar day found within
      ``lookback_years`` ending at ``end_year``. This is a sample maximum over
      the data period, **not** a return-period design storm.
    * ``explicit_period`` — a user-given day or inclusive hourly window inside
      one calendar month.
    """

    review_id: str
    mode: Literal["historical_max", "explicit_period"]
    station_count: int | None = Field(default=None, ge=1, le=100, strict=True)
    lookback_years: int = Field(default=30, ge=1, le=100)
    end_year: int | None = Field(default=None, ge=1900, le=2200)
    date: str | None = None
    start_datetime: str | None = None
    end_datetime: str | None = None
    source_timezone: str = "Asia/Shanghai"

    @field_validator("date")
    @classmethod
    def _valid_date(cls, value):
        if value is not None:
            from datetime import date
            date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def _consistent_period(self) -> "RainfallRequestPayload":
        if self.mode == "historical_max":
            if self.date or self.start_datetime or self.end_datetime:
                raise ValueError("historical_max 不接受具体日期或时段")
            return self
        if self.date and (self.start_datetime or self.end_datetime):
            raise ValueError("date 与 start/end 不能同时给出")
        if not self.date and not (self.start_datetime and self.end_datetime):
            raise ValueError("explicit_period 需要 date，或同时给出开始与结束时间")
        if bool(self.start_datetime) != bool(self.end_datetime):
            raise ValueError("开始与结束时间必须同时给出")
        if self.start_datetime and self.end_datetime:
            from datetime import datetime

            start = datetime.fromisoformat(self.start_datetime)
            end = datetime.fromisoformat(self.end_datetime)
            if end < start:
                raise ValueError("结束时间不能早于开始时间")
            if (start.year, start.month) != (end.year, end.month):
                raise ValueError("单次事件必须位于同一自然月")
            if (end - start).total_seconds() // 3600 + 1 > 247:
                raise ValueError("事件超过雨站包支持的 247 个时次")
        return self


class HydroRunRequestPayload(BaseModel):
    """The ``plan_hydro_run`` / ``start_hydro_run`` request.

    ``parameter_overrides`` is empty unless the user explicitly asked for a
    change and the workflow confirmed the values against the server's reported
    bounds. The agent never fills it in on its own.
    """

    input_id: str
    mode: RunMode = "build_run_qa"
    through: str | None = None
    parameter_preset: ParameterPreset = "medium"
    parameter_overrides: dict[str, float] = Field(default_factory=dict)
    notes: str = ""

    @field_validator("through")
    @classmethod
    def _whitelisted_stage_only(cls, value: str | None) -> str | None:
        """The stop-after stage must be one the service actually runs.

        The service validates this too; checking here stops a free-text value —
        a script name, a shell fragment — before it ever reaches the wire.
        """
        if value is None:
            return None
        if value not in WHITELIST_STAGES:
            raise ValueError(f"未知阶段：{value}；允许值：{list(WHITELIST_STAGES)}")
        return value

    @field_validator("parameter_overrides")
    @classmethod
    def _known_parameters_only(cls, value: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(value) - set(OVERRIDABLE_PARAMETERS))
        if unknown:
            raise ValueError(f"不支持的参数：{unknown}；可覆盖的参数：{sorted(OVERRIDABLE_PARAMETERS)}")
        return {key: float(item) for key, item in value.items()}

    @field_validator("notes")
    @classmethod
    def _bounded_notes(cls, value: str) -> str:
        return value.strip()[:2000]


class VisualizationRequest(BaseModel):
    """The ``create_hydro_visualization`` request.

    ``kind`` is validated by the server against its own catalog; the agent only
    checks that it looks like a catalog name, never that it is one, so the
    server stays the single source of truth.
    """

    run_id: str
    kind: str = Field(min_length=3, max_length=64, pattern=r"^[a-z0-9_]+$")


class ArtifactFilter(BaseModel):
    run_id: str
    stage_id: str | None = Field(default=None, pattern=r"^[a-z0-9_]+$")


#: Fields of an artifact descriptor the agent declares. The service publishes
#: more than this; declaring the useful subset means a field the service adds
#: later is dropped on the way in rather than stored or forwarded. In particular
#: ``internal_path`` is not declared, so it cannot reach a session file or a
#: browser response even if a service ever published it.
class ArtifactDescriptor(BaseModel):
    """One downloadable file behind a figure or a run.

    ``formats`` lists what can be downloaded now (``original`` when the file is
    on disk); ``exportable`` lists formats that can still be produced on
    request. A caller offers a button for one and not for the other until it
    has been asked for.
    """

    model_config = ConfigDict(extra="ignore")

    artifact_id: str
    title: str | None = None
    role: str | None = None
    format: str = "binary"
    media_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    units: str | None = None
    crs: str | None = None
    qa_status: str | None = None
    stage_id: str | None = None
    ready: bool = False
    formats: list[str] = Field(default_factory=list)
    exportable: list[str] = Field(default_factory=list)
    producer: str | None = None
    derived_from_artifact_ids: list[str] = Field(default_factory=list)
    #: Which column a table's time axis lives in, when the service could tell.
    time_column: str | None = None
    rows: int | None = None
    column_count: int | None = None

    @model_validator(mode="after")
    def _normalise_legacy_metadata(self) -> "ArtifactDescriptor":
        """Repair metadata omitted or misclassified by older registries.

        The title names the original file the service will send, so its known
        suffix is authoritative for MIME. Units are filled only for files whose
        whole content is one named quantity; multi-column tables remain unset.
        """
        suffix = Path(str(self.title or "")).suffix.lower()
        known_media = {
            ".csv": "text/csv", ".tsv": "text/tab-separated-values",
            ".json": "application/json", ".geojson": "application/geo+json",
            ".txt": "text/plain", ".log": "text/plain", ".md": "text/markdown",
            ".inp": "text/plain", ".rpt": "text/plain",
            ".out": "application/octet-stream", ".tif": "image/tiff",
            ".tiff": "image/tiff", ".png": "image/png", ".pdf": "application/pdf",
            ".zip": "application/zip",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
        if suffix in known_media:
            self.media_type = known_media[suffix]
        if not self.units:
            name = Path(str(self.title or "")).name.lower()
            file_units = (
                ("minimum_safety_factor", "dimensionless"),
                ("maximum_flow_depth_m", "m"),
                ("maximum_velocity_ms", "m/s"),
                ("maximum_pore_water_pressure_kpa", "kPa"),
                ("unstable_duration_h", "h"),
                ("deposit_thickness_m", "m"),
                ("saturated_cohesion_kpa", "kPa"),
                ("saturated_friction_angle_deg", "degree"),
                ("calculation_resolution_m", "m"),
            )
            self.units = next((unit for token, unit in file_units if token in name), None)
        return self

    @property
    def diagnostic_only(self) -> bool:
        """True when this file may be looked at but not relied on.

        The quality gate's ``failed`` verdict is a property of the run, so a
        file carrying it is a diagnostic file — it can still be downloaded, and
        the interface must say what it is.
        """
        return (self.qa_status or "").lower() in {"failed", "qa_failed"}


class VisualizationResult(BaseModel):
    """A figure and the real files it was drawn from.

    The session stores this — ids and public metadata only — so a reopened
    conversation can offer the same downloads without keeping a single byte of
    the files themselves.
    """

    model_config = ConfigDict(extra="ignore")

    visualization_id: str
    kind: str | None = None
    title: str | None = None
    run_id: str | None = None
    status: str | None = None
    qa_failed: bool = False
    limitations: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactDescriptor] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    #: Where this figure's preview sits in the reply's ``images`` list, if it
    #: rendered one. The image bytes themselves are never copied in here.
    image_index: int | None = None

    def public_view(self) -> dict[str, Any]:
        """What may travel to a browser: no path, no image data."""
        return {
            "visualization_id": self.visualization_id,
            "kind": self.kind,
            "title": self.title,
            "run_id": self.run_id,
            "status": self.status,
            "qa_failed": self.qa_failed,
            "limitations": list(self.limitations),
            "warnings": list(self.warnings),
            "artifacts": [item.model_dump(mode="json") for item in self.artifacts],
        }


def drop_none(payload: BaseModel) -> dict[str, Any]:
    """Serialise for the wire: omit unset optional fields, keep explicit falses."""
    return payload.model_dump(mode="json", exclude_none=True)


# --------------------------------------------------------------- session state


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


class ConversationTurn(BaseModel):
    """One exchange, kept so pronouns and follow-ups stay resolvable."""

    role: Literal["user", "agent"]
    text: str
    at: str = Field(default_factory=_now)
    #: Tool calls this turn produced, for the event log and for debugging.
    tools: list[str] = Field(default_factory=list)
    #: Optional provenance; older transcripts simply have no source marker.
    answer_source: str | None = None

    @field_validator("text")
    @classmethod
    def _bounded(cls, value: str) -> str:
        return value.strip()


#: Bumped whenever the persisted shape changes, so an old file can be migrated.
SESSION_SCHEMA_VERSION = 4


#: What a session keeps from a run plan. The service returns more than this —
#: including ``engine.directory``, which names a place on this machine — and
#: none of it is needed: later steps address the run by id, and "start" replays
#: the preset and the overrides, not the engine's location. A whitelist is used
#: rather than saving the whole response and deleting fields afterwards,
#: because a field the service adds later must not be persisted by default.
PLAN_STAGE_FIELDS: tuple[str, ...] = ("stage_id", "title")
PLAN_INPUT_FIELDS: tuple[str, ...] = ("input_id", "origin", "event", "study_area")
PLAN_ENGINE_FIELDS: tuple[str, ...] = ("ok", "files")
PLAN_SCALAR_FIELDS: tuple[str, ...] = ("mode", "through", "input_id")


def _pick(source: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    return {key: source[key] for key in fields if isinstance(source, dict) and key in source}


def minimise_plan(plan: Any) -> dict[str, Any] | None:
    """Project a run plan onto the fields a session may persist."""
    if not isinstance(plan, dict):
        return None
    stages: list[Any] = []
    for item in plan.get("stages") or []:
        stages.append(_pick(item, PLAN_STAGE_FIELDS) if isinstance(item, dict) else item)
    minimised: dict[str, Any] = {
        "input": _pick(plan.get("input"), PLAN_INPUT_FIELDS),
        "stages": stages,
        "parameter_preset": plan.get("parameter_preset"),
        "parameter_overrides": dict(plan.get("parameter_overrides") or {}),
        "parameter_bounds": dict(plan.get("parameter_bounds") or {}),
        "engine": _pick(plan.get("engine"), PLAN_ENGINE_FIELDS),
    }
    for key in PLAN_SCALAR_FIELDS:
        if key in plan:
            minimised[key] = plan[key]
    return minimised


class AgentSession(BaseModel):
    """Everything the agent knows about one conversation.

    One directory per session under the configured root:

    ``state`` is validated on assignment, so a bad transition cannot be written
    silently — it fails at the point of the mistake rather than on the next load.
    Ids are always the values the MCP servers returned; the agent never mints
    one.
    """

    model_config = ConfigDict(validate_assignment=True)

    schema_version: int = SESSION_SCHEMA_VERSION
    session_id: str
    state: str
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    language: Literal["zh", "en"] = "zh"

    intent: str | None = None
    river_name: str | None = None
    administrative_area: str | None = None
    selection_mode: str | None = None
    assessment_mode: Literal["regional_screening","catchment_check"] = "regional_screening"
    selection_context: dict = Field(default_factory=dict)
    study_area_coordinates: list[list[float]] = Field(default_factory=list)
    study_area_draft: dict[str, Any] = Field(default_factory=dict)
    study_area_qa: dict[str, Any] = Field(default_factory=dict)
    rainfall_context: dict[str, Any] = Field(default_factory=dict)
    run_status_detail: dict[str, Any] = Field(default_factory=dict)
    requested_outlet: dict | None = None
    # A model's approximate map centre, kept separate from the user's chosen outlet.
    location_hint: dict | None = None
    rainfall_request: dict | None = None
    parameter_preset: Literal["low", "medium", "high"] = "medium"
    #: A deliberate parameter change, always validated against the bounds the
    #: hydro service reports before it reaches a run plan. Empty by default:
    #: the agent never fills this in on its own.
    parameter_overrides: dict[str, float] = Field(default_factory=dict)

    outlet_candidates: list[dict] = Field(default_factory=list)
    review_id: str | None = None
    watershed_confirmed: bool = False
    #: The delineation's own spatial QA verdict. A watershed that failed it must
    #: not be confirmable, so the agent refuses before asking the server.
    watershed_qa_passed: bool | None = None
    reviewer_note: str | None = None

    rainfall_package_id: str | None = None
    #: The exported package, by the id the spatial service minted. Its directory
    #: is a machine path and never enters the session; later steps address the
    #: package by this id alone.
    model_input_package_id: str | None = None
    hydro_input_id: str | None = None
    hydro_run_id: str | None = None
    hydro_run_status: str | None = None
    qa_passed: bool | None = None
    planned_input_signature: str | None = None
    run_input_signature: str | None = None

    # --- facts the workflow gates depend on -------------------------------
    #: Whether the imported input passed validation. ``start_hydro_run`` is
    #: unreachable until this is true.
    hydro_input_validated: bool = False
    #: Whether a run plan has been generated and shown. Starting is unreachable
    #: until this is true, so nothing runs that the user has not seen.
    hydro_planned: bool = False
    #: The exact plan that was shown, so "start" runs what was described.
    last_plan: dict | None = None
    #: The run mode the agent requested. The plan payload does not carry it, so
    #: it is recorded here at plan time — without it the assessment cannot say
    #: which pipeline produced its numbers.
    run_mode: str | None = None
    #: Figures created for the current run: the id the server minted plus the
    #: public descriptors of the files behind them. No path, no image bytes —
    #: a resumed session can offer its downloads without holding any file.
    visualizations: list[dict] = Field(default_factory=list)
    #: Every registered source/result file for the current run, as a public
    #: descriptor. This is broader than the files associated with one figure:
    #: it lets a user request any process artifact in its original format.
    #: Paths and bytes never enter the session.
    process_artifacts: list[dict] = Field(default_factory=list)
    #: The run's SWMM model files (INP / RPT / OUT / review tables) as the
    #: service published them — public descriptors only — plus the model package
    #: once one has been prepared. Kept so a reopened conversation can offer the
    #: same downloads without holding any file.
    swmm_model: dict | None = None

    pending_question: str | None = None
    #: Unfulfilled task inputs survive intervening knowledge questions. Values
    #: are validated user inputs, not permissions to advance the workflow.
    pending_workflow: dict[str, Any] = Field(default_factory=dict)
    last_error: dict | None = None
    conversation: list[ConversationTurn] = Field(default_factory=list)

    # --- conversation routing (V2) ----------------------------------------
    #: Which route this turn took: ``workflow_action``, ``clarification``,
    #: ``knowledge_chat`` or ``out_of_scope``. Kept apart from ``intent`` so a
    #: chat turn cannot overwrite the last workflow intent the session holds.
    last_route: str | None = None
    #: The single question the agent asked last. The next turn is read as an
    #: answer to it — "历史最大的那个" is a rainfall answer, not a new request.
    last_clarification: str | None = None
    #: A rolling summary of the conversation. Text only: no path, key or id.
    conversation_summary: str | None = None

    @field_validator("state")
    @classmethod
    def _known_state(cls, value: str) -> str:
        from .state_machine import is_known_state

        if not is_known_state(value):
            raise ValueError(f"未知会话状态：{value}")
        return value

    @field_validator("session_id")
    @classmethod
    def _valid_session_id(cls, value: str) -> str:
        if not value.startswith("session-") or len(value) < 10:
            raise ValueError("会话编号必须以 session- 开头")
        return value

    # ------------------------------------------------------------------ facts
    def record_turn(self, role: str, text: str, *, tools: list[str] | None = None,
                    answer_source: str | None = None,
                    maximum_turns: int = 20, maximum_chars: int = 2000) -> None:
        """Append a turn and trim the oldest ones.

        Only the text is trimmed. Every structured field above is kept in full,
        so a resumed session still knows its ids regardless of how long the
        conversation ran.
        """
        self.conversation.append(ConversationTurn(role=role, text=text[:maximum_chars],
                                                  tools=list(tools or []), answer_source=answer_source))
        if len(self.conversation) > maximum_turns:
            self.conversation = self.conversation[-maximum_turns:]
        self.updated_at = _now()

    def recent_conversation(self, turns: int = 8) -> list[ConversationTurn]:
        return self.conversation[-turns:]

    def pending_action(self) -> str:
        from .state_machine import pending_action

        if self.selection_mode=="drawn_study_area":
            if self.state in {"new","needs_location","outlet_candidates_ready","watershed_rejected"}:
                return "在定位后的 DEM 地图上绘制闭合评价边界，并提交覆盖检查。"
            if self.state=="watershed_qa_failed":
                return "评价范围覆盖或水文检查未通过，请调整边界或补齐数据；当前范围不能确认或计算。"
            if self.state=="awaiting_watershed_confirmation":
                return "核对绘制边界、覆盖和水文合理性检查后，确认、拒绝或继续修改边界。"
        if (self.state == "awaiting_watershed_confirmation" and self.watershed_qa_passed is False):
            return "流域质量检查未通过，请在地图河道上重新选出水口；当前结果不能确认或用于计算。"
        return self.pending_question or pending_action(self.state)

    def refresh_summary(self) -> str:
        """A one-line digest of what this session has established.

        Recomputed from the fields rather than appended to, so it cannot drift
        from what it describes, and it holds only opaque ids, names and dates —
        never a path, a key or a transcript.
        """
        parts: list[str] = []
        if self.administrative_area and self.selection_mode=="drawn_study_area":
            parts.append(f"评价区域 {self.administrative_area}（用户绘制边界）")
        if self.river_name:
            parts.append(f"河流 {self.river_name}")
        if self.selection_context.get("location_mode")=="coordinates":
            parts.append("经纬度定位（用户绘制评价范围）")
        if self.review_id:
            parts.append(f"流域审核 {self.review_id}")
        if self.watershed_confirmed:
            parts.append("边界已确认")
        request = self.rainfall_request or {}
        if request.get("mode") == "historical_max":
            parts.append("降雨用资料期内样本最大日")
        elif request.get("date"):
            parts.append(f"降雨日期 {request['date']}")
        elif request.get("start_datetime"):
            parts.append(f"降雨时段 {request['start_datetime']} 起")
        if self.hydro_run_id:
            parts.append(f"任务 {self.hydro_run_id}（{self.hydro_run_status or '未知'}）")
        self.conversation_summary = "；".join(parts) or None
        return self.conversation_summary or ""

    def clear_error(self) -> None:
        self.last_error = None

    def note_error(self, *, code: str, message: str, next_step: str = "") -> None:
        self.last_error = {"code": code, "message": message, "next_step": next_step,
                           "at": _now()}

    def has_pending_watershed(self) -> bool:
        """True when exactly one delineated watershed is waiting for a decision.

        A "confirm" with nothing pending, or with a watershed already decided,
        must not misfire.
        """
        from .state_machine import WorkflowState

        return (self.state == WorkflowState.AWAITING_WATERSHED_CONFIRMATION.value
                and bool(self.review_id)
                and not self.watershed_confirmed)

    def reset_for_new_run(self) -> None:
        """Clear the per-run facts before planning a new run in the same session."""
        self.hydro_planned = False
        self.last_plan = None
        self.run_mode = None
        self.hydro_run_id = None
        self.hydro_run_status = None
        self.planned_input_signature = None
        self.run_input_signature = None
        self.qa_passed = None
        self.visualizations = []
        self.process_artifacts = []
        self.swmm_model = None
        self.clear_error()
