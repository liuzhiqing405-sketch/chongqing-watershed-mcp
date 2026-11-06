"""Session persistence.

One directory per session under the configured root:

    session.json           structured state, written atomically
    events.jsonl           append-only record of tool calls and transitions
    evidence.json          the evidence object behind the current assessment
    final_assessment.json  the assessment itself

Sessions live under ``runs/`` and are never committed. Nothing here writes an
API key: :class:`AgentSession` has no field for one, and the event writer drops
credential-shaped keys before a record is appended.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .config import SessionSettings, load_settings
from .models import AgentSession
from .state_machine import WorkflowState

SESSION_ID_PREFIX = "session-"
SESSION_ID_PATTERN = re.compile(r"^session-[0-9a-f]{12}$")

#: Keys whose values never belong in an event record.
CREDENTIAL_KEYS = frozenset({
    "api_key", "apikey", "secret", "password", "token", "authorization",
    "access_token", "refresh_token", "key",
})


class SessionError(RuntimeError):
    """A session could not be created, loaded or written."""

    next_step = "请用 new-session 建立新的会话。"

    def user_message(self) -> str:
        return f"{self}\n下一步：{self.next_step}"


def new_session_id() -> str:
    return f"{SESSION_ID_PREFIX}{uuid4().hex[:12]}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_session_id(value: str) -> str:
    if not isinstance(value, str) or not SESSION_ID_PATTERN.fullmatch(value):
        raise SessionError(f"会话编号格式无效：{value!r}")
    return value


class EventLog:
    """Append-only ``events.jsonl`` for one session."""

    def __init__(self, path: Path, *, maximum_payload_chars: int = 4000) -> None:
        self.path = path
        self.maximum_payload_chars = maximum_payload_chars

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {"at": utc_now(), "event": event, **_scrub(fields)}
        encoded = json.dumps(record, ensure_ascii=False)
        if len(encoded) > self.maximum_payload_chars:
            record = {"at": record["at"], "event": event,
                      "truncated": True, "note": "记录过长已截断"}
            encoded = json.dumps(record, ensure_ascii=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
        return record

    def read(self, limit: int = 200) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows[-limit:]


_AUTHORIZATION_VALUE = re.compile(
    r"(?i)\b(?:authorization\s*:\s*)?bearer\s+[^\s,;]+"
    r"|\b(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*[^\s,;]+")
_OPENAI_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b")


def _scrub(value: Any) -> Any:
    """Remove credential-shaped keys and values before persistence or logging."""
    if isinstance(value, dict):
        return {key: ("***" if str(key).lower() in CREDENTIAL_KEYS else _scrub(item))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(item) for item in value]
    if isinstance(value, str):
        from .response_writer import scrub_paths

        cleaned = _AUTHORIZATION_VALUE.sub("***REDACTED***", value)
        cleaned = _OPENAI_KEY.sub("***REDACTED***", cleaned)
        cleaned = scrub_paths(cleaned)
        if len(cleaned) > 4000:
            return cleaned[:4000] + "…"
        return cleaned
    return value


def sanitize_user_text(text: str) -> str:
    """Text safe to retain or send to the optional intent model."""
    cleaned = _scrub(str(text))
    return cleaned if isinstance(cleaned, str) else ""


#: Windows refuses ``os.replace`` onto a file another process currently has
#: open for reading (WinError 5 or 32). ``status`` and ``inspect`` read session
#: files while the agent may be writing them, so the swap is retried rather
#: than allowed to fail on a momentary overlap.
_REPLACE_ATTEMPTS = 40
_REPLACE_DELAY_S = 0.05


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write ``payload`` so a reader never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(_REPLACE_DELAY_S)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


#: A string that names a place on this machine. Only strings that *are* a path,
#: or that embed one, are touched — coordinates, negative numbers and ordinary
#: scientific text are left exactly as they are.
_ABSOLUTE_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/])"            # C:\ or C:/
    r"|(?:\\\\[^\\/\s]+[\\/])"       # \\server\share
    r"|(?:/(?:home|usr|opt|var|etc|mnt|tmp)/)"  # POSIX root
)


def _strip_paths(value: Any) -> Any:
    """Drop path-shaped strings; scrub a path embedded in longer text."""
    if isinstance(value, str):
        if not _ABSOLUTE_PATH.search(value):
            return value
        if _ABSOLUTE_PATH.match(value):
            # The whole value was a path; there is nothing left worth keeping.
            return None
        from .response_writer import scrub_paths

        return scrub_paths(value)
    if isinstance(value, dict):
        return {key: _strip_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_paths(item) for item in value]
    return value


def migrate_session_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Bring an old session file up to the current schema.

    Unknown fields are dropped rather than carried, because a field the current
    model does not know about cannot be validated — and an old
    ``model_input_package_ref`` held a machine directory. Returns the payload and
    whether it changed.
    """
    from .models import SESSION_SCHEMA_VERSION, AgentSession

    if not isinstance(payload, dict):
        raise SessionError("会话文件不是 JSON 对象")
    if payload.get("schema_version") == SESSION_SCHEMA_VERSION:
        cleaned = _strip_paths(_scrub(payload))
        return cleaned, cleaned != payload
    known = set(AgentSession.model_fields)
    migrated = {key: _strip_paths(_scrub(value))
                for key, value in payload.items() if key in known}
    migrated["schema_version"] = SESSION_SCHEMA_VERSION
    return migrated, True


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        if default is not None:
            return default
        raise SessionError(f"文件不存在：{path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SessionError(f"文件不是合法 JSON：{path.name}（{exc}）") from exc


class SessionStore:
    """Creates, loads and lists sessions, and owns their event logs."""

    def __init__(self, settings: SessionSettings | None = None) -> None:
        self.settings = settings or load_settings().sessions

    # ------------------------------------------------------------------ paths
    @property
    def root(self) -> Path:
        return self.settings.resolved_root()

    def directory(self, session_id: str) -> Path:
        return self.root / validate_session_id(session_id)

    def session_path(self, session_id: str) -> Path:
        return self.directory(session_id) / "session.json"

    def events_path(self, session_id: str) -> Path:
        return self.directory(session_id) / "events.jsonl"

    def evidence_path(self, session_id: str) -> Path:
        return self.directory(session_id) / "evidence.json"

    def assessment_path(self, session_id: str) -> Path:
        return self.directory(session_id) / "final_assessment.json"

    # ----------------------------------------------------------------- create
    def create(self, *, language: str = "en", session_id: str | None = None) -> AgentSession:
        identifier = validate_session_id(session_id) if session_id else new_session_id()
        directory = self.directory(identifier)
        if directory.exists():
            raise SessionError(f"会话已存在：{identifier}")
        directory.mkdir(parents=True)
        session = AgentSession(session_id=identifier, state=WorkflowState.NEW.value,
                               language=language)
        self.save(session)
        self.events(identifier).append("session_created", state=session.state)
        return session

    # ------------------------------------------------------------------- load
    def exists(self, session_id: str) -> bool:
        try:
            return self.session_path(session_id).is_file()
        except SessionError:
            return False

    def load(self, session_id: str) -> AgentSession:
        """Read a session, migrating and minimising it on the way in.

        An older file is brought up to the current schema and rewritten in
        place: unknown fields are dropped, and any machine path is removed. A
        file that cannot be migrated at all is reported, not silently emptied.
        """
        path = self.session_path(session_id)
        if not path.is_file():
            raise SessionError(f"找不到会话：{session_id}")
        payload = read_json(path)
        try:
            migrated, changed = migrate_session_payload(payload)
            session = AgentSession.model_validate(migrated)
        except Exception as exc:  # noqa: BLE001 - report, never traceback
            raise SessionError(f"会话文件无法读取：{session_id}（{exc}）") from exc
        if changed:
            self.save(session)
        return session

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for directory in sorted(self.root.iterdir()):
            try:
                validate_session_id(directory.name)
            except SessionError:
                continue
            path = directory / "session.json"
            if not directory.is_dir() or not path.is_file():
                continue
            try:
                payload = read_json(path)
            except SessionError:
                continue
            conversation = payload.get("conversation") or []
            first_user_text = next((
                str(turn.get("text") or "").strip()
                for turn in conversation
                if isinstance(turn, dict) and turn.get("role") == "user"
                and str(turn.get("text") or "").strip()
            ), "")
            summary = str(payload.get("conversation_summary") or "").strip()
            river_name = str(payload.get("river_name") or "").strip()
            title = summary or river_name or first_user_text or "New session"
            rows.append({
                "session_id": payload.get("session_id", directory.name),
                "state": payload.get("state"),
                "updated_at": payload.get("updated_at"),
                "created_at": payload.get("created_at"),
                "river_name": payload.get("river_name"),
                "hydro_run_id": payload.get("hydro_run_id"),
                "hydro_run_status": payload.get("hydro_run_status"),
                "pending_action": payload.get("pending_question"),
                "conversation_summary": payload.get("conversation_summary"),
                "message_count": len(conversation),
                "title": title[:80],
            })
        rows.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return rows[:limit]

    def latest_session_id(self) -> str | None:
        rows = self.list_sessions(limit=1)
        return rows[0]["session_id"] if rows else None

    # ------------------------------------------------------------------- save
    def save(self, session: AgentSession) -> AgentSession:
        session.updated_at = utc_now()
        payload = _strip_paths(_scrub(session.model_dump(mode="json")))
        write_json_atomic(self.session_path(session.session_id), payload)
        return session

    # ----------------------------------------------------------------- events
    def events(self, session_id: str) -> EventLog:
        return EventLog(self.events_path(session_id),
                        maximum_payload_chars=self.settings.max_event_payload_chars)

    def read_events(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        return self.events(session_id).read(limit=limit)

    # -------------------------------------------------------------- artefacts
    def save_evidence(self, session_id: str, payload: dict[str, Any]) -> None:
        write_json_atomic(self.evidence_path(session_id), _strip_paths(_scrub(payload)))

    def load_evidence(self, session_id: str) -> dict[str, Any] | None:
        path = self.evidence_path(session_id)
        return read_json(path) if path.is_file() else None

    def save_assessment(self, session_id: str, payload: dict[str, Any]) -> None:
        write_json_atomic(self.assessment_path(session_id), _strip_paths(_scrub(payload)))

    def load_assessment(self, session_id: str) -> dict[str, Any] | None:
        path = self.assessment_path(session_id)
        return read_json(path) if path.is_file() else None

    # ----------------------------------------------------------------- delete
    def delete(self, session_id: str) -> None:
        """Permanently remove one validated session directory.

        The Web layer separately refuses deletion while a hydro job is active.
        This storage primitive only establishes that the target is exactly one
        ordinary session directory directly below the configured root; it will
        neither follow a symlink nor accept a path-shaped identifier.
        """
        identifier = validate_session_id(session_id)
        directory = self.directory(identifier)
        if not directory.exists():
            raise SessionError(f"找不到会话：{identifier}")
        if (directory.is_symlink() or getattr(directory, "is_junction", lambda: False)()
                or not directory.is_dir()):
            raise SessionError(f"会话目录不安全，拒绝删除：{identifier}")
        root = self.root.resolve()
        resolved = directory.resolve()
        if resolved.parent != root:
            raise SessionError(f"会话目录超出存储范围，拒绝删除：{identifier}")
        shutil.rmtree(directory)

    def clear_artifacts(self, session_id: str, names: Iterable[str]) -> None:
        """Remove derived files, leaving ``session.json`` and the event log."""
        for name in names:
            (self.directory(session_id) / name).unlink(missing_ok=True)
