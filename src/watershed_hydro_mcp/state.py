"""Shared state helpers for the hydro/stability MCP."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from watershed_exchange.paths import safe_id
from watershed_exchange.paths import PathSafetyError


#: Stable-id prefixes used across the service.
RUN_ID_PREFIX = "hydro-run-"
INPUT_ID_PREFIX = "hydro-input-"
ARTIFACT_ID_PREFIX = "artifact-"

ID_PATTERN = re.compile(r"^[0-9A-Za-z_\-]{6,120}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    from uuid import uuid4

    return f"{prefix}{uuid4().hex[:12]}"


def check_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise PathSafetyError(f"{label}格式无效")
    return value


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.is_file():
        if default is not None:
            return default
        raise ValueError(f"记录不存在：{path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


#: Windows refuses ``os.replace`` onto a file that another process currently
#: has open for reading, failing with ``PermissionError`` (WinError 5 or 32).
#: Polling a job's status does exactly that, so without a retry a worker can die
#: on its final status write purely because someone looked at it at the wrong
#: instant — losing a finished computation. The contention lasts as long as one
#: read, so a short retry is enough; the swap stays atomic either way.
_REPLACE_ATTEMPTS = 40
_REPLACE_DELAY_S = 0.05


def replace_with_retry(temporary: str | Path, path: Path) -> None:
    """``os.replace``, tolerating a reader holding the destination open."""
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_DELAY_S)


def write_json(path: Path, value: Any) -> None:
    """Write JSON atomically so a half-written file is never read back."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        replace_with_retry(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def append_json_line(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def envelope(
    *,
    ok: bool,
    status: str,
    summary: str,
    facts: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    limitations: list[str] | None = None,
    next_actions: list[str] | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The single response shape every tool returns."""
    return {
        "ok": ok,
        "status": status,
        "summary": summary,
        "facts": facts or {},
        "warnings": warnings or [],
        "limitations": limitations or [],
        "next_actions": next_actions or [],
        "data": data or {},
    }


def failure(summary: str, *, status: str = "failed", **kwargs: Any) -> dict[str, Any]:
    return envelope(ok=False, status=status, summary=summary, **kwargs)


def find_job_id(settings, run_id: str) -> str:
    check_id(run_id, "任务编号")
    return run_id
