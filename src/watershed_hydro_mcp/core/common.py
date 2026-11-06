"""Shared helpers for the watershed hydro/stability computation core.

Paths are resolved against a per-run working root rather than the code
directory, so a single installed copy of this package can serve many
isolated runs. The MCP server creates the run root and points
``WHS_RUN_ROOT`` at it; nothing a client sends can change these values.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
import re

import yaml
from pyproj import CRS


#: Directory holding this module (``watershed_hydro_mcp/core``).
CORE_ROOT = Path(__file__).resolve().parent

#: Code root of the installed package.
PACKAGE_ROOT = CORE_ROOT.parent


def _resolve_run_root() -> Path:
    value = os.environ.get("WHS_RUN_ROOT", "").strip()
    return Path(value).expanduser().resolve() if value else PACKAGE_ROOT


#: Working directory for one hydro run; holds ``config.yaml`` and all outputs.
RUN_ROOT = _resolve_run_root()

#: Configuration file driving the current run.
CONFIG = Path(os.environ.get("WHS_CONFIG", "").strip() or (RUN_ROOT / "config.yaml")).resolve()


def resolve_path(value: str | Path | None) -> Path | None:
    """Resolve run-relative paths without tying the project to a drive letter."""
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (RUN_ROOT / path).resolve()


def native_dir() -> Path:
    """Bundled native components (the SWMM engine), relative to no drive letter."""
    configured = os.environ.get("WHS_NATIVE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (PACKAGE_ROOT.parents[1] / "native").resolve()


def engine_executable() -> Path:
    """The bundled SWMM command-line engine."""
    return native_dir() / "swmm" / "runswmm.exe"


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def config() -> dict:
    """Read the run configuration once per process.

    Importing a stage module must not require a configuration file to exist:
    one server process serves many runs and reads the config when a stage
    actually executes.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE


_CONFIG_CACHE: dict | None = None


class LazyConfig:
    """Dict-like view that loads the run configuration on first access.

    Stage modules keep referring to a module-level ``CFG``; binding it to this
    proxy defers the file read until a stage actually runs, so importing the
    package never depends on a run directory existing.
    """

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: dict | None = None

    def _load(self) -> dict:
        if self._data is None:
            self._data = config()
        return self._data

    def reset(self) -> None:
        self._data = None

    def __getitem__(self, key):
        return self._load()[key]

    def __setitem__(self, key, value) -> None:
        self._load()[key] = value

    def __delitem__(self, key) -> None:
        del self._load()[key]

    def __contains__(self, key) -> bool:
        return key in self._load()

    def __iter__(self):
        return iter(self._load())

    def __len__(self) -> int:
        return len(self._load())

    def __getattr__(self, name):
        return getattr(self._load(), name)

    def __repr__(self) -> str:
        return f"LazyConfig({self._load()!r})"


def save_config(cfg: dict) -> None:
    CONFIG.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def slugify(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_\-]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "event"


def event_slug(cfg: dict) -> str:
    event = cfg.get("event", {})
    name = slugify(str(event.get("name") or cfg["project"].get("name") or "event"))
    date = str(event.get("date") or "").replace("-", "")
    return f"{name}_{date}" if date and date not in name else name


def parse_dt(value: str) -> datetime:
    value = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported datetime format: {value}")


def event_times(cfg: dict) -> tuple[datetime, datetime, datetime]:
    event = cfg.get("event", {})
    start = parse_dt(event.get("start_datetime") or f"{event.get('date')} 00:00:00")
    end = parse_dt(event.get("end_datetime") or f"{event.get('date')} 23:00:00")
    sim_end = parse_dt(event.get("simulation_end_datetime") or (end + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"))
    return start, end, sim_end


def prj_crs(shp_path: Path) -> CRS | None:
    prj = shp_path.with_suffix(".prj")
    if not prj.exists():
        return None
    try:
        return CRS.from_wkt(prj.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None
