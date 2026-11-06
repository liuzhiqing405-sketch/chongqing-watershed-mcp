"""Artifact registration.

Every file a run produces is described once, in one shape, and afterwards only
its ``artifact_id`` is used. Binary payloads never travel inside JSON: previews
go as MCP images and full files as resource links.
"""

from __future__ import annotations

import csv
import mimetypes
from pathlib import Path
from typing import Any, Iterable

import rasterio

from watershed_exchange.paths import safe_relative_path, sha256_file

from .settings import SETTINGS, Settings
from .state import new_id, read_json, utc_now, write_json


ARTIFACT_ID_PREFIX = "artifact-"

#: Kinds a run can register, with the stage that produces them.
STAGE_KINDS = {
    "delineate": "partition",
    "rainfall": "rainfall",
    "rainfall_plot": "figure",
    "build": "swmm_input",
    "swmm": "swmm_result",
    "safety": "stability",
    "analysis": "analysis",
    "export": "export",
    "comparison": "comparison",
}

#: Raster artifacts can be rendered; text artifacts can be read back.
RASTER_SUFFIXES = {".tif", ".tiff"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
TEXT_SUFFIXES = {".txt", ".log", ".csv", ".md", ".json", ".inp", ".rpt"}

#: The content types this service actually produces, stated rather than guessed.
#:
#: ``mimetypes.guess_type`` consults the Windows registry, where ``.csv`` is
#: commonly claimed by a spreadsheet application. That made the registered type
#: depend on the machine — the same run recorded ``application/vnd.ms-excel``
#: here and would record ``text/csv`` elsewhere — and a browser handed that
#: value saves a comma-separated file under an ``.xls`` name. A file whose type
#: we know should not have its type read off the host's file associations.
MEDIA_TYPES: dict[str, str] = {
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".geojson": "application/geo+json",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".log": "text/plain",
    ".prj": "text/plain",
    ".cpg": "text/plain",
    # SWMM: two text formats the engine writes and one native binary.
    ".inp": "text/plain",
    ".rpt": "text/plain",
    ".out": "application/octet-stream",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".pdf": "application/pdf",
    ".zip": "application/zip",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    # Shapefile components are not self-describing; the set is the artifact.
    ".shp": "application/octet-stream",
    ".shx": "application/octet-stream",
    ".dbf": "application/octet-stream",
}

#: The unit a whole file carries, by the quantity in its name. Only files whose
#: *entire* content is one quantity appear here: a unit belongs to a column, and
#: putting one on a multi-column table would be an invention rather than a fact.
FILE_UNITS: tuple[tuple[str, str], ...] = (
    ("minimum_safety_factor", "dimensionless"),
    ("maximum_flow_depth_m", "m"),
    ("maximum_velocity_ms", "m/s"),
    ("maximum_pore_water_pressure_kpa", "kPa"),
    ("unstable_duration_h", "h"),
    ("deposit_thickness_m", "m"),
    ("saturated_cohesion_kpa", "kPa"),
    ("saturated_friction_angle_deg", "degree"),
    ("calculation_resolution_m", "m"),
    ("dem", "m"),
)

MAXIMUM_TEXT_BYTES = 20 * 1024
MAXIMUM_PREVIEW_BYTES = 5 * 1024 * 1024
#: How much of a table is read to describe it. The row count of a large CSV is
#: not worth loading the whole file for; past this the count is reported as
#: "at least N" rather than being guessed.
TABLE_SCAN_BYTES = 2 * 1024 * 1024


def media_type(path: Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in MEDIA_TYPES:
        return MEDIA_TYPES[suffix]
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def units_for(path: Path) -> str | None:
    """The unit of a file whose whole content is one quantity, else ``None``."""
    name = Path(path).name.lower()
    for token, unit in FILE_UNITS:
        if token in name:
            return unit
    return None


def _raster_facts(path: Path) -> dict[str, Any]:
    try:
        with rasterio.open(path) as source:
            transform = source.transform
            return {
                "crs": source.crs.to_string() if source.crs is not None else None,
                "bbox": [round(value, 6) for value in source.bounds],
                "width": source.width,
                "height": source.height,
                "count": source.count,
                "dtype": source.dtypes[0] if source.dtypes else None,
                "nodata": source.nodata,
                # The six affine coefficients, so a consumer can rebuild the grid
                # exactly rather than inferring it from the bounding box.
                "transform": [float(value) for value in tuple(transform)[:6]],
                "resolution": [abs(float(transform.a)), abs(float(transform.e))],
            }
    except Exception:  # noqa: BLE001 - a non-raster or unreadable file is still registrable
        return {}


def _table_facts(path: Path) -> dict[str, Any]:
    """What a delimited text file holds: its columns and, when cheap, its rows.

    Only the head of the file is read. Reporting a row count for a table too
    large to scan would mean either loading it anyway or inventing a number, and
    a download button does not need either.
    """
    facts: dict[str, Any] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            sample = stream.read(TABLE_SCAN_BYTES)
    except (OSError, UnicodeDecodeError):
        return facts
    lines = sample.splitlines()
    if not lines:
        return facts
    try:
        header = next(csv.reader([lines[0]]))
    except (csv.Error, StopIteration):
        return facts
    columns = [column.strip() for column in header if column.strip()]
    if not columns:
        return facts
    facts["columns"] = columns[:64]
    facts["column_count"] = len(columns)
    # A time column is what makes a table a series; note it when one is named.
    # Matched on a substring so "datetime_beijing" counts and "rain_mm" does not.
    facts["time_column"] = next(
        (name for name in columns
         if any(token in name.lower() for token in ("datetime", "date", "time", "时间"))),
        None)
    if len(sample) < TABLE_SCAN_BYTES:
        # The whole file was read, so the count is exact.
        facts["rows"] = max(0, len(lines) - 1)
    else:
        facts["rows_at_least"] = max(0, len(lines) - 1)
    return facts


def _available_renderers(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in RASTER_SUFFIXES:
        return ["raster_png", "raster_map"]
    if suffix in IMAGE_SUFFIXES:
        return ["passthrough"]
    # Checked before the text test on purpose: ``.csv`` is in TEXT_SUFFIXES, and
    # testing for text first made the table branch below unreachable.
    if suffix in {".csv", ".tsv"}:
        return ["text", "table"]
    if suffix in TEXT_SUFFIXES:
        return ["text"]
    return []


def describe(
    path: Path,
    *,
    run_id: str,
    stage_id: str,
    project_root: Path,
    kind: str | None = None,
    title: str | None = None,
    derived_from: Iterable[str] | None = None,
    producer: str | None = None,
    producer_version: str | None = None,
    derivation: dict[str, Any] | None = None,
    units: str | None = None,
) -> dict[str, Any]:
    path = Path(path).resolve()
    relative = path.relative_to(Path(project_root).resolve()).as_posix()
    suffix = path.suffix.lower()
    if suffix in RASTER_SUFFIXES:
        facts = _raster_facts(path)
    elif suffix in {".csv", ".tsv"}:
        facts = _table_facts(path)
    else:
        facts = {}
    return {
        "artifact_id": new_id(ARTIFACT_ID_PREFIX),
        "hydro_run_id": run_id,
        "stage_id": stage_id,
        "kind": kind or STAGE_KINDS.get(stage_id, "other"),
        "title": title or path.name,
        "media_type": media_type(path),
        "internal_path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "created_at": utc_now(),
        "ready": True,
        "visualizable": bool(_available_renderers(path)),
        "available_renderers": _available_renderers(path),
        "crs": facts.get("crs"),
        "bbox": facts.get("bbox"),
        # A derived file states the unit it inherits; only a file that holds one
        # quantity on its own gets a unit from its name.
        "units": units if units is not None else units_for(path),
        # Filled by the QA stage, which is the only place that knows the
        # verdict. Until then the artefact's quality is genuinely unknown, and
        # saying so is better than guessing "passed".
        "qa_status": None,
        # A derived file says what it came from and how, so a reader can redo
        # it. An original file has neither.
        "derived_from_artifact_ids": list(derived_from or []),
        "producer": producer,
        "producer_version": producer_version,
        "derivation": dict(derivation or {}) or None,
        **{key: value for key, value in facts.items() if key not in {"crs", "bbox"}},
    }


class ArtifactRegistry:
    """Append-only artifact index for one job."""

    def __init__(self, job_dir: Path, run_id: str, project_root: Path) -> None:
        self.job_dir = Path(job_dir)
        self.run_id = run_id
        self.project_root = Path(project_root)
        self.path = self.job_dir / "artifacts.json"
        self._records: list[dict[str, Any]] = []
        if self.path.is_file():
            payload = read_json(self.path)
            self._records = list(payload.get("artifacts") or [])

    def register(self, path: Path, stage_id: str, *, kind: str | None = None,
                 title: str | None = None,
                 derived_from: Iterable[str] | None = None,
                 producer: str | None = None,
                 producer_version: str | None = None,
                 derivation: dict[str, Any] | None = None,
                 units: str | None = None) -> dict[str, Any] | None:
        path = Path(path)
        if not path.is_file():
            return None
        record = describe(path, run_id=self.run_id, stage_id=stage_id,
                          project_root=self.project_root, kind=kind, title=title,
                          derived_from=derived_from, producer=producer,
                          producer_version=producer_version, derivation=derivation,
                          units=units)
        # Replace an earlier record for the same file so a re-run does not duplicate it.
        self._records = [item for item in self._records if item["internal_path"] != record["internal_path"]]
        self._records.append(record)
        self.flush()
        return record

    def register_many(self, paths, stage_id: str, *, kind: str | None = None) -> list[dict[str, Any]]:
        registered = []
        for path in paths:
            record = self.register(path, stage_id, kind=kind)
            if record:
                registered.append(record)
        return registered

    def flush(self) -> None:
        write_json(self.path, {"run_id": self.run_id, "artifacts": self._records})

    def stamp_qa(self, verdict: str) -> int:
        """Record the run's QA verdict on every registered artifact.

        Called once, by the stage that runs the gates, because that is the only
        place that knows the verdict. The stamp is never written back to
        "unknown": a file that has been judged must not later look unjudged.
        """
        stamped = 0
        for record in self._records:
            if record.get("qa_status") != verdict:
                record["qa_status"] = verdict
                stamped += 1
        if stamped:
            self.flush()
        return stamped

    def by_path_and_hash(self, relative_path: str,
                         sha256: str | None = None) -> dict[str, Any] | None:
        """The record for a file, matched on its path and — when given — hash.

        The hash is what makes this a lookup rather than a guess: a figure
        recorded its sources by path *and* digest, and a file that has changed
        since is not the file the figure was drawn from.
        """
        for item in self._records:
            if item.get("internal_path") != relative_path:
                continue
            if sha256 and item.get("sha256") != sha256:
                continue
            return item
        return None

    def all(self) -> list[dict[str, Any]]:
        return list(self._records)

    def by_id(self, artifact_id: str) -> dict[str, Any]:
        for record in self._records:
            if record["artifact_id"] == artifact_id:
                return record
        raise ValueError(f"未登记成果：{artifact_id}")

    def resolve(self, record: dict[str, Any]) -> Path:
        relative = safe_relative_path(record["internal_path"])
        candidate = (self.project_root / relative).resolve()
        if candidate != self.project_root and self.project_root not in candidate.parents:
            raise ValueError("成果路径超出运行目录")
        return candidate


def load_artifacts(job_dir: Path) -> list[dict[str, Any]]:
    path = Path(job_dir) / "artifacts.json"
    if not path.is_file():
        return []
    payload = read_json(path)
    return list(payload.get("artifacts") or [])


def registry_for(job_dir: Path, run_id: str, settings: Settings = SETTINGS) -> ArtifactRegistry:
    return ArtifactRegistry(job_dir, run_id, Path(job_dir) / "project")
