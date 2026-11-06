"""Import and registration of hydro model inputs.

A package from the spatial MCP and data from any other source enter through the
same gate: paths are checked, an archive is extracted into a scratch directory,
the manifest is validated, every hash is verified, the spatial and temporal
content is inspected, and only then is the result moved into the managed input
directory. After that the run never reads the source directory again.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import rasterio
from pyproj import CRS
from shapely.geometry import shape as shapely_shape

from watershed_exchange.paths import PathSafetyError, safe_id, safe_relative_path
from watershed_exchange.schemas import SHAPEFILE_COMPANIONS
from watershed_exchange.validation import (
    PackageValidationError,
    load_manifest,
    verify_package,
)

from .settings import SETTINGS, Settings
from .state import new_id, read_json, utc_now, write_json


INPUT_ID_PREFIX = "hydro-input-"

#: Role -> member file names expected inside an imported package directory.
PACKAGE_MEMBERS = {
    "dem": ["dem.tif"],
    "boundary": ["basin.geojson"],
    "rainfall": ["station_rainfall.zip"],
    "landuse": ["landuse.shp", f"landuse{SHAPEFILE_COMPANIONS[0]}"],
    "river": ["river.shp", f"river{SHAPEFILE_COMPANIONS[0]}"],
}


class InputRegistrationError(ValueError):
    """Raised when input data cannot be accepted."""


def inbox_roots(settings: Settings = SETTINGS) -> list[Path]:
    """Directories a package may be imported from.

    Both are plain data directories inside this project; neither is another
    service's run state, so importing works whether or not the spatial MCP has
    ever run.
    """
    configured = os.environ.get("WHS_INBOX_ROOTS", "").strip()
    roots = [settings.root / "exchange" / "exports", settings.root / "exchange" / "imports"]
    if configured:
        roots = [Path(part).expanduser() for part in configured.split(os.pathsep) if part.strip()]
    return [Path(root).resolve() for root in roots]


def find_package_source(name: str, settings: Settings = SETTINGS) -> Path:
    """Resolve a package id or archive name inside the allowed inbox roots."""
    text = str(name).strip()
    if not text or "/" in text or "\\" in text or ".." in text:
        raise PathSafetyError("包名不能包含路径分隔符")
    for root in inbox_roots(settings):
        for candidate in (root / text, root / f"{text}.zip"):
            if candidate.exists():
                return candidate
    raise InputRegistrationError(
        f"未在收件目录中找到包：{text}；已搜索 {[str(root) for root in inbox_roots(settings)]}"
    )


def _stage_directory(source: Path, scratch: Path) -> tuple[Path, bool]:
    """Place the package's files in ``scratch``; return (directory, was_archive)."""
    if source.is_dir():
        return source, False
    if source.suffix.lower() != ".zip":
        raise InputRegistrationError(f"不支持的输入类型：{source.name}")
    target = scratch / "package"
    from watershed_exchange.paths import safe_extract_zip

    safe_extract_zip(source, target)
    return target, True


def _nested_package_root(directory: Path) -> Path:
    """A zip made from a package folder may hold exactly one top-level folder."""
    if (directory / "manifest.json").is_file():
        return directory
    children = [item for item in directory.iterdir() if item.is_dir()]
    files = [item for item in directory.iterdir() if item.is_file()]
    if len(children) == 1 and not files and (children[0] / "manifest.json").is_file():
        return children[0]
    return directory


def _spatial_report(directory: Path) -> dict[str, Any]:
    """Read the DEM and boundary to confirm they describe the same place."""
    report: dict[str, Any] = {}
    dem_path = directory / "dem.tif"
    with rasterio.open(dem_path) as source:
        if source.crs is None:
            raise InputRegistrationError("DEM 缺少坐标参考系统")
        report["dem"] = {
            "crs": source.crs.to_string(),
            "width": source.width,
            "height": source.height,
            "resolution": [abs(source.transform.a), abs(source.transform.e)],
            "bounds": [source.bounds.left, source.bounds.bottom, source.bounds.right, source.bounds.top],
            "nodata": source.nodata,
        }
    boundary_path = directory / "basin.geojson"
    payload = json.loads(boundary_path.read_text(encoding="utf-8"))
    geometries = [
        shapely_shape(feature["geometry"])
        for feature in (payload.get("features") or [])
        if feature.get("geometry")
    ]
    if not geometries:
        raise InputRegistrationError("流域边界没有有效几何")
    from shapely.ops import unary_union

    merged = unary_union(geometries)
    if merged.is_empty or merged.area <= 0:
        raise InputRegistrationError("流域边界面积为零")
    report["boundary"] = {
        "bounds": [round(value, 7) for value in merged.bounds],
        "geometry_type": merged.geom_type,
        "crs": "OGC:CRS84",
    }
    return report


def _temporal_report(directory: Path) -> dict[str, Any]:
    """Confirm the rainfall archive is a readable station package."""
    import shapefile

    from watershed_exchange.vector_io import open_reader_bytes

    archive = directory / "station_rainfall.zip"
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        dbf_names = [name for name in names if name.lower().endswith(".dbf")]
        if not dbf_names:
            raise InputRegistrationError("降雨压缩包中没有站点 DBF 文件")
        dbf_name = dbf_names[0]
        shp_name = next(
            (name for name in names if name.lower() == dbf_name[:-4].lower() + ".shp"), None
        )
        dbf_bytes = package.read(dbf_name)
        shp_bytes = package.read(shp_name) if shp_name else None
    try:
        reader = open_reader_bytes(shp_bytes, dbf_bytes)
    except ValueError as exc:
        raise InputRegistrationError(f"降雨站点 DBF 无法解析：{exc}") from exc
    fields = [field[0] for field in reader.fields[1:]]
    record_count = len(reader)
    if record_count < 1:
        raise InputRegistrationError("降雨站点记录为空")
    return {"station_dbf": dbf_name, "station_count": record_count,
            "field_count": len(fields), "fields_sample": fields[:8]}


def _accept_package(directory: Path, settings: Settings) -> dict[str, Any]:
    manifest = load_manifest(directory / "manifest.json")
    verification = verify_package(directory, manifest)
    if not verification["ok"]:
        raise InputRegistrationError("输入包校验失败：" + "；".join(verification["errors"]))
    spatial = _spatial_report(directory)
    temporal = _temporal_report(directory)
    input_id = new_id(INPUT_ID_PREFIX)
    target = settings.input_dir(input_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise InputRegistrationError("输入目录已存在")
    # Atomic move into the managed directory: a partially copied input is never visible.
    temporary = target.with_name(f".{input_id}.staging")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(directory, temporary)
    os.replace(temporary, target)
    record = {
        "input_id": input_id,
        "origin": "model_input_package",
        "schema_version": manifest.schema_version,
        "source_package_id": manifest.package_id,
        "created_at": utc_now(),
        "event": manifest.event.model_dump(),
        "study_area": manifest.study_area.model_dump(),
        "limitations": manifest.limitations,
        "provenance": manifest.provenance,
        "qa": manifest.qa,
        "verification": verification,
        "spatial": spatial,
        "temporal": temporal,
        "files": sorted(
            item.relative_to(target).as_posix() for item in target.rglob("*") if item.is_file()
        ),
        "directory": str(target),
    }
    write_json(target / "input_record.json", record)
    return record


def import_package(name: str, settings: Settings = SETTINGS) -> dict[str, Any]:
    """Import a Model Input Package v1 by id or archive name from an inbox root."""
    source = find_package_source(name, settings)
    with tempfile.TemporaryDirectory(prefix="whs-import-") as scratch:
        directory, was_archive = _stage_directory(source, Path(scratch))
        directory = _nested_package_root(directory)
        record = _accept_package(directory, settings)
    record["imported_from"] = str(source)
    record["imported_from_archive"] = was_archive
    write_json(settings.input_dir(record["input_id"]) / "input_record.json", record)
    return record


def _resolve_inbox(source_root: str, settings: Settings) -> Path:
    candidate = Path(source_root).expanduser()
    if not candidate.is_absolute():
        candidate = settings.root / candidate
    candidate = candidate.resolve()
    for root in inbox_roots(settings):
        if candidate == root or root in candidate.parents:
            return candidate
    raise PathSafetyError(
        f"来源目录必须位于收件目录内：{[str(root) for root in inbox_roots(settings)]}"
    )


def _copy_member(source_root: Path, relative: str, target: Path) -> dict[str, Any]:
    relative_path = Path(safe_relative_path(relative))
    source = source_root / relative_path
    if not source.is_file():
        raise InputRegistrationError(f"来源文件不存在：{relative}")
    destination = target / relative_path.name
    shutil.copy2(source, destination)
    return {"member": relative_path.name, "size_bytes": destination.stat().st_size}


def register_inputs(request, settings: Settings = SETTINGS) -> dict[str, Any]:
    """Copy caller-supplied files from an inbox directory into a managed input."""
    source_root = _resolve_inbox(request.source_root, settings)
    if not source_root.is_dir():
        raise InputRegistrationError(f"来源目录不存在：{source_root}")
    input_id = new_id(INPUT_ID_PREFIX)
    target = settings.input_dir(input_id)
    temporary = target.with_name(f".{input_id}.staging")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        members = {
            "dem": request.dem,
            "boundary": request.boundary,
            "rainfall": request.rainfall_zip,
            "landuse": request.landuse,
        }
        if request.river:
            members["river"] = request.river
        copied: dict[str, Any] = {}
        for role, relative in members.items():
            copied[role] = _copy_member(source_root, relative, temporary)
        # Shapefile companions travel with their .shp.
        for role in ("landuse", "river"):
            if role not in members:
                continue
            stem = Path(safe_relative_path(members[role])).with_suffix("").name
            for suffix in SHAPEFILE_COMPANIONS:
                companion = source_root / f"{Path(safe_relative_path(members[role])).parent.as_posix()}/{stem}{suffix}"
                if companion.is_file():
                    shutil.copy2(companion, temporary / f"{stem}{suffix}")
        for original, canonical in (("dem", "dem.tif"), ("boundary", "basin.geojson"),
                                    ("rainfall", "station_rainfall.zip"),
                                    ("landuse", "landuse.shp"), ("river", "river.shp")):
            if original not in copied:
                continue
            produced = temporary / Path(safe_relative_path(members[original])).name
            canonical_path = temporary / canonical
            if produced != canonical_path:
                os.replace(produced, canonical_path)
        missing = [role for role in ("dem", "boundary", "rainfall", "landuse") if role not in copied]
        if missing:
            raise InputRegistrationError(f"缺少必需数据：{missing}")
        spatial = _spatial_report(temporary)
        temporal = _temporal_report(temporary)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    record = {
        "input_id": input_id,
        "origin": "registered",
        "schema_version": "registered-1.0",
        "created_at": utc_now(),
        "source_root": str(source_root),
        "labels": dict(request.labels),
        "limitations": [
            "Input data was registered from an external source; this service did not produce or verify its provenance.",
            "The spatial and temporal checks confirm readability and consistency, not scientific suitability.",
        ],
        "spatial": spatial,
        "temporal": temporal,
        "files": sorted(
            item.relative_to(target).as_posix() for item in target.rglob("*") if item.is_file()
        ),
        "directory": str(target),
    }
    write_json(target / "input_record.json", record)
    return record


def load_record(input_id: str, settings: Settings = SETTINGS) -> dict[str, Any]:
    safe_id(input_id, "输入编号")
    directory = settings.input_dir(input_id)
    if not directory.is_dir():
        raise InputRegistrationError(f"输入不存在：{input_id}")
    return read_json(directory / "input_record.json")


def list_inputs(settings: Settings = SETTINGS) -> list[dict[str, Any]]:
    if not settings.inputs_root.is_dir():
        return []
    records = []
    for directory in sorted(settings.inputs_root.iterdir()):
        record_path = directory / "input_record.json"
        if not directory.is_dir() or not record_path.is_file():
            continue
        record = read_json(record_path)
        records.append({
            "input_id": record.get("input_id"),
            "origin": record.get("origin"),
            "created_at": record.get("created_at"),
            "schema_version": record.get("schema_version"),
            "source_package_id": record.get("source_package_id"),
            "event": record.get("event"),
            "study_area": record.get("study_area"),
        })
    return records


def validate_input(input_id: str, settings: Settings = SETTINGS) -> dict[str, Any]:
    """Re-check a stored input and report anything that no longer holds."""
    record = load_record(input_id, settings)
    directory = settings.input_dir(input_id)
    errors: list[str] = []
    warnings: list[str] = []
    for name in ("dem.tif", "basin.geojson", "station_rainfall.zip", "landuse.shp"):
        if not (directory / name).is_file():
            errors.append(f"缺少文件：{name}")
    if record.get("origin") == "model_input_package":
        verification = verify_package(directory)
        errors.extend(verification["errors"])
        warnings.extend(verification["warnings"])
    spatial: dict[str, Any] = {}
    temporal: dict[str, Any] = {}
    if (directory / "dem.tif").is_file() and (directory / "basin.geojson").is_file():
        try:
            spatial = _spatial_report(directory)
        except Exception as exc:  # noqa: BLE001 - report the reader's message
            errors.append(f"空间检查失败：{exc}")
    if (directory / "station_rainfall.zip").is_file():
        try:
            temporal = _temporal_report(directory)
        except Exception as exc:  # noqa: BLE001 - report the reader's message
            errors.append(f"降雨检查失败：{exc}")
    event = record.get("event") or {}
    if not event.get("start_datetime") or not event.get("end_datetime"):
        warnings.append("事件起止时间缺失；SWMM 运行时间将取默认值")
    return {
        "input_id": input_id,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "spatial": spatial or record.get("spatial") or {},
        "temporal": temporal or record.get("temporal") or {},
        "event": event,
        "limitations": record.get("limitations", []),
    }
