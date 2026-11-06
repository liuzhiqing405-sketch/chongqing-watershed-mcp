"""Verification of a Model Input Package v1 before it enters a managed run."""

from __future__ import annotations

from io import BytesIO
import json
import math
from pathlib import Path
from typing import Any
import zipfile

from pydantic import ValidationError
import shapefile
from shapely.geometry import Point, box, shape as shapely_shape
from shapely.ops import unary_union

from .paths import (
    PathSafetyError,
    contains_link,
    find_secrets,
    safe_relative_path,
    sha256_file,
)
from .schemas import ModelInputPackage, SHAPEFILE_COMPANIONS


class PackageValidationError(ValueError):
    """Raised when a package is structurally valid JSON but unusable."""


def load_manifest(manifest_path: Path) -> ModelInputPackage:
    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PackageValidationError(f"manifest.json 不是合法JSON：{exc}") from exc
    secrets = find_secrets(payload)
    if secrets:
        raise PackageValidationError(f"manifest 包含疑似凭据字段：{secrets}")
    try:
        return ModelInputPackage.model_validate(payload)
    except ValidationError as exc:
        raise PackageValidationError(f"manifest 不符合 Model Input Package v1：{exc}") from exc


def _shapefile_set(root: Path, entry) -> list[str]:
    """Every companion a shapefile entry needs, checked case-insensitively."""
    relative = Path(entry.relative_path)
    if relative.suffix.lower() != ".shp":
        return []
    base = relative.with_suffix("")
    present = {item.name.lower() for item in (root / base.parent).glob(f"{base.name}.*")}
    missing = [
        suffix for suffix in SHAPEFILE_COMPANIONS
        if f"{base.name}{suffix}".lower() not in present
    ]
    return missing


def _virtual_station_spatial_errors(root: Path, manifest: ModelInputPackage) -> list[str]:
    """Require exported virtual-station coordinates to lie inside the basin."""
    boundary_entry = manifest.file_for("boundary")
    rainfall_entry = manifest.file_for("rainfall")
    if boundary_entry is None or rainfall_entry is None:
        return []
    boundary_path = root / boundary_entry.relative_path
    rainfall_path = root / rainfall_entry.relative_path
    try:
        payload = json.loads(boundary_path.read_text(encoding="utf-8"))
        geometries = [
            shapely_shape(feature["geometry"])
            for feature in payload.get("features", [])
            if feature.get("geometry")
        ]
        basin = unary_union(geometries)
        if basin.is_empty:
            return ["流域边界没有有效几何，无法校验虚拟雨站位置"]
        with zipfile.ZipFile(rainfall_path) as archive:
            names = archive.namelist()
            source_selection = {}
            if 'rainfall_source_selection.json' in names:
                selection = json.loads(archive.read('rainfall_source_selection.json'))
                if selection.get('policy') != 'existing_era5_grid_first_then_inside_virtual':
                    return ['降雨包雨量源选取规则无效']
                for source in selection['stations']:
                    key = source['station_key']
                    if key in source_selection:
                        return ['降雨源编号重复']
                    source_selection[key] = source
            shp_name = next((name for name in names if name.lower().endswith(".shp")), None)
            if shp_name is None:
                return ["降雨包中缺少虚拟雨站 Shapefile"]
            stem = shp_name[:-4]
            by_lower = {name.lower(): name for name in names}
            shx_name = by_lower.get((stem + ".shx").lower())
            dbf_name = by_lower.get((stem + ".dbf").lower())
            if shx_name is None or dbf_name is None:
                return ["降雨包中的虚拟雨站 Shapefile 组件不完整"]
            cpg_name = by_lower.get((stem + ".cpg").lower())
            encoding = "utf-8"
            if cpg_name:
                encoding = archive.read(cpg_name).decode("ascii", errors="ignore").strip() or encoding
            reader = shapefile.Reader(
                shp=BytesIO(archive.read(shp_name)),
                shx=BytesIO(archive.read(shx_name)),
                dbf=BytesIO(archive.read(dbf_name)),
                encoding=encoding,
            )
            outside: list[str] = []
            try:
                for index, item in enumerate(reader.iterShapeRecords(), start=1):
                    if not item.shape.points:
                        outside.append(f"#{index}(无坐标)")
                        continue
                    lon, lat = item.shape.points[0][:2]
                    station_id = str(item.record[0]) if item.record else f"#{index}"
                    source = source_selection.get(station_id) or {}
                    if source_selection and (not source or abs(source['lon']-lon)>1e-7 or abs(source['lat']-lat)>1e-7):
                        return ['降雨源坐标或编号与源选取记录不一致']
                    if not basin.covers(Point(float(lon), float(lat))):
                        if source.get('selection_basis') == 'existing_era5_grid':
                            bounds = source.get('support_bounds') or []
                            if (len(bounds)==4 and all(math.isfinite(float(v)) for v in bounds)
                                    and bounds[0]<bounds[2] and bounds[1]<bounds[3]
                                    and abs((bounds[0]+bounds[2])/2-lon)<1e-7
                                    and abs((bounds[1]+bounds[3])/2-lat)<1e-7
                                    and basin.intersection(box(*bounds)).area>1e-15):
                                continue
                        outside.append(f"{station_id}({lon:.5f},{lat:.5f})")
            finally:
                reader.close()
            if reader.numRecords < 1:
                return ["降雨包至少需要一个有效虚拟雨站"]
            if outside:
                return ["虚拟雨站不在流域范围内：" + "、".join(outside[:10])]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, zipfile.BadZipFile, shapefile.ShapefileException) as exc:
        return [f"虚拟雨站空间校验失败：{exc}"]
    return []



#: Files a service writes beside the package once it has taken custody of it.
#: They are bookkeeping, not package content, so they are never reported as
#: undeclared members.
SERVICE_OWNED_FILES = frozenset({"input_record.json"})


def verify_package(package_root: Path, manifest: ModelInputPackage | None = None) -> dict[str, Any]:
    """Check every declared file: presence, safety, size and SHA-256.

    Returns a report instead of raising so callers can surface partial results;
    any entry in ``errors`` makes the package unusable.
    """
    root = Path(package_root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checked: list[dict[str, Any]] = []
    if manifest is None:
        try:
            manifest = load_manifest(root / "manifest.json")
        except PackageValidationError as exc:
            return {"ok": False, "errors": [str(exc)], "warnings": [], "files": []}

    missing_roles = manifest.missing_roles()
    if missing_roles:
        errors.append(f"缺少必需文件角色：{missing_roles}")

    for entry in manifest.files:
        try:
            relative = safe_relative_path(entry.relative_path)
        except PathSafetyError as exc:
            errors.append(str(exc))
            continue
        if contains_link(root, relative):
            errors.append(f"文件路径包含符号链接或联接点：{relative}")
            continue
        target = root / relative
        if not target.is_file():
            errors.append(f"声明文件缺失：{relative}")
            continue
        size = target.stat().st_size
        if size != entry.size_bytes:
            errors.append(f"文件大小与清单不符：{relative}（磁盘 {size}，清单 {entry.size_bytes}）")
            continue
        digest = sha256_file(target)
        if digest != entry.sha256:
            errors.append(f"SHA-256 不匹配：{relative}")
            continue
        missing = _shapefile_set(root, entry)
        if missing:
            errors.append(f"Shapefile组件缺失：{relative} 缺少 {missing}")
            continue
        checked.append({"role": entry.role, "relative_path": relative,
                        "size_bytes": size, "sha256": digest})

    declared = {entry.relative_path for entry in manifest.files}
    for extra in sorted(root.rglob("*")):
        if not extra.is_file():
            continue
        relative = extra.relative_to(root).as_posix()
        if relative == "manifest.json" or relative in declared:
            continue
        if Path(relative).name in SERVICE_OWNED_FILES:
            continue
        if any(relative == f"{Path(entry.relative_path).with_suffix(suffix).as_posix()}"
               for entry in manifest.files for suffix in SHAPEFILE_COMPANIONS):
            continue
        warnings.append(f"包内存在未在清单声明的文件：{relative}")

    limitations = list(manifest.limitations)
    if "虚拟站" not in " ".join(limitations) and "virtual" not in " ".join(limitations).lower():
        warnings.append("限制说明未标记ERA5虚拟站并非实测站")

    if not errors:
        errors.extend(_virtual_station_spatial_errors(root, manifest))

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "files": checked,
        "package_id": manifest.package_id,
        "created_at": manifest.created_at,
        "study_area": manifest.study_area.model_dump(),
        "event": manifest.event.model_dump(),
    }


def read_text_limited(path: Path, limit_bytes: int = 20 * 1024) -> dict[str, Any]:
    """Read a text artifact with a hard byte cap so logs cannot flood a client."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(limit_bytes)
    truncated = size > limit_bytes
    return {
        "text": raw.decode("utf-8", errors="replace"),
        "size_bytes": size,
        "returned_bytes": len(raw),
        "truncated": truncated,
        "limit_bytes": limit_bytes,
    }
