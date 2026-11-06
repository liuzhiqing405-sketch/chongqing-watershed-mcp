from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pyogrio
import rasterio
from pyproj import CRS, Transformer

from .settings import SETTINGS, Settings


DATASET_LABELS = {
    "dem": "重庆DEM",
    "river_network": "重庆市水系",
    "landuse": "2023年土地利用",
    "administrative_boundary": "重庆市边界",
    "hourly_rainfall": "ERA5逐时降雨宽表",
}

SHAPEFILE_COMPONENTS = (".shp", ".shx", ".dbf", ".prj")


def rainfall_files(settings: Settings = SETTINGS) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in settings.rainfall_root.glob(settings.rainfall_pattern):
        match = re.search(r"(\d{4})$", path.stem)
        if match:
            result[int(match.group(1))] = path
    return dict(sorted(result.items()))


def dataset_paths(settings: Settings = SETTINGS) -> dict[str, Path]:
    return {
        "dem": settings.dem,
        "river_network": settings.rivers,
        "landuse": settings.landuse,
        "administrative_boundary": settings.boundary,
        "hourly_rainfall": settings.rainfall_root,
    }


def list_catalog(settings: Settings = SETTINGS) -> dict[str, Any]:
    paths = dataset_paths(settings)
    years = list(rainfall_files(settings))
    items = []
    for dataset_id, path in paths.items():
        item = {
            "dataset_id": dataset_id,
            "label": DATASET_LABELS[dataset_id],
            "path": str(path),
            "exists": path.exists(),
        }
        if dataset_id == "hourly_rainfall":
            item.update({"years": years, "year_count": len(years)})
        items.append(item)
    return {"datasets": items, "managed_only": True}


def _raster_metadata(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as source:
        crs = source.crs
        extent = None
        if crs is not None:
            wgs84 = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
            left, bottom = wgs84.transform(source.bounds.left, source.bounds.bottom)
            right, top = wgs84.transform(source.bounds.right, source.bounds.top)
            extent = [[round(left, 7), round(bottom, 7)], [round(right, 7), round(top, 7)]]
        return {
            "driver": source.driver,
            "size": [source.width, source.height],
            "band_count": source.count,
            "dtype": source.dtypes[0],
            "nodata": source.nodata,
            "coordinate_system": crs.to_wkt() if crs is not None else None,
            "crs_epsg": crs.to_epsg() if crs is not None else None,
            "geo_transform": list(source.transform)[:6],
            "bounds_native": {
                "left": round(source.bounds.left, 4), "bottom": round(source.bounds.bottom, 4),
                "right": round(source.bounds.right, 4), "top": round(source.bounds.top, 4),
            },
            "resolution": [abs(source.transform.a), abs(source.transform.e)],
            "wgs84_extent": extent,
        }


def _as_list(value: Any) -> list[Any]:
    """pyogrio returns numpy arrays; JSON needs plain lists."""
    if value is None:
        return []
    return [item.item() if hasattr(item, "item") else item for item in value]


def _vector_metadata(path: Path) -> dict[str, Any]:
    info = pyogrio.read_info(path)
    crs = info.get("crs")
    bounds = info.get("total_bounds")
    return {
        "driver": info.get("driver"),
        "layer": path.stem,
        "feature_count": info.get("features"),
        "geometry_type": info.get("geometry_type"),
        "fields": _as_list(info.get("fields")),
        "coordinate_system": None if crs is None else str(crs),
        "total_bounds": [round(float(value), 4) for value in _as_list(bounds)],
    }


def inspect(dataset_id: str, settings: Settings = SETTINGS) -> dict[str, Any]:
    paths = dataset_paths(settings)
    if dataset_id not in paths:
        raise ValueError(f"未知数据集：{dataset_id}；允许值：{sorted(paths)}")
    path = paths[dataset_id]
    if dataset_id == "hourly_rainfall":
        files = rainfall_files(settings)
        if not files:
            return {"dataset_id": dataset_id, "exists": False, "path": str(path)}
        sample = files[max(files)]
        with sample.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream))
        return {
            "dataset_id": dataset_id,
            "path": str(path),
            "years": list(files),
            "year_count": len(files),
            "grid_point_count": max(0, len(header) - 1),
            "time_column": header[0],
            "sample_file": str(sample),
        }
    if not path.is_file():
        return {"dataset_id": dataset_id, "exists": False, "path": str(path)}
    if path.suffix.lower() in {".tif", ".tiff"}:
        metadata = _raster_metadata(path)
    else:
        metadata = _vector_metadata(path)
    return {"dataset_id": dataset_id, "path": str(path), "exists": True, **metadata}


def validate_bundle(settings: Settings = SETTINGS) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    for dataset_id, path in dataset_paths(settings).items():
        if not path.exists():
            errors.append(f"缺少 {DATASET_LABELS[dataset_id]}：{path}")
    for shp in (settings.rivers, settings.landuse, settings.boundary):
        for suffix in SHAPEFILE_COMPONENTS:
            if not shp.with_suffix(suffix).is_file():
                errors.append(f"Shapefile组件缺失：{shp.with_suffix(suffix)}")
    years = list(rainfall_files(settings))
    if years:
        missing = [year for year in range(min(years), max(years) + 1) if year not in years]
        if missing:
            warnings.append(f"逐时降雨年份不连续：{missing}")
    else:
        errors.append("没有找到ERA5逐时降雨CSV")
    for label, path in (("DEM", settings.dem), ("水系", settings.rivers),
                        ("土地利用", settings.landuse), ("边界", settings.boundary)):
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() in {".tif", ".tiff"}:
                with rasterio.open(path) as source:
                    if source.crs is None:
                        errors.append(f"{label}缺少坐标参考系统：{path}")
            else:
                info = pyogrio.read_info(path)
                if not info.get("crs"):
                    errors.append(f"{label}缺少坐标参考系统：{path}")
        except Exception as exc:  # noqa: BLE001 - surface the reader's own message
            errors.append(f"{label}无法读取：{exc}")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "rainfall_years": years,
        "next_step": "recommend_outlets" if not errors else "修复缺失项后重新验证",
    }
