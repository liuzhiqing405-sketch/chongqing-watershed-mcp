from __future__ import annotations

from pathlib import Path
from typing import Any

from watershed_hydro_mcp.core import runtime_bootstrap  # noqa: F401

import numpy as np
import rasterio
from rasterio.features import rasterize
from pyproj import CRS, Transformer
import shapefile
from shapely.geometry import mapping, shape
from shapely.ops import transform as transform_geometry

from watershed_exchange.vector_io import attribute_encoding

from .common import prj_crs


def apply_calculation_mask(valid, transform, crs, project: Path):
    """Use the same retained domain for every downstream raster calculation."""
    from rasterio.warp import reproject, Resampling
    path = Path(project) / "inputs/dem_derived/calculation_mask.tif"
    diagnostic = path.with_name('dem_derived_subcatchments_diagnostics.json')
    if not path.is_file():
        if diagnostic.is_file():
            import json
            if json.loads(diagnostic.read_text(encoding='utf-8')).get('schema_version') in {2, 3}:
                raise RuntimeError('Filtered partition calculation mask is missing')
        return valid
    target = np.zeros(valid.shape, dtype='uint8')
    with rasterio.open(path) as source:
        reproject(source.read(1), target, src_transform=source.transform, src_crs=source.crs,
                  dst_transform=transform, dst_crs=crs, resampling=Resampling.nearest,
                  src_nodata=0, dst_nodata=0)
    return valid & (target == 1)


def _transform_points(points: list[tuple[float, float]], src: CRS | None, dst: CRS) -> list[tuple[float, float]]:
    if not src or src == dst:
        return [(float(x), float(y)) for x, y in points]
    tx = Transformer.from_crs(src, dst, always_xy=True)
    return [tx.transform(float(x), float(y)) for x, y in points]


def _close_ring(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if ring and ring[0] != ring[-1]:
        ring = ring + [ring[0]]
    return ring


def shapefile_polygon_features(path: Path, target_crs: CRS) -> list[dict[str, Any]]:
    # The declared encoding in the .cpg decides how attributes decode; layers
    # without one default to GBK, which is how the managed network is stored.
    reader = shapefile.Reader(str(path), encoding=attribute_encoding(path))
    fields = [f[0] for f in reader.fields[1:]]
    src_crs = prj_crs(path)
    out: list[dict[str, Any]] = []
    try:
        for sr in reader.iterShapeRecords():
            record = dict(zip(fields, sr.record))
            geometry = shape(sr.shape.__geo_interface__)
            if src_crs and src_crs != target_crs:
                tx = Transformer.from_crs(src_crs, target_crs, always_xy=True)
                geometry = transform_geometry(tx.transform, geometry)
            if geometry.is_empty:
                continue
            if geometry.geom_type not in {"Polygon", "MultiPolygon"} or not geometry.is_valid:
                raise ValueError(f"Invalid polygon geometry: {path}")
            # SWMM's GUI polygon is a single outline. It is only a preview;
            # all areas, raster masks and rainfall weights use the full geometry.
            exterior = (geometry if geometry.geom_type == "Polygon"
                        else max(geometry.geoms, key=lambda part: part.area)).exterior
            out.append({"geometry": mapping(geometry), "points": list(exterior.coords), "record": record})
    finally:
        reader.close()
    return out


def landuse_name(record: dict[str, Any]) -> str:
    return str(
        record.get("XDMMC")
        or record.get("DLMC")
        or record.get("地类名称")
        or record.get("NAME")
        or record.get("name")
        or "unknown"
    ).strip()


def landuse_mask_for_grid(shape: tuple[int, int], transform, crs, landuse_path: Path) -> np.ndarray:
    if not landuse_path.exists():
        return np.zeros(shape, dtype=bool)
    features = shapefile_polygon_features(landuse_path, CRS.from_user_input(crs))
    if not features:
        return np.zeros(shape, dtype=bool)
    return rasterize(
        [(item["geometry"], 1) for item in features],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)


def landuse_class_grid(
    shape: tuple[int, int],
    transform,
    crs,
    landuse_path: Path,
) -> tuple[np.ndarray, dict[int, str]]:
    if not landuse_path.exists():
        return np.zeros(shape, dtype=np.int16), {}
    features = shapefile_polygon_features(landuse_path, CRS.from_user_input(crs))
    names = sorted({landuse_name(item["record"]) for item in features if landuse_name(item["record"])})
    lookup = {idx + 1: name for idx, name in enumerate(names)}
    reverse = {name: idx for idx, name in lookup.items()}
    shapes = [(item["geometry"], reverse.get(landuse_name(item["record"]), 0)) for item in features]
    grid = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int16", all_touched=True)
    return grid, lookup
