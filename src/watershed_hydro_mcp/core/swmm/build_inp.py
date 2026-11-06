from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import sys

from watershed_hydro_mcp.core import runtime_bootstrap  # noqa: F401

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import Affine
from rasterio.windows import from_bounds, Window
import shapefile
from pyproj import CRS, Transformer
import yaml
from shapely.geometry import shape, Point

from ..common import event_times, prj_crs
from ..spatial_domain import landuse_class_grid, landuse_name, shapefile_polygon_features
from ..spatial_domain import landuse_mask_for_grid
from ..rainfall.station_rainfall import (
    idw_mean_weights_by_zone,
    idw_weights_at_points,
    read_station_observations,
    station_coordinates,
)
from .landuse_response import build_units, seal_files, FILES as RESPONSE_FILES


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from watershed_hydro_mcp.core.common import LazyConfig, RUN_ROOT as ROOT

CFG = LazyConfig()

LANDUSE_PARAMS = {
    "耕地": {"imperv": 8.0, "n_perv": 0.18, "maxrate": 45.0, "minrate": 6.0, "decay": 4.0},
    "有林地": {"imperv": 3.0, "n_perv": 0.35, "maxrate": 70.0, "minrate": 10.0, "decay": 3.5},
    "灌木林地": {"imperv": 4.0, "n_perv": 0.30, "maxrate": 60.0, "minrate": 8.0, "decay": 3.8},
    "林地": {"imperv": 3.0, "n_perv": 0.35, "maxrate": 70.0, "minrate": 10.0, "decay": 3.5},
    "草地": {"imperv": 5.0, "n_perv": 0.24, "maxrate": 55.0, "minrate": 8.0, "decay": 3.8},
    "园地": {"imperv": 6.0, "n_perv": 0.28, "maxrate": 60.0, "minrate": 8.0, "decay": 3.8},
    "建设用地": {"imperv": 65.0, "n_perv": 0.08, "maxrate": 20.0, "minrate": 2.0, "decay": 5.0},
    "房屋建筑（区）": {"imperv": 65.0, "n_perv": 0.08, "maxrate": 20.0, "minrate": 2.0, "decay": 5.0},
    "住宅用地": {"imperv": 55.0, "n_perv": 0.10, "maxrate": 22.0, "minrate": 2.0, "decay": 5.0},
    "交通运输用地": {"imperv": 75.0, "n_perv": 0.06, "maxrate": 15.0, "minrate": 1.5, "decay": 5.0},
    "水域及水利设施用地": {"imperv": 100.0, "n_perv": 0.03, "maxrate": 0.0, "minrate": 0.0, "decay": 0.0},
    "水域": {"imperv": 100.0, "n_perv": 0.03, "maxrate": 0.0, "minrate": 0.0, "decay": 0.0},
}
DEFAULT_PARAM = {"imperv": 10.0, "n_perv": 0.25, "maxrate": 50.0, "minrate": 6.0, "decay": 4.0}


def transform_points(points: list[tuple[float, float]], src: CRS | None, dst: CRS) -> list[tuple[float, float]]:
    if not src or src == dst:
        return [(float(x), float(y)) for x, y in points]
    tx = Transformer.from_crs(src, dst, always_xy=True)
    return [tx.transform(float(x), float(y)) for x, y in points]


def polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    return sum(x for x, _ in points) / len(points), sum(y for _, y in points) / len(points)


def polygon_bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def bboxes_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def point_in_poly(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_at_y = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_at_y:
                inside = not inside
        j = i
    return inside


def read_polygon_records(path: Path, target_crs: CRS) -> list[dict]:
    return [dict(item, id=str(index)) for index, item in enumerate(
        shapefile_polygon_features(path, target_crs), start=1)]


def full_geometry(points: list[tuple[float, float]], geometry: dict | None = None) -> dict:
    return geometry if geometry is not None else {"type": "Polygon", "coordinates": [points]}


def dominant_landuse(cx: float, cy: float, landuse: list[dict]) -> tuple[str, float]:
    for item in landuse:
        rec = item["record"]
        if "geometry" in item:
            if shape(item["geometry"]).covers(Point(cx, cy)):
                return landuse_name(rec), 1.0
            continue
        if "points" in item:
            pts = item["points"]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            if min(xs) <= cx <= max(xs) and min(ys) <= cy <= max(ys) and point_in_poly(cx, cy, pts):
                return landuse_name(rec), 1.0
    return "unknown", 0.0


def dominant_landuse_by_overlap(points: list[tuple[float, float]], landuse: list[dict]) -> tuple[str, float]:
    cx, cy = polygon_centroid(points)
    name, frac = dominant_landuse(cx, cy, landuse)
    if name != "unknown":
        return name, frac
    sample_step = max(1, len(points) // 12)
    votes: dict[str, int] = {}
    for x, y in points[::sample_step]:
        sample_name, _ = dominant_landuse(x, y, landuse)
        if sample_name != "unknown":
            votes[sample_name] = votes.get(sample_name, 0) + 1
    if not votes:
        return "unknown", 0.0
    best, count = max(votes.items(), key=lambda kv: kv[1])
    return best, count / max(len(points[::sample_step]), 1)


def load_landuse_grid(dem_path: Path, target_crs: CRS, landuse_path: Path) -> tuple[np.ndarray, dict[int, str], Affine]:
    resolution = float(CFG["swmm"].get("delineation_resolution_m") or 20)
    with rasterio.open(dem_path) as src:
        scale_x = max(1, int(round(resolution / abs(src.transform.a))))
        scale_y = max(1, int(round(resolution / abs(src.transform.e))))
        out_h = max(1, src.height // scale_y)
        out_w = max(1, src.width // scale_x)
        transform = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
    grid, lookup = landuse_class_grid((out_h, out_w), transform, target_crs, landuse_path)
    return grid, lookup, transform


def dominant_landuse_from_grid(
    points: list[tuple[float, float]],
    class_grid: np.ndarray,
    lookup: dict[int, str],
    transform: Affine,
    geometry: dict | None = None,
) -> tuple[str, float]:
    if class_grid.size == 0 or not lookup:
        return "unknown", 0.0
    geom = full_geometry(points, geometry)
    window = from_bounds(*shape(geom).bounds, transform=transform).round_offsets().round_lengths()
    row0 = max(0, int(window.row_off))
    col0 = max(0, int(window.col_off))
    row1 = min(class_grid.shape[0], row0 + max(1, int(window.height)))
    col1 = min(class_grid.shape[1], col0 + max(1, int(window.width)))
    if row0 >= row1 or col0 >= col1:
        return "unknown", 0.0
    sub = class_grid[row0:row1, col0:col1]
    win_transform = transform * Affine.translation(col0, row0)
    mask = geometry_mask([geom], out_shape=sub.shape, transform=win_transform, invert=True, all_touched=True)
    vals = sub[mask & (sub > 0)]
    if vals.size == 0:
        return "unknown", 0.0
    labels, counts = np.unique(vals, return_counts=True)
    idx = int(np.argmax(counts))
    label = int(labels[idx])
    return lookup.get(label, "unknown"), float(counts[idx] / max(vals.size, 1))


def minimum_dem_outlet(geometry: dict, source) -> dict:
    """Lowest native, finite DEM cell inside the full polygon, excluding holes.

    Equal minima use native row/column order for reproducible coordinates.
    No resampling, filled terrain or representative-point fallback is used.
    """
    requested = from_bounds(*shape(geometry).bounds, transform=source.transform)
    row0, col0 = max(0, math.floor(requested.row_off)), max(0, math.floor(requested.col_off))
    row1 = min(source.height, math.ceil(requested.row_off + requested.height))
    col1 = min(source.width, math.ceil(requested.col_off + requested.width))
    if row0 >= row1 or col0 >= col1:
        raise ValueError("分区没有与DEM重叠的有效像元，无法选择最低点出口")
    window = Window(col0, row0, col1-col0, row1-row0)
    data = source.read(1, window=window, masked=True).astype('float64')
    mask = geometry_mask([geometry], out_shape=data.shape, transform=source.window_transform(window), invert=True)
    usable = mask & ~np.ma.getmaskarray(data) & np.isfinite(np.ma.getdata(data))
    if not usable.any():
        raise ValueError("分区内没有有效DEM像元，无法选择最低点出口")
    values = np.where(usable, np.ma.getdata(data), np.inf)
    row, col = np.unravel_index(np.argmin(values), values.shape)
    native_row, native_col = int(row0+row), int(col0+col)
    x, y = rasterio.transform.xy(source.transform, native_row, native_col, offset='center')
    return {'outlet': (float(x), float(y)), 'min_elev': float(values[row, col]),
            'outlet_dem_row': native_row, 'outlet_dem_col': native_col,
            'outlet_method': 'minimum_dem_elevation'}


def raster_stats(points: list[tuple[float, float]], dem_path: Path, geometry: dict | None = None) -> dict:
    geom = full_geometry(points, geometry)
    polygon = shape(geom)
    with rasterio.open(dem_path) as src:
        minimum = minimum_dem_outlet(geom, src)
        window = from_bounds(*polygon.bounds, transform=src.transform).round_offsets().round_lengths()
        resolution = float(CFG["swmm"].get("delineation_resolution_m") or 20)
        out_w = max(1, int(round((window.width * abs(src.transform.a)) / resolution)))
        out_h = max(1, int(round((window.height * abs(src.transform.e)) / resolution)))
        data = src.read(1, window=window, out_shape=(out_h, out_w), masked=True).astype("float64")
        transform = src.window_transform(window) * Affine.scale(window.width / out_w, window.height / out_h)
        mask = geometry_mask([geom], out_shape=data.shape, transform=transform, invert=True)
        arr = np.ma.array(data, mask=np.ma.getmaskarray(data) | ~mask | ~np.isfinite(np.ma.getdata(data)))
        valid = arr.compressed()
        if valid.size == 0:
            return dict(minimum, mean_elev=minimum['min_elev'], slope_pct=1.0)
        filled = arr.filled(np.nan)
        if filled.shape[0] < 2 or filled.shape[1] < 2:
            return dict(minimum, mean_elev=float(np.nanmean(valid)), slope_pct=1.0)
        dy, dx = np.gradient(filled, abs(transform.e), abs(transform.a))
        slope = np.sqrt(dx * dx + dy * dy) * 100.0
        slope_vals = slope[np.isfinite(slope) & mask]
        slope_pct = max(float(np.nanmean(slope_vals)) if slope_vals.size else 1.0, 0.1)
        return dict(minimum, mean_elev=float(np.nanmean(valid)), slope_pct=slope_pct)


def drainage_outlet_record(record: dict) -> tuple[float | str, float | str]:
    """Read the delineation's flow exit, retaining it in the parameter review."""
    try:
        x = float(record.get('out_x', record.get('OUT_X', 0)) or 0)
        y = float(record.get('out_y', record.get('OUT_Y', 0)) or 0)
        if math.isfinite(x) and math.isfinite(y) and (x, y) != (0, 0):
            return x, y
    except (TypeError, ValueError):
        pass
    return '', ''


def drainage_exit_outlet(record: dict, geometry: dict, source) -> dict | None:
    x, y = drainage_outlet_record(record)
    if x == '' or y == '':
        return None
    if not shape(geometry).covers(Point(x, y)):
        raise ValueError('沿流向确定的分区出口不在该分区内，请复核分区与出口')
    row, col = source.index(x, y)
    if not (0 <= row < source.height and 0 <= col < source.width):
        raise ValueError('分区排水出口超出DEM范围')
    data = source.read(1, window=Window(col, row, 1, 1), masked=True)
    if np.ma.getmaskarray(data)[0, 0] or not np.isfinite(data[0, 0]):
        raise ValueError('分区排水出口没有有效DEM高程')
    return {'outlet': (x, y), 'min_elev': float(data[0, 0]), 'outlet_dem_row': row,
            'outlet_dem_col': col, 'outlet_method': 'drainage_exit'}


def section_options() -> list[str]:
    event_start, _, sim_end = event_times(CFG)
    return [
        "[OPTIONS]",
        "FLOW_UNITS           CMS",
        "INFILTRATION         HORTON",
        "FLOW_ROUTING         KINWAVE",
        f"START_DATE           {event_start:%m/%d/%Y}",
        f"START_TIME           {event_start:%H:%M:%S}",
        f"REPORT_START_DATE    {event_start:%m/%d/%Y}",
        f"REPORT_START_TIME    {event_start:%H:%M:%S}",
        f"END_DATE             {sim_end:%m/%d/%Y}",
        f"END_TIME             {sim_end:%H:%M:%S}",
        "SWEEP_START          01/01",
        "SWEEP_END            12/31",
        "DRY_DAYS             0",
        f"REPORT_STEP          {CFG['swmm'].get('report_step', '0:05:00')}",
        f"WET_STEP             {CFG['swmm'].get('wet_step', '0:05:00')}",
        f"DRY_STEP             {CFG['swmm'].get('dry_step', '1:00:00')}",
        f"ROUTING_STEP         {CFG['swmm'].get('routing_step', '0:00:05')}",
        "ALLOW_PONDING        YES",
        "MIN_SLOPE            0",
    ]


def write_hourly_rainfall(path: Path, datetimes: list, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["datetime_beijing", "rain_mm"])
        writer.writeheader()
        for stamp, value in zip(datetimes, values):
            writer.writerow({"datetime_beijing": stamp.strftime("%Y-%m-%d %H:%M:%S"), "rain_mm": f"{float(value):.4f}"})


def spatial_station_rainfall(
    model_rows: list[dict],
    model_items: list[dict],
    dem_path: Path,
    landuse_path: Path,
    target_crs: CRS,
    out_dir: Path,
    rainfall_csv: Path,
) -> tuple[list[str], np.ndarray]:
    rain_cfg = CFG.get("rainfall", {})
    observations_path = Path(rain_cfg.get("station_observations_csv") or "")
    if not observations_path.is_file():
        raise RuntimeError(
            "Multi-station observations are missing. Run the station-rainfall preparation stage first: "
            f"{observations_path}"
        )
    observations = read_station_observations(observations_path)
    station_x, station_y = station_coordinates(observations, target_crs)
    resolution = float(rain_cfg.get("station_idw_resolution_m", 5.0))
    power = float(rain_cfg.get("station_idw_power", 2.0))
    minimum_distance = float(rain_cfg.get("station_idw_min_distance_m", 100.0))

    with rasterio.open(dem_path) as source:
        width = max(1, int(round(source.width * abs(source.transform.a) / resolution)))
        height = max(1, int(round(source.height * abs(source.transform.e) / resolution)))
        dem = source.read(1, out_shape=(height, width), masked=True, resampling=Resampling.nearest)
        transform = source.transform * Affine.scale(source.width / width, source.height / height)
        crs = source.crs
    valid = ~np.ma.getmaskarray(dem)
    # DEMs arrive as integer or floating point; cast before using NaN as the fill.
    valid &= np.isfinite(np.ma.getdata(dem).astype("float64"))
    if CFG.get("study_area", {}).get("calculation_domain") == "dem_landuse_intersection":
        valid &= landuse_mask_for_grid(valid.shape, transform, crs, landuse_path)

    shapes = []
    for zone_id, item in enumerate(model_items, start=1):
        geometry = full_geometry(item["points"], item.get("geometry"))
        shapes.append((geometry, zone_id))
    zone_grid = rasterize(
        shapes,
        out_shape=valid.shape,
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )
    valid &= zone_grid > 0
    weights, cell_counts = idw_mean_weights_by_zone(
        zone_grid,
        valid,
        transform,
        station_x,
        station_y,
        zone_count=len(model_rows),
        power=power,
        minimum_distance_m=minimum_distance,
    )

    missing_zones = np.flatnonzero(cell_counts == 0)
    if missing_zones.size:
        centroids = np.asarray([shape(full_geometry(model_items[index]["points"], model_items[index].get("geometry"))).representative_point().coords[0] for index in missing_zones], dtype="float64")
        weights[missing_zones] = idw_weights_at_points(
            centroids[:, 0],
            centroids[:, 1],
            station_x,
            station_y,
            power=power,
            minimum_distance_m=minimum_distance,
        )

    rainfall_matrix = weights @ observations.rainfall_mm
    area = np.asarray([float(row["area_ha"]) for row in model_rows], dtype="float64")
    domain_series = np.average(rainfall_matrix, axis=0, weights=area)
    write_hourly_rainfall(rainfall_csv, observations.datetimes, domain_series)
    station_csv = Path(rain_cfg.get("station_csv") or "")
    if station_csv:
        write_hourly_rainfall(station_csv, observations.datetimes, domain_series)

    raingage_lines = [
        "[RAINGAGES]",
        ";;Name           Format    Interval SCF      Source     SeriesName",
    ]
    timeseries_lines = [
        "[TIMESERIES]",
        ";;Name           Date       Time       Value",
    ]
    for index, row in enumerate(model_rows):
        gage = f"G{index + 1:04d}"
        series_name = f"TS{index + 1:04d}"
        row["rain_gage"] = gage
        row["rainfall_total_mm"] = float(np.sum(rainfall_matrix[index]))
        row["rainfall_peak_mm_h"] = float(np.max(rainfall_matrix[index]))
        dominant = int(np.argmax(weights[index]))
        row["dominant_rain_station"] = observations.names[dominant]
        row["dominant_station_weight"] = float(weights[index, dominant])
        row["idw_cell_count"] = int(cell_counts[index])
        raingage_lines.append(f"{gage:<16} VOLUME    1:00     1.0      TIMESERIES {series_name}")
        for stamp, value in zip(observations.datetimes, rainfall_matrix[index]):
            timeseries_lines.append(
                f"{series_name:<16} {stamp:%m/%d/%Y} {stamp:%H:%M:%S} {float(value):.4f}"
            )

    weights_csv = out_dir / "auto_swmm_station_idw_weights.csv"
    with weights_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["subcatchment", "rain_gage", "station_key", "station_id", "station_name", "idw_area_weight"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for subcatchment_index, row in enumerate(model_rows):
            for station_index, station_key in enumerate(observations.keys):
                writer.writerow(
                    {
                        "subcatchment": row["subcatchment"],
                        "rain_gage": row["rain_gage"],
                        "station_key": station_key,
                        "station_id": observations.station_ids[station_index],
                        "station_name": observations.names[station_index],
                        "idw_area_weight": f"{weights[subcatchment_index, station_index]:.8f}",
                    }
                )

    rainfall_review = out_dir / "auto_swmm_rainfall_review.csv"
    review_fields = [
        "subcatchment",
        "rain_gage",
        "area_ha",
        "rainfall_total_mm",
        "rainfall_peak_mm_h",
        "dominant_rain_station",
        "dominant_station_weight",
        "idw_cell_count",
    ]
    with rainfall_review.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=review_fields)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in review_fields} for row in model_rows])

    print(
        f"[INFO] Spatial rainfall: stations={len(observations.keys)}; grid={width}x{height}; "
        f"IDW power={power:.2f}; area-mean resolution={resolution:.1f} m"
    )
    fallback_area = float(np.sum(area[missing_zones])) if missing_zones.size else 0.0
    fallback_fraction = 100.0 * fallback_area / max(float(np.sum(area)), 1e-12)
    print(
        f"[INFO] Subcatchment IDW fallback to centroid: {missing_zones.size}; "
        f"area={fallback_area:.4f} ha ({fallback_fraction:.6f}% of model area)"
    )
    print(
        f"[INFO] Subcatchment rainfall totals={np.min(np.sum(rainfall_matrix, axis=1)):.3f}-"
        f"{np.max(np.sum(rainfall_matrix, axis=1)):.3f} mm; domain area-weighted total={np.sum(domain_series):.3f} mm"
    )
    return raingage_lines + [""] + timeseries_lines, domain_series


def write_review_shp(rows: list[dict], target_crs: CRS, out_dir: Path) -> None:
    base = out_dir / "auto_swmm_outfalls"
    writer = shapefile.Writer(str(base), shapeType=shapefile.POINT)
    writer.field("outfall", "C", 24)
    writer.field("subcat", "C", 24)
    writer.field("area_ha", "F", 12, 4)
    writer.field("slope", "F", 12, 4)
    for r in rows:
        writer.point(r["outlet_x"], r["outlet_y"])
        writer.record(r["outfall"], r["subcatchment"], r["area_ha"], r["slope_pct"])
    writer.close()
    base.with_suffix(".prj").write_text(target_crs.to_wkt("WKT1_GDAL"), encoding="utf-8")


def main() -> None:
    dem = Path(CFG["terrain"]["dem"])
    landuse_path = Path(CFG["study_area"].get("landuse_shp") or "")
    derived_boundary = Path(
        CFG["swmm"].get("dem_derived_boundary_shp")
        or ROOT / "inputs/dem_derived/dem_derived_subcatchments.shp"
    )
    if derived_boundary.exists():
        boundary = derived_boundary
        boundary_source = "DEM-derived subcatchments"
    elif CFG["swmm"].get("auto_delineate_subcatchments", True):
        raise FileNotFoundError(f"DEM-derived subcatchments were not written: {derived_boundary}")
    else:
        boundary = Path(CFG["study_area"]["boundary_shp"])
        boundary_source = "user-provided watershed boundary"
    if not boundary.exists():
        raise RuntimeError(f"SWMM partition boundary not found: {boundary}")
    out_inp = Path(os.environ.get("PIPELINE_SWMM_INP") or CFG["swmm"]["working_inp"])
    out_dir = out_inp.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem) as src:
        target_crs = CRS.from_user_input(src.crs)
        dem_bbox = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
    calculation_bbox = dem_bbox
    scope_source = f"DEM extent; partitions={boundary_source}"
    watersheds = read_polygon_records(boundary, target_crs)
    landuse = shapefile_polygon_features(landuse_path, target_crs) if landuse_path.is_file() else []
    landuse_grid, landuse_lookup, landuse_transform = (
        load_landuse_grid(dem, target_crs, landuse_path) if landuse_path.is_file() else (np.zeros((0, 0), dtype=np.int16), {}, None)
    )

    model_rows = []
    model_items = []
    kept_items = []
    skipped = 0
    min_area_ha = float(CFG["swmm"].get("swmm_model_min_area_ha", 0) or 0)
    outlet_selection = str(CFG['swmm'].get('outlet_selection') or 'drainage_exit')
    if outlet_selection not in {'drainage_exit', 'minimum_dem_elevation'}:
        raise ValueError(f'Unknown outlet selection method: {outlet_selection}')
    for item in watersheds:
        if bboxes_intersect(shape(item["geometry"]).bounds, calculation_bbox):
            kept_items.append(item)
        else:
            skipped += 1

    for i, item in enumerate(kept_items, start=1):
        points = item["points"]
        geometry = item["geometry"]
        area_m2 = shape(geometry).area
        if area_m2 <= 0 or area_m2 < min_area_ha * 10000.0:
            continue
        cx, cy = polygon_centroid(points)
        stats = raster_stats(points, dem, geometry)
        land_name, land_frac = dominant_landuse_from_grid(points, landuse_grid, landuse_lookup, landuse_transform, geometry)
        if land_name == "unknown":
            land_name, land_frac = dominant_landuse_by_overlap(points, landuse)
        if CFG.get("study_area", {}).get("calculation_domain", "dem_landuse_intersection") == "dem_landuse_intersection" and land_frac <= 0:
            skipped += 1
            continue
        params = LANDUSE_PARAMS.get(land_name, DEFAULT_PARAM)
        rec = item.get("record", {})
        drainage_x, drainage_y = drainage_outlet_record(rec)
        if outlet_selection == 'drainage_exit':
            with rasterio.open(dem) as source:
                selected = drainage_exit_outlet(rec, geometry, source)
            if selected:
                stats.update(selected)
        flow_length = max(float(np.sqrt(area_m2)), 1.0)
        width = max(area_m2 / flow_length, 5.0)
        model_rows.append(
            {
                "subcatchment": f"S{i:03d}",
                "outfall": f"O{i:03d}",
                "source_name": item["record"].get("name") or item["record"].get("NAME") or item["record"].get("OBJECTID") or i,
                "area_ha": area_m2 / 10000.0,
                "imperv_pct": params["imperv"],
                "width_m": width,
                "slope_pct": stats["slope_pct"],
                "mean_elev_m": stats["mean_elev"],
                "outfall_elev_m": stats["min_elev"],
                "outlet_x": stats["outlet"][0],
                "outlet_y": stats["outlet"][1],
                "outlet_method": stats["outlet_method"],
                "outlet_dem_row": stats["outlet_dem_row"],
                "outlet_dem_col": stats["outlet_dem_col"],
                "drainage_outlet_x": drainage_x,
                "drainage_outlet_y": drainage_y,
                "flow_length_m": flow_length,
                "dominant_landuse": land_name,
                "dominant_landuse_fraction": land_frac,
                "n_imperv": 0.015,
                "n_perv": params["n_perv"],
                "maxrate_mm_h": params["maxrate"],
                "minrate_mm_h": params["minrate"],
                "decay_h": params["decay"],
                "calculation_scope": scope_source,
            }
        )
        model_items.append(item)

    if not model_rows:
        raise RuntimeError("No valid watershed polygons were found for SWMM auto generation.")

    rainfall_csv = Path(os.environ.get("PIPELINE_RAINFALL_CSV") or CFG["rainfall"]["csv"])
    rainfall_section, domain_rainfall = spatial_station_rainfall(
        model_rows,
        model_items,
        dem,
        landuse_path,
        target_crs,
        out_dir,
        rainfall_csv,
    )
    representation = str(CFG["swmm"].get("landuse_representation", "response_units"))
    if representation not in {"response_units", "dominant"}:
        raise ValueError(f"Unknown land-use representation: {representation}")
    if representation == "response_units":
        calculation_rows = build_units(model_rows, model_items, dem, landuse_path, out_dir,
                                       ROOT, LANDUSE_PARAMS, DEFAULT_PARAM)
    else:
        calculation_rows = model_rows
        # Explicit legacy comparisons must not inherit a previous build's mapping.
        for name in RESPONSE_FILES:
            (out_dir / name).unlink(missing_ok=True)
    lines: list[str] = []
    lines.append(f"; LANDUSE_REPRESENTATION {representation}")
    lines.extend(section_options())
    lines.extend(["", "[REPORT]", "INPUT      NO", "CONTROLS   NO", "SUBCATCHMENTS ALL", "NODES ALL", "LINKS ALL", ""])
    lines.extend(rainfall_section)
    lines.extend(["", "[SUBCATCHMENTS]", ";;Name  RainGage  Outlet  Area_ha  %Imperv  Width_m  Slope_%  CurbLen"])
    for r in calculation_rows:
        lines.append(f"{r['subcatchment']:<12} {r['rain_gage']:<8} {r['outfall']:<8} {r['area_ha']:.6f} {r['imperv_pct']:.6f} {r['width_m']:.6f} {r['slope_pct']:.4f} 0")
    lines.extend(["", "[SUBAREAS]", ";;Subcatchment  N-Imperv  N-Perv  S-Imperv  S-Perv  PctZero  RouteTo  PctRouted"])
    for r in calculation_rows:
        lines.append(f"{r['subcatchment']:<12} {r['n_imperv']:.3f} {r['n_perv']:.3f} 1.5 5.0 25 OUTLET")
    lines.extend(["", "[INFILTRATION]", ";;Subcatchment  MaxRate  MinRate  Decay  DryTime  MaxInfil"])
    for r in calculation_rows:
        lines.append(f"{r['subcatchment']:<12} {r['maxrate_mm_h']:.3f} {r['minrate_mm_h']:.3f} {r['decay_h']:.3f} 7.0 0")
    lines.extend(["", "[OUTFALLS]", ";;Name  Elevation  Type  StageData  Gated  RouteTo"])
    for r in model_rows:
        lines.append(f"{r['outfall']:<8} {r['outfall_elev_m']:.3f} FREE  NO")
    lines.extend(["", "[COORDINATES]", ";;Node  X-Coord  Y-Coord"])
    for r in model_rows:
        lines.append(f"{r['outfall']:<8} {r['outlet_x']:.3f} {r['outlet_y']:.3f}")
    lines.extend(["", "[POLYGONS]", ";;Subcatchment  X-Coord  Y-Coord"])
    for item, r in zip(model_items, model_rows):
        if r.get("response_unit_count", 1) > 1:
            # SWMM display polygons cannot encode disjoint subgrid patches.
            # The native index raster is authoritative; the UI draws parents.
            continue
        for x, y in item["points"]:
            lines.append(f"{r['subcatchment']:<12} {x:.3f} {y:.3f}")
    lines.extend(["", "[TAGS]", ""])
    out_inp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if representation == "response_units":
        seal_files(out_dir, out_inp)

    review_csv = out_dir / "auto_swmm_parameter_review.csv"
    with review_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(model_rows[0].keys()))
        writer.writeheader()
        writer.writerows(model_rows)
    write_review_shp(model_rows, target_crs, out_dir)

    print(f"Written SWMM input: {out_inp}")
    print(f"Written parameter review: {review_csv}")
    print(f"Subcatchments: {len(model_rows)}")
    print(f"Land-use representation: {representation}; internal response units: {len(calculation_rows)}")
    print(f"Outlet selection: {outlet_selection}; missing drainage coordinates fall back to native DEM minimum.")
    print(f"Skipped outside DEM extent: {skipped}")
    print(f"Total area: {sum(r['area_ha'] for r in model_rows):.3f} ha")
    print(
        "Calculation bbox: "
        f"{calculation_bbox[0]:.3f}, {calculation_bbox[1]:.3f}, {calculation_bbox[2]:.3f}, {calculation_bbox[3]:.3f}"
    )
    print(f"Area-weighted rainfall total: {float(np.sum(domain_rainfall)):.3f} mm")


if __name__ == "__main__":
    main()
