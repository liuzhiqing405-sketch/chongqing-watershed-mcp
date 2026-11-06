from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import sys
import warnings
from datetime import datetime

from watershed_hydro_mcp.core import runtime_bootstrap  # noqa: F401

import numpy as np
import rasterio
from rasterio.enums import WktVersion
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import Resampling
import shapefile
import yaml

from ..spatial_domain import landuse_class_grid, landuse_mask_for_grid, apply_calculation_mask
from ..swmm.output_reader import hourly_runoff_volumes, read_subcatchment_runoff


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from watershed_hydro_mcp.core.common import LazyConfig, RUN_ROOT as ROOT

CFG = LazyConfig()


def apply_env_overrides() -> None:
    """Honour optional path overrides the run planner may set for one run."""
    if os.environ.get("PIPELINE_RAINFALL_CSV"):
        CFG["rainfall"]["csv"] = os.environ["PIPELINE_RAINFALL_CSV"]
    if os.environ.get("PIPELINE_SWMM_RPT"):
        CFG["swmm"]["report"] = os.environ["PIPELINE_SWMM_RPT"]
    if os.environ.get("PIPELINE_SWMM_OUT"):
        CFG["swmm"]["output"] = os.environ["PIPELINE_SWMM_OUT"]
    if os.environ.get("PIPELINE_PROJECT_NAME"):
        CFG["project"]["name"] = os.environ["PIPELINE_PROJECT_NAME"]


NODATA = -9999.0

LANDUSE_STABILITY_PARAMS = {
    "耕地": {"deposit_thickness_m": 1.2, "saturated_cohesion_kpa": 5.0, "saturated_friction_angle_deg": 15.0, "initial_saturation": 0.40},
    "有林地": {"deposit_thickness_m": 1.5, "saturated_cohesion_kpa": 7.0, "saturated_friction_angle_deg": 18.0, "initial_saturation": 0.35},
    "灌木林地": {"deposit_thickness_m": 1.6, "saturated_cohesion_kpa": 5.5, "saturated_friction_angle_deg": 16.0, "initial_saturation": 0.38},
    "房屋建筑（区）": {"deposit_thickness_m": 0.8, "saturated_cohesion_kpa": 10.0, "saturated_friction_angle_deg": 20.0, "initial_saturation": 0.30},
    "水域及水利设施用地": {"deposit_thickness_m": 0.5, "saturated_cohesion_kpa": 20.0, "saturated_friction_angle_deg": 25.0, "initial_saturation": 1.00},
}

D8 = [
    (-1, 0, 1.0),
    (-1, 1, math.sqrt(2.0)),
    (0, 1, 1.0),
    (1, 1, math.sqrt(2.0)),
    (1, 0, 1.0),
    (1, -1, math.sqrt(2.0)),
    (0, -1, 1.0),
    (-1, -1, math.sqrt(2.0)),
]


def prj_wkt(crs) -> str:
    try:
        return crs.to_wkt(version=WktVersion.WKT1_ESRI)
    except Exception:
        return ""


def read_dem_grid() -> tuple[np.ndarray, np.ndarray, Affine, object]:
    dem_path = Path(CFG["terrain"]["dem"])
    resolution = float(CFG.get("stability", {}).get("calculation_resolution_m") or CFG["swmm"].get("delineation_resolution_m") or 20)
    with rasterio.open(dem_path) as src:
        scale_x = max(1, int(round(resolution / abs(src.transform.a))))
        scale_y = max(1, int(round(resolution / abs(src.transform.e))))
        out_h = max(1, src.height // scale_y)
        out_w = max(1, src.width // scale_x)
        data = src.read(1, out_shape=(out_h, out_w), masked=True, resampling=Resampling.bilinear).astype("float64")
        transform = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
        valid = ~np.ma.getmaskarray(data)
        arr = data.filled(np.nan)
        valid &= np.isfinite(arr)
        if CFG.get("study_area", {}).get("calculation_domain", "dem_landuse_intersection") == "dem_landuse_intersection":
            landuse_path = Path(CFG.get("study_area", {}).get("landuse_shp") or "")
            if landuse_path.exists():
                lu_mask = landuse_mask_for_grid(arr.shape, transform, src.crs, landuse_path)
                valid &= lu_mask
                print(f"[INFO] Calculation domain: DEM and land-use intersection; valid cells={int(np.count_nonzero(valid))}")
            else:
                print("[WARN] Land-use layer not found; using DEM valid area only.")
        valid = apply_calculation_mask(valid, transform, src.crs, ROOT)
        return arr, valid, transform, src.crs


def terrain_grids(dem: np.ndarray, transform: Affine, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    filled = dem.copy()
    filled[~np.isfinite(filled)] = float(np.nanmean(filled[valid]))
    cell = float((abs(transform.a) + abs(transform.e)) / 2.0)
    dzdy, dzdx = np.gradient(filled, abs(transform.e), abs(transform.a))
    slope_tan = np.clip(np.sqrt(dzdx * dzdx + dzdy * dzdy), 0.001, 2.0)

    receiver_r = np.full(dem.shape, -1, dtype=np.int32)
    receiver_c = np.full(dem.shape, -1, dtype=np.int32)
    local_drop = np.zeros(dem.shape, dtype="float32")
    rows, cols = dem.shape
    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue
            best_drop = 0.0
            best = None
            z = filled[r, c]
            for dr, dc, dist_factor in D8:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols or not valid[rr, cc]:
                    continue
                drop = (z - filled[rr, cc]) / max(cell * dist_factor, 1e-6)
                if drop > best_drop:
                    best_drop = drop
                    best = (rr, cc)
            if best:
                receiver_r[r, c], receiver_c[r, c] = best
                local_drop[r, c] = best_drop

    order = np.argsort(filled.ravel())[::-1]
    acc = np.zeros(dem.shape, dtype="float64")
    acc[valid] = 1.0
    for idx in order:
        r, c = divmod(int(idx), cols)
        if not valid[r, c]:
            continue
        rr, cc = receiver_r[r, c], receiver_c[r, c]
        if rr >= 0:
            acc[rr, cc] += acc[r, c]

    acc_area_m2 = acc * cell * cell
    slope_eff = np.where(local_drop > 0, local_drop, slope_tan)
    slope_eff = np.clip(slope_eff, 0.001, 2.0)
    for arr in [slope_tan, slope_eff, acc_area_m2]:
        arr[~valid] = np.nan
    return slope_tan, slope_eff, acc_area_m2, filled


def downhill_weights(filled_dem: np.ndarray, valid: np.ndarray, transform: Affine) -> tuple[list[np.ndarray], np.ndarray]:
    cell = float((abs(transform.a) + abs(transform.e)) / 2.0)
    weights = []
    total = np.zeros(filled_dem.shape, dtype="float64")
    for dr, dc, dist_factor in D8:
        shifted = np.roll(filled_dem, shift=(-dr, -dc), axis=(0, 1))
        shifted_valid = np.roll(valid, shift=(-dr, -dc), axis=(0, 1))
        if dr < 0:
            shifted_valid[dr:, :] = False
        elif dr > 0:
            shifted_valid[:dr, :] = False
        if dc < 0:
            shifted_valid[:, dc:] = False
        elif dc > 0:
            shifted_valid[:, :dc] = False
        drop = np.maximum((filled_dem - shifted) / max(cell * dist_factor, 1e-6), 0.0)
        w = np.where(valid & shifted_valid & (drop > 0), np.power(drop, 1.5), 0.0)
        weights.append(w)
        total += w
    total = np.where(total > 0, total, 1.0)
    return weights, total


def read_rainfall() -> list[tuple[str, float]]:
    rain_path = Path(CFG["rainfall"]["csv"])
    time_col = CFG["rainfall"].get("time_column", "datetime_beijing")
    value_col = CFG["rainfall"].get("value_column", "rain_mm")
    if not rain_path.exists():
        raise RuntimeError(f"Rainfall csv not found: {rain_path}. Run rainfall scenario first.")
    values = []
    with rain_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                values.append((row.get(time_col, ""), max(0.0, float(row[value_col]))))
            except Exception:
                continue
    if not values:
        raise RuntimeError(f"No rainfall values found in {rain_path}.")
    return values


def parse_swmm_runoff_summary() -> dict[str, dict[str, float]]:
    rpt_path = Path(CFG["swmm"]["report"])
    if not rpt_path.exists():
        return {}
    rows: dict[str, dict[str, float]] = {}
    in_table = False
    for raw in rpt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line == "Subcatchment Runoff Summary":
            in_table = True
            continue
        if not in_table:
            continue
        if line.startswith("********") and rows:
            break
        parts = line.split()
        if len(parts) < 11 or not parts[0].startswith("S"):
            continue
        try:
            rows[parts[0]] = {
                "precip_mm": float(parts[1]),
                "infil_mm": float(parts[4]),
                "runoff_mm": float(parts[7]),
                "runoff_million_l": float(parts[8]),
                "peak_runoff_cms": float(parts[9]),
                "runoff_coeff": float(parts[10]),
            }
        except ValueError:
            continue
    return rows


def read_swmm_partition_lookup() -> dict[str, str]:
    model_dir = Path(os.environ.get("PIPELINE_SWMM_MODEL_DIR") or ROOT / "swmm" / "model")
    review_path = model_dir / "auto_swmm_parameter_review.csv"
    lookup: dict[str, str] = {}
    if not review_path.exists():
        return lookup
    with review_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sub = str(row.get("subcatchment") or "").strip()
            source = str(row.get("source_name") or "").strip()
            if sub:
                lookup[sub] = sub
            if source and sub:
                lookup[source] = sub
    return lookup


def shape_geometry(shape) -> dict:
    # Ring orientation identifies holes; each shapefile record is one partition.
    return shape.__geo_interface__


def rasterize_landuse_response(directory, transform, grid_shape, valid, rainfall, summary):
    """Use each cell's land-use hydrograph, without spreading it across its parent."""
    from ..swmm.landuse_response import runoff_mapping
    with rasterio.open(CFG["terrain"]["dem"]) as source:
        coeff, qha, ratio, index, names, areas = runoff_mapping(
            directory, transform, grid_shape, source.crs, valid, summary)
    output = Path(CFG["swmm"]["output"])
    series = read_subcatchment_runoff(output)
    positions = {name: i for i, name in enumerate(series.subcatchment_names)}
    if any(name not in positions for name in names):
        raise ValueError("SWMM二进制结果缺少内部地类单元，不能退回估算产流")
    selected = np.array([positions[name] for name in names])
    volumes = hourly_runoff_volumes(series)
    hourly = np.zeros((len(rainfall), len(names)), dtype="float32")
    for i, (stamp, _) in enumerate(rainfall):
        if stamp in volumes:
            hourly[i] = volumes[stamp][selected]/np.asarray(areas)
    positive = qha[valid & (qha > 0)]
    reference = max(float(np.median(positive)) if positive.size else 1e-6, 1e-6)
    print(f"[OK] Land-use hydrographs mapped: units={len(names)}; cells={int(valid.sum())}; zero runoff retained")
    return {"available": True, "dynamic_available": True,
            "runoff_coeff": np.clip(coeff, 0, 1), "peak_q_ha": qha,
            "runoff_depth_ratio": np.clip(ratio, 0, 1),
            "peak_scale": np.clip(qha/reference, .25, 4),
            "subcatchment_index": index, "hourly_runoff_depth_m": hourly,
            "source": str(CFG["swmm"]["report"]), "dynamic_source": str(output),
            "subcatchments": len(names), "landuse_response_units": True}


def rasterize_swmm_runoff(
    transform: Affine,
    shape: tuple[int, int],
    valid: np.ndarray,
    rainfall: list[tuple[str, float]],
) -> dict[str, object]:
    if not CFG.get("hydraulic_response", {}).get("use_swmm_runoff_summary", True):
        one = np.ones(shape, dtype="float32")
        return {"available": False, "runoff_coeff": one, "peak_q_ha": one, "runoff_depth_ratio": one, "source": "disabled"}

    summary = parse_swmm_runoff_summary()
    model_path = Path(CFG.get("swmm", {}).get("working_inp") or ROOT / "swmm/model/model.inp")
    model_directory = Path(os.environ.get("PIPELINE_SWMM_MODEL_DIR") or model_path.parent)
    if model_path.is_file() and "; LANDUSE_REPRESENTATION response_units" in model_path.read_text(encoding="utf-8"):
        return rasterize_landuse_response(model_directory, transform, shape, valid, rainfall, summary)
    shp_path = Path(CFG["swmm"]["dem_derived_boundary_shp"])
    if not summary or not shp_path.exists():
        one = np.ones(shape, dtype="float32")
        return {"available": False, "runoff_coeff": one, "peak_q_ha": one, "runoff_depth_ratio": one, "source": "missing"}

    reader = shapefile.Reader(str(shp_path))
    fields = [f[0] for f in reader.fields[1:]]
    lookup = read_swmm_partition_lookup()
    coeff_shapes = []
    qha_shapes = []
    ratio_shapes = []
    index_shapes = []
    subcatchment_ids: list[str] = []
    subcatchment_area_m2: list[float] = []
    for sr in reader.iterShapeRecords():
        rec = dict(zip(fields, sr.record))
        raw_id = str(
            rec.get("swmm_id")
            or rec.get("SWMM_ID")
            or rec.get("name")
            or rec.get("NAME")
            or rec.get("id")
            or rec.get("ID")
            or ""
        ).strip()
        sid = lookup.get(raw_id, raw_id)
        stats = summary.get(sid)
        if not stats:
            continue
        try:
            area_ha = float(rec.get("area_ha") or rec.get("AREA_HA") or 0)
        except Exception:
            area_ha = 0.0
        peak_q_ha = stats["peak_runoff_cms"] / area_ha if area_ha > 0 else 0.0
        ratio = stats["runoff_mm"] / max(stats["precip_mm"], 1e-6)
        geom = shape_geometry(sr.shape)
        coeff_shapes.append((geom, float(stats["runoff_coeff"])))
        qha_shapes.append((geom, float(peak_q_ha)))
        ratio_shapes.append((geom, float(ratio)))
        index_shapes.append((geom, len(subcatchment_ids)))
        subcatchment_ids.append(sid)
        subcatchment_area_m2.append(max(area_ha * 10000.0, 1.0))

    if not coeff_shapes:
        one = np.ones(shape, dtype="float32")
        return {"available": False, "runoff_coeff": one, "peak_q_ha": one, "runoff_depth_ratio": one, "source": "unmatched"}

    runoff_coeff = rasterize(coeff_shapes, out_shape=shape, transform=transform, fill=np.nan, dtype="float32")
    peak_q_ha = rasterize(qha_shapes, out_shape=shape, transform=transform, fill=np.nan, dtype="float32")
    runoff_depth_ratio = rasterize(ratio_shapes, out_shape=shape, transform=transform, fill=np.nan, dtype="float32")
    for arr in [runoff_coeff, peak_q_ha, runoff_depth_ratio]:
        median = float(np.nanmedian(arr[valid])) if np.any(np.isfinite(arr[valid])) else 1.0
        arr[~np.isfinite(arr)] = median
        arr[~valid] = 0.0

    # When every subcatchment has a zero peak (a very small event), the median of
    # the positive peaks is undefined; keep a finite reference so the scaling
    # below cannot turn the whole runoff grid into NaN.
    positive_peaks = peak_q_ha[valid & (peak_q_ha > 0)]
    peak_ref = max(float(np.nanmedian(positive_peaks)) if positive_peaks.size else 1e-6, 1e-6)
    peak_scale = np.clip(peak_q_ha / peak_ref, 0.25, 4.0).astype("float32")
    subcatchment_index = rasterize(
        index_shapes,
        out_shape=shape,
        transform=transform,
        fill=-1,
        dtype="int32",
    )
    subcatchment_index[~valid] = -1

    dynamic_available = False
    hourly_depth = np.zeros((len(rainfall), len(subcatchment_ids)), dtype="float32")
    output_path = Path(CFG.get("swmm", {}).get("output") or "")
    if output_path.exists():
        try:
            series = read_subcatchment_runoff(output_path)
            volumes = hourly_runoff_volumes(series)
            output_lookup = {name: index for index, name in enumerate(series.subcatchment_names)}
            area = np.asarray(subcatchment_area_m2, dtype="float64")
            matched = np.asarray([output_lookup.get(sid, -1) for sid in subcatchment_ids], dtype="int32")
            available = matched >= 0
            for time_index, (stamp, _) in enumerate(rainfall):
                volume = volumes.get(stamp)
                if volume is None:
                    continue
                hourly_depth[time_index, available] = (
                    volume[matched[available]] / area[available]
                ).astype("float32")
            # A successfully read all-zero hydrograph is still a valid result.
            dynamic_available = bool(np.any(available))
            if dynamic_available:
                print(
                    f"[OK] SWMM binary hydrographs loaded: periods={len(series.datetimes)}; "
                    f"report_step={series.report_step_seconds}s; matched={int(np.count_nonzero(available))}"
                )
        except Exception as exc:
            print(f"[WARN] SWMM binary hydrograph could not be read; using event summary: {exc}")
    return {
        "available": True,
        "dynamic_available": dynamic_available,
        "runoff_coeff": np.clip(runoff_coeff, 0.05, 0.95).astype("float32"),
        "peak_q_ha": peak_q_ha.astype("float32"),
        "runoff_depth_ratio": np.clip(runoff_depth_ratio, 0.05, 0.95).astype("float32"),
        "peak_scale": peak_scale,
        "subcatchment_index": subcatchment_index,
        "hourly_runoff_depth_m": hourly_depth,
        "source": str(CFG["swmm"]["report"]),
        "dynamic_source": str(output_path) if dynamic_available else "",
        "subcatchments": len(coeff_shapes),
    }


def effective_stability_params() -> dict[str, float | str]:
    st_cfg = CFG.get("stability", {})
    preset = str(st_cfg.get("parameter_preset", "custom"))
    params = {
        "mechanics_model": str(st_cfg.get("mechanics_model", "reference_equation_9")),
        "deposit_thickness_m": float(st_cfg.get("deposit_thickness_m", 1.5)),
        "natural_density_g_cm3": float(st_cfg.get("natural_density_g_cm3", 1.85)),
        "saturated_density_g_cm3": float(st_cfg.get("saturated_density_g_cm3", 2.1)),
        "water_density_g_cm3": float(st_cfg.get("water_density_g_cm3", 1.0)),
        "runoff_density_g_cm3": float(st_cfg.get("runoff_density_g_cm3", 1.5)),
        "friction_velocity_coefficient_a": float(st_cfg.get("friction_velocity_coefficient_a", 0.6)),
        "saturated_cohesion_kpa": float(st_cfg.get("saturated_cohesion_kpa", 6.0)),
        "saturated_friction_angle_deg": float(st_cfg.get("saturated_friction_angle_deg", 16.0)),
        "initial_saturation": float(st_cfg.get("initial_saturation", 0.35)),
        "parameter_preset": preset,
    }
    presets = st_cfg.get("presets", {})
    if preset != "custom" and isinstance(presets, dict) and preset in presets:
        for key, value in presets[preset].items():
            params[key] = float(value)
    return params


def stability_parameter_grids(
    shape: tuple[int, int],
    valid: np.ndarray,
    transform: Affine,
    crs,
) -> dict[str, object]:
    base = effective_stability_params()
    h = np.full(shape, float(base["deposit_thickness_m"]), dtype="float32")
    c = np.full(shape, float(base["saturated_cohesion_kpa"]), dtype="float32")
    phi = np.full(shape, float(base["saturated_friction_angle_deg"]), dtype="float32")
    sat = np.full(shape, float(base["initial_saturation"]), dtype="float32")
    class_grid = np.zeros(shape, dtype="int16")
    lookup: dict[int, str] = {}
    used_zones = False

    st_cfg = CFG.get("stability", {})
    if bool(st_cfg.get("use_landuse_zones", False)):
        landuse_path = Path(CFG.get("study_area", {}).get("landuse_shp") or "")
        class_grid, lookup = landuse_class_grid(shape, transform, crs, landuse_path)
        configured = st_cfg.get("landuse_mechanical_params", {})
        for class_id, name in lookup.items():
            params = {}
            params.update(LANDUSE_STABILITY_PARAMS.get(name, {}))
            if isinstance(configured, dict):
                params.update(configured.get(name, {}) or {})
            mask = valid & (class_grid == class_id)
            if not np.any(mask):
                continue
            h[mask] = float(params.get("deposit_thickness_m", base["deposit_thickness_m"]))
            c[mask] = float(params.get("saturated_cohesion_kpa", base["saturated_cohesion_kpa"]))
            phi[mask] = float(params.get("saturated_friction_angle_deg", base["saturated_friction_angle_deg"]))
            sat[mask] = float(params.get("initial_saturation", base["initial_saturation"]))
        used_zones = bool(lookup)

    for arr in [h, c, phi, sat]:
        arr[~valid] = NODATA
    class_grid[~valid] = 0
    return {
        "base": base,
        "h": h,
        "cohesion": c,
        "phi": phi,
        "initial_saturation": sat,
        "class_grid": class_grid,
        "lookup": lookup,
        "used_zones": used_zones,
    }


def hydraulic_width(acc_area_m2: np.ndarray) -> np.ndarray:
    acc_ha = np.maximum(acc_area_m2 / 10000.0, 0.01)
    return np.clip(1.2 * np.power(acc_ha, 0.45), 1.0, 35.0)


def select_representative_samples(
    sample_indices: np.ndarray,
    fs_history: np.ndarray,
    depth_history: np.ndarray,
    velocity_history: np.ndarray,
    rainfall_values: np.ndarray,
    shape: tuple[int, int],
) -> list[dict[str, float | int | str]]:
    """Select contrasting response types from the same spatial sample used in analysis."""
    min_fs = np.nanmin(fs_history, axis=0)
    max_depth = np.nanmax(depth_history, axis=0)
    max_velocity = np.nanmax(velocity_history, axis=0)
    score = (2.0 - min_fs) * (max_depth + 0.05) * (max_velocity + 0.05)
    positive_rain = np.flatnonzero(rainfall_values > 0)
    pre_event_index = max(0, int(positive_rain[0]) - 1) if positive_rain.size else 0
    pre_event_fs = fs_history[pre_event_index]
    fs_drop = pre_event_fs - min_fs

    r1_pool = (pre_event_fs >= 1.0) & (min_fs < 1.0)
    event_score = fs_drop * (max_depth + 0.05) * (max_velocity + 0.05)
    if np.any(r1_pool):
        r1_pos = int(np.nanargmax(np.where(r1_pool, event_score, np.nan)))
    else:
        r1_pos = int(np.nanargmax(event_score))
    r2_pool = (pre_event_fs < 1.0) & (min_fs < 1.0) & (max_depth < np.nanpercentile(max_depth, 50))
    if np.any(r2_pool):
        r2_pos = int(np.nanargmax(np.where(r2_pool, 2.0 - min_fs, np.nan)))
    else:
        r2_pos = int(np.nanargmax(2.0 - min_fs))
    n1_pool = (min_fs >= 1.0) & (event_score > np.nanpercentile(event_score, 75))
    if np.any(n1_pool):
        n1_pos = int(np.nanargmax(np.where(n1_pool, event_score, np.nan)))
    else:
        n1_pos = int(np.nanargmax(min_fs))

    selected = [
        ("R1", r1_pos, "event-induced transition across Fs=1"),
        ("R2", r2_pos, "persistent low Fs with weaker surface-water response"),
        ("N1", n1_pos, "hydrodynamic response while remaining at Fs>=1"),
    ]
    rows: list[dict[str, float | int | str]] = []
    for code, sample_position, label in selected:
        flat_index = int(sample_indices[sample_position])
        row, col = divmod(flat_index, shape[1])
        rows.append(
            {
                "code": code,
                "sample_position": sample_position,
                "flat_index": flat_index,
                "row": row,
                "col": col,
                "type": label,
                "min_fs": float(min_fs[sample_position]),
                "pre_event_fs": float(pre_event_fs[sample_position]),
                "fs_drop": float(fs_drop[sample_position]),
                "max_depth_m": float(max_depth[sample_position]),
                "max_velocity_ms": float(max_velocity[sample_position]),
            }
        )
    return rows


def simulate_response(
    rainfall: list[tuple[str, float]],
    valid: np.ndarray,
    slope_tan: np.ndarray,
    slope_eff: np.ndarray,
    acc_area_m2: np.ndarray,
    filled_dem: np.ndarray,
    transform: Affine,
    swmm_grid: dict[str, np.ndarray | bool | str],
    stability_grid: dict[str, object],
    tracked_flat_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int | str]:
    st_params = stability_grid["base"]
    h_soil = np.asarray(stability_grid["h"], dtype="float64")
    mechanics_model = str(st_params.get("mechanics_model", "reference_equation_9"))
    rho_s = float(st_params["natural_density_g_cm3"])
    rho_sat = float(st_params["saturated_density_g_cm3"])
    rho_w = float(st_params["water_density_g_cm3"])
    rho_runoff = float(st_params["runoff_density_g_cm3"])
    friction_a = float(st_params["friction_velocity_coefficient_a"])
    if not 0.0 < friction_a < 1.0:
        raise ValueError("friction_velocity_coefficient_a must be between 0 and 1.")
    gamma_w = float(st_params["water_density_g_cm3"]) * 9.81
    gamma_sat = float(st_params["saturated_density_g_cm3"]) * 9.81
    gamma_runoff = float(st_params["runoff_density_g_cm3"]) * 9.81
    cohesion = np.asarray(stability_grid["cohesion"], dtype="float64")
    phi = np.deg2rad(np.asarray(stability_grid["phi"], dtype="float64"))
    initial_saturation = np.asarray(stability_grid["initial_saturation"], dtype="float64")
    hydro_cfg = CFG.get("hydraulic_response", {})
    infiltration_coeff = float(hydro_cfg.get("infiltration_coefficient", 0.55))
    recession = float(hydro_cfg.get("recession_factor", 0.72))
    n = float(hydro_cfg.get("manning_n", 0.06))
    substeps = max(1, int(hydro_cfg.get("substeps_per_hour", 6)))
    max_depth_m = float(hydro_cfg.get("max_depth_m", 8.0))
    max_velocity_ms = float(hydro_cfg.get("max_velocity_ms", 12.0))
    max_outflow_fraction = float(hydro_cfg.get("max_outflow_fraction_per_step", 0.65))
    dt = 3600.0 / substeps
    recession_step = np.power(np.clip(recession, 0.0, 1.0), 1.0 / substeps)
    cell = float((abs(transform.a) + abs(transform.e)) / 2.0)

    slope_angle = np.arctan(slope_tan)
    sin_t = np.sin(slope_angle)
    cos_t = np.cos(slope_angle)
    tan_phi = np.tan(phi)
    weights, weight_total = downhill_weights(filled_dem, valid, transform)
    runoff_coeff_grid = np.asarray(swmm_grid["runoff_coeff"], dtype="float64")
    runoff_ratio_grid = np.asarray(swmm_grid["runoff_depth_ratio"], dtype="float64")
    peak_scale_grid = np.asarray(swmm_grid.get("peak_scale", np.ones_like(slope_tan)), dtype="float64")
    dynamic_swmm = bool(swmm_grid.get("dynamic_available", False))
    subcatchment_index = np.asarray(
        swmm_grid.get("subcatchment_index", np.full(slope_tan.shape, -1)),
        dtype="int32",
    )
    hourly_runoff_depth = np.asarray(
        swmm_grid.get("hourly_runoff_depth_m", np.empty((0, 0))),
        dtype="float64",
    )

    max_depth = np.zeros(slope_tan.shape, dtype="float64")
    max_velocity = np.zeros(slope_tan.shape, dtype="float64")
    max_pore = np.zeros(slope_tan.shape, dtype="float64")
    min_fs = np.full(slope_tan.shape, 2.0, dtype="float64")
    unstable_hours = np.zeros(slope_tan.shape, dtype="float64")
    water_depth = np.zeros(slope_tan.shape, dtype="float64")
    cumulative_effective = initial_saturation * h_soil * 0.35
    cumulative_effective[~valid] = 0.0

    acc_ref = max(float(np.nanmedian(acc_area_m2[valid])), 1.0)
    acc_norm = np.log1p(acc_area_m2 / acc_ref)
    acc_norm = acc_norm / max(float(np.nanpercentile(acc_norm[valid], 98)), 1e-6)
    acc_norm = np.clip(acc_norm, 0.05, 3.0)

    peak_rain = max(v for _, v in rainfall)
    peak_time = ""
    min_mean_fs = 2.0
    min_mean_fs_time = ""
    max_unstable_fraction = 0.0
    max_unstable_time = ""
    time_series: list[dict[str, float | str]] = []

    analysis_cfg = CFG.get("analysis", {})
    requested_sample_cells = max(1000, int(analysis_cfg.get("sample_cells", 30000)))
    valid_flat_indices = np.flatnonzero(valid.ravel())
    sample_count = min(requested_sample_cells, int(valid_flat_indices.size))
    sample_rng = np.random.default_rng(int(analysis_cfg.get("random_seed", 20260908)))
    if sample_count < valid_flat_indices.size:
        analysis_sample_indices = np.sort(sample_rng.choice(valid_flat_indices, size=sample_count, replace=False))
    else:
        analysis_sample_indices = valid_flat_indices
    critical_state: dict[str, np.ndarray | str] | None = None
    sampled_fs_history: list[np.ndarray] = []
    sampled_depth_history: list[np.ndarray] = []
    sampled_velocity_history: list[np.ndarray] = []
    sampled_pore_history: list[np.ndarray] = []
    sampled_friction_velocity_history: list[np.ndarray] = []
    sampled_initiation_shear_history: list[np.ndarray] = []
    sampled_resisting_shear_history: list[np.ndarray] = []
    tracked_indices = np.asarray(
        [] if tracked_flat_indices is None else tracked_flat_indices,
        dtype=np.int64,
    )
    if tracked_indices.size:
        tracked_indices = np.unique(tracked_indices)
        if tracked_indices.min() < 0 or tracked_indices.max() >= valid.size:
            raise ValueError("tracked_flat_indices contains an index outside the calculation grid.")
        if not np.all(valid.ravel()[tracked_indices]):
            raise ValueError("tracked_flat_indices contains a cell outside the valid calculation domain.")
    tracked_fs_history: list[np.ndarray] = []
    tracked_depth_history: list[np.ndarray] = []

    for time_index, (stamp, rain_mm) in enumerate(rainfall):
        if dynamic_swmm and time_index < hourly_runoff_depth.shape[0]:
            runoff_supply = np.zeros_like(water_depth)
            mapped = valid & (subcatchment_index >= 0)
            runoff_supply[mapped] = hourly_runoff_depth[
                time_index,
                subcatchment_index[mapped],
            ]
        else:
            runoff_supply = rain_mm / 1000.0 * runoff_coeff_grid * peak_scale_grid
        mean_runoff_input_mm = float(np.nanmean(runoff_supply[valid]) * 1000.0)
        for _ in range(substeps):
            water_depth += runoff_supply / substeps
            water_depth[~valid] = 0.0

            velocity = (1.0 / n) * np.power(np.maximum(water_depth, 0.0), 2.0 / 3.0) * np.sqrt(slope_eff)
            velocity = np.clip(velocity, 0.0, max_velocity_ms)
            outflow = water_depth * np.minimum(max_outflow_fraction, velocity * dt / max(cell, 1e-6))
            outflow = np.where(valid, np.minimum(outflow, water_depth), 0.0)
            inflow = np.zeros_like(water_depth)
            for (dr, dc, _), w in zip(D8, weights):
                part = outflow * (w / weight_total)
                moved = np.roll(part, shift=(dr, dc), axis=(0, 1))
                if dr < 0:
                    moved[dr:, :] = 0.0
                elif dr > 0:
                    moved[:dr, :] = 0.0
                if dc < 0:
                    moved[:, dc:] = 0.0
                elif dc > 0:
                    moved[:, :dc] = 0.0
                inflow += moved
            water_depth = np.maximum(water_depth - outflow + inflow, 0.0) * recession_step
            water_depth = np.clip(water_depth, 0.0, max_depth_m)

        depth = water_depth
        velocity = (1.0 / n) * np.power(np.maximum(depth, 0.0), 2.0 / 3.0) * np.sqrt(slope_eff)
        velocity = np.clip(velocity, 0.0, max_velocity_ms)

        rain_m_h = rain_mm / 1000.0
        cumulative_effective = cumulative_effective * 0.985 + rain_m_h * infiltration_coeff * (1.0 - runoff_ratio_grid) * (0.55 + 0.45 * acc_norm)
        saturation = np.clip(cumulative_effective / np.maximum(h_soil * 0.35, 0.05), 0.0, 1.0)
        if mechanics_model == "reference_equation_9":
            # Wang et al.-style initiation mechanics used by the reference study.
            friction_velocity = -0.44 * (1.0 - 1.0 / friction_a) * velocity
            gravity_shear = (rho_s - rho_w) * 9.81 * h_soil * sin_t
            seepage_shear = rho_w * 9.81 * h_soil * slope_tan
            scour_shear = rho_runoff * np.square(friction_velocity)
            initiation_shear = gravity_shear + seepage_shear + scour_shear

            total_stress = rho_runoff * 9.81 * depth + rho_sat * 9.81 * h_soil
            pore = (rho_s - rho_w) * 9.81 * h_soil + rho_runoff * 9.81 * depth
            resistance = cohesion + np.maximum(total_stress - pore, 0.0) * tan_phi
            fs = np.clip(resistance / np.maximum(initiation_shear, 1e-6), 0.0, 2.0)
        else:
            friction_velocity = np.zeros_like(velocity)
            pore = gamma_w * saturation * h_soil * cos_t * cos_t
            normal = gamma_sat * h_soil * cos_t * cos_t
            driving = gamma_sat * h_soil * sin_t * cos_t
            runoff_shear = gamma_runoff * depth * sin_t
            initiation_shear = driving + runoff_shear
            resistance = cohesion + np.maximum(normal - pore, 0.0) * tan_phi
            fs = np.clip(resistance / np.maximum(initiation_shear, 1e-6), 0.0, 2.0)

        max_depth = np.maximum(max_depth, depth)
        max_velocity = np.maximum(max_velocity, velocity)
        max_pore = np.maximum(max_pore, pore)
        min_fs = np.minimum(min_fs, fs)
        unstable_hours += (fs < 1.0).astype("float64")

        fs_valid = fs[valid]
        mean_fs = float(np.nanmean(fs_valid))
        unstable_fraction = float(np.nanmean((fs_valid < 1.0).astype("float64")))
        sampled_fs = fs.ravel()[analysis_sample_indices]
        sampled_fs_history.append(sampled_fs.astype("float32", copy=True))
        sampled_depth_history.append(depth.ravel()[analysis_sample_indices].astype("float32", copy=True))
        sampled_velocity_history.append(velocity.ravel()[analysis_sample_indices].astype("float32", copy=True))
        sampled_pore_history.append(pore.ravel()[analysis_sample_indices].astype("float32", copy=True))
        sampled_friction_velocity_history.append(friction_velocity.ravel()[analysis_sample_indices].astype("float32", copy=True))
        sampled_initiation_shear_history.append(initiation_shear.ravel()[analysis_sample_indices].astype("float32", copy=True))
        sampled_resisting_shear_history.append(resistance.ravel()[analysis_sample_indices].astype("float32", copy=True))
        if tracked_indices.size:
            tracked_fs_history.append(fs.ravel()[tracked_indices].astype("float32", copy=True))
            tracked_depth_history.append(depth.ravel()[tracked_indices].astype("float32", copy=True))
        time_series.append(
            {
                "datetime": stamp,
                "rain_mm": float(rain_mm),
                "swmm_effective_runoff_mm": mean_runoff_input_mm,
                "mean_fs": mean_fs,
                "sample_median_fs": float(np.nanmedian(sampled_fs)),
                "sample_p05_fs": float(np.nanpercentile(sampled_fs, 5)),
                "sample_p95_fs": float(np.nanpercentile(sampled_fs, 95)),
                "unstable_fraction": unstable_fraction,
                "unstable_area_km2": unstable_fraction * float(np.count_nonzero(valid)) * cell * cell / 1_000_000.0,
                "fraction_fs_lt_1": unstable_fraction,
                "fraction_fs_1_to_1p25": float(np.nanmean(((fs_valid >= 1.0) & (fs_valid < 1.25)).astype("float64"))),
                "fraction_fs_1p25_to_1p5": float(np.nanmean(((fs_valid >= 1.25) & (fs_valid < 1.5)).astype("float64"))),
                "fraction_fs_ge_1p5": float(np.nanmean((fs_valid >= 1.5).astype("float64"))),
            }
        )
        if mean_fs < min_mean_fs:
            min_mean_fs = mean_fs
            min_mean_fs_time = stamp
        if critical_state is None or unstable_fraction > max_unstable_fraction:
            max_unstable_fraction = unstable_fraction
            max_unstable_time = stamp
            critical_state = {
                "critical_time": stamp,
                "flat_index": analysis_sample_indices.copy(),
                "slope_angle_deg": np.rad2deg(slope_angle).ravel()[analysis_sample_indices].copy(),
                "flow_depth_m": depth.ravel()[analysis_sample_indices].copy(),
                "flow_velocity_ms": velocity.ravel()[analysis_sample_indices].copy(),
                "friction_velocity_ms": friction_velocity.ravel()[analysis_sample_indices].copy(),
                "pore_pressure_kpa": pore.ravel()[analysis_sample_indices].copy(),
                "initiation_shear_kpa": initiation_shear.ravel()[analysis_sample_indices].copy(),
                "resisting_shear_kpa": resistance.ravel()[analysis_sample_indices].copy(),
                "cohesion_kpa": cohesion.ravel()[analysis_sample_indices].copy(),
                "friction_angle_deg": np.rad2deg(phi).ravel()[analysis_sample_indices].copy(),
                "soil_thickness_m": h_soil.ravel()[analysis_sample_indices].copy(),
                "natural_density_g_cm3": np.full(sample_count, rho_s, dtype="float64"),
                "saturated_density_g_cm3": np.full(sample_count, rho_sat, dtype="float64"),
                "runoff_density_g_cm3": np.full(sample_count, rho_runoff, dtype="float64"),
                "base_fs": sampled_fs.copy(),
            }
        if rain_mm >= peak_rain:
            peak_time = stamp

    fs_history = np.stack(sampled_fs_history, axis=0)
    depth_history = np.stack(sampled_depth_history, axis=0)
    velocity_history = np.stack(sampled_velocity_history, axis=0)
    pore_history = np.stack(sampled_pore_history, axis=0)
    friction_velocity_history = np.stack(sampled_friction_velocity_history, axis=0)
    initiation_shear_history = np.stack(sampled_initiation_shear_history, axis=0)
    resisting_shear_history = np.stack(sampled_resisting_shear_history, axis=0)
    tracked_fs = (
        np.stack(tracked_fs_history, axis=0)
        if tracked_indices.size
        else np.empty((len(rainfall), 0), dtype="float32")
    )
    tracked_depth = (
        np.stack(tracked_depth_history, axis=0)
        if tracked_indices.size
        else np.empty((len(rainfall), 0), dtype="float32")
    )
    representative_points = select_representative_samples(
        analysis_sample_indices,
        fs_history,
        depth_history,
        velocity_history,
        np.asarray([value for _, value in rainfall], dtype="float64"),
        valid.shape,
    )
    representative_codes = np.full(sample_count, "", dtype="U2")
    for point in representative_points:
        representative_codes[int(point["sample_position"])] = str(point["code"])
    if critical_state is not None:
        critical_state["representative_code"] = representative_codes

    representative_time_series: list[dict[str, float | str]] = []
    for time_index, ((stamp, rain_mm), regional_row) in enumerate(zip(rainfall, time_series)):
        row: dict[str, float | str] = {
            "datetime": stamp,
            "rain_mm": float(rain_mm),
            "swmm_effective_runoff_mm": float(regional_row["swmm_effective_runoff_mm"]),
            "domain_mean_fs": float(regional_row["mean_fs"]),
            "domain_unstable_fraction": float(regional_row["unstable_fraction"]),
        }
        for point in representative_points:
            pos = int(point["sample_position"])
            code = str(point["code"])
            row[f"{code}_fs"] = float(fs_history[time_index, pos])
            row[f"{code}_depth_m"] = float(depth_history[time_index, pos])
            row[f"{code}_velocity_ms"] = float(velocity_history[time_index, pos])
            row[f"{code}_friction_velocity_ms"] = float(friction_velocity_history[time_index, pos])
            row[f"{code}_pore_pressure_kpa"] = float(pore_history[time_index, pos])
            row[f"{code}_initiation_shear_kpa"] = float(initiation_shear_history[time_index, pos])
            row[f"{code}_resisting_shear_kpa"] = float(resisting_shear_history[time_index, pos])
        representative_time_series.append(row)

    for arr in [max_depth, max_velocity, max_pore, min_fs, unstable_hours]:
        arr[~valid] = NODATA

    return {
        "max_depth": max_depth.astype("float32"),
        "max_velocity": max_velocity.astype("float32"),
        "max_pore": max_pore.astype("float32"),
        "min_fs": min_fs.astype("float32"),
        "unstable_hours": unstable_hours.astype("float32"),
        "peak_rain": peak_rain,
        "peak_time": peak_time,
        "min_mean_fs": min_mean_fs,
        "min_mean_fs_time": min_mean_fs_time,
        "max_unstable_fraction": max_unstable_fraction,
        "max_unstable_time": max_unstable_time,
        "rain_hours": len(rainfall),
        "swmm_runoff_used": bool(swmm_grid.get("available")),
        "swmm_dynamic_hydrograph_used": dynamic_swmm,
        "swmm_runoff_source": str(swmm_grid.get("source", "")),
        "swmm_dynamic_source": str(swmm_grid.get("dynamic_source", "")),
        "swmm_subcatchments": int(swmm_grid.get("subcatchments", 0) or 0),
        "mechanics_model": mechanics_model,
        "natural_density_g_cm3": rho_s,
        "saturated_density_g_cm3": rho_sat,
        "runoff_density_g_cm3": rho_runoff,
        "friction_velocity_coefficient_a": friction_a,
        "parameter_preset": str(st_params["parameter_preset"]),
        "landuse_zones_used": bool(stability_grid.get("used_zones")),
        "cohesion_kpa": float(np.nanmean(cohesion[valid])),
        "friction_angle_deg": float(np.nanmean(np.rad2deg(phi)[valid])),
        "deposit_thickness_m": float(np.nanmean(h_soil[valid])),
        "initial_saturation": float(np.nanmean(initial_saturation[valid])),
        "time_series": time_series,
        "critical_state": critical_state,
        "representative_points": representative_points,
        "representative_time_series": representative_time_series,
        "analysis_sample_cells": sample_count,
        "tracked_flat_indices": tracked_indices,
        "tracked_fs_history": tracked_fs,
        "tracked_depth_history": tracked_depth,
    }


def write_raster(path: Path, data: np.ndarray, transform: Affine, crs) -> Path:
    final_path = path
    if path.exists():
        try:
            path.unlink()
        except PermissionError:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_path = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
            print(f"[WARN] Existing raster is locked; writing a new file: {final_path.name}")
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": NODATA,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(final_path, "w", **profile) as dst:
        dst.write(data, 1)
    return final_path


def write_representative_points(out_dir: Path, data: dict, valid: np.ndarray, transform: Affine, crs) -> None:
    base = out_dir / f"{CFG['project']['name']}_representative_response_points"
    writer = shapefile.Writer(str(base), shapeType=shapefile.POINT)
    writer.field("code", "C", 8)
    writer.field("type", "C", 80)
    writer.field("min_fs", "F", 10, 3)
    writer.field("depth_m", "F", 10, 3)
    writer.field("vel_ms", "F", 10, 3)
    for point in data["representative_points"]:
        code = str(point["code"])
        r = int(point["row"])
        c = int(point["col"])
        label = str(point["type"])
        x, y = transform * (c + 0.5, r + 0.5)
        writer.point(float(x), float(y))
        writer.record(
            code,
            label,
            float(point["min_fs"]),
            float(point["max_depth_m"]),
            float(point["max_velocity_ms"]),
        )
    writer.close()
    base.with_suffix(".prj").write_text(prj_wkt(crs), encoding="utf-8")


def write_landuse_parameter_outputs(out_dir: Path, stability_grid: dict[str, object], valid: np.ndarray, transform: Affine, crs) -> list[tuple[Path, np.ndarray, str]]:
    slug = CFG["project"]["name"]
    outputs: list[tuple[Path, np.ndarray, str]] = []
    class_grid = np.asarray(stability_grid["class_grid"], dtype="float32")
    class_grid[~valid] = NODATA
    outputs.append((write_raster(out_dir / f"{slug}_landuse_mechanical_zone_id.tif", class_grid, transform, crs), class_grid, "landuse_mechanical_zone_id"))
    for key, name in [
        ("h", "landuse_deposit_thickness_m"),
        ("cohesion", "landuse_saturated_cohesion_kpa"),
        ("phi", "landuse_saturated_friction_angle_deg"),
        ("initial_saturation", "landuse_initial_saturation"),
    ]:
        arr = np.asarray(stability_grid[key], dtype="float32")
        outputs.append((write_raster(out_dir / f"{slug}_{name}.tif", arr, transform, crs), arr, name))

    table = out_dir / f"{slug}_landuse_mechanical_parameter_lookup.csv"
    lookup = stability_grid.get("lookup", {})
    with table.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["zone_id", "landuse", "pixel_count", "deposit_thickness_m", "saturated_cohesion_kpa", "saturated_friction_angle_deg", "initial_saturation"])
        for zone_id, label in sorted(lookup.items()):
            mask = valid & (np.asarray(stability_grid["class_grid"]) == int(zone_id))
            if not np.any(mask):
                continue
            writer.writerow(
                [
                    zone_id,
                    label,
                    int(np.count_nonzero(mask)),
                    f"{float(np.nanmean(np.asarray(stability_grid['h'])[mask])):.3f}",
                    f"{float(np.nanmean(np.asarray(stability_grid['cohesion'])[mask])):.3f}",
                    f"{float(np.nanmean(np.asarray(stability_grid['phi'])[mask])):.3f}",
                    f"{float(np.nanmean(np.asarray(stability_grid['initial_saturation'])[mask])):.3f}",
                ]
            )
    print(f"[OK] Land-use mechanical parameter lookup: {table}")
    return outputs


def main() -> None:
    dem, valid, transform, crs = read_dem_grid()
    slope_tan, slope_eff, acc_area_m2, filled_dem = terrain_grids(dem, transform, valid)
    rainfall = read_rainfall()
    swmm_grid = rasterize_swmm_runoff(transform, dem.shape, valid, rainfall)
    stability_grid = stability_parameter_grids(dem.shape, valid, transform, crs)

    out_dir = Path(os.environ.get("PIPELINE_FS_OUT_DIR") or ROOT / "rapid_safety_factor_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = CFG["project"]["name"]
    data = simulate_response(rainfall, valid, slope_tan, slope_eff, acc_area_m2, filled_dem, transform, swmm_grid, stability_grid)

    outputs = [
        (out_dir / f"{slug}_enhanced_maximum_flow_depth_m.tif", data["max_depth"], "maximum_flow_depth_m"),
        (out_dir / f"{slug}_enhanced_maximum_velocity_ms.tif", data["max_velocity"], "maximum_velocity_ms"),
        (out_dir / f"{slug}_enhanced_maximum_pore_water_pressure_kpa.tif", data["max_pore"], "maximum_pore_water_pressure_kpa"),
        (out_dir / f"{slug}_enhanced_minimum_safety_factor.tif", data["min_fs"], "minimum_safety_factor"),
        (out_dir / f"{slug}_enhanced_unstable_duration_h.tif", data["unstable_hours"], "unstable_duration_h"),
    ]
    written_outputs = []
    for path, arr, name in outputs:
        written_outputs.append((write_raster(path, arr, transform, crs), arr, name))
    outputs = written_outputs
    parameter_outputs = write_landuse_parameter_outputs(out_dir, stability_grid, valid, transform, crs)

    write_representative_points(out_dir, data, valid, transform, crs)

    summary = out_dir / f"{slug}_enhanced_safety_factor_summary.csv"
    with summary.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["rainfall_hours", data["rain_hours"]])
        writer.writerow(["peak_hourly_rain_mm", f"{data['peak_rain']:.3f}"])
        writer.writerow(["peak_rain_time", data["peak_time"]])
        writer.writerow(["min_mean_fs", f"{data['min_mean_fs']:.3f}"])
        writer.writerow(["min_mean_fs_time", data["min_mean_fs_time"]])
        writer.writerow(["max_unstable_fraction", f"{data['max_unstable_fraction']:.6f}"])
        writer.writerow(["max_unstable_time", data["max_unstable_time"]])
        writer.writerow(["swmm_runoff_used", data["swmm_runoff_used"]])
        writer.writerow(["swmm_dynamic_hydrograph_used", data["swmm_dynamic_hydrograph_used"]])
        writer.writerow(["swmm_runoff_source", data["swmm_runoff_source"]])
        writer.writerow(["swmm_dynamic_source", data["swmm_dynamic_source"]])
        writer.writerow(["swmm_subcatchments", data["swmm_subcatchments"]])
        writer.writerow(["mechanics_model", data["mechanics_model"]])
        writer.writerow(["natural_density_g_cm3", f"{data['natural_density_g_cm3']:.3f}"])
        writer.writerow(["saturated_density_g_cm3", f"{data['saturated_density_g_cm3']:.3f}"])
        writer.writerow(["runoff_density_g_cm3", f"{data['runoff_density_g_cm3']:.3f}"])
        writer.writerow(["friction_velocity_coefficient_a", f"{data['friction_velocity_coefficient_a']:.3f}"])
        writer.writerow(["parameter_preset", data["parameter_preset"]])
        writer.writerow(["landuse_zones_used", data["landuse_zones_used"]])
        writer.writerow(["deposit_thickness_m", f"{data['deposit_thickness_m']:.3f}"])
        writer.writerow(["saturated_cohesion_kpa", f"{data['cohesion_kpa']:.3f}"])
        writer.writerow(["saturated_friction_angle_deg", f"{data['friction_angle_deg']:.3f}"])
        writer.writerow(["initial_saturation", f"{data['initial_saturation']:.3f}"])
        writer.writerow(["analysis_sample_cells", data["analysis_sample_cells"]])
        for _, arr, name in outputs:
            vals = arr[valid & (arr != NODATA)]
            # A global minimum matters most for the safety factor; the other
            # layers use the same five-number summary for consistency.
            writer.writerow([f"{name}_min", f"{float(np.nanmin(vals)):.4f}"])
            writer.writerow([f"{name}_mean", f"{float(np.nanmean(vals)):.4f}"])
            writer.writerow([f"{name}_p95", f"{float(np.nanpercentile(vals, 95)):.4f}"])
            writer.writerow([f"{name}_max", f"{float(np.nanmax(vals)):.4f}"])

    time_series_path = out_dir / f"{slug}_safety_factor_time_series.csv"
    time_series_rows = data["time_series"]
    with time_series_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(time_series_rows[0].keys()))
        writer.writeheader()
        writer.writerows(time_series_rows)

    representative_time_series_path = out_dir / f"{slug}_representative_point_time_series.csv"
    representative_time_series_rows = data["representative_time_series"]
    with representative_time_series_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(representative_time_series_rows[0].keys()))
        writer.writeheader()
        writer.writerows(representative_time_series_rows)

    critical_state_path = out_dir / f"{slug}_critical_state_parameter_sample.csv"
    critical_state = data["critical_state"]
    if not isinstance(critical_state, dict):
        raise RuntimeError("Critical-state sample was not generated.")
    critical_fields = [
        "critical_time",
        "representative_code",
        "flat_index",
        "slope_angle_deg",
        "flow_depth_m",
        "flow_velocity_ms",
        "friction_velocity_ms",
        "pore_pressure_kpa",
        "initiation_shear_kpa",
        "resisting_shear_kpa",
        "cohesion_kpa",
        "friction_angle_deg",
        "soil_thickness_m",
        "natural_density_g_cm3",
        "saturated_density_g_cm3",
        "runoff_density_g_cm3",
        "base_fs",
    ]
    with critical_state_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(critical_fields)
        sample_arrays = [np.asarray(critical_state[name]) for name in critical_fields[1:]]
        for values in zip(*sample_arrays):
            writer.writerow([critical_state["critical_time"], *values])

    readme = out_dir / "README_增强版安全系数结果说明.md"
    readme.write_text(
        "\n".join(
            [
                "# 增强版安全系数模拟结果说明",
                "",
                "本步骤采用 DEM 约束的多方向栅格降雨-径流路由，并按参考文献式（3）至式（9）的启动模型计算事件尺度安全系数。水深、流速、摩阻流速、孔压、起动力和抗力同步输出，便于核查安全系数突变的物理来源。",
                "",
                "当前版本直接读取 SWMM 二进制输出中的 5 分钟子汇水区径流过程，按体积守恒聚合为逐时有效径流深并栅格化回 DEM；二进制过程不可用时才退回事件汇总参数。坡面平均流速通过 a=0.6 转换为摩阻流速，进入冲刷起动力；DEM 坡度控制浮重与渗流力；土地利用分区控制土层厚度、饱和黏聚力和饱和内摩擦角。",
                "",
                "## 主要输出",
                "",
                f"- `{outputs[3][0].name}`：最小安全系数，Fs < 1 表示模拟中达到不稳定状态。",
                f"- `{outputs[4][0].name}`：不稳定持续时间，单位 h。",
                f"- `{outputs[0][0].name}`：过程变量，最大水深，单位 m。",
                f"- `{outputs[1][0].name}`：过程变量，最大流速，单位 m/s。",
                f"- `{outputs[2][0].name}`：过程变量，最大孔隙水压力，单位 kPa。",
                f"- `{parameter_outputs[0][0].name}`：土地利用力学参数分区编号。",
                f"- `{slug}_landuse_mechanical_parameter_lookup.csv`：土地利用类别与力学参数赋值表。",
                f"- `{slug}_representative_response_points.shp`：自动选取的 R1、R2、N1 代表点。",
                f"- `{summary.name}`：本次模拟的统计摘要。",
                f"- `{time_series_path.name}`：逐时平均 Fs、抽样分位数、不稳定比例和不稳定面积。",
                f"- `{representative_time_series_path.name}`：R1、R2、N1 的降雨、水深、流速、摩阻流速、孔压、起动力、抗力和 Fs 同步过程。",
                f"- `{critical_state_path.name}`：最大不稳定面积时刻的均匀空间样本，用于参数敏感性与贡献分析。",
                "",
                "## 方法定位",
                "",
                "该结果服务于灾后快速安全系数判别和现场核查优先区识别。它不处理桥涵、建筑阻水和复杂回水过程，不能替代精细工程水动力模型。",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[OK] Enhanced safety-factor results written: {out_dir}")
    print(f"[OK] Summary: {summary}")
    print(f"[OK] Safety-factor time series: {time_series_path}")
    print(f"[OK] Representative-point time series: {representative_time_series_path}")
    print(f"[OK] Critical-state parameter sample: {critical_state_path}")
    print(f"[INFO] Peak rain={data['peak_rain']:.3f} mm/h at {data['peak_time']}")
    print(f"[INFO] SWMM runoff used={data['swmm_runoff_used']}; subcatchments={data['swmm_subcatchments']}")
    print(f"[INFO] SWMM dynamic hydrograph used={data['swmm_dynamic_hydrograph_used']}; source={data['swmm_dynamic_source']}")
    print(f"[INFO] Mechanics model={data['mechanics_model']}; friction-velocity coefficient a={data['friction_velocity_coefficient_a']:.3f}")
    print(f"[INFO] Parameter preset={data['parameter_preset']}; land-use zones used={data['landuse_zones_used']}")
    print(f"[INFO] Mean parameters: c={data['cohesion_kpa']:.2f} kPa; phi={data['friction_angle_deg']:.2f} deg; H={data['deposit_thickness_m']:.2f} m")
    print(f"[INFO] Minimum mean Fs={data['min_mean_fs']:.3f} at {data['min_mean_fs_time']}")
    print(f"[INFO] Maximum unstable fraction={data['max_unstable_fraction']:.6f} at {data['max_unstable_time']}")


if __name__ == "__main__":
    main()
