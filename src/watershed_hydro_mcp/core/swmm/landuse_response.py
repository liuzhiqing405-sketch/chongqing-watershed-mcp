"""Subgrid land-use responses, retaining the original drainage partitions.

One native DEM cell belongs to exactly one (partition, land-use) pair. Units
share their parent's rainfall, slope, effective flow length and outlet. Their
width is proportional to area; Horton parameters are never averaged.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling

from ..spatial_domain import apply_calculation_mask, landuse_class_grid

REVIEW = "auto_swmm_landuse_response_units.csv"
GRID = "auto_swmm_landuse_response_index.tif"
SUMMARY = "auto_swmm_landuse_response_summary.json"
COMPARISON = "auto_swmm_landuse_area_review.csv"
FILES = (REVIEW, GRID, SUMMARY, COMPARISON)


def read_units(directory: Path) -> list[dict]:
    path = Path(directory) / REVIEW
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_units(parents, items, dem_path, landuse_path, directory, project, parameters, fallback):
    """Build native-grid units; fail instead of dropping unrepresented cells."""
    with rasterio.open(dem_path) as source:
        dem = source.read(1, masked=True)
        valid = ~np.ma.getmaskarray(dem) & np.isfinite(np.ma.getdata(dem))
        valid = apply_calculation_mask(valid, source.transform, source.crs, project)
        classes, lookup = (landuse_class_grid(source.shape, source.transform, source.crs, landuse_path)
                           if landuse_path.is_file() else (np.zeros(source.shape, dtype="int16"), {}))
        zones = rasterize([(item["geometry"], i) for i, item in enumerate(items, 1)],
                          out_shape=source.shape, transform=source.transform, fill=0,
                          dtype="int32", all_touched=False)
        valid &= zones > 0
        stride = max(lookup, default=0) + 1
        codes = zones.astype("int64") * stride + classes
        values, counts = np.unique(codes[valid], return_counts=True)
        by_parent = defaultdict(list)
        for value, count in zip(values, counts):
            by_parent[int(value) // stride].append((int(value) % stride, int(count), int(value)))
        table = np.zeros((len(parents) + 1) * stride, dtype="int32")
        units = []
        original_areas = defaultdict(float)
        assigned_areas = defaultdict(float)
        pixel_area_ha = abs(source.transform.determinant) / 10000
        for index, parent in enumerate(parents, 1):
            parts = by_parent[index]
            total = sum(count for _, count, _ in parts)
            if not total:
                raise ValueError(f"分区 {parent['subcatchment']} 没有可映射的原始DEM像元，不能分配地类响应")
            parent_area = float(parent["area_ha"])
            allocated = 0.0
            for part_index, (label, count, code) in enumerate(parts, 1):
                name = lookup.get(label, "unknown")
                param = parameters.get(name, fallback)
                area = parent_area * count / total if part_index < len(parts) else parent_area - allocated
                allocated += area
                fraction = area / parent_area
                unit = dict(parent)
                unit.update(subcatchment=(parent["subcatchment"] if len(parts) == 1
                            else f"{parent['subcatchment']}_L{part_index:02d}"),
                            parent_subcatchment=parent["subcatchment"], response_unit_index=len(units)+1,
                            landuse=name, landuse_fraction=fraction, cell_count=count,
                            parent_area_ha=parent_area, area_ha=area,
                            width_m=float(parent["width_m"]) * fraction,
                            dominant_landuse=name, dominant_landuse_fraction=1.0,
                            imperv_pct=param["imperv"], n_perv=param["n_perv"],
                            maxrate_mm_h=param["maxrate"], minrate_mm_h=param["minrate"],
                            decay_h=param["decay"], landuse_representation="response_units")
                units.append(unit)
                table[code] = len(units)
                original_areas[name] += count * pixel_area_ha
                assigned_areas[name] += area
            selected = units[-len(parts):]
            parent.update(imperv_pct=sum(u["imperv_pct"]*u["landuse_fraction"] for u in selected),
                          landuse_representation="response_units", response_unit_count=len(parts),
                          parameter_scope="parent_summary_only")
            if len(parts) > 1:
                # There is no single equivalent Horton curve or Manning n.
                for key in ("maxrate_mm_h", "minrate_mm_h", "decay_h", "n_perv"):
                    parent[key] = ""
        index_grid = np.zeros(source.shape, dtype="int32")
        index_grid[valid] = table[codes[valid]]
        profile = source.profile.copy()
        profile.update(dtype="int32", count=1, nodata=0, compress="deflate", predictor=2)
        with rasterio.open(directory / GRID, "w", **profile) as destination:
            destination.write(index_grid, 1)
            destination.update_tags(meaning="1-based index in landuse response CSV; zero outside model")
        write_csv(directory / REVIEW, units)
        original_total, assigned_total = sum(original_areas.values()), sum(assigned_areas.values())
        comparison = [dict(landuse=name, native_area_ha=original_areas[name],
                          modeled_area_ha=assigned_areas[name],
                          native_percent=100*original_areas[name]/original_total,
                          modeled_percent=100*assigned_areas[name]/assigned_total)
                      for name in sorted(original_areas)]
        write_csv(directory / COMPARISON, comparison)
        summary = dict(schema_version=1, representation="response_units",
                       parent_count=len(parents), response_unit_count=len(units),
                       total_area_ha=assigned_total, native_area_ha=original_total,
                       native_valid_cells=int(valid.sum()),
                       unknown_area_ha=assigned_areas.get("unknown", 0),
                       class_areas=comparison,
                       spatial_resolution="native_dem", rainfall="inherited_from_parent",
                       geometry="parent_slope_and_flow_length; width scaled by area fraction",
                       parameter_status="existing class assumptions; not calibrated by this change",
                       display="original drainage partitions and outlets; internal units stored in index raster")
        (directory / SUMMARY).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return units


def seal_files(directory: Path, inp: Path) -> None:
    path = directory / SUMMARY
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["sha256"] = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                         for name in (REVIEW, GRID, COMPARISON)}
    summary["inp_sha256"] = hashlib.sha256(inp.read_bytes()).hexdigest()
    summary["inp_name"] = inp.name
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_units(directory: Path, model_rows: list[list[str]], parents: list[dict]) -> dict:
    """Check actual INP against the complete spatial/parameter mapping."""
    summary = json.loads((directory/SUMMARY).read_text(encoding="utf-8"))
    if hashlib.sha256((directory/summary["inp_name"]).read_bytes()).hexdigest() != summary["inp_sha256"]:
        raise ValueError("SWMM输入已改变，与地类响应审计记录不一致")
    for name in (REVIEW, GRID, COMPARISON):
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != summary["sha256"].get(name):
            raise ValueError(f"地类响应映射文件校验失败：{name}")
    units = read_units(directory)
    by_id = {u["subcatchment"]: u for u in units}
    model_ids = [r[0] for r in model_rows]
    if len(by_id) != len(units) or len(set(model_ids)) != len(model_ids) or set(by_id) != set(model_ids):
        raise ValueError("SWMM内部地类单元缺失、重复或与映射表不一致")
    parent_ids = {p["subcatchment"] for p in parents}
    totals = defaultdict(float)
    impervious = defaultdict(float)
    expected_indices = list(range(1, len(units)+1))
    if [int(u["response_unit_index"]) for u in units] != expected_indices:
        raise ValueError("地类单元栅格索引不连续或重复")
    for row in model_rows:
        unit = by_id[row[0]]
        area = float(row[3])
        parent = unit["parent_subcatchment"]
        if (parent not in parent_ids or area <= 0 or not math.isfinite(area)
                or abs(area-float(unit["area_ha"])) > .00000051
                or row[1] != unit["rain_gage"] or row[2] != unit["outfall"]
                or abs(float(row[4])-float(unit["imperv_pct"])) > .0000051):
            raise ValueError(f"地类单元 {row[0]} 面积、降雨、出口或不透水率不一致")
        totals[parent] += area
        impervious[parent] += area*float(row[4])/100
    for parent in parents:
        sid, area = parent["subcatchment"], float(parent["area_ha"])
        tolerance = max(1e-5, sum(u["parent_subcatchment"] == sid for u in units)*.00000051)
        if abs(totals[sid]-area) > tolerance:
            raise ValueError(f"分区 {sid} 的内部地类面积不守恒")
        if abs(impervious[sid]-area*float(parent["imperv_pct"])/100) > tolerance:
            raise ValueError(f"分区 {sid} 的不透水面积不守恒")
    with rasterio.open(directory/GRID) as source:
        grid = source.read(1)
        if np.any(grid < 0) or np.any(grid > len(units)):
            raise ValueError("地类响应栅格含未知索引")
        counts = np.bincount(grid.ravel(), minlength=len(units)+1)[1:]
        if not np.array_equal(counts, [int(u["cell_count"]) for u in units]):
            raise ValueError("地类响应空间像元数与面积记录不一致")
    if len(parents) != summary["parent_count"] or len(units) != summary["response_unit_count"]:
        raise ValueError("地类响应单元数量与审计摘要不一致")
    return dict(passed=True, parent_count=len(parents), response_unit_count=len(units),
                total_area_ha=sum(totals.values()), unknown_area_ha=summary["unknown_area_ha"])


def runoff_mapping(directory, transform, shape, crs, valid, summary):
    """Map each cell to its own land-use response, including exact zero flow."""
    units = read_units(directory)
    if not units:
        raise ValueError("内部地类响应单元表缺失")
    index = np.zeros(shape, dtype="int32")
    with rasterio.open(directory/GRID) as source:
        reproject(source.read(1), index, src_transform=source.transform, src_crs=source.crs,
                  dst_transform=transform, dst_crs=crs, src_nodata=0, dst_nodata=0,
                  resampling=Resampling.nearest)
    if np.any(valid & ((index <= 0) | (index > len(units)))):
        raise ValueError("稳定性计算范围存在未匹配的地类响应单元，请重新建模")
    names, areas = [], []
    coeff, qha, ratio = [np.zeros(len(units)+1, dtype="float32") for _ in range(3)]
    for i, unit in enumerate(units, 1):
        sid = unit["subcatchment"]
        if int(unit["response_unit_index"]) != i or sid not in summary:
            raise ValueError(f"SWMM未返回地类单元 {sid} 的产流结果")
        stat, area = summary[sid], float(unit["area_ha"])
        names.append(sid)
        areas.append(area*10000)
        coeff[i] = stat["runoff_coeff"]
        qha[i] = stat["peak_runoff_cms"]/area
        ratio[i] = stat["runoff_mm"]/max(stat["precip_mm"], 1e-6)
    cell_index = index-1
    cell_index[~valid] = -1
    return coeff[index], qha[index], ratio[index], cell_index, names, areas


def aggregate_series(series, directory):
    """Sum synchronous child discharges, never the individual peak values."""
    from .output_reader import SwmmRunoffSeries
    units = read_units(directory)
    if not units:
        return series
    parents = list(dict.fromkeys(u["parent_subcatchment"] for u in units))
    parent_index = {name:i for i, name in enumerate(parents)}
    child_index = {name:i for i, name in enumerate(series.subcatchment_names)}
    result = np.zeros((len(series.datetimes), len(parents)), dtype="float64")
    for unit in units:
        result[:, parent_index[unit["parent_subcatchment"]]] += series.runoff_cms[:, child_index[unit["subcatchment"]]]
    return SwmmRunoffSeries(parents, series.datetimes, result, series.report_step_seconds)


def write_parent_runoff(output: Path, directory: Path) -> None:
    if not read_units(directory):
        return
    from .output_reader import read_subcatchment_runoff
    series = aggregate_series(read_subcatchment_runoff(output), directory)
    rows = []
    for i, name in enumerate(series.subcatchment_names):
        values = series.runoff_cms[:, i]
        peak = int(np.argmax(values))
        rows.append(dict(parent_subcatchment=name, peak_runoff_cms=float(values[peak]),
                         peak_time=series.datetimes[peak].isoformat(),
                         reporting_step_integrated_volume_m3=float(values.sum()*series.report_step_seconds),
                         volume_note="reporting-step numerical integral; use native RPT for solver water balance"))
    write_csv(output.parent/"auto_swmm_parent_runoff.csv", rows)
