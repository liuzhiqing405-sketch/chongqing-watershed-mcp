"""D8 diagnostics on surrounding *source* DEM, never the polygon-masked DEM.

This screens local rainfall domains, not total outlet discharge or flood risk.
Thresholds are configurable screening tolerances, not engineering standards.
"""
from __future__ import annotations

import math

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from scipy import ndimage
from shapely.geometry import mapping

from workers.hydrology_worker import (
    accumulation_and_order, build_receivers, fill_depressions_priority_flood,
)


def unavailable_check(reason: str, *, fatal: bool = False) -> dict:
    return {"checked": False, "passed": False, "classification": "indeterminate",
            "errors": [reason], "warnings": [], "outlets": [], "inlets": [],
            "complete_natural_watershed_verified": False,"blocking_input_error":fatal}


def analyze_grid(dem, valid, selected, transform, crs, *, max_external_fraction=.05,
                 min_channel_area_km2=.02, min_outlet_fraction=.01, include_routing_signature=False) -> dict:
    """Count unique outside cells reaching the domain and local first-exit areas.

    First-exit labels stop at the cutline; recrossing a nonconvex polygon never
    counts the same local origin twice. Outside origins stop at their first
    entry. A cell count is a terrain contribution area, not a runoff volume.
    """
    valid = np.asarray(valid, dtype=bool) & np.isfinite(dem)
    selected = np.asarray(selected, dtype=bool)
    if dem.shape != selected.shape or dem.shape != valid.shape:
        raise ValueError("水文检查栅格尺寸不一致")
    if not selected.any() or np.any(selected & ~valid):
        return unavailable_check("选区地形不完整，无法进行可靠的水文检查。",fatal=True)
    if (not all(math.isfinite(v) for v in (max_external_fraction, min_outlet_fraction, min_channel_area_km2))
            or not 0 <= max_external_fraction <= 1 or not 0 < min_outlet_fraction <= 1 or min_channel_area_km2 < 0):
        raise ValueError("水文检查阈值无效")
    count = int(selected.sum())
    cell_area = abs(transform.a * transform.e - transform.b * transform.d)
    relief = float(np.ptp(dem[selected]))
    if relief <= .01:
        return unavailable_check("选区地形近乎平坦，无法可靠确定汇流方向；请核对 DEM 或补充水面流向资料。")
    filled, parents = fill_depressions_priority_flood(dem, valid)
    receivers, _ = build_receivers(filled, valid, parents,
                                  cell_width=abs(transform.a), cell_height=abs(transform.e))
    _, order = accumulation_and_order(receivers, valid)
    if len(order) != int(valid.sum()):
        return unavailable_check("DEM 汇流关系存在环路，不能核对上游与出口。",fatal=True)
    inside = selected.ravel()
    active = valid.ravel()
    reaches = inside.copy()
    exits = np.full(receivers.size, -1, dtype=np.int64)
    for index in reversed(order):
        receiver = int(receivers[index])
        if receiver >= 0:
            reaches[index] |= reaches[receiver]
        if inside[index]:
            exits[index] = (exits[receiver] if receiver >= 0 and inside[receiver] else index)
    external = active & ~inside & reaches
    local_counts = np.bincount(exits[inside], minlength=receivers.size)
    # Track only outside origins; propagation stops at the first selected cell.
    outside_counts = (active & ~inside).astype(np.int64)
    inlet_counts = np.zeros(receivers.size, dtype=np.int64)
    for index in order:
        receiver = int(receivers[index])
        if inside[index] or receiver < 0:
            continue
        if inside[receiver]:
            inlet_counts[receiver] += outside_counts[index]
        else:
            outside_counts[receiver] += outside_counts[index]
    edge = valid & ndimage.binary_dilation(~valid, structure=np.ones((3, 3)))
    edge[0, :] |= valid[0, :]; edge[-1, :] |= valid[-1, :]
    edge[:, 0] |= valid[:, 0]; edge[:, -1] |= valid[:, -1]
    # Original downhill directions at edge cells reveal inflow from beyond the
    # read window; a NoData hole is also an unknown contributing-data boundary.
    upstream_edge = edge.ravel() & reaches & active
    threshold = max(min_channel_area_km2 * 1e6 / cell_area, count * min_outlet_fraction)
    to_wgs = Transformer.from_crs(crs, 4326, always_xy=True)

    def locations(weights):
        groups, number = ndimage.label((weights >= threshold).reshape(dem.shape), structure=np.ones((3, 3)))
        result = []
        for label in range(1, number + 1):
            indices = np.flatnonzero(groups.ravel() == label)
            index = int(indices[np.argmax(weights[indices])])
            row, col = divmod(index, dem.shape[1])
            lon, lat = to_wgs.transform(*(transform * (col + .5, row + .5)))
            cells = int(weights[indices].sum())
            result.append({"longitude": lon, "latitude": lat, "contributing_cells": cells,
                           "area_km2": round(cells * cell_area / 1e6, 6), "fraction_of_domain": cells / count})
        return sorted(result, key=lambda item: item["contributing_cells"], reverse=True)

    outlets, inlets = locations(local_counts), locations(inlet_counts)
    external_fraction = int(external.sum()) / count
    errors, warnings = [], []
    fill_fraction = float(np.count_nonzero((filled - dem)[selected] > .1) / count)
    if fill_fraction > .2:
        warnings.append("较大比例地形参与填洼，汇流结果可能受水库、平坦区或 DEM 误差影响；请重点核对水面与出口。")
    if external_fraction > max_external_fraction:
        errors.append("边界切断了明显的上游汇水，当前模型未提供外部入流；请沿分水岭纳入上游，或另行补充边界入流建模。")
    if upstream_edge.any():
        errors.append("影响选区的上游汇流接触检查窗口边缘或 DEM 数据缺口，外部汇水面积可能被低估；需要扩大检查范围或补齐地形。")
    if external.any() and external_fraction <= max_external_fraction:
        warnings.append("存在少量框外汇水，未超过当前筛查容差；模型仅计算框内降雨产流，不包含框外入流。")
    if len(outlets) > 1:
        warnings.append("选区存在多个主要出流位置，可作为多个子汇水区的评价范围；不能把产流之和当作单一河口流量。")
    if not outlets:
        warnings.append("未识别到集中的主要出流位置，可能以分散坡面出流为主；请核对边界与地形。")
    dominant = outlets[0]["fraction_of_domain"] if outlets else 0.
    classification = ("upstream_truncated" if external_fraction > max_external_fraction else
                      "context_incomplete" if upstream_edge.any() else
                      "single_outlet_candidate" if len(outlets) == 1 and dominant >= .7 else "multiple_or_diffuse_outlets")
    # Kept only during context comparison and removed from the public report.
    exit_weights = {}
    for index in np.flatnonzero(local_counts):
        row, col = divmod(int(index), dem.shape[1])
        x, y = transform * (col + .5, row + .5)
        exit_weights[(round(x, 6), round(y, 6))] = int(local_counts[index])
    return {"checked": True, "passed": not errors, "classification": classification,
            **({"_exit_weights": exit_weights} if include_routing_signature else {}),
            "complete_natural_watershed_verified": False, "method": "source_dem_priority_flood_d8",
            "selected_cells": count, "resolution_m": [abs(transform.a), abs(transform.e)],
            "domain_area_km2": count * cell_area / 1e6,
            "external_upstream_cells": int(external.sum()),
            "external_upstream_area_km2": round(int(external.sum()) * cell_area / 1e6, 6),
            "external_area_fraction": external_fraction, "upstream_context_edge_cells": int(upstream_edge.sum()),
            "major_outlet_count": len(outlets), "dominant_outlet_fraction": dominant,
            "diffuse_exit_fraction": 1 - sum(item["fraction_of_domain"] for item in outlets),
            "depression_fill_fraction": fill_fraction,
            "outlets": outlets, "inlets": inlets, "errors": errors, "warnings": warnings,
            "thresholds": {"maximum_external_area_fraction": max_external_fraction,
                           "minimum_channel_area_km2": min_channel_area_km2,
                           "minimum_outlet_area_fraction": min_outlet_fraction}}


def check_polygon(source_path, polygon, options: dict) -> dict:
    """Expand surrounding native-resolution source windows, without resampling.

    No successful result is labelled a verified complete natural watershed:
    finite-window fill and real culverts/reservoirs still need human review.
    """
    buffer_m = float(options.get("initial_buffer_m", 1000))
    attempts = int(options.get("maximum_attempts", 3))
    budget = int(options.get("maximum_cells", 4_000_000))
    if not math.isfinite(buffer_m) or buffer_m <= 0 or not 2 <= attempts <= 6 or budget <= 0:
        raise ValueError("水文检查窗口配置无效")
    history = []
    result = None
    previous = None
    previous_result = None
    with rasterio.open(source_path) as source:
        crs = CRS.from_user_input(source.crs)
        if (not crs.is_projected or any(abs(a.unit_conversion_factor - 1) > 1e-9 for a in crs.axis_info[:2])
                or source.transform.b != 0 or source.transform.d != 0):
            return unavailable_check("水文检查要求以米为单位、无旋转的投影 DEM，请先转换地形数据。",fatal=True)
        west, south, east, north = polygon.bounds
        for attempt in range(attempts):
            margin = buffer_m * 2 ** attempt
            raw = from_bounds(west - margin, south - margin, east + margin, north + margin, source.transform)
            left, top = max(0, math.floor(raw.col_off)), max(0, math.floor(raw.row_off))
            right, bottom = min(source.width, math.ceil(raw.col_off + raw.width)), min(source.height, math.ceil(raw.row_off + raw.height))
            window = Window(left, top, right - left, bottom - top)
            if window.width * window.height > budget:
                if result is None:
                    result = unavailable_check("原分辨率下的水文检查范围超过栅格预算，请缩小选区或提高检查预算；未降低 DEM 分辨率。")
                else:
                    result["errors"].append("扩大窗口复核超过原分辨率栅格预算，水文检查尚未完成；请缩小选区或提高预算。")
                    result["passed"] = False
                break
            key = (left, top, right, bottom)
            if key == previous:
                break
            previous = key
            values = source.read(1, window=window, masked=True).astype("float32")
            transform = source.window_transform(window)
            selected = geometry_mask([mapping(polygon)], values.shape, transform, invert=True)
            result = analyze_grid(values.filled(np.nan), ~np.ma.getmaskarray(values), selected, transform, crs,
                                  max_external_fraction=float(options.get("maximum_external_area_fraction", .05)),
                                  min_channel_area_km2=float(options.get("minimum_channel_area_km2", .02)),
                                  min_outlet_fraction=float(options.get("minimum_outlet_area_fraction", .01)),
                                  include_routing_signature=True)
            result["context_bounds_projected"] = list(rasterio.windows.bounds(window, source.transform))
            history.append({"buffer_m": margin, "cells": int(window.width * window.height),
                            "external_upstream_cells": result.get("external_upstream_cells"),
                            "classification": result["classification"]})
            # A definite major upstream cut is enough to refuse this domain.
            # Clean domains always get a second, larger-context check.
            if not result["checked"] or result["classification"] == "upstream_truncated":
                break
            if len(history) >= 2 and not result.get("upstream_context_edge_cells"):
                tolerance = float(options.get("context_stability_tolerance", .01))
                if not math.isfinite(tolerance) or not 0 <= tolerance <= 1:
                    raise ValueError("水文检查稳定性容差无效")
                before, after = previous_result["_exit_weights"], result["_exit_weights"]
                change = sum(abs(before.get(key, 0) - after.get(key, 0)) for key in before.keys() | after.keys())
                result["context_difference_fraction"] = change / (2 * result["selected_cells"])
                result["context_stability_verified"] = result["context_difference_fraction"] <= tolerance
                if (abs(result["external_area_fraction"] - previous_result["external_area_fraction"]) > tolerance
                        or abs(result["dominant_outlet_fraction"] - previous_result["dominant_outlet_fraction"]) > tolerance
                        or result["major_outlet_count"] != previous_result["major_outlet_count"]
                        or not result["context_stability_verified"]):
                    result["passed"] = False
                    result["context_stability_verified"] = False
                    result["classification"] = "context_unstable"
                    result["errors"].append("扩大检查范围后汇流诊断仍发生明显变化，需扩大地形范围复核，不能确认当前边界。")
                break
            previous_result = result
        result["context_attempts"] = history
        result.pop("_exit_weights", None)
        result.setdefault("context_stability_verified", False)
        result["source_dem"] = str(source_path)
        result["limitations"] = ["D8 地形诊断不包含涵洞、管网、水库调度与地下水流向；检查通过仍需人工核对。",
                                 "框外贡献面积是当前检查窗口内识别的下限，不是实际边界入流量。"]
        return result
