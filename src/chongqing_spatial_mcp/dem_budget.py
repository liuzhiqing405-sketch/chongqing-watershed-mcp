"""Exact native-grid budgets shared by boundary intake and input export."""
from __future__ import annotations

import math

from pyproj import CRS
from rasterio.features import geometry_window
from rasterio.windows import Window, bounds, from_bounds
from shapely.geometry import mapping
from watershed_exchange.dem_budget import DEFAULT_MAXIMUM_PROCESSING_CELLS, cell_budget


def processing_options(settings):
    defaults = settings.defaults
    options = defaults.get("model_inputs") or {}
    maximum = (defaults.get("delineation") or {}).get("maximum_processing_cells", DEFAULT_MAXIMUM_PROCESSING_CELLS)
    # Preserve an explicitly configured older, stricter export cap at intake too.
    limits = [maximum]
    if "drainage_context_maximum_cells" in options:
        limits.append(options["drainage_context_maximum_cells"])
    limits = [cell_budget(value) for value in limits]
    buffer_m = float(options.get("drainage_context_buffer_m", 1000))
    if not math.isfinite(buffer_m) or buffer_m <= 0:
        raise ValueError("外围DEM缓冲距离必须为正值")
    return buffer_m, min(limits)


def drainage_window(source, crop_bounds, settings):
    buffer_m, maximum = processing_options(settings)
    if not source.crs or not source.crs.is_projected or source.transform.b or source.transform.d:
        raise ValueError("外围排水DEM需要无旋转的投影栅格")
    unit = CRS.from_user_input(source.crs).axis_info[0].unit_conversion_factor
    distance = buffer_m / unit
    left, bottom, right, top = crop_bounds
    raw = from_bounds(left-distance, bottom-distance, right+distance, top+distance, source.transform)
    x0, y0 = max(0, math.floor(raw.col_off)), max(0, math.floor(raw.row_off))
    x1, y1 = min(source.width, math.ceil(raw.col_off+raw.width)), min(source.height, math.ceil(raw.row_off+raw.height))
    window = Window(x0, y0, x1-x0, y1-y0)
    cells = int(window.width*window.height)
    if cells > maximum:
        raise ValueError(f"所画范围及外围DEM需要{cells:,}像元，超过{maximum:,}像元预算；"
                         "请缩小选区、分区计算或提高明确配置的预算")
    return window, {"buffer_m": buffer_m, "cells": cells, "maximum_cells": maximum,
                    "width": int(window.width), "height": int(window.height),
                    "source_extent_reached": x0 == 0 or y0 == 0 or x1 == source.width or y1 == source.height,
                    "scope": "terrain context only; external rainfall and inflow excluded"}


def preflight_geometry(source, geometry, settings):
    """Inspect dimensions only, before loading terrain or creating artifacts."""
    crop = geometry_window(source, [mapping(geometry)])
    _, facts = drainage_window(source, bounds(crop, source.transform), settings)
    review = geometry_window(source, [mapping(geometry)], pad_x=1, pad_y=1)
    review_cells = int(review.width*review.height)
    if review_cells > facts["maximum_cells"]:
        raise ValueError(f"边界检查DEM需要{review_cells:,}像元，超过{facts['maximum_cells']:,}像元预算；请分区计算")
    return {**facts, "clipped_cells": int(crop.width*crop.height), "review_cells": review_cells}
