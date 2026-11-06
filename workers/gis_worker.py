"""GIS worker for the spatial MCP.

Runs as a subprocess of the MCP server using the project's own interpreter and
dependencies (rasterio, pyproj, pyshp, matplotlib). No QGIS, GRASS or external
Python runtime is required, and no drive letter is assumed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
import shapefile
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.features import shapes as raster_shapes
from rasterio.transform import Affine
from rasterio.windows import from_bounds
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from watershed_exchange.vector_io import attribute_encoding


#: 选址底图的输出分辨率；前端按 image_width/height 与 axes_rect 换算点击位置，
#: 所以它必须和这里的 dpi 保持一致。
OUTLET_MAP_DPI = 135


#: Fallbacks used when no --config is supplied. The service passes
#: config/spatial.default.yaml so these thresholds stay configurable.
DEFAULT_DELINEATION: dict[str, object] = {
    "minimum_area_km2": 1.8,
    "maximum_river_distance_m": 300.0,
    "maximum_named_river_snap_distance_m": 500.0,
    "maximum_processing_cells": 8_000_000,
    "minimum_cell_size_m": 30.0,
    "edge_clearance_km": 10.0,
    "candidate_spacing_km": 2.5,
    "river_vertex_sample": 300,
    # The processing window adapts to the basin instead of being a fixed 40 km
    # square: a basin whose longest flow path is longer than the requested
    # half-window is cut by the window edge, and the QA gate then refuses the
    # boundary — which no choice of outlet can fix, because every candidate on
    # the same river is cut the same way. The window is therefore grown until
    # the basin fits, bounded by four ceilings: attempts, half-window size, grid
    # resolution (growing a window with a fixed cell budget coarsens the grid)
    # and a wall-clock budget.
    "adaptive_window": True,
    "window_growth_factor": 1.5,
    "maximum_window_attempts": 7,
    "maximum_half_window_km": 100.0,
    "maximum_cell_size_m": 100.0,
    "window_time_budget_s": 900.0,
    "minimum_nodata_contact_fraction": 0.005,
    #: 交互式选址底图的范围：短河也要留出点选余地，所以除了按河段包围盒外扩，
    #: 还有一个图幅下限（度）。太小就只能在一两公里的框里点，用户没法挑位置。
    "outlet_map_min_span_deg": 0.06,
    "outlet_map_min_pad_deg": 0.012,
    #: 判定"流域已完整"的方式不是"没触边"，而是"范围再大一圈后流域不变"：
    #: 每次确认把四边各外扩这么多公里，再比一次（面积与包围盒）。
    "window_confirm_margin_km": 5.0,
    "window_stability_tolerance": 0.01,
    #: 流域必须离范围四边至少这么远（km）才算可信：范围边缘处的填洼与平坦区
    #: 流向会失真，"仅仅装得下"会给出假完整（实测 253 km² vs 1011 km²）。
    "window_interior_margin_km": 10.0,
}


def load_defaults(path: Path | None) -> dict:
    """Delineation and rainfall thresholds, from the service's own config file."""
    defaults: dict = dict(DEFAULT_DELINEATION)
    if path is not None and Path(path).is_file():
        import yaml

        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        defaults.update(payload.get("delineation") or {})
        defaults["rainfall"] = payload.get("rainfall") or {}
    return defaults


def _dataset_crs(source: rasterio.DatasetReader) -> CRS:
    if source.crs is None:
        raise ValueError("DEM has no coordinate reference system")
    return CRS.from_user_input(source.crs)


def _transformer(source: CRS, target: CRS) -> Transformer:
    return Transformer.from_crs(source, target, always_xy=True)


def _layer_crs(path: Path) -> CRS | None:
    """CRS of a shapefile from its sidecar, or of a raster from its header."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as source:
            return CRS.from_user_input(source.crs) if source.crs else None
    prj = path.with_suffix(".prj")
    if not prj.is_file():
        return None
    try:
        return CRS.from_wkt(prj.read_text(encoding="utf-8", errors="ignore"))
    except Exception:  # noqa: BLE001 - an unreadable .prj means "unknown CRS"
        return None


def _feature_points(geometry_dict: dict) -> list[tuple[float, float]]:
    """Every vertex of a polygon/multipolygon geometry, outer ring and holes."""
    kind = geometry_dict.get("type")
    if kind == "Polygon":
        polygons = [geometry_dict.get("coordinates", [])]
    elif kind == "MultiPolygon":
        polygons = geometry_dict.get("coordinates", [])
    else:
        return []
    points: list[tuple[float, float]] = []
    for polygon in polygons:
        for ring in polygon:
            points.extend((float(x), float(y)) for x, y in ring)
    return points


def _river_features(path: Path, query: str, sample_count: int = 300
                  ) -> tuple[list[tuple[float, float, str]], list[str]]:
    """(x, y, river name) samples for every river matching the query name."""
    samples: list[tuple[float, float, str]] = []
    matched: set[str] = set()
    for name, rings in _matched_river_features(path, query):
        matched.add(name)
        points = [point for ring in rings for point in ring]
        stride = max(1, len(points) // max(1, sample_count))
        samples.extend((x, y, name) for x, y in points[::stride])
    return samples, sorted(matched)


def normalise_river_query(name: object) -> str:
    """The one place a spoken river name becomes a layer query.

    ``recommend`` and ``outlet_map`` must match the same features, so the
    normalisation lives here rather than being repeated at each call site.
    """
    return str(name).strip().replace("流域", "").replace("水系", "")


def recommend(args: argparse.Namespace) -> dict:
    query = normalise_river_query(args.name)
    if not query:
        raise ValueError("river name is empty after normalisation")
    defaults = getattr(args, "defaults", DEFAULT_DELINEATION)
    samples, matched_names = _river_features(
        args.rivers, query, int(defaults.get("river_vertex_sample", 300)))
    if not samples:
        return {
            "river_name": args.name, "candidates": [], "confidence": "none",
            "message": "本地水系未找到该名称，或匹配河段没有可用DEM像元；需要用户在地图上选点。",
        }
    with rasterio.open(args.dem) as source:
        dem_crs = _dataset_crs(source)
        to_dem = _transformer(_river_layer_crs(args.rivers), dem_crs)
        to_wgs = _transformer(dem_crs, CRS.from_epsg(4326))
        transform = source.transform
        nodata = source.nodata
        height, width = source.height, source.width
        projected = [to_dem.transform(x, y) for x, y, _ in samples]
        columns = [int(round((x - transform.c)) / transform.a) for x, _ in projected]
        rows = [int(round((y - transform.f)) / transform.e) for _, y in projected]
        keep = [index for index, (column, row) in enumerate(zip(columns, rows))
                if 0 <= column < width and 0 <= row < height]
        if not keep:
            return {
                "river_name": args.name, "matched_names": matched_names, "candidates": [],
                "confidence": "none",
                "message": "匹配到的河段不在当前DEM有效范围内；需要用户重新选点。",
            }
        values = list(source.sample(
            [(projected[index][0], projected[index][1]) for index in keep], indexes=1
        ))
    samples_by_index = {index: value[0] for index, value in zip(keep, values)}
    ranked: list[tuple[float, float, str, float, float, str]] = []
    for index in keep:
        elevation = samples_by_index.get(index)
        if elevation is None:
            continue
        elevation = float(elevation)
        if nodata is not None and elevation == nodata:
            continue
        if not math.isfinite(elevation):
            continue
        column, row = columns[index], rows[index]
        edge_m = min(column, row, width - 1 - column, height - 1 - row) * abs(transform.a)
        if (edge_m / 1000.0) < float(defaults.get("edge_clearance_km", 10.0)):
            continue
        x, y = projected[index]
        ranked.append((elevation, -edge_m, x, y, samples[index][2]))
    if not ranked:
        return {
            "river_name": args.name, "matched_names": matched_names, "candidates": [],
            "confidence": "none",
            "message": "匹配河段没有满足“远离DEM边界”的可用像元；需要用户在地图上选点。",
        }
    candidates: list[dict] = []
    for elevation, negative_edge, x, y, name in sorted(ranked):
        spacing_m = float(defaults.get("candidate_spacing_km", 2.5)) * 1000.0
        if any(math.hypot(x - item["projected_x"], y - item["projected_y"]) < spacing_m
               for item in candidates):
            continue
        longitude, latitude = to_wgs.transform(x, y)
        candidates.append({
            "longitude": round(longitude, 7), "latitude": round(latitude, 7),
            "projected_x": round(x, 2), "projected_y": round(y, 2),
            "elevation_m": round(elevation, 2), "dem_edge_distance_km": round(-negative_edge / 1000, 2),
            "matched_river": name, "confidence": "candidate_only",
        })
        if len(candidates) >= args.limit:
            break
    return {
        "river_name": args.name, "matched_names": matched_names, "candidates": candidates,
        "confidence": "candidate_only", "requires_delineation_and_human_confirmation": True,
        "ranking_note": "低高程、远离DEM边界且彼此分散的河流采样点；不是唯一水文出口。",
    }


def _matched_river_features(path: Path, query: str) -> list[tuple[str, list[list[tuple[float, float]]]]]:
    """Exact river-name matches, allowing only an omitted final 河.

    Rings are in the layer's own coordinates; the caller reprojects.
    """
    reader = shapefile.Reader(str(path), encoding=attribute_encoding(path))
    matched: list[tuple[str, list[list[tuple[float, float]]]]] = []
    try:
        for record in reader.iterShapeRecords():
            attributes = record.record.as_dict()
            name = str(attributes.get("Name") or "").strip()
            kind = str(attributes.get("kind") or "").strip()
            if kind and kind != "0123":
                continue
            if not name or name not in {query, query + "河"}:
                continue
            points = record.shape.points
            parts = list(record.shape.parts) + [len(points)]
            rings = [[(float(x), float(y)) for x, y in points[start:end]]
                     for start, end in zip(parts[:-1], parts[1:])]
            if rings:
                matched.append((name, rings))
    finally:
        reader.close()
    return matched


def _all_river_rings(path: Path, west: float, south: float, east: float, north: float
                    ) -> list[list[tuple[float, float]]]:
    """Rings of every feature overlapping the given WGS84 box (drawn as context)."""
    reader = shapefile.Reader(str(path), encoding=attribute_encoding(path))
    rings: list[list[tuple[float, float]]] = []
    try:
        for record in reader.iterShapeRecords():
            box = record.shape.bbox
            if box[0] > east or box[2] < west or box[1] > north or box[3] < south:
                continue
            points = record.shape.points
            parts = list(record.shape.parts) + [len(points)]
            rings.extend([(float(x), float(y)) for x, y in points[start:end]]
                         for start, end in zip(parts[:-1], parts[1:]))
    finally:
        reader.close()
    return rings


def _river_layer_crs(path: Path) -> CRS:
    """CRS of the river layer from its .prj, defaulting to WGS84 (the managed layer)."""
    prj = path.with_suffix(".prj")
    if prj.is_file():
        try:
            return CRS.from_user_input(prj.read_text(encoding="utf-8", errors="ignore"))
        except Exception:  # noqa: BLE001 - an unreadable .prj means "unknown"
            pass
    return CRS.from_epsg(4326)


def outlet_map(args: argparse.Namespace) -> dict:
    """推荐出水口候选 + 一张可点选的选址底图（PNG base64 与地理范围）。"""
    import base64
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    _configure_fonts(plt)

    defaults = getattr(args, "defaults", DEFAULT_DELINEATION)
    longitude, latitude = getattr(args, "lon", None), getattr(args, "lat", None)
    if (longitude is None) != (latitude is None):
        raise ValueError("地图中心必须同时提供经度和纬度")
    coordinate_centre = longitude is not None
    if coordinate_centre and (not math.isfinite(longitude) or not math.isfinite(latitude)
                              or not -180 <= longitude <= 180 or not -90 <= latitude <= 90):
        raise ValueError("地图中心经纬度无效")
    match_ns = argparse.Namespace(dem=args.dem, rivers=args.rivers, name=args.name,
                                  limit=args.limit, defaults=defaults)
    recommendation = (recommend(match_ns) if args.limit and not coordinate_centre else {
        "river_name": args.name, "candidates": [], "confidence": "manual_selection",
    })
    matched = _matched_river_features(args.rivers, normalise_river_query(args.name))
    if not matched and not coordinate_centre:
        return dict(recommendation, map=None, matched_names=[],
                    message=recommendation.get("message")
                    or "本地水系未找到该名称，无法出图；请换一个河名或直接给出坐标。")
    recommendation["matched_names"] = sorted({name for name, _ in matched})

    layer_crs = _river_layer_crs(args.rivers)
    layer_to_wgs = Transformer.from_crs(layer_crs, CRS.from_epsg(4326), always_xy=True)
    # 范围按匹配河段的经纬度包围盒定，四边再外扩，让短河也有上下文。
    lonlats = ([(longitude, latitude)] if coordinate_centre else [layer_to_wgs.transform(x, y)
               for _, rings in matched for ring in rings for x, y in ring])
    if coordinate_centre:
        recommendation.update(confidence="coordinate_centre", message=
                              "按推测位置加载地图；河名位置尚未核实，请在河道上选实际出水口。")
    lons = [point[0] for point in lonlats]
    lats = [point[1] for point in lonlats]
    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)
    span = max(east - west, north - south, 0.02)
    pad = max(span * 0.25, float(defaults.get("outlet_map_min_pad_deg", 0.012)))
    west, east, south, north = west - pad, east + pad, south - pad, north + pad
    # 短河也要留出点选余地：图幅不足下限时以河段为中心扩到下限（约 6.7 km）。
    minimum_span = float(defaults.get("outlet_map_min_span_deg", 0.06))
    centre_lon, centre_lat = (west + east) / 2, (south + north) / 2
    half_lon = max((east - west) / 2, minimum_span / 2)
    half_lat = max((north - south) / 2, minimum_span / 2)
    if coordinate_centre:
        # A point locates a drawing viewport; it does not define a watershed.
        # Leave enough context to draw a small study domain around the point.
        half_lat=max(half_lat,20/111.32)
        half_lon=max(half_lon,20/(111.32*math.cos(math.radians(centre_lat))))
    west, east = centre_lon - half_lon, centre_lon + half_lon
    south, north = centre_lat - half_lat, centre_lat + half_lat

    with rasterio.open(args.dem) as source:
        crs = source.crs
        to_raster = Transformer.from_crs(layer_crs, crs, always_xy=True)
        to_wgs = Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)
        wgs_to_raster = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
        if coordinate_centre:
            cx, cy = wgs_to_raster.transform(longitude, latitude)
            if not (source.bounds.left <= cx < source.bounds.right
                    and source.bounds.bottom < cy <= source.bounds.top):
                return dict(recommendation, map=None, message=
                            "推测位置超出当前 DEM 覆盖范围。请补充所在区县或附近地标，或提供该区域数据。")
            sample = next(source.sample([(cx, cy)], masked=True))
            if np.ma.getmaskarray(sample).any() or not np.isfinite(sample).all():
                return dict(recommendation, map=None, message=
                            "推测位置没有有效地形数据。请补充所在区县或附近地标。")
        corners = [wgs_to_raster.transform(lon, lat)
                   for lon, lat in ((west, south), (east, south), (west, north), (east, north))]
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        window = from_bounds(min(xs), min(ys), max(xs), max(ys), source.transform)
        resolution = max(abs(source.transform.a), (max(xs) - min(xs)) / 1400.0)
        height = max(1, int(round((max(ys) - min(ys)) / resolution)))
        width = max(1, int(round((max(xs) - min(xs)) / resolution)))
        # Render in the same geographic coordinates that the browser interpolates.
        # A UTM rectangle with only a geographic bounding box shifts clicks off
        # the river, especially on large map frames.
        from rasterio.warp import reproject
        from rasterio.transform import from_bounds as geographic_transform
        geographic_grid = geographic_transform(west, south, east, north, width, height)
        dem = np.full((height, width), np.nan, dtype="float64")
        reproject(source=rasterio.band(source, 1), destination=dem,
                  src_transform=source.transform, src_crs=crs, src_nodata=source.nodata,
                  dst_transform=geographic_grid, dst_crs="EPSG:4326", dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
        nodata = source.nodata
    if nodata is not None:
        dem[dem == nodata] = np.nan
    finite = dem[np.isfinite(dem)]
    if not finite.size or min(dem.shape) < 2:
        return dict(recommendation, map=None, message="图幅没有有效地形数据，请补充位置或该区域数据。")
    gy, gx = np.gradient(np.nan_to_num(dem, nan=float(np.nanmean(finite)) if finite.size else 0.0))
    slope = np.arctan(np.hypot(gx / resolution, gy / resolution))
    aspect = np.arctan2(-gx, gy)
    azimuth, altitude = np.radians(315.0), np.radians(45.0)
    shade = (np.sin(altitude) * np.cos(slope)
             + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect))
    shade = (shade - np.nanmin(shade)) / max(float(np.nanmax(shade) - np.nanmin(shade)), 1e-9)

    def to_xy(rings):
        return [[layer_to_wgs.transform(lon, lat) for lon, lat in ring] for ring in rings]

    figure, axis = plt.subplots(figsize=(9.6, 7.6), constrained_layout=True)
    extent = (west, east, south, north)
    axis.imshow(shade, extent=extent, cmap="gray", origin="upper", alpha=0.92, zorder=1)
    axis.imshow(dem, extent=extent, cmap="terrain", origin="upper", alpha=0.45, zorder=2,
                vmin=float(np.percentile(finite, 2)) if finite.size else None,
                vmax=float(np.percentile(finite, 98)) if finite.size else None)

    wgs_to_layer = Transformer.from_crs(CRS.from_epsg(4326), layer_crs, always_xy=True)
    box_in_layer = [wgs_to_layer.transform(lon, lat)
                    for lon, lat in ((west, south), (east, south), (west, north), (east, north))]
    box_lons = [point[0] for point in box_in_layer]
    box_lats = [point[1] for point in box_in_layer]
    for ring in _all_river_rings(args.rivers, min(box_lons), min(box_lats),
                                 max(box_lons), max(box_lats)):
        projected = [layer_to_wgs.transform(lon, lat) for lon, lat in ring]
        axis.fill([point[0] for point in projected], [point[1] for point in projected],
                  color="#2f6ea6", alpha=0.85, linewidth=0, zorder=4)
    for _, rings in matched:
        for ring in to_xy(rings):
            xs2, ys2 = zip(*ring)
            axis.fill(xs2, ys2, color="#00d0ff", alpha=0.95, linewidth=0.6,
                      edgecolor="white", zorder=5)

    candidates = recommendation.get("candidates") or []
    reference_legend = None
    if getattr(args,"display_rivers",None) is not None:
        from chongqing_spatial_mcp.map_hydrography import draw_reference
        reference_legend = draw_reference(axis,args.display_rivers,(west,south,east,north),
                                          args.display_rivers_label,width,height)
    for index, item in enumerate(candidates, start=1):
        cx, cy = item["longitude"], item["latitude"]
        axis.plot([cx], [cy], marker="o", ms=13, mfc="#ffd400", mec="#111", mew=1.6, zorder=8)
        axis.annotate(str(index), (cx, cy), color="#111", fontsize=10, fontweight="bold",
                      ha="center", va="center", zorder=9)

    if candidates:
        mid = candidates[0]
        mx, my = mid["longitude"], mid["latitude"]
        axis.annotate(f"{mid['longitude']:.5f}E\n{mid['latitude']:.5f}N",
                      (mx, my), xytext=(12, 12), textcoords="offset points", fontsize=8.5,
                      zorder=10, bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                           ec="#111", alpha=0.9))

    # 经纬度刻度：用 UTM 坐标画，标签换算成经纬度，便于用户把图上位置说出来。
    xticks = np.linspace(extent[0], extent[1], 7)
    yticks = np.linspace(extent[2], extent[3], 7)
    axis.set_xticks(xticks)
    axis.set_yticks(yticks)
    axis.set_xticklabels([f"{x:.4f}"
                          for x in xticks], fontsize=8.5)
    axis.set_yticklabels([f"{y:.4f}"
                          for y in yticks], fontsize=8.5)
    axis.set_xlabel("经度 (°E)")
    axis.set_ylabel("纬度 (°N)")
    axis.grid(True, ls=":", color="white", alpha=0.35)

    km_per_degree = 111.32 * math.cos(math.radians(centre_lat))
    span_km = (east - west) * km_per_degree
    bar = max(1.0, float(round(span_km / 5)))
    axis.add_patch(plt.Rectangle(
        (extent[0] + (extent[1] - extent[0]) * 0.06, extent[2] + (extent[3] - extent[2]) * 0.06),
        bar / km_per_degree, (extent[3] - extent[2]) * 0.008, color="#111", zorder=10))
    axis.annotate(f"{bar:g} km",
                  (extent[0] + (extent[1] - extent[0]) * 0.06 + bar / km_per_degree / 2,
                   extent[2] + (extent[3] - extent[2]) * 0.06),
                  xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9, zorder=10)

    legend_handles = [
        Patch(fc="#2f6ea6", label="其他河流（底图）"),
    ]
    if reference_legend is not None:
        legend_handles.append(reference_legend)
    if matched:
        legend_handles.insert(0, Patch(fc="#00d0ff", ec="white", label="匹配到的河段"))
    if candidates:
        legend_handles.append(Line2D([], [], color="#ffd400", marker="o", ms=11,
                                     ls="none", mec="#111", label="推荐出水口候选"))
    axis.legend(handles=legend_handles, loc="upper right", fontsize=9)
    axis.set_xlim(west, east)
    axis.set_ylim(south, north)
    axis.set_title(f"「{args.name}」评价范围绘制底图\n"
                   "河名或坐标用于定位，请逐点绘制闭合评价边界",
                   fontsize=11.5)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=OUTLET_MAP_DPI)
    axes_rect = list(axis.get_position().bounds)  # 图内绘图区，比例坐标，原点在左下
    size_inches = figure.get_size_inches()
    plt.close(figure)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    image_width = int(round(float(size_inches[0]) * OUTLET_MAP_DPI))
    image_height = int(round(float(size_inches[1]) * OUTLET_MAP_DPI))

    return {
        "river_name": recommendation.get("river_name", args.name),
        "matched_names": recommendation.get("matched_names", []),
        "candidates": candidates,
        "confidence": recommendation.get("confidence"),
        "ranking_note": recommendation.get("ranking_note"),
        "message": recommendation.get("message"),
        "map": {
            "media_type": "image/png",
            "image_base64": image_b64,
            "image_width": image_width,
            "image_height": image_height,
            "bbox_wgs84": [round(west, 7), round(south, 7), round(east, 7), round(north, 7)],
            #: 前端换算点击位置用：先按图像尺寸归一化，再按 axes_rect 裁到绘图区。
            "axes_rect": [round(value, 6) for value in axes_rect],
            "crs": "EPSG:4326",
            "display_crs": "EPSG:4326",
            "note": "底图与河道均以 WGS84 经纬度绘制。bbox_wgs84 对应实际绘图区；"
                    "axes_rect=[左,下,宽,高]，原点左下，点击按这两项换算经纬度。",
        },
    }


def _write_window(
    source: rasterio.DatasetReader,
    destination: Path,
    west: float,
    south: float,
    east: float,
    north: float,
    resolution: float,
) -> tuple[Affine, int, int]:
    window = from_bounds(west, south, east, north, source.transform)
    width = max(1, int(round((east - west) / resolution)))
    height = max(1, int(round((north - south) / resolution)))
    data = source.read(
        1, window=window, out_shape=(height, width),
        resampling=Resampling.bilinear, masked=True,
    )
    transform = Affine.translation(west, north) * Affine.scale(
        (east - west) / width, -(north - south) / height
    )
    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": data.dtype.name, "crs": source.crs, "transform": transform,
        "nodata": -9999.0, "compress": "lzw",
    }
    array = data.astype("float32").filled(-9999.0)
    with rasterio.open(destination, "w", **profile) as target:
        target.write(array, 1)
    return transform, width, height


def _polygonize_basin(basin_path: Path, destination: Path, basin_crs: CRS) -> None:
    with rasterio.open(basin_path) as source:
        mask = source.read(1) == 1
        polygons = [
            shape(geometry)
            for geometry, value in raster_shapes(mask.astype("uint8"), mask=mask, transform=source.transform)
            if value == 1
        ]
    if not polygons:
        raise ValueError("No watershed polygon was generated")
    merged = unary_union(polygons)
    to_wgs = _transformer(basin_crs, CRS.from_epsg(4326))
    from shapely.ops import transform as shapely_transform

    geographic = shapely_transform(lambda x, y, z=None: to_wgs.transform(x, y), merged)
    destination.write_text(
        json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {}, "geometry": mapping(geographic)}
        ]}, ensure_ascii=False),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class CutReport:
    """How a basin was cut off: which window edges, and how much DEM no-data.

    The count describes contact inside the available DEM; it does not estimate
    missing upstream area outside the DEM. Named-river completeness therefore
    uses a stricter source-coverage check even for a small contact fraction.
    """

    sides: tuple[str, ...] = ()
    nodata_contact_cells: int = 0
    basin_cells: int = 0
    #: The measured basin's projected bounding box (west, south, east, north).
    #: The next crop is built from this: an uncut side only needs to cover the
    #: basin plus a margin, and the cells saved there pay for a finer grid on
    #: the side that still has to grow.
    basin_bounds: tuple[float, float, float, float] | None = None

    @property
    def nodata_fraction(self) -> float:
        return self.nodata_contact_cells / self.basin_cells if self.basin_cells else 0.0

    @property
    def touches_dem_nodata(self) -> bool:
        return self.nodata_contact_cells > 0


def _boundary_truncated(basin_path: Path, window_path: Path) -> CutReport:
    """Which window edges the basin reaches, and how much DEM no-data it meets.

    The sides matter: a basin cut on its northern edge needs a window that
    reaches further north, while a side that is already the DEM's own edge
    cannot be fixed by any window — there is no data beyond it to route from.
    """
    with rasterio.open(basin_path) as basin_source:
        basin = basin_source.read(1) == 1
        basin_transform = basin_source.transform
        rows = np.flatnonzero(basin.any(axis=1))
        cols = np.flatnonzero(basin.any(axis=0))
        if rows.size and cols.size:
            left, top = basin_transform * (float(cols[0]), float(rows[0]))
            right, bottom = basin_transform * (float(cols[-1] + 1), float(rows[-1] + 1))
            basin_bounds = (left, bottom, right, top)
        else:
            basin_bounds = None
    sides: list[str] = []
    # The window is written north-up, so row 0 is the northern edge.
    if basin[0].any():
        sides.append("north")
    if basin[-1].any():
        sides.append("south")
    if basin[:, 0].any():
        sides.append("west")
    if basin[:, -1].any():
        sides.append("east")
    with rasterio.open(window_path) as window_source:
        masked = window_source.read(1, masked=True)
        nodata = np.ma.getmaskarray(masked)
    adjacent = nodata.copy()
    for delta_row in (-1, 0, 1):
        for delta_column in (-1, 0, 1):
            adjacent |= np.roll(np.roll(nodata, delta_row, axis=0), delta_column, axis=1)
    return CutReport(sides=tuple(sides),
                     nodata_contact_cells=int(np.count_nonzero(basin & adjacent)),
                     basin_cells=int(np.count_nonzero(basin)),
                     basin_bounds=basin_bounds)


@dataclass(frozen=True)
class WindowPlan:
    """One attempt's window: the square around the outlet, clamped to the DEM."""

    half_km: float
    west: float
    south: float
    east: float
    north: float
    resolution_m: float
    cells: int
    clamped_sides: tuple[str, ...] = ()
    #: The per-side extents this attempt asked for, in km from the outlet.
    extent_km: dict[str, float] = field(default_factory=dict)

    @property
    def width_km(self) -> float:
        return (self.east - self.west) / 1000.0

    @property
    def height_km(self) -> float:
        return (self.north - self.south) / 1000.0


def _plan_window(bounds, x: float, y: float, extent: dict[str, float],
                 defaults: dict) -> WindowPlan | None:
    """The processing range around the outlet, clamped into the DEM.

    ``extent`` is per side and in kilometres, so the range can take the basin's
    own shape instead of being a square. That matters because the cell budget
    is fixed: an elongated basin leaves most of a square's cells on empty
    corners, and those wasted cells are exactly what forces the grid coarser on
    the side that still has to grow.

    ``None`` means the DEM cannot host even the smallest useful range here —
    an outlet outside the data, or a sliver pressed against its edge. Clamping
    is what lets an outlet *near* the edge be processed at all, and the clamped
    sides are recorded so a basin cut by the data boundary is told apart from
    one that is merely larger than the range.
    """
    west = x - float(extent["west"]) * 1000.0
    east = x + float(extent["east"]) * 1000.0
    south = y - float(extent["south"]) * 1000.0
    north = y + float(extent["north"]) * 1000.0
    clamped: list[str] = []
    if west < bounds.left:
        west, clamped = bounds.left, [*clamped, "west"]
    if east > bounds.right:
        east, clamped = bounds.right, [*clamped, "east"]
    if south < bounds.bottom:
        south, clamped = bounds.bottom, [*clamped, "south"]
    if north > bounds.top:
        north, clamped = bounds.top, [*clamped, "north"]
    # ``None`` only for the degenerate cases: an outlet the DEM does not cover
    # at all, or a window that is a sliver against the data boundary. An outlet
    # merely *near* the boundary is not refused any more — the window is
    # clamped, and if that costs the basin its headwaters, the truncation check
    # says so in terms of the data range rather than as an opaque refusal.
    if not (bounds.left <= x <= bounds.right and bounds.bottom <= y <= bounds.top):
        return None
    if min(east - west, north - south) < float(defaults.get("minimum_window_m", 2000.0)):
        return None
    minimum_cell = float(defaults.get("minimum_cell_size_m", 30.0))
    maximum_cells = float(defaults.get("maximum_processing_cells", 8_000_000))
    # The grid follows the geometric-mean side, so a rectangle is charged for
    # the cells it actually needs rather than for its longest side.
    span = math.sqrt(max(1.0, (east - west) * (north - south)))
    resolution = max(minimum_cell, math.ceil(span / math.sqrt(maximum_cells)))
    cells = int(round((east - west) / resolution)) * int(round((north - south) / resolution))
    return WindowPlan(half_km=min(float(extent["west"]), float(extent["east"]),
                                 float(extent["south"]), float(extent["north"])),
                      west=west, south=south, east=east, north=north,
                      resolution_m=float(resolution), cells=cells,
                      clamped_sides=tuple(clamped), extent_km=dict(extent))


def _next_extent(plan: WindowPlan, report: CutReport, x: float, y: float, *,
                 attempt: int, elapsed_s: float, defaults: dict) -> dict[str, float] | None:
    """The processing range to try next, or ``None`` when growing cannot help.

    A side the basin reached has to **grow**: past that edge its true extent is
    unknown. A side it did *not* reach is left where it is — never tightened.

    Tightening is unsound here: the flow field itself depends on the range,
    because depressions are filled outward from the raster's own boundary and
    flat cells take their direction from that fill. A range shrunk around a
    basin can therefore *lose* headwaters that drain across a flat — and then
    report a smaller, fully contained basin, which passes the edge check while
    being wrong. (Measured on the real DEM: the same outlet and the same
    snapped point give a "contained, stable" 253 km² in a 60×50 km range and
    1011 km² — still cut to the north and east — in an 80×80 km one.)

    So the range only ever grows, on two grounds: a side the basin reached needs
    more room, and a side the basin came within ``window_interior_margin_km`` of
    needs clearance as well. Clearance is the part that rules out the false
    pass, and the stability check on top of it confirms the basin does not
    change when the range grows again.

    Growing is still bounded on four sides at once — attempts, per-side
    ceiling, grid resolution and wall-clock budget — because with a fixed cell
    budget a larger range can only be read at a coarser grid, and a boundary
    traced on a coarser grid is a different boundary.
    """
    if not report.sides:
        return None
    if not bool(defaults.get("adaptive_window", True)):
        return None
    if attempt >= int(defaults.get("maximum_window_attempts", 5)):
        return None
    if elapsed_s >= float(defaults.get("window_time_budget_s", 600.0)):
        return None
    if set(report.sides) <= set(plan.clamped_sides):
        # Every cut side is already the DEM's own edge: a bigger range would
        # only add empty space, never the missing headwater.
        return None

    growth = float(defaults.get("window_growth_factor", 1.5))
    ceiling_m = float(defaults.get("maximum_half_window_km", 100.0)) * 1000.0
    interior_margin_m = float(defaults.get("window_interior_margin_km", 10.0)) * 1000.0
    bbox = report.basin_bounds
    index_by_side = {"west": 0, "south": 1, "east": 2, "north": 3}
    axis_by_side = {"west": x, "east": x, "south": y, "north": y}
    sign_by_side = {"west": -1.0, "east": 1.0, "south": -1.0, "north": 1.0}

    extent: dict[str, float] = {}
    for side, index in index_by_side.items():
        current_m = float(plan.extent_km.get(side, plan.half_km)) * 1000.0
        wanted_m = current_m * growth if side in report.sides else current_m
        if bbox is not None:
            # Clearance, not just containment: the flow field is distorted near
            # the range's own edges (depressions are filled outward from them),
            # so a basin that merely fits is not yet trustworthy. Measured: the
            # same outlet and snap gives a "contained" 253 km² in a 60×50 km
            # range and 1011 km² in an 80×80 km one.
            needed_m = sign_by_side[side] * (bbox[index] - axis_by_side[side]) + interior_margin_m
            wanted_m = max(wanted_m, needed_m)
        extent[side] = min(wanted_m, ceiling_m) / 1000.0
    # No progress is a reason to stop: a range that is identical to the one
    # just delineated would only repeat the same work.
    current = {side: float(plan.extent_km.get(side, plan.half_km)) for side in extent}
    if all(abs(extent[side] - current[side]) < 1e-6 for side in extent):
        return None
    return extent


def _basin_area_km2(basin_path: Path) -> float:
    """The basin's area, straight from its raster."""
    with rasterio.open(basin_path) as source:
        basin = source.read(1) == 1
        transform = source.transform
    return float(np.count_nonzero(basin)) * abs(transform.a * transform.e) / 1_000_000


def _grow_all_sides(plan: WindowPlan, defaults: dict) -> dict[str, float] | None:
    """Every side out by the confirmation margin, for the stability check.

    A basin that does not touch the range edges may still be incomplete — the
    flow field depends on the range — so the only honest test is whether it
    stays the same when the range grows. ``None`` means the range cannot grow
    any further, and the check cannot be run.
    """
    margin_km = float(defaults.get("window_confirm_margin_km", 5.0))
    ceiling_km = float(defaults.get("maximum_half_window_km", 100.0))
    extent = {side: float(plan.extent_km.get(side, plan.half_km)) + margin_km
              for side in ("west", "east", "south", "north")}
    if max(extent.values()) > ceiling_km:
        return None
    return extent


def _basin_is_stable(current: dict, previous: dict | None, defaults: dict) -> bool:
    """Whether two ranges give the same basin, within the stated tolerances.

    Area alone could agree between two different basins, so the bounding box is
    checked too — as far as the grids allow, since the two attempts may have
    been read at different cell sizes.
    """
    if not previous:
        return False
    area, previous_area = current.get("area_km2"), previous.get("area_km2")
    if not area or not previous_area:
        return False
    tolerance = float(defaults.get("window_stability_tolerance", 0.01))
    if abs(float(area) - float(previous_area)) > tolerance * max(float(area), float(previous_area)):
        return False
    current_box, previous_box = current.get("bounds"), previous.get("bounds")
    if current_box is None or previous_box is None:
        return True
    slack_m = 5.0 * max(float(current.get("resolution_m") or 0.0),
                        float(previous.get("resolution_m") or 0.0))
    return all(abs(float(a) - float(b)) <= slack_m for a, b in zip(current_box, previous_box))


def _window_findings(plan: WindowPlan, report: CutReport, *,
                     attempts: int, defaults: dict,
                     at_ceiling: bool = False,
                     confirmed_by_growth: bool = False) -> tuple[list[str], list[str]]:
    """What to tell the reviewer about a basin that was cut: ``(errors, notes)``.

    Two different problems must not be reported as one. A window that could
    still grow is a matter of grid resolution and patience; a basin that runs
    past the DEM's own edge needs different data; a basin that runs past the
    window *ceiling* needs a different outlet — one further upstream, whose
    upstream area is smaller; and a flank grazing a no-data cell is neither, so
    it is a note rather than a refusal to confirm.
    """
    errors: list[str] = []
    notes: list[str] = []
    data_sides = [side for side in report.sides if side in plan.clamped_sides]
    window_sides = [side for side in report.sides if side not in plan.clamped_sides]
    if window_sides:
        tried = f"已自适应尝试 {attempts} 次" if attempts > 1 else "本次"
        errors.append(
            f"流域触及处理范围边界（{'、'.join(window_sides)} 方向；{tried}，"
            f"处理范围 {plan.width_km:g}×{plan.height_km:g} km，栅格 {plan.resolution_m:g} m），"
            "范围可能不完整")
    if data_sides:
        errors.append(
            f"流域向{'、'.join(data_sides)}方向延伸超出 DEM 数据范围，"
            "现有数据无法补全该部分（需要覆盖更完整的 DEM）")
    if at_ceiling and window_sides:
        errors.append(
            f"处理范围已到上限（{attempts} 次尝试；本次 {plan.width_km:g}×{plan.height_km:g} km）；"
            "该流域的上游流程超出本工具一次能覆盖的范围，"
            "请改选更靠上游、汇水面积更小的出水口（或分段分析）")
    if not report.sides and not confirmed_by_growth:
        # The edge check passed, but the flow field depends on the range, so a
        # basin that merely fits is not the same as a basin that is complete.
        notes.append("未能在更大的处理范围上确认流域稳定（范围或尝试次数已到上限），"
                     "边界可能仍不完整")
    if report.touches_dem_nodata:
        fraction = report.nodata_fraction
        # "像元" is reserved by the numeric audit for the basin's own cell count;
        # a different count under that word is reported as a field mismatch, so
        # this one is written as grid cells.
        text = (f"流域边缘与 DEM 无数据区接触 {report.nodata_contact_cells} 个栅格单元"
                f"（占流域 {fraction * 100:.2f}%）")
        threshold = float(defaults.get("minimum_nodata_contact_fraction", 0.005))
        if fraction >= threshold:
            errors.append(text + "，存在跨界截断")
        else:
            notes.append(text + f"，低于截断判定阈值 {threshold * 100:.2f}%，仅作提示")
    return errors, notes


def _river_snap_geometry(rivers: Path | None, river_name: str | None, dem_crs,
                         x: float, y: float, maximum_distance_m: float):
    """Named water polygon, excluding other named rivers near the click.

    A one-cell corridor is added later for raster alignment. Other river water
    polygons themselves remain excluded, particularly at confluences.
    """
    if not river_name or rivers is None:
        return None, None
    from shapely.geometry import Point
    from shapely.ops import transform as project_geometry
    query = normalise_river_query(river_name)
    tx = Transformer.from_crs(_river_layer_crs(rivers), dem_crs, always_xy=True)
    inverse = Transformer.from_crs(dem_crs, _river_layer_crs(rivers), always_xy=True)
    region = Point(x,y).buffer(maximum_distance_m + 200.0)
    corners = [inverse.transform(xx,yy) for xx in (region.bounds[0],region.bounds[2])
               for yy in (region.bounds[1],region.bounds[3])]
    west,east = min(xx for xx,yy in corners),max(xx for xx,yy in corners)
    south,north = min(yy for xx,yy in corners),max(yy for xx,yy in corners)
    target, others = [], []
    matched = False
    reader = shapefile.Reader(str(rivers),encoding=attribute_encoding(rivers))
    try:
        for record in reader.iterShapeRecords():
            attributes = record.record.as_dict()
            name = str(attributes.get("Name") or "").strip()
            kind = str(attributes.get("kind") or "").strip()
            if kind and kind != "0123":
                continue
            is_target = name in {query,query+"河"}
            matched |= is_target
            bbox = record.shape.bbox
            if bbox[0]>east or bbox[2]<west or bbox[1]>north or bbox[3]<south:
                continue
            geometry = project_geometry(tx.transform,shape(record.shape.__geo_interface__).buffer(0))
            clipped = geometry.intersection(region)
            if not clipped.is_empty:
                (target if is_target else others).append(clipped)
    finally:
        reader.close()
    if not matched:
        return None,None
    return unary_union(target),unary_union(others)


def _write_river_snap_mask(window_dem: Path, output: Path, geometry, other_geometry) -> Path:
    from rasterio.features import rasterize
    with rasterio.open(window_dem) as source:
        # Half a cell diagonal accounts for cell-centre/vector alignment only;
        # the former 300 m proximity tolerance cannot establish channel identity.
        tolerance = math.hypot(source.res[0],source.res[1]) / 2.0
        corridor = geometry.buffer(tolerance).difference(other_geometry)
        mask = (rasterize([(mapping(corridor),1)],out_shape=source.shape,transform=source.transform,
                          dtype="uint8") if not corridor.is_empty
                else np.zeros(source.shape,dtype="uint8"))
        profile = source.profile.copy()
        profile.update(dtype="uint8",count=1,nodata=0,compress="lzw")
    path = output / "river_snap_mask.tif"
    with rasterio.open(path,"w",**profile) as destination:
        destination.write(mask,1)
    return path


def _run_hydrology(worker: Path, window_dem: Path, output: Path, *,
                   x: float, y: float, snap_radius: int, river_mask: Path | None = None,
                   maximum_snap_distance_m: float = 500.0, minimum_area_km2: float = 1.8) -> None:
    """Delineate on the window DEM, as its own process and its own timeout."""
    import subprocess

    command = [
        sys.executable, str(worker),
        "--dem", str(window_dem), "--output", str(output), "--x", str(x), "--y", str(y),
        "--snap-radius", str(snap_radius),
    ]
    if river_mask is not None:
        command += ["--river-mask",str(river_mask),
                    "--maximum-snap-distance-m",str(maximum_snap_distance_m),
                    "--minimum-area-km2",str(minimum_area_km2)]
    process = subprocess.run(command, cwd=output, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=360)
    (output / "hydrology.log").write_text(
        process.stdout + "\n--- STDERR ---\n" + process.stderr, encoding="utf-8"
    )
    if process.returncode:
        raise RuntimeError(
            "Native hydrology delineation failed; see hydrology.log: "
            + (process.stderr or process.stdout)[-1000:]
        )


def _river_outlet_check(rivers: Path | None, river_name: str | None,
                        snapped_x: float, snapped_y: float, dem_crs,
                        defaults: dict) -> dict:
    """Keep a valid tributary basin from being mistaken for the named river."""
    if not river_name or rivers is None:
        return {"checked": False, "passed": None}
    matched = _matched_river_features(rivers, normalise_river_query(river_name))
    if not matched:
        return {"checked": False, "passed": None, "river_name": river_name,
                "note": "本地水系未匹配该河名，河段身份仍需在地图上人工核对。"}
    from shapely.geometry import Point, Polygon
    from shapely.ops import transform as project_geometry
    tx = Transformer.from_crs(_river_layer_crs(rivers), dem_crs, always_xy=True)
    polygons = [project_geometry(tx.transform, Polygon(ring).buffer(0))
                for _, rings in matched for ring in rings if len(ring) >= 3]
    river = unary_union(polygons)
    if river.is_empty:
        raise ValueError("匹配河段没有有效几何，不能核对目标河流")
    distance = float(river.distance(Point(snapped_x, snapped_y)))
    maximum = float(defaults.get("maximum_river_distance_m", 300.0))
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("maximum_river_distance_m 必须为正的有限距离")
    return {"checked": True, "passed": distance <= maximum, "river_name": river_name,
            "river_distance_m": round(distance, 2), "maximum_river_distance_m": maximum,
            "matched_names": sorted({name for name, _ in matched})}


def _named_river_coverage_check(river_name: str | None, report: CutReport) -> dict:
    """A basin grazing NoData cannot establish a named river's full headwaters.

    The relaxed local boundary-contact fraction is useful diagnostically, but
    it is not a measure of the area missing beyond the source DEM.
    """
    if not river_name:
        return {"checked":False,"passed":None}
    return {"checked":True,"passed":not report.touches_dem_nodata,
            "source_nodata_contact_cells":report.nodata_contact_cells,
            "scope":"selected_outlet_upstream_within_source_dem"}


def delineate(args: argparse.Namespace) -> dict:
    # Resolve both paths up front: the hydrology step runs with ``cwd=output``,
    # so a relative --output (or --dem) would be re-interpreted inside that
    # directory and fail to open the window this function just wrote.
    output: Path = Path(args.output).resolve()
    args.dem = Path(args.dem).resolve()
    output.mkdir(parents=True, exist_ok=True)
    defaults = getattr(args, "defaults", DEFAULT_DELINEATION)
    hydrology_worker = Path(__file__).with_name("hydrology_worker.py")
    window_dem = output / "dem_processing_window.tif"
    started = time.monotonic()
    attempts: list[dict] = []
    plan: WindowPlan | None = None
    report = CutReport()
    previous: dict | None = None
    confirming = False
    confirmed_by_growth = False
    half_km = float(args.half_window_km)
    extent = {side: half_km for side in ("west", "east", "south", "north")}
    with rasterio.open(args.dem) as source:
        dem_crs = _dataset_crs(source)
        to_dem = _transformer(CRS.from_epsg(4326), dem_crs)
        to_wgs = _transformer(dem_crs, CRS.from_epsg(4326))
        x, y = to_dem.transform(args.lon, args.lat)
        maximum_snap = float(defaults.get("maximum_named_river_snap_distance_m",500.0))
        if not math.isfinite(maximum_snap) or maximum_snap <= 0:
            raise ValueError("maximum_named_river_snap_distance_m 必须为正的有限距离")
        river_geometry, other_geometry = _river_snap_geometry(
            getattr(args,"rivers",None),getattr(args,"name",None),dem_crs,x,y,maximum_snap)
        bounds = source.bounds
        while True:
            candidate = _plan_window(bounds, x, y, extent, defaults)
            if candidate is None:
                if plan is None:
                    raise ValueError(
                        f"出水口离 DEM 数据边界过近：{half_km:g} km 的处理范围放不进现有 DEM。"
                        "请换一个更靠内的出水口，或提供覆盖更完整的 DEM")
                break
            _write_window(source, window_dem, candidate.west, candidate.south,
                          candidate.east, candidate.north, candidate.resolution_m)
            river_mask = (_write_river_snap_mask(window_dem,output,river_geometry,other_geometry)
                          if river_geometry is not None else None)
            _run_hydrology(hydrology_worker, window_dem, output,
                           x=x, y=y, snap_radius=args.snap_radius,river_mask=river_mask,
                           maximum_snap_distance_m=maximum_snap,
                           minimum_area_km2=float(defaults.get("minimum_area_km2",1.8)))
            report = _boundary_truncated(output / "basin.tif", window_dem)
            measured = {"area_km2": _basin_area_km2(output / "basin.tif"),
                        "bounds": report.basin_bounds, "resolution_m": candidate.resolution_m}
            plan = candidate
            attempts.append({
                "mode": "confirm" if confirming else "grow",
                "extent_km": {side: round(value, 3) for side, value in candidate.extent_km.items()},
                "width_km": round(candidate.width_km, 3),
                "height_km": round(candidate.height_km, 3),
                "resolution_m": candidate.resolution_m,
                "cells": candidate.cells, "clamped_sides": list(candidate.clamped_sides),
                "touched_sides": list(report.sides),
                "nodata_contact_cells": report.nodata_contact_cells,
                "area_km2": round(measured["area_km2"], 4),
                "seconds": round(time.monotonic() - started, 1),
            })
            if confirming and not report.sides and _basin_is_stable(measured, previous, defaults):
                # The basin survived a strictly larger range unchanged: it is
                # complete, and this is the answer.
                confirmed_by_growth = True
                break
            next_extent: dict[str, float] | None = None
            if report.sides:
                confirming = False
                next_extent = _next_extent(candidate, report, x, y, attempt=len(attempts),
                                           elapsed_s=time.monotonic() - started,
                                           defaults=defaults)
            else:
                # Not cut — but a range that is merely big enough is not proof,
                # so grow every side once and see whether the basin holds still.
                confirming = True
                next_extent = _grow_all_sides(candidate, defaults)
            if next_extent is not None:
                # Growing is only worth it while the grid stays fine enough for
                # the boundary to mean the same thing.
                probe = _plan_window(bounds, x, y, next_extent, defaults)
                if probe is None or probe.resolution_m > float(
                        defaults.get("maximum_cell_size_m", 100.0)):
                    next_extent = None
            if next_extent is None:
                break
            previous = measured
            extent = next_extent
    if plan is None:  # pragma: no cover - the first attempt plans or raises
        raise RuntimeError("处理窗口规划失败：没有任何尝试被执行")
    basin_path = output / "basin.tif"
    with rasterio.open(basin_path) as basin_source:
        basin = basin_source.read(1) == 1
        basin_transform = basin_source.transform
    selected_cells = int(np.count_nonzero(basin))
    area_km2 = selected_cells * abs(basin_transform.a * basin_transform.e) / 1_000_000
    boundary = output / "basin.geojson"
    _polygonize_basin(basin_path, boundary, dem_crs)
    outlet = json.loads((output / "outlet_projected.json").read_text(encoding="utf-8"))
    requested_lon, requested_lat = to_wgs.transform(outlet["requested_x"], outlet["requested_y"])
    snapped_lon, snapped_lat = to_wgs.transform(outlet["snapped_x"], outlet["snapped_y"])
    outlet.update({
        "requested_longitude": round(requested_lon, 7), "requested_latitude": round(requested_lat, 7),
        "snapped_longitude": round(snapped_lon, 7), "snapped_latitude": round(snapped_lat, 7),
    })
    qa_errors = []
    minimum_area = float(defaults.get("minimum_area_km2", 1.8))
    nodata_threshold = float(defaults.get("minimum_nodata_contact_fraction", 0.005))
    if area_km2 < minimum_area:
        # The threshold travels with the verdict: a reply that quotes it must be
        # able to point at the record it came from.
        qa_errors.append(f"流域面积{area_km2:.3f} km2，小于当前{minimum_area} km2下限")
    ceilings = (len(attempts) >= int(defaults.get("maximum_window_attempts", 5))
                or min(plan.extent_km.values() or [0.0]) >= float(
                    defaults.get("maximum_half_window_km", 100.0)))
    window_errors, window_notes = _window_findings(
        plan, report, attempts=len(attempts), defaults=defaults,
        at_ceiling=bool(report.sides) and ceilings,
        confirmed_by_growth=confirmed_by_growth)
    qa_errors.extend(window_errors)
    river_check = _river_outlet_check(getattr(args, "rivers", None), getattr(args, "name", None),
                                     outlet["snapped_x"], outlet["snapped_y"], dem_crs, defaults)
    if river_check.get("passed") is False:
        qa_errors.append(f"吸附点距指定河流「{river_check['river_name']}」的本地河段"
                         f"{river_check['river_distance_m']:.1f} m，未落在目标河流附近；请重新选点")
    if river_check.get("note"):
        window_notes.append(river_check["note"])
    channel_check = outlet.get("channel_check") or {"checked":False,"passed":None}
    if channel_check.get("passed") is False:
        qa_errors.append("目标河段附近未找到与水系图层一致的有效汇流河道；"
                         "河道图层与 DEM 汇流路径可能不一致，不能把附近其他河流或局部坡面的汇水区当作目标流域")
    coverage_check = _named_river_coverage_check(getattr(args,"name",None),report)
    if coverage_check.get("passed") is False:
        qa_errors.append("目标汇水区接触 DEM 无数据边界，无法确认上游覆盖完整；"
                         "需补齐上游 DEM 后重新划分，单纯放大处理窗口不能补全缺失数据")
    return {
        "outlet": outlet,
        "qa": {"passed": not qa_errors, "errors": qa_errors, "notes": window_notes,
               "river_check": river_check,
               "channel_check": channel_check,
               "coverage_check": coverage_check,
               "area_km2": round(area_km2, 4),
               "selected_cells": selected_cells,
               "touches_processing_window": bool(report.sides),
               "touches_dem_nodata": report.touches_dem_nodata,
               "touched_sides": list(report.sides),
               "nodata_contact_cells": report.nodata_contact_cells,
               "nodata_contact_fraction": round(report.nodata_fraction, 6),
               "minimum_area_km2": minimum_area,
               "minimum_nodata_contact_fraction": nodata_threshold},
        "artifacts": {
            "original_dem": str(window_dem), "filled_dem": str(output / "dem_filled.tif"),
            "fill_depth": str(output / "fill_depth.tif"), "flow_direction": str(output / "flow_direction.tif"),
            "flow_accumulation": str(output / "flow_accumulation.tif"), "basin_raster": str(basin_path),
            "boundary_geojson": str(boundary), "hydrology_log": str(output / "hydrology.log"),
        },
        "processing": {"method": "Priority-Flood + D8", "resolution_m": plan.resolution_m,
                       "window_extent_km": {side: round(value, 3)
                                            for side, value in plan.extent_km.items()},
                       "maximum_processing_cells": int(defaults.get("maximum_processing_cells", 2_000_000)),
                       "window_km": round(plan.half_km, 3),
                       "window_width_km": round(plan.width_km, 3),
                       "window_height_km": round(plan.height_km, 3),
                       "window_attempts": len(attempts),
                       "window_attempts_detail": attempts,
                       "confirmed_by_growth": confirmed_by_growth,
                       "window_clamped_sides": list(plan.clamped_sides),
                       "adaptive_window": bool(defaults.get("adaptive_window", True)),
                       "native_cell_size_m": float(defaults.get("minimum_cell_size_m", 30.0)),
                       "grid_coarsened": plan.resolution_m > float(
                           defaults.get("minimum_cell_size_m", 30.0))},
    }


def _configure_fonts(plt) -> None:
    """Chinese labels need a CJK-capable family before Arial or DejaVu."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                            "Source Han Sans SC", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })


def _read_station_observations(rainfall_dir: Path) -> list[dict[str, str]]:
    """Per-station hourly values from the virtual-station package."""
    import csv

    long_table = next(iter(sorted(rainfall_dir.glob("*observations_long.csv"))), None)
    if long_table is None or not long_table.is_file():
        raise ValueError(f"降雨包中没有站点逐时长表：{rainfall_dir}")
    with long_table.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _basin_rings(boundary_path: Path) -> list[list[list[float]]]:
    payload = json.loads(boundary_path.read_text(encoding="utf-8"))
    rings: list[list[list[float]]] = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        kind = geometry.get("type")
        polygons = [geometry.get("coordinates", [])] if kind == "Polygon" else geometry.get("coordinates", [])
        for polygon in polygons:
            for ring in polygon:
                rings.append([(float(x), float(y)) for x, y in ring])
    return rings


def _render_virtual_stations(review: dict, rainfall_dir: Path, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_fonts(plt)
    observations = _read_station_observations(rainfall_dir)
    stations: dict[str, tuple[float, float, float]] = {}
    for row in observations:
        key = row.get("station_key") or row.get("station_id") or row.get("station_name")
        if key and key not in stations:
            stations[key] = (float(row.get("lon") or 0.0), float(row.get("lat") or 0.0),
                             sum(float(item.get("rain_mm") or 0.0)
                                 for item in observations if (item.get("station_key") or item.get("station_id") or item.get("station_name")) == key))
    figure, axis = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    for ring in _basin_rings(Path(review["artifacts"]["boundary_geojson"])):
        axis.plot([point[0] for point in ring], [point[1] for point in ring],
                  color="#727B84", linewidth=1.4)
    outlet = review["outlet"]
    axis.scatter(outlet.get("snapped_longitude"), outlet.get("snapped_latitude"),
                 c="red", marker="*", s=150, zorder=4)
    for key, (lon, lat, total) in stations.items():
        axis.scatter(lon, lat, s=70, color="#2F6EA6", edgecolor="white", zorder=3)
        axis.annotate(f"{key}  {total:.1f} mm", (lon, lat), textcoords="offset points",
                      xytext=(6, 6), fontsize=7)
    axis.set_xlabel("经度")
    axis.set_ylabel("纬度")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title(f"流域内虚拟雨站（{len(stations)} 个；ERA5 网格插值，非实测站）")
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _render_rainfall_series(review: dict, rainfall_dir: Path, destination: Path,
                            cumulative: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_fonts(plt)
    observations = _read_station_observations(rainfall_dir)
    stamps: list[str] = []
    series: dict[str, list[float]] = {}
    for row in observations:
        key = row.get("station_key") or row.get("station_id") or row.get("station_name") or "?"
        stamp = row.get("datetime_beijing") or row.get("datetime") or ""
        if stamp and stamp not in stamps:
            stamps.append(stamp)
        series.setdefault(key, []).append(float(row.get("rain_mm") or 0.0))
    figure, axis = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
    for key, values in series.items():
        plotted = list(np.cumsum(values)) if cumulative else values
        axis.plot(range(len(plotted)), plotted, linewidth=1.1, label=key)
    axis.set_xlabel("时次")
    axis.set_ylabel("累积雨量 (mm)" if cumulative else "小时雨量 (mm)")
    axis.set_title(("累积降雨" if cumulative else "逐时降雨过程")
                   + f"（{len(series)} 个雨站，{len(stamps)} 个时次）")
    axis.legend(ncol=2)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def _read_polygon_layer(path: Path) -> list[list[tuple[float, float]]]:
    if not path.is_file():
        return []
    reader = shapefile.Reader(str(path), encoding=attribute_encoding(path))
    rings: list[list[tuple[float, float]]] = []
    try:
        for record in reader.iterShapeRecords():
            points = record.shape.points
            parts = list(record.shape.parts) + [len(points)]
            for start, end in zip(parts[:-1], parts[1:]):
                rings.append([(float(x), float(y)) for x, y in points[start:end]])
    finally:
        reader.close()
    return rings


def _ring_to_raster_pixels(
    ring: list[tuple[float, float]],
    source_crs,
    raster_crs,
    transform: Affine,
    stride: int = 1,
) -> tuple[list[float], list[float]]:
    """Reproject a map-coordinate ring and return display pixel coordinates."""
    tx = Transformer.from_crs(source_crs, raster_crs, always_xy=True)
    points = [tx.transform(x, y) for x, y in ring]
    columns = [(x - transform.c) / transform.a / stride for x, _ in points]
    rows = [(y - transform.f) / transform.e / stride for _, y in points]
    return columns, rows


def _overlay_layers(model_input_dir: Path) -> tuple[list[tuple[str, str, CRS]], list[str]]:
    """The vector overlays that can be drawn, and those skipped for a missing .prj.

    A layer without a .prj is usually in degrees while the DEM is projected:
    reprojecting with "assume it already matches" plots every ring at map
    coordinates read as pixels — that is, far off the canvas, with nothing drawn
    and no error. Such a layer is reported instead of silently mis-drawn.
    """
    drawable: list[tuple[str, str, CRS]] = []
    missing: list[str] = []
    for name, colour in (("landuse.shp", "#D9DEE3"), ("river.shp", "#2F6EA6")):
        layer = model_input_dir / name
        if not layer.is_file():
            continue
        layer_crs = _layer_crs(layer)
        if layer_crs is None:
            missing.append(name)
            continue
        drawable.append((str(layer), colour, layer_crs))
    return drawable, missing


def _render_overlay(review: dict, model_input_dir: Path, destination: Path,
                    extent_only: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _configure_fonts(plt)
    dem_path = model_input_dir / "dem.tif"
    if not dem_path.is_file():
        raise ValueError(f"模型输入包中没有 DEM：{dem_path}")
    with rasterio.open(dem_path) as source:
        array = source.read(1, masked=True).astype("float64")
        transform = source.transform
        crs = source.crs
    data = array.filled(np.nan)
    stride = max(1, math.ceil(max(data.shape) / 1200))
    finite = data[np.isfinite(data)]
    figure, axis = plt.subplots(figsize=(8, 6.4), constrained_layout=True)
    image = axis.imshow(data[::stride, ::stride], cmap="terrain",
                        vmin=float(np.percentile(finite, 2)) if finite.size else None,
                        vmax=float(np.percentile(finite, 98)) if finite.size else None)
    figure.colorbar(image, ax=axis, shrink=0.8, label="高程 (m)")
    drawable, missing_crs = _overlay_layers(model_input_dir) if not extent_only else ([], [])
    for layer_path, colour, layer_crs in drawable:
        for ring in _read_polygon_layer(Path(layer_path)):
            columns, rows = _ring_to_raster_pixels(ring, layer_crs, crs, transform, stride)
            axis.plot(columns, rows, color=colour,
                      linewidth=0.8 if layer_path.endswith("river.shp") else 0.4, alpha=0.9)
    boundary = model_input_dir / "basin.geojson"
    if boundary.is_file():
        rings = _basin_rings(boundary)
        if rings:
            # Basin GeoJSON is always OGC:CRS84; the cropped DEM normally uses
            # UTM 49N. Reproject before converting map coordinates to pixels.
            for ring in rings:
                columns, rows = _ring_to_raster_pixels(
                    ring, CRS.from_epsg(4326), crs, transform, stride
                )
                axis.plot(columns, rows, color="#C94C4C", linewidth=1.4)
    axis.set_xlabel("栅格列")
    axis.set_ylabel("栅格行")
    title = "模型输入包覆盖范围" if extent_only else "DEM、土地利用、水系与流域边界叠加"
    if missing_crs:
        title += "（" + "、".join(missing_crs) + " 缺少 .prj，未叠加）"
    axis.set_title(title)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def render(args: argparse.Namespace) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    review = json.loads((args.review_dir / "review.json").read_text(encoding="utf-8"))
    artifact = args.artifact
    visual_dir = Path(args.visual_dir) if args.visual_dir else args.review_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    destination = visual_dir / f"{artifact}.png"

    figure_artifacts = {
        "virtual_stations": lambda: _render_virtual_stations(review, args.rainfall_package, destination),
        "rainfall_series": lambda: _render_rainfall_series(review, args.rainfall_package, destination, False),
        "rainfall_cumulative": lambda: _render_rainfall_series(review, args.rainfall_package, destination, True),
        "overlay": lambda: _render_overlay(review, args.model_input, destination, False),
        "package_extent": lambda: _render_overlay(review, args.model_input, destination, True),
    }
    if artifact in figure_artifacts:
        if artifact in {"virtual_stations", "rainfall_series", "rainfall_cumulative"}:
            if args.rainfall_package is None:
                raise ValueError("该图件需要降雨包：请提供 rainfall_package_id")
        elif args.model_input is None:
            raise ValueError("该图件需要模型输入包：请提供 model_input_package_id")
        figure_artifacts[artifact]()
        return {"image": str(destination), "artifact": artifact, "status": review["status"]}

    _configure_fonts(plt)
    allowed = {
        "original_dem": ("original_dem", "terrain", "Elevation (m)"),
        "filled_dem": ("filled_dem", "terrain", "Filled elevation (m)"),
        "fill_depth": ("fill_depth", "magma", "Fill depth (m)"),
        "flow_direction": ("flow_direction", "twilight", "Flow direction code"),
        "flow_accumulation": ("flow_accumulation", "viridis", "Upstream cells"),
        "watershed": ("original_dem", "terrain", "Elevation (m)"),
    }
    if artifact not in allowed:
        raise ValueError(f"Unsupported artifact: {artifact}")
    key, cmap, label = allowed[artifact]
    with rasterio.open(review["artifacts"][key]) as source:
        array = np.asarray(source.read(1), dtype="float64")
        nodata = source.nodata
        transform = source.transform
    if nodata is not None:
        array[array == nodata] = np.nan
    stride = max(1, math.ceil(max(array.shape) / 1600))
    view = array[::stride, ::stride]
    figure, axis = plt.subplots(figsize=(10, 8), constrained_layout=True)
    if artifact == "flow_accumulation":
        positive = view[np.isfinite(view) & (view > 0)]
        norm = LogNorm(vmin=max(1.0, float(np.percentile(positive, 2))),
                       vmax=float(np.percentile(positive, 99.8))) if positive.size else None
        image = axis.imshow(view, cmap=cmap, norm=norm)
    else:
        image = axis.imshow(view, cmap=cmap)
    figure.colorbar(image, ax=axis, shrink=0.78, label=label)
    if artifact == "watershed":
        with rasterio.open(review["artifacts"]["basin_raster"]) as basin_source:
            full_basin = basin_source.read(1) > 0
        # A small failed basin must remain visible in an overview. Sampling
        # every nth pixel can erase its entire footprint.
        from scipy.ndimage import maximum_filter
        basin = maximum_filter(full_basin, size=stride)[::stride, ::stride]
        axis.contour(basin.astype(float), levels=[0.5], colors=["#00ffff"], linewidths=1.5)
        outlet = review["outlet"]
        column = (outlet["snapped_x"] - transform.c) / transform.a
        row = (outlet["snapped_y"] - transform.f) / transform.e
        axis.scatter(column / stride, row / stride, c="red", marker="*", s=140, label="Snapped outlet")
        requested_col = (outlet["requested_x"] - transform.c) / transform.a
        requested_row = (outlet["requested_y"] - transform.f) / transform.e
        axis.scatter(requested_col / stride, requested_row / stride, c="white", marker="x",
                     s=70, label="Requested outlet")
        if not (review.get("qa") or {}).get("passed"):
            zoom = axis.inset_axes([0.02, 0.02, 0.47, 0.47])
            r0 = max(0, int(min(row, requested_row)) - 35)
            r1 = min(array.shape[0], int(max(row, requested_row)) + 36)
            c0 = max(0, int(min(column, requested_col)) - 35)
            c1 = min(array.shape[1], int(max(column, requested_col)) + 36)
            zoom.imshow(array[r0:r1, c0:c1], cmap="terrain", extent=(c0, c1, r1, r0))
            small_basin = full_basin[r0:r1, c0:c1]
            from matplotlib.colors import ListedColormap
            zoom.imshow(np.ma.masked_where(~small_basin, small_basin), cmap=ListedColormap(["#00ffff"]),
                        alpha=0.65, vmin=0, vmax=1, extent=(c0, c1, r1, r0))
            with rasterio.open(review["artifacts"]["flow_accumulation"]) as accumulation_source:
                drainage = accumulation_source.read(1, window=rasterio.windows.Window(
                    c0, r0, c1 - c0, r1 - r0))
            threshold = (review.get("qa", {}).get("minimum_area_km2", 1.8) * 1_000_000
                         / abs(transform.a * transform.e))
            zoom.contour(np.arange(c0, c1) + 0.5, np.arange(r0, r1) + 0.5,
                         (drainage >= threshold).astype(float), levels=[0.5],
                         colors=["#0050ff"], linewidths=1.2)
            zoom.scatter(requested_col, requested_row, c="white", edgecolors="black",
                         marker="o", s=50, label="请求点")
            zoom.scatter(column, row, c="red", marker="*", s=100, label="吸附点")
            zoom.set_xlim(c0, c1); zoom.set_ylim(r1, r0)
            zoom.set_title("选点局部：青色为划分区域，蓝线为较大汇流通道", fontsize=8)
            zoom.legend(fontsize=7, loc="lower right")
            zoom.tick_params(labelsize=7)
        axis.legend(loc="lower right")
    qa_label = "PASS" if review.get("qa", {}).get("passed") else "FAIL"
    if artifact == "watershed":
        data_failure = any((review.get("qa", {}).get(field) or {}).get("passed") is False
                           for field in ("channel_check","coverage_check"))
        axis.set_title("流域划分：质量检查通过，等待人工核对边界" if qa_label == "PASS"
                       else "流域划分：质量检查未通过，请核对空间数据" if data_failure
                       else "流域划分：质量检查未通过，不能确认，请重新选点")
    else:
        axis.set_title(f"{artifact.replace('_', ' ').title()} | status: {review['status']} | QA: {qa_label}")
    axis.set_xlabel("Raster column")
    axis.set_ylabel("Raster row")
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return {"image": str(destination), "artifact": artifact, "status": review["status"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None,
                        help="config/spatial.default.yaml; thresholds fall back to built-in values")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("recommend")
    rec.add_argument("--dem", type=Path, required=True)
    rec.add_argument("--rivers", type=Path, required=True)
    rec.add_argument("--name", required=True)
    rec.add_argument("--limit", type=int, default=5)
    delineation = sub.add_parser("delineate")
    delineation.add_argument("--dem", type=Path, required=True)
    delineation.add_argument("--output", type=Path, required=True)
    delineation.add_argument("--lon", type=float, required=True)
    delineation.add_argument("--lat", type=float, required=True)
    delineation.add_argument("--snap-radius", type=int, default=5)
    delineation.add_argument("--rivers", type=Path, default=None)
    delineation.add_argument("--name", default=None)
    delineation.add_argument("--half-window-km", type=float, default=20)
    outletmap = sub.add_parser("outlet_map")
    outletmap.add_argument("--dem", type=Path, required=True)
    outletmap.add_argument("--rivers", type=Path, required=True)
    outletmap.add_argument("--display-rivers", type=Path, default=None)
    outletmap.add_argument("--display-rivers-label", default="参考河网")
    outletmap.add_argument("--name", required=True)
    outletmap.add_argument("--limit", type=int, default=0)
    outletmap.add_argument("--lon", type=float, default=None)
    outletmap.add_argument("--lat", type=float, default=None)
    renderer = sub.add_parser("render")
    renderer.add_argument("--review-dir", type=Path, required=True)
    renderer.add_argument("--artifact", required=True)
    renderer.add_argument("--rainfall-package", type=Path, default=None)
    renderer.add_argument("--model-input", type=Path, default=None)
    renderer.add_argument("--visual-dir", type=Path, default=None)
    # Every subcommand accepts the same config flag, so it may be placed either
    # side of the subcommand name.
    for action in sub.choices.values():
        action.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    args.defaults = load_defaults(args.config)
    if args.command == "recommend":
        result = recommend(args)
    elif args.command == "outlet_map":
        result = outlet_map(args)
    elif args.command == "delineate":
        result = delineate(args)
    else:
        result = render(args)
    # Keep the batch-wrapper wire ASCII-only so Windows console code pages cannot
    # corrupt Chinese names before the MCP process decodes the JSON.
    sys.stdout.write(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
