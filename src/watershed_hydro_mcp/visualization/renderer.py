"""Renderers for the visualization catalog.

Every figure is drawn from files the run actually produced, records the SHA-256
of each source, and carries the renderer version. Figures drawn from a run
whose quality gates failed are stamped with a visible diagnostic marker so a
failed result can never be mistaken for a validated one.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import shapefile  # noqa: E402

from watershed_exchange.paths import sha256_file  # noqa: E402

from ..artifact_registry import load_artifacts  # noqa: E402
from ..qa import read_gates  # noqa: E402


RENDERER_VERSION = "1.7"

#: Preview bounds: the plan caps previews at 1600x1200 and about 5 MB.
MAXIMUM_PREVIEW_PIXELS = (1600, 1200)
MAXIMUM_PREVIEW_BYTES = 5 * 1024 * 1024

COLORS = {
    "blue": "#2F6EA6", "red": "#C94C4C", "gold": "#D49A32",
    "green": "#3D8062", "grey": "#727B84", "light": "#D9DEE3",
}


@dataclass
class RenderContext:
    job_dir: Path
    project: Path
    run_id: str
    status: dict[str, Any]

    @property
    def artifacts(self) -> list[dict[str, Any]]:
        return load_artifacts(self.job_dir)

    @property
    def qa_failed(self) -> bool:
        report = read_gates(self.job_dir)
        return bool(report is not None and not report.get("passed", True))

    def find(self, *patterns: str) -> list[Path]:
        found: list[Path] = []
        for pattern in patterns:
            found.extend(sorted(self.project.glob(pattern)))
        return [path for path in found if path.is_file()]

    def first(self, *patterns: str) -> Path | None:
        matches = self.find(*patterns)
        return matches[0] if matches else None


def _style() -> None:
    # Figure labels are Chinese, so a CJK-capable family has to come first:
    # matplotlib silently renders missing glyphs as empty boxes.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
            "WenQuanYi Zen Hei", "Arial", "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "figure.facecolor": "white",
    })


def _stamp(figure, context: RenderContext, title: str) -> None:
    figure.suptitle(title, fontsize=10)
    footer = f"run {context.run_id} · renderer v{RENDERER_VERSION}"
    if context.qa_failed:
        footer += "  ·  QA FAILED — 结果未通过质量门槛，不得作为结论使用"
        figure.text(
            0.5, 0.5, "QA FAILED", fontsize=44, color="#C94C4C", alpha=0.18,
            ha="center", va="center", rotation=28, transform=figure.transFigure,
        )
    figure.text(0.01, 0.005, footer, fontsize=6, color="#727B84")


def _save(figure, destination: Path) -> dict[str, Any]:
    """Write a preview bounded by size and weight, reducing again if needed."""
    width, height = MAXIMUM_PREVIEW_PIXELS
    inches = figure.get_size_inches()
    dpi = 150
    if inches[0] * dpi > width:
        dpi = max(72, int(width / inches[0]))
    if inches[1] * dpi > height:
        dpi = min(dpi, max(72, int(height / inches[1])))
    figure.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    for attempt, reduced in enumerate((0.75, 0.55, 0.4)):
        size = destination.stat().st_size
        if size <= MAXIMUM_PREVIEW_BYTES:
            break
        figure.savefig(destination, dpi=max(50, int(dpi * reduced)),
                       bbox_inches="tight", facecolor="white")
    plt.close(figure)
    final_size = destination.stat().st_size
    return {
        "size_bytes": final_size,
        "within_size_limit": final_size <= MAXIMUM_PREVIEW_BYTES,
        "dpi": dpi,
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _numeric(value: Any, default: float = float("nan")) -> float:
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def _read_shapes(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    from watershed_exchange.vector_io import open_reader

    reader = open_reader(path)
    records = []
    try:
        for record in reader.iterShapeRecords():
            points = record.shape.points
            parts = list(record.shape.parts) + [len(points)]
            rings = [points[start:end] for start, end in zip(parts[:-1], parts[1:])]
            records.append({"record": record.record.as_dict(), "rings": rings})
    finally:
        reader.close()
    crs = None
    prj = path.with_suffix(".prj")
    if prj.is_file():
        crs = prj.read_text(encoding="utf-8", errors="ignore")[:400]
    return records, crs


def _plot_rings(axis, rings, **kwargs) -> None:
    for ring in rings:
        if len(ring) < 3:
            continue
        xs = [point[0] for point in ring]
        ys = [point[1] for point in ring]
        axis.plot(xs, ys, **kwargs)


def _rings_in_crs(rings, path: Path, target_crs):
    """Project reference vectors before overlaying a metric DEM."""
    from pyproj import CRS, Transformer
    prj=path.with_suffix('.prj')
    if not prj.is_file():
        raise ValueError(f"图层缺少坐标系，无法可靠叠加：{path.name}")
    source_crs=CRS.from_wkt(prj.read_text(encoding='utf-8'))
    if source_crs==CRS.from_user_input(target_crs):
        return rings
    transform=Transformer.from_crs(source_crs,target_crs,always_xy=True)
    return [list(zip(*transform.transform(*zip(*ring)))) for ring in rings if ring]


def _raster_preview(axis, path: Path, cmap: str, label: str, log: bool = False) -> None:
    with rasterio.open(path) as source:
        array = source.read(1, masked=True).astype("float64")
        bounds = source.bounds
        crs = source.crs
    data = array.filled(np.nan)
    stride = max(1, math.ceil(max(data.shape) / 1200))
    view = data[::stride, ::stride]
    if log:
        finite = view[np.isfinite(view) & (view > 0)]
        if finite.size:
            from matplotlib.colors import LogNorm

            image = axis.imshow(view, cmap=cmap, origin="upper",
                                extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                                norm=LogNorm(vmin=max(finite.min(), 1e-6),
                                             vmax=np.percentile(finite, 99.5)))
        else:
            image = axis.imshow(view, cmap=cmap, origin="upper",
                                extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    else:
        finite = view[np.isfinite(view)]
        vmin = float(np.percentile(finite, 2)) if finite.size else None
        vmax = float(np.percentile(finite, 98)) if finite.size else None
        image = axis.imshow(view, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
                            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    axis.figure.colorbar(image, ax=axis, shrink=0.8, label=label)
    axis.set_xlabel("投影坐标 X")
    axis.set_ylabel("投影坐标 Y")
    axis.text(0.01, 0.01, f"CRS: {crs or '未知'}", transform=axis.transAxes,
              fontsize=6, color=COLORS["grey"], ha="left", va="bottom",
              bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})
    if crs is not None and crs.is_projected:
        width = bounds.right - bounds.left
        length = 10 ** math.floor(math.log10(max(width * 0.2, 1e-9)))
        if length * 5 <= width * 0.3:
            length *= 5
        elif length * 2 <= width * 0.3:
            length *= 2
        x0 = bounds.left + width * 0.06
        y0 = bounds.bottom + (bounds.top - bounds.bottom) * 0.06
        axis.plot([x0, x0 + length], [y0, y0], color="black", linewidth=2.2)
        axis.text(x0 + length / 2, y0, f"{length:g} m", ha="center", va="bottom",
                  fontsize=7, color="black")


# --------------------------------------------------------------------- stages


def render_run_manifest(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    job_path = context.job_dir / "job.json"
    config_path = context.project / "config.yaml"
    sources = [path for path in (job_path, config_path) if path.is_file()]
    figure, axis = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    axis.axis("off")
    if job_path.is_file():
        job = json.loads(job_path.read_text(encoding="utf-8"))
        validation = job.get("input_validation") or {}
        rows = [
            ("任务编号", job.get("run_id")),
            ("输入编号", job.get("input_id")),
            ("运行模式", job.get("mode")),
            ("阶段", " → ".join(job.get("stages") or [])),
            ("参数预设", job.get("parameter_preset")),
            ("输入校验", "通过" if validation.get("ok") else "未记录/未通过"),
            ("校验摘要哈希", validation.get("sha256", "")[:20]),
            ("创建时间", job.get("created_at")),
        ]
        table = axis.table(cellText=[[str(a), str(b or "")] for a, b in rows],
                           colWidths=[0.18, 0.76], loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.55)
        axis.set_title("运行配置与输入校验快照", pad=12)
    else:
        axis.text(0.5, 0.5, "缺少 job.json", ha="center", transform=axis.transAxes)
    _stamp(figure, context, "运行配置快照")
    return {"figure": figure, "sources": sources}


def render_input_coverage(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    sources: list[Path] = []
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    dem = context.first("inputs/user_data/dem.tif")
    if dem:
        sources.append(dem)
        with rasterio.open(dem) as source:
            preview = source.read(1, masked=True, out_shape=(
                min(source.height, 900), min(source.width, 900)))
            axes[0].imshow(preview, cmap="terrain")
            axes[0].set_title(f"DEM 覆盖（{source.width}×{source.height}，分辨率 "
                              f"{abs(source.transform.a):.1f} m）")
            axes[0].set_xlabel("列"); axes[0].set_ylabel("行")
    else:
        axes[0].text(0.5, 0.5, "缺少 DEM", ha="center", transform=axes[0].transAxes)
    boundary = context.first("inputs/user_data/basin.geojson")
    landuse = context.first("inputs/user_data/landuse.shp")
    if boundary:
        sources.append(boundary)
        payload = json.loads(boundary.read_text(encoding="utf-8"))
        for feature in payload.get("features", []):
            geometry = feature.get("geometry") or {}
            polygons = [geometry.get("coordinates", [])] if geometry.get("type") == "Polygon" else geometry.get("coordinates", [])
            for polygon in polygons:
                for ring in polygon:
                    xs = [point[0] for point in ring]
                    ys = [point[1] for point in ring]
                    axes[1].plot(xs, ys, color=COLORS["red"], linewidth=1.4)
    if landuse:
        sources.append(landuse)
        try:
            records, _ = _read_shapes(landuse)
            for item in records[:400]:
                _plot_rings(axes[1], item["rings"], color=COLORS["light"], linewidth=0.3)
        except Exception:  # noqa: BLE001 - the boundary view still renders
            pass
    axes[1].set_title("流域边界与土地利用范围（WGS84）")
    axes[1].set_xlabel("经度"); axes[1].set_ylabel("纬度")
    axes[1].set_aspect("equal", adjustable="datalim")
    _stamp(figure, context, "输入覆盖与分辨率检查")
    return {"figure": figure, "sources": sources}


def _excluded_boundary_layer(axis, context, sources):
    from matplotlib.colors import ListedColormap
    path = context.first('inputs/dem_derived/excluded_boundary_mask.tif')
    diagnostic = context.first('inputs/dem_derived/dem_derived_subcatchments_diagnostics.json')
    if not path or not diagnostic:
        return None
    facts = json.loads(diagnostic.read_text(encoding='utf-8')).get('boundary_filter') or {}
    if facts.get('excluded_count', 0):
        with rasterio.open(path) as source:
            data = source.read(1, masked=True)
            b = source.bounds
            axis.imshow(data, extent=(b.left, b.right, b.bottom, b.top), origin='upper',
                        cmap=ListedColormap(['#777777']), interpolation='nearest', zorder=2, alpha=.85)
        sources.extend([path, diagnostic])
    return facts


def render_subcatchments(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    path = context.first("inputs/dem_derived/*subcatchments.shp")
    sources = [path] if path else []
    figure, axis = plt.subplots(figsize=(9, 6.2), constrained_layout=True)
    if path:
        from matplotlib.lines import Line2D
        records, _ = _read_shapes(path)
        terrain = context.first("inputs/user_data/dem.tif")
        if terrain:
            _raster_preview(axis, terrain, "terrain", "高程 (m)")
            sources.append(terrain)
            with rasterio.open(terrain) as dem:
                bounds, target_crs=dem.bounds,dem.crs
        else:
            bounds=None
            target_crs=path.with_suffix('.prj').read_text(encoding='utf-8')
        diagnostic = context.first("inputs/dem_derived/dem_derived_subcatchments_diagnostics.json")
        verified = diagnostic is not None
        if diagnostic:
            sources.append(diagnostic)
        small_boundary = 0
        for item in records:
            is_boundary = bool(item["record"].get("boundary"))
            small_boundary += int(is_boundary)
            _plot_rings(axis, _rings_in_crs(item["rings"],path,target_crs),
                        color="#FF7900" if is_boundary else "#1745C1",
                        linewidth=0.45, alpha=0.9)
        river = context.first("inputs/user_data/river.shp")
        legend = [Line2D([0],[0],color="#1745C1",label="沿排水关系划分的单元")]
        filtered = _excluded_boundary_layer(axis, context, sources)
        if filtered and filtered.get('excluded_count'):
            legend.append(Line2D([0],[0],color='#777777',linewidth=5,label='灰色：边缘小区已排除计算'))
        if small_boundary:
            legend.append(Line2D([0],[0],color="#FF7900",label="含独立边界出流的单元"))
        if river:
            rivers, _ = _read_shapes(river)
            for item in rivers:
                _plot_rings(axis,_rings_in_crs(item["rings"],river,target_crs),color="#E400D5",linewidth=.8,alpha=.9)
            sources.append(river)
            legend.append(Line2D([0],[0],color="#E400D5",label="输入河网（参考）"))
        axis.legend(handles=legend,loc="upper left",fontsize=7,framealpha=.9)
        suffix = "局部区域，保留独立边界出流" if verified else "缺少新版排水验证，请重新划分"
        if filtered:
            suffix = f"保留 {filtered.get('retained_area_ha', 0)/100:.3f} km²；排除 {filtered.get('excluded_count', 0)} 个边缘小区、{filtered.get('excluded_fraction', 0):.2%} 面积"
        axis.set_title(f"子汇水区划分（{len(records)} 个）\n{suffix}")
        if bounds:
            axis.set_xlim(bounds.left,bounds.right);axis.set_ylim(bounds.bottom,bounds.top)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("投影坐标 X (m)"); axis.set_ylabel("投影坐标 Y (m)")
    else:
        axis.text(0.5, 0.5, "缺少子汇水区文件", ha="center", transform=axis.transAxes)
    _stamp(figure, context, "SWMM 子汇水区划分")
    return {"figure": figure, "sources": sources}


def render_area_histogram(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    path = context.first("inputs/dem_derived/*.shp")
    sources = [path] if path else []
    figure, axis = plt.subplots(figsize=(7, 4.4), constrained_layout=True)
    if path:
        records, _ = _read_shapes(path)
        areas = [value for value in (_numeric(item["record"].get("area_ha")) for item in records)
                 if value == value]
        axis.hist(areas, bins=min(40, max(6, len(areas) // 6)), color=COLORS["blue"], edgecolor="white")
        axis.axvline(np.median(areas), color=COLORS["red"], linestyle="--", linewidth=1.2,
                     label=f"中位数 {np.median(areas):.2f} ha")
        axis.set_xlabel("子汇水区面积 (ha)"); axis.set_ylabel("数量")
        axis.set_title(f"面积分布（{len(areas)} 个分区）")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "缺少子汇水区文件", ha="center", transform=axis.transAxes)
    _stamp(figure, context, "子汇水区面积分布")
    return {"figure": figure, "sources": sources}


def render_coverage_gap(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    path = context.first("inputs/dem_derived/*.shp")
    sources = [path] if path else []
    figure, axis = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    if path:
        records, _ = _read_shapes(path)
        total = 0.0
        for item in records:
            total += max(0.0, _numeric(item["record"].get("area_ha"), 0.0))
            _plot_rings(axis, item["rings"], color=COLORS["green"], linewidth=0.5, alpha=0.8)
        coverage = context.first("inputs/rainfall/*station_domain_idw_weights.csv")
        label = f"划入子汇水区面积合计 {total:,.1f} ha"
        if coverage:
            sources.append(coverage)
            rows = _read_csv_rows(coverage)
            label += f"；权重表覆盖 {len(rows)} 条记录"
        axis.set_title("划分覆盖检查\n" + label)
        axis.set_aspect("equal", adjustable="datalim")
    else:
        axis.text(0.5, 0.5, "缺少子汇水区文件", ha="center", transform=axis.transAxes)
    _stamp(figure, context, "划分覆盖缺口")
    return {"figure": figure, "sources": sources}


def render_rainfall_series(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    path = context.first("inputs/rainfall/*station_hourly.csv")
    sources = [path] if path else []
    figure, axis = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
    if path:
        rows = _read_csv_rows(path)
        time_column = next(iter(rows[0]), "datetime_beijing") if rows else "datetime_beijing"
        stations = [key for key in rows[0] if key != time_column] if rows else []
        xs = [row.get(time_column, "") for row in rows]
        for station in stations:
            axis.plot(range(len(xs)), [_numeric(row.get(station), 0.0) for row in rows],
                      linewidth=1.0, label=station)
        axis.set_xlabel("时次"); axis.set_ylabel("小时雨量 (mm)")
        axis.set_title(f"逐时降雨过程（{len(stations)} 个雨站，{len(rows)} 个时次）")
        if len(stations) <= 12:
            axis.legend(ncol=2)
    else:
        axis.text(0.5, 0.5, "缺少站点逐时降雨表", ha="center", transform=axis.transAxes)
    _stamp(figure, context, "逐时降雨过程")
    return {"figure": figure, "sources": sources}


def render_rainfall_cumulative(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    station = context.first("inputs/rainfall/*station_hourly.csv")
    active = context.first("inputs/rainfall/*active_hourly.csv")
    sources = [path for path in (station, active) if path]
    figure, axis = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
    if station:
        rows = _read_csv_rows(station)
        time_column = next(iter(rows[0]), "datetime_beijing") if rows else "datetime_beijing"
        for key in [column for column in rows[0] if column != time_column][:12]:
            values = np.cumsum([_numeric(row.get(key), 0.0) for row in rows])
            axis.plot(range(len(values)), values, linewidth=1.0, label=key)
    if active:
        rows = _read_csv_rows(active)
        values = np.cumsum([_numeric(row.get("rain_mm"), 0.0) for row in rows])
        axis.plot(range(len(values)), values, color="black", linewidth=1.8,
                  linestyle="--", label="流域面平均")
    if not (station or active):
        axis.text(0.5, 0.5, "缺少降雨数据", ha="center", transform=axis.transAxes)
    axis.set_xlabel("时次"); axis.set_ylabel("累积雨量 (mm)")
    axis.set_title("累积降雨")
    axis.legend(ncol=2)
    _stamp(figure, context, "累积降雨")
    return {"figure": figure, "sources": sources}


def render_rainfall_station_map(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    from matplotlib.lines import Line2D
    observations = context.first("inputs/rainfall/*station_observations_long.csv")
    boundary = context.first("inputs/user_data/basin.geojson")
    summary = context.first("inputs/rainfall/*station_rainfall_summary.csv")
    selection_path = context.first('inputs/rainfall/station_source/rainfall_source_selection.json')
    source_types = {}
    if selection_path:
        source_types = {s['station_key']:s.get('selection_basis') for s in
                        json.loads(selection_path.read_text(encoding='utf-8')).get('stations',[])}
    sources = [path for path in (observations, boundary, summary, selection_path) if path]
    figure, axis = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    if boundary:
        payload = json.loads(boundary.read_text(encoding="utf-8"))
        for feature in payload.get("features", []):
            geometry = feature.get("geometry") or {}
            polygons = [geometry.get("coordinates", [])] if geometry.get("type") == "Polygon" else geometry.get("coordinates", [])
            for polygon in polygons:
                for ring in polygon:
                    axis.plot([point[0] for point in ring], [point[1] for point in ring],
                              color=COLORS["grey"], linewidth=1.2)
    points: dict[str, tuple[float, float, float]] = {}
    if summary:
        for row in _read_csv_rows(summary):
            key = row.get("station_key") or row.get("station_id") or row.get("station_name")
            if key:
                points[key] = (_numeric(row.get("lon")), _numeric(row.get("lat")), _numeric(row.get("total_mm"), 0.0))
                if row.get('station_id') in source_types:
                    source_types[key] = source_types[row['station_id']]
    elif observations:
        for row in _read_csv_rows(observations):
            key = row.get("station_key") or row.get("station_name")
            if key:
                previous = points.get(key, (0, 0, 0))[2]
                points[key] = (_numeric(row.get("lon")), _numeric(row.get("lat")),
                               previous + _numeric(row.get("rain_mm"), 0.0))
    for key, (lon, lat, value) in points.items():
        existing = source_types.get(key)=='existing_era5_grid'
        axis.scatter(lon, lat, s=70, marker='s' if existing else 'o',
                     color='#0072B2' if existing else '#8D2FC0', zorder=3, edgecolor="white")
        axis.annotate(f"{key}\n{value:.1f} mm", (lon, lat), textcoords="offset points",
                      xytext=(6, 6), fontsize=6.5)
    axis.set_title(f"模型雨量点（{len(points)} 个；ERA5源代理点及补充虚拟站，非实测站）")
    axis.legend(handles=[Line2D([0], [0], marker='s',color='#0072B2',linestyle='none',label='现有ERA5源的区内代理点'),
                         Line2D([0], [0], marker="o", color="#8D2FC0", linestyle="none", label="补充虚拟站（标注事件总雨量）")],
                fontsize=7,loc='upper right',framealpha=.95)
    axis.set_xlabel("经度"); axis.set_ylabel("纬度")
    axis.set_aspect("equal", adjustable="datalim")
    _stamp(figure, context, "降雨源分布")
    return {"figure": figure, "sources": sources}


def render_station_weights(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    weights = context.first("inputs/rainfall/*station_domain_idw_weights.csv",
                            "swmm/model/*station_idw_weights.csv")
    sources = [weights] if weights else []
    figure, axis = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    if weights:
        rows = _read_csv_rows(weights)
        columns = list(rows[0]) if rows else []
        station_columns = [column for column in columns if column.lower().startswith(("vs", "station"))]
        if station_columns:
            matrix = np.array([[_numeric(row.get(column), 0.0) for column in station_columns]
                               for row in rows[:200]])
            image = axis.imshow(matrix, aspect="auto", cmap="YlGnBu")
            axis.set_xticks(range(len(station_columns)))
            axis.set_xticklabels(station_columns, rotation=45, ha="right")
            axis.set_ylabel("子汇水区 / 网格行")
            axis.set_title("站点空间权重（IDW）")
            figure.colorbar(image, ax=axis, shrink=0.85, label="权重")
        else:
            axis.text(0.5, 0.5, "权重表列名未识别", ha="center", transform=axis.transAxes)
    else:
        axis.text(0.5, 0.5, "缺少权重文件", ha="center", transform=axis.transAxes)
    _stamp(figure, context, "站点空间权重")
    return {"figure": figure, "sources": sources}


def _inp_sections(path: Path) -> dict[str, list[list[str]]]:
    sections: dict[str, list[list[str]]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped.strip("[]").upper()
            sections.setdefault(current, [])
        elif current and stripped and not stripped.startswith(";;"):
            sections[current].append(stripped.split())
    return sections


def render_model_structure(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    from matplotlib.lines import Line2D
    from pyproj import CRS, Transformer
    inp = context.first("swmm/model/*.inp")
    sources = [inp] if inp else []
    figure, axis = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    if inp:
        sections = _inp_sections(inp)
        coordinates = {row[0]: (float(row[1]), float(row[2]))
                       for row in sections.get("COORDINATES", []) if len(row) >= 3}
        polygons = sections.get("POLYGONS", [])
        subcatchments = sections.get("SUBCATCHMENTS", [])
        outfalls = sections.get("OUTFALLS", [])
        boundary = context.first("inputs/dem_derived/*subcatchments.shp")
        if boundary:
            sources.append(boundary)
            shapes, _ = _read_shapes(boundary)
            for item in shapes:
                for ring in item["rings"]:
                    if len(ring) >= 3:
                        xs = [point[0] for point in ring]
                        ys = [point[1] for point in ring]
                        axis.plot(xs, ys, color=COLORS["blue"], linewidth=0.45, alpha=0.65, zorder=1)
        else:
            grouped: dict[str, list[tuple[float, float]]] = {}
            for row in polygons:
                if len(row) >= 3:
                    grouped.setdefault(row[0], []).append((float(row[1]), float(row[2])))
            for points in grouped.values():
                if len(points) >= 3:
                    xs, ys = zip(*points)
                    axis.fill(xs, ys, facecolor="#DDEAF4", edgecolor=COLORS["blue"],
                              linewidth=0.55, alpha=0.55, zorder=1)
        outfall_names = {row[0] for row in outfalls if row}
        outlets = sorted({point for name, point in coordinates.items() if name in outfall_names})
        if outlets:
            axis.scatter(*zip(*outlets), s=14, marker="v", color=COLORS["red"], alpha=.75, zorder=3)
        nodes = [point for name, point in coordinates.items() if name not in outfall_names]
        if nodes:
            axis.scatter(*zip(*nodes), s=10, color=COLORS["blue"], zorder=2)
        handles = [Line2D([0], [0], color=COLORS["blue"], label="子汇水区边界"),
                   Line2D([0], [0], marker="v", color=COLORS["red"], linestyle="none", label="分区出水口（红色倒三角）")]
        filtered = _excluded_boundary_layer(axis, context, sources)
        if filtered and filtered.get('excluded_count'):
            handles.append(Line2D([0],[0],color='#777777',linewidth=5,
                label=f"灰色排除范围 / {filtered['excluded_fraction']:.2%} 面积"))
        summary = context.first("inputs/rainfall/*station_rainfall_summary.csv")
        dem = context.first("inputs/user_data/dem.tif")
        target_crs = None
        if dem:
            with rasterio.open(dem) as raster:
                target_crs = raster.crs
        elif boundary and boundary.with_suffix(".prj").is_file():
            target_crs = CRS.from_wkt(boundary.with_suffix(".prj").read_text(encoding="utf-8"))
        station_count = 0
        if summary and target_crs:
            sources.append(summary)
            if dem:
                sources.append(dem)
            tx = Transformer.from_crs(4326, target_crs, always_xy=True)
            for row in _read_csv_rows(summary):
                lon, lat = _numeric(row.get("lon")), _numeric(row.get("lat"))
                if not math.isfinite(lon) or not math.isfinite(lat):
                    continue
                x, y = tx.transform(lon, lat)
                axis.scatter(x, y, s=65, marker="o", color="#8D2FC0", edgecolor="white", zorder=5)
                axis.annotate(row.get("station_key") or row.get("station_id") or "雨源点", (x, y),
                              xytext=(5, 5), textcoords="offset points", fontsize=6, color="#6C1C94")
                station_count += 1
            handles.append(Line2D([0], [0], marker="o", color="#8D2FC0", linestyle="none", label="降雨来源点（紫色圆点，ERA5代理点）"))
        else:
            handles.append(Line2D([], [], linestyle="none", label="缺少雨源位置，未绘制雨站"))
        handles.append(Line2D([], [], linestyle="none", label="分区插值雨量计是降雨输入，不另画为站点"))
        parameters_path = context.first("swmm/model/auto_swmm_parameter_review.csv")
        parent_count = len(subcatchments)
        if parameters_path:
            sources.append(parameters_path)
            parameters = _read_csv_rows(parameters_path)
            if parameters and parameters[0].get('landuse_representation') == 'response_units':
                parent_count = len(parameters)
                handles.append(Line2D([], [], linestyle="none", label=f"{len(subcatchments)} 个内部地类响应单元，共用原分区出口"))
            if parameters and all(row.get('outlet_method') == 'minimum_dem_elevation' for row in parameters):
                handles.append(Line2D([], [], linestyle="none", label="出口定位：分区内原始DEM最低有效像元"))
            elif parameters and all(row.get('outlet_method') == 'drainage_exit' for row in parameters):
                handles.append(Line2D([], [], linestyle="none", label="出口定位：沿流向确定的分区出流点"))
        axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2, fontsize=7)
        axis.set_title(
            f"SWMM 结构：{parent_count} 个分区 · {len(outfalls)} 个出水口\n"
            f"{station_count} 个雨源点（ERA5代理点，非实测站） · {len(sections.get('RAINGAGES', []))} 个分区插值雨量计"
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("投影坐标 X (m)"); axis.set_ylabel("投影坐标 Y (m)")
    else:
        axis.text(0.5, 0.5, "缺少 SWMM 输入文件", ha="center", transform=axis.transAxes)
    _stamp(figure, context, "SWMM 模型结构")
    return {"figure": figure, "sources": sources}


def render_area_impervious(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    inp = context.first("swmm/model/*.inp")
    sources = [inp] if inp else []
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    if inp:
        rows = _inp_sections(inp).get("SUBCATCHMENTS", [])
        parent_path = context.first("swmm/model/auto_swmm_parameter_review.csv")
        parents = _read_csv_rows(parent_path) if parent_path else []
        if parents and parents[0].get('landuse_representation') == 'response_units':
            sources.append(parent_path)
            areas = [_numeric(row['area_ha']) for row in parents]
            impervious = [_numeric(row['imperv_pct']) for row in parents]
        else:
            areas = [_numeric(row[3]) for row in rows if len(row) > 4]
            impervious = [_numeric(row[4]) for row in rows if len(row) > 4]
        axes[0].hist([value for value in areas if value == value], bins=30,
                     color=COLORS["blue"], edgecolor="white")
        axes[0].set_xlabel("面积 (ha)"); axes[0].set_ylabel("数量"); axes[0].set_title("子汇水区面积")
        axes[1].hist([value for value in impervious if value == value], bins=30,
                     color=COLORS["gold"], edgecolor="white")
        axes[1].set_xlabel("不透水率 (%)"); axes[1].set_ylabel("数量"); axes[1].set_title("不透水率")
    else:
        axes[0].text(0.5, 0.5, "缺少 SWMM 输入文件", ha="center", transform=axes[0].transAxes)
        axes[1].axis("off")
    _stamp(figure, context, "面积与不透水率")
    return {"figure": figure, "sources": sources}


def render_infiltration(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    inp = context.first("swmm/model/*.inp")
    sources = [inp] if inp else []
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    if inp:
        sections = _inp_sections(inp)
        rows = sections.get("INFILTRATION", [])
        values = [_numeric(row[1]) for row in rows if len(row) > 1]
        axes[0].hist([value for value in values if value == value], bins=30,
                     color=COLORS["green"], edgecolor="white")
        axes[0].set_xlabel("下渗参数取值"); axes[0].set_ylabel("数量"); axes[0].set_title("渗透参数分布")
        subareas = sections.get("SUBAREAS", [])
        widths = [_numeric(row[1]) for row in subareas if len(row) > 1]
        axes[1].hist([value for value in widths if value == value], bins=30,
                     color=COLORS["grey"], edgecolor="white")
        axes[1].set_xlabel("漫流宽度 / N-Imperv 取值"); axes[1].set_ylabel("数量")
        axes[1].set_title("子汇水区参数")
    else:
        axes[0].text(0.5, 0.5, "缺少 SWMM 输入文件", ha="center", transform=axes[0].transAxes)
        axes[1].axis("off")
    _stamp(figure, context, "渗透参数")
    return {"figure": figure, "sources": sources}


def _runoff_series(context: RenderContext):
    output = context.first("swmm/results/*.out")
    if not output:
        return None, None
    from ..core.swmm.output_reader import read_subcatchment_runoff
    from ..core.swmm.landuse_response import aggregate_series
    inp = context.first("swmm/model/*.inp")
    series = read_subcatchment_runoff(output)
    if inp:
        series = aggregate_series(series, inp.parent)
    return series, output


def render_swmm_runoff(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    series, path = _runoff_series(context)
    sources = [path] if path else []
    figure, axis = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
    if series is not None:
        total = series.runoff_cms.sum(axis=1)
        axis.plot(series.datetimes, total, color=COLORS["blue"], linewidth=1.4, label="各子汇水区汇总")
        top = np.argsort(series.runoff_cms.max(axis=0))[-3:]
        for index in top:
            axis.plot(series.datetimes, series.runoff_cms[:, index], linewidth=0.9,
                      label=series.subcatchment_names[index])
        axis.set_ylabel("流量 (m³/s)"); axis.set_xlabel("时间")
        axis.set_title("径流过程（子汇水区汇总，非出口路由流量过程）")
        axis.legend(ncol=2)
        figure.autofmt_xdate()
    else:
        axis.text(0.5, 0.5, "缺少 SWMM 输出文件或尚未运行", ha="center", transform=axis.transAxes)
    _stamp(figure, context, "径流过程")
    return {"figure": figure, "sources": sources}


def render_swmm_peak(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    series, path = _runoff_series(context)
    sources = [path] if path else []
    figure, axis = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    if series is not None:
        peaks = series.runoff_cms.max(axis=0)
        axis.bar(range(len(peaks)), peaks, color=COLORS["red"])
        axis.set_xlabel("子汇水区序号"); axis.set_ylabel("峰值流量 (m³/s)")
        axis.set_title(f"峰值径流（{len(peaks)} 个子汇水区）")
    else:
        axis.text(0.5, 0.5, "缺少 SWMM 输出文件", ha="center", transform=axis.transAxes)
    _stamp(figure, context, "峰值径流")
    return {"figure": figure, "sources": sources}


def render_swmm_cumulative(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    series, path = _runoff_series(context)
    sources = [path] if path else []
    figure, axis = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    if series is not None:
        step = series.report_step_seconds
        cumulative = np.cumsum(series.runoff_cms, axis=0) * step
        axis.plot(series.datetimes, cumulative.sum(axis=1), color=COLORS["green"], linewidth=1.6)
        axis.set_ylabel("累积径流体积 (m³)"); axis.set_xlabel("时间")
        axis.set_title("累积径流")
        figure.autofmt_xdate()
    else:
        axis.text(0.5, 0.5, "缺少 SWMM 输出文件", ha="center", transform=axis.transAxes)
    _stamp(figure, context, "累积径流")
    return {"figure": figure, "sources": sources}


def render_swmm_continuity(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    report = context.first("swmm/results/*.rpt")
    sources = [report] if report else []
    figure, axis = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    if report:
        from ..result_builder import _continuity

        continuity = _continuity(report.read_text(encoding="utf-8", errors="replace"))
        labels = []
        values = []
        for key, entry in continuity.items():
            if "continuity_error_percent" in entry:
                labels.append("径流" if key == "runoff" else "演算")
                values.append(entry["continuity_error_percent"])
        if values:
            bars = axis.bar(labels, values,
                            color=[COLORS["green"] if abs(value) <= 10 else COLORS["red"] for value in values])
            axis.axhline(10, color=COLORS["red"], linestyle="--", linewidth=1, label="10% 门槛")
            axis.axhline(-10, color=COLORS["red"], linestyle="--", linewidth=1)
            for bar, value in zip(bars, values):
                axis.annotate(f"{value:.2f}%", (bar.get_x() + bar.get_width()/2, value),
                              textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7)
            axis.set_ylabel("连续性误差 (%)")
            axis.set_title("SWMM 连续性误差")
            axis.legend()
        else:
            axis.text(0.5, 0.5, "报告中未找到连续性误差块", ha="center", transform=axis.transAxes)
    else:
        axis.text(0.5, 0.5, "缺少 SWMM 报告文件", ha="center", transform=axis.transAxes)
    _stamp(figure, context, "连续性误差")
    return {"figure": figure, "sources": sources}


def render_stability_raster(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    metric = options.get("metric", "minimum_safety_factor")
    path = context.first(f"rapid_safety_factor_results/*{metric}.tif")
    sources = [path] if path else []
    figure, axis = plt.subplots(figsize=(7.5, 6.2), constrained_layout=True)
    if path:
        log = metric == "unstable_duration_h"
        _raster_preview(axis, path, "RdYlBu" if "safety" in metric else "viridis",
                        metric.replace("_", " "), log=log)
        with rasterio.open(path) as source:
            data = source.read(1, masked=True)
        finite = data.compressed()
        if finite.size:
            axis.set_title(f"{metric}\n"
                           f"最小 {float(finite.min()):.4g} · 中位数 {float(np.median(finite)):.4g} · "
                           f"最大 {float(finite.max()):.4g}")
    else:
        axis.text(0.5, 0.5, f"缺少 {metric} 栅格", ha="center", transform=axis.transAxes)
    _stamp(figure, context, metric)
    return {"figure": figure, "sources": sources}


def render_safety_time_series(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    path = context.first("rapid_safety_factor_results/*safety_factor_time_series.csv")
    sources = [path] if path else []
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    if path:
        rows = _read_csv_rows(path)
        columns = list(rows[0]) if rows else []
        time_column = next((column for column in columns if "time" in column.lower()), columns[0] if columns else "")
        for axis, column, label in (
            (axes[0], next((c for c in columns if "mean" in c.lower() and "fs" in c.lower()), None),
             "流域平均安全系数"),
            (axes[1], next((c for c in columns if "unstable" in c.lower()), None), "失稳面积比例"),
        ):
            if column is None:
                axis.text(0.5, 0.5, "缺少对应列", ha="center", transform=axis.transAxes)
                continue
            values = [_numeric(row.get(column)) for row in rows]
            axis.plot(range(len(values)), values, color=COLORS["blue"], linewidth=1.4)
            axis.set_xlabel(f"时次（起始 {rows[0].get(time_column, '')}）")
            axis.set_ylabel(label); axis.set_title(label)
    else:
        axes[0].text(0.5, 0.5, "缺少安全系数时序文件", ha="center", transform=axes[0].transAxes)
        axes[1].axis("off")
    _stamp(figure, context, "安全系数时序")
    return {"figure": figure, "sources": sources}


def render_sensitivity_ranking(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    path = context.first("analysis_results/parameter_sensitivity_ranking.csv")
    sources = [path] if path else []
    figure, axis = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    if path:
        rows = _read_csv_rows(path)
        parameter_column = next((column for column in rows[0] if "param" in column.lower()), list(rows[0])[0])
        value_column = next((column for column in rows[0]
                             if column != parameter_column and "rank" not in column.lower()), list(rows[0])[-1])
        pairs = sorted(((_numeric(row.get(value_column), 0.0), str(row.get(parameter_column))) for row in rows),
                       reverse=True)
        axis.barh([name for _, name in pairs], [value for value, _ in pairs], color=COLORS["blue"])
        axis.set_xlabel(value_column)
        axis.set_title("参数敏感性排序")
    else:
        axis.text(0.5, 0.5, "缺少敏感性结果", ha="center", transform=axis.transAxes)
    _stamp(figure, context, "参数敏感性排序")
    return {"figure": figure, "sources": sources}


def render_sobol_contribution(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    path = context.first("analysis_results/parameter_contribution_sobol.csv")
    sources = [path] if path else []
    figure, axis = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    if path:
        rows = _read_csv_rows(path)
        columns = list(rows[0]) if rows else []
        parameter_column = next((column for column in columns if "param" in column.lower()), columns[0])
        value_column = next((column for column in columns if "total" in column.lower() or "sobol" in column.lower()),
                            columns[-1])
        pairs = sorted(((_numeric(row.get(value_column), 0.0), str(row.get(parameter_column))) for row in rows),
                       reverse=True)
        axis.barh([name for _, name in pairs], [value for value, _ in pairs], color=COLORS["gold"])
        axis.set_xlabel(value_column); axis.set_title("Sobol 参数贡献")
    else:
        axis.text(0.5, 0.5, "缺少 Sobol 贡献结果", ha="center", transform=axis.transAxes)
    _stamp(figure, context, "参数贡献")
    return {"figure": figure, "sources": sources}


def render_representative_series(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    path = context.first("rapid_safety_factor_results/*representative_point_time_series.csv")
    sources = [path] if path else []
    figure, axis = plt.subplots(figsize=(9.5, 4.4), constrained_layout=True)
    if path:
        rows = _read_csv_rows(path)
        columns = list(rows[0]) if rows else []
        code_column = next((column for column in columns if "code" in column.lower()), None)
        value_column = next((column for column in columns if "fs" in column.lower() or "safety" in column.lower()), None)
        if code_column and value_column:
            groups: dict[str, list[float]] = {}
            for row in rows:
                groups.setdefault(str(row.get(code_column)), []).append(_numeric(row.get(value_column)))
            for code, values in groups.items():
                axis.plot(range(len(values)), values, linewidth=1.3, label=code)
            axis.legend(ncol=3)
        axis.set_xlabel("时次"); axis.set_ylabel("安全系数")
        axis.set_title("代表点安全系数时序")
    else:
        axis.text(0.5, 0.5, "缺少代表点时序文件", ha="center", transform=axis.transAxes)
    _stamp(figure, context, "关键位置时间序列")
    return {"figure": figure, "sources": sources}


def render_stage_comparison(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    path = context.first("analysis_results/staged_stability_comparison.csv")
    sources = [path] if path else []
    figure, axis = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    if path:
        rows = _read_csv_rows(path)
        columns = list(rows[0]) if rows else []
        label_column = columns[0]
        labels = [str(row.get(label_column)) for row in rows]
        value_columns = [column for column in columns[1:]][:4]
        width = 0.8 / max(1, len(value_columns))
        for index, column in enumerate(value_columns):
            positions = [position + index * width for position in range(len(labels))]
            axis.bar(positions, [_numeric(row.get(column), 0.0) for row in rows], width=width, label=column)
        axis.set_xticks([position + width * (len(value_columns) - 1) / 2 for position in range(len(labels))])
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.set_title("稳定性等级面积统计")
        axis.legend()
    else:
        axis.text(0.5, 0.5, "缺少分级对比结果", ha="center", transform=axis.transAxes)
    _stamp(figure, context, "等级面积统计")
    return {"figure": figure, "sources": sources}


def render_qa_dashboard(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    from ..qa import read_gates

    report = read_gates(context.job_dir)
    sources: list[Path] = []
    gate_file = context.job_dir / "qa.json"
    if gate_file.is_file():
        sources.append(gate_file)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    if report:
        gates = report.get("gates", [])
        names = [item["name"] for item in gates]
        passed = [1 if item["passed"] else 0 for item in gates]
        axes[0].bar(names, passed,
                    color=[COLORS["green"] if value else COLORS["red"] for value in passed])
        axes[0].set_ylim(0, 1.2); axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["未通过", "通过"])
        axes[0].set_title("质量门槛通过情况")
        facts = {}
        for item in gates:
            for key, value in (item.get("facts") or {}).items():
                if isinstance(value, (int, float)):
                    facts.setdefault(key, value)
        keys = list(facts)[:10]
        axes[1].barh(keys, [facts[key] for key in keys], color=COLORS["blue"])
        axes[1].set_title("关键指标")
    else:
        axes[0].text(0.5, 0.5, "尚未执行质量检查", ha="center", transform=axes[0].transAxes)
        axes[1].axis("off")
    _stamp(figure, context, "QA 仪表盘")
    return {"figure": figure, "sources": sources}


def render_qa_coverage(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    path = context.first("inputs/dem_derived/*.shp")
    sources = [path] if path else []
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)
    if path:
        records, _ = _read_shapes(path)
        areas = [_numeric(item["record"].get("area_ha"), 0.0) for item in records]
        for item in records:
            _plot_rings(axes[0], item["rings"], color=COLORS["blue"], linewidth=0.5, alpha=0.8)
        axes[0].set_title(f"划分覆盖（{len(records)} 个分区）")
        axes[0].set_aspect("equal", adjustable="datalim")
        tiny = [value for value in areas if 0 < value < 1.0]
        axes[1].hist(areas, bins=30, color=COLORS["grey"], edgecolor="white")
        axes[1].axvline(1.0, color=COLORS["red"], linestyle="--", linewidth=1.2, label="1 ha")
        axes[1].set_xlabel("面积 (ha)"); axes[1].set_ylabel("数量")
        axes[1].set_title(f"单像元/异常小分区检查（< 1 ha 共 {len(tiny)} 个）")
        axes[1].legend()
    else:
        axes[0].text(0.5, 0.5, "缺少子汇水区文件", ha="center", transform=axes[0].transAxes)
        axes[1].axis("off")
    _stamp(figure, context, "空间覆盖检查")
    return {"figure": figure, "sources": sources}


def render_export_index(context: RenderContext, options: dict[str, Any]) -> dict[str, Any]:
    index = context.first("final_results/final_results_index.csv")
    sources = [index] if index else []
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    if index:
        rows = _read_csv_rows(index)
        roles: dict[str, int] = {}
        for row in rows:
            role = row.get("role") or "other"
            roles[role] = roles.get(role, 0) + 1
        axes[0].bar(list(roles), list(roles.values()), color=COLORS["blue"])
        axes[0].set_title(f"成果构成（共 {len(rows)} 项）")
        axes[0].tick_params(axis="x", rotation=30)
        sizes = [_numeric(row.get("size_bytes"), 0.0) / 1024.0 for row in rows]
        axes[1].hist([value for value in sizes if value >= 0], bins=min(30, max(5, len(sizes) // 3)),
                     color=COLORS["gold"], edgecolor="white")
        axes[1].set_xlabel("文件大小 (KB)"); axes[1].set_ylabel("数量"); axes[1].set_title("成果体积分布")
    else:
        axes[0].text(0.5, 0.5, "缺少最终成果索引", ha="center", transform=axes[0].transAxes)
        axes[1].axis("off")
    _stamp(figure, context, "最终成果索引")
    return {"figure": figure, "sources": sources}


RENDERERS: dict[str, Callable[[RenderContext, dict[str, Any]], dict[str, Any]]] = {
    "run_manifest": render_run_manifest,
    "input_coverage": render_input_coverage,
    "subcatchments": render_subcatchments,
    "area_histogram": render_area_histogram,
    "coverage_gap": render_coverage_gap,
    "rainfall_series": render_rainfall_series,
    "rainfall_cumulative": render_rainfall_cumulative,
    "rainfall_station_map": render_rainfall_station_map,
    "station_weights": render_station_weights,
    "model_structure": render_model_structure,
    "area_impervious": render_area_impervious,
    "infiltration": render_infiltration,
    "swmm_runoff": render_swmm_runoff,
    "swmm_peak": render_swmm_peak,
    "swmm_cumulative": render_swmm_cumulative,
    "swmm_continuity": render_swmm_continuity,
    "stability_raster": render_stability_raster,
    "safety_time_series": render_safety_time_series,
    "sensitivity_ranking": render_sensitivity_ranking,
    "sobol_contribution": render_sobol_contribution,
    "representative_series": render_representative_series,
    "stage_comparison": render_stage_comparison,
    "qa_dashboard": render_qa_dashboard,
    "qa_coverage": render_qa_coverage,
    "export_index": render_export_index,
}
