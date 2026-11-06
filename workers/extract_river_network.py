"""从 DEM 抽取河网，输出与 data/重庆市_水系.shp **同字段**的面状水系图层。

方法就是项目自己那套（workers/hydrology_worker.py）：
    Priority-Flood 填洼 → D8 最陡下降（平区用填洼父节点链定向）→ 汇流累积
    → 河道 = 累积面积 ≥ 阈值 → 矢量化成面

必须分块跑（一次 8M 格上限），块间留 overlap，**只取块的内区**，以减轻接边不一致。
注意：填洼从栅格边界播种，平坦河段的方向受范围影响（项目已知的"坑 1"），
所以接边处尤其是低洼平区的河道走向仍可能与本底不符——这一点无法靠 overlap 消除。

用法：
  python workers/extract_river_network.py \
      --dem data/重庆.tif --output .scratch/river_network_dem.shp \
      --threshold-ha 10 --col 2209 --row 21052 --size 5656
省略 --col/--row/--size 则跑全幅（172 块，约 3 小时）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import shapefile
from rasterio.features import shapes as raster_shapes
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import shape as shapely_shape
from shapely.geometry.polygon import orient as shapely_orient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hydrology_worker import (  # noqa: E402
    accumulation_and_order,
    build_receivers,
    fill_depressions_priority_flood,
)

#: 与 data/重庆市_水系.shp 一致的字段，便于直接替换而不改读取代码。
#: DBF 字段名上限 10 字符——超了会被 pyshp 静默截断，之后按名取值就取不到。
FIELDS = [("Name", "C", 60), ("pyname", "C", 24), ("kind", "C", 8),
          ("up_km2", "F", 18, 4), ("src_len_km", "F", 18, 4)]


def _open_writer(base: Path) -> shapefile.Writer:
    """A writer with the drop-in field layout of data/重庆市_水系.shp."""
    writer = shapefile.Writer(str(base), shapeType=shapefile.POLYGON, encoding="gbk")
    for name, kind, *rest in FIELDS:
        writer.field(name, kind, *rest)
    return writer


def _write_prj(base: Path, crs) -> None:
    if crs is not None:
        base.with_suffix(".prj").write_text(crs.to_wkt(), encoding="utf-8")


def _merge_parts(parts_dir: Path, output: Path, crs,
                 min_length_m: float = 0.0, simplify_m: float = 0.0) -> tuple[int, int]:
    """把每个块的小 shapefile 合成一个；分块文件保留，便于中断后重跑合并。

    返回 (分块文件数, 写出的要素数)。``min_length_m`` 丢掉过短的退化面，
    ``simplify_m`` 按容差简化轮廓——两者都可以在 --remix-parts 下事后重来。
    """
    parts = sorted(parts_dir.glob("tile_*.shp"))
    writer = _open_writer(output.with_suffix(""))
    written = dropped = 0
    for part in parts:
        reader = shapefile.Reader(str(part), encoding="gbk")
        try:
            for record in reader.iterShapeRecords():
                fields = record.record.as_dict()
                # 分块文件可能是用 11 字符字段名写的（DBF 上限 10，pyshp 会截断），
                # 所以按名取不到时退回到截断名。
                values = [fields.get(name, fields.get(name[:10])) for name, *_ in FIELDS]
                length_km = float(values[4] or 0.0)
                if min_length_m and length_km * 1000.0 < min_length_m:
                    dropped += 1
                    continue
                points = record.shape.points
                ring_bounds = list(record.shape.parts) + [len(points)]
                rings = [[(x, y) for x, y in points[start:end]]
                         for start, end in zip(ring_bounds[:-1], ring_bounds[1:])]
                if simplify_m > 0:
                    polygon = shapely_orient(shapely_shape({"type": "Polygon",
                                                            "coordinates": rings}), sign=-1.0)
                    simplified = polygon.simplify(simplify_m)
                    if simplified.is_empty or simplified.geom_type != "Polygon":
                        dropped += 1
                        continue
                    rings = [list(simplified.exterior.coords)]
                writer.poly(rings)
                writer.record(*values)
                written += 1
        finally:
            reader.close()
    writer.close()
    _write_prj(output.with_suffix(""), crs)
    return len(parts), written, dropped


def _channel_length_by_label(channel: np.ndarray, receivers: np.ndarray,
                             labels: np.ndarray, count: int,
                             shape: tuple[int, int], cell_m: float) -> np.ndarray:
    """每个河段的河长（米）：沿 D8 流向累加，斜向一步算 √2 格。

    ``receivers`` 是 ``build_receivers`` 返回的**展平**索引（长 size = h*w）。
    只有流向仍在同一河段内的那一步才计入，所以河长与"河道像元数×格距"同源、
    不会被多边形周长带偏。
    """
    height, width = shape
    flat_receivers = np.asarray(receivers).reshape(-1)
    flat_channel = channel.reshape(-1)
    flat_labels = labels.reshape(-1)
    length = np.zeros(count + 1, dtype="float64")
    index = np.flatnonzero(flat_channel & (flat_receivers >= 0))
    if index.size == 0:
        return length
    target = flat_receivers[index]
    inside = flat_channel[target]
    index, target = index[inside], target[inside]
    rows_from, cols_from = np.divmod(index, width)
    rows_to, cols_to = np.divmod(target, width)
    diagonal = (rows_from != rows_to) & (cols_from != cols_to)
    step = np.where(diagonal, np.sqrt(2.0), 1.0) * cell_m
    np.add.at(length, flat_labels[index], step)
    return length


def tile_origins(col: int, row: int, width: int, height: int,
                 size: int) -> list[tuple[int, int]]:
    """块左上角（col, row）；最后一块对齐到区域边界，保证覆盖整个区域。"""
    cols = list(range(col, max(col + 1, col + width - size + 1), size))
    rows = list(range(row, max(row + 1, row + height - size + 1), size))
    if cols[-1] + size < col + width:
        cols.append(max(col, col + width - size))
    if rows[-1] + size < row + height:
        rows.append(max(row, row + height - size))
    return [(c, r) for r in rows for c in cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-ha", type=float, default=10.0,
                        help="河道判定阈值：上游汇水面积达到这么多公顷就算河道")
    parser.add_argument("--tile-cells", type=int, default=2828,
                        help="单次填洼的块边长（格）。2828² = 8.0M，与管线上限一致")
    parser.add_argument("--region", default="",
                        help="只跑一块区域：col,row,width,height（格）。省略=全幅")
    parser.add_argument("--overlap", type=int, default=160,
                        help="块间重叠（格，默认 160 ≈ 2 km）；只在重叠环以外的内区取河道")
    parser.add_argument("--min-length-m", type=float, default=0.0,
                        help="丢弃河长短于该值的孤立面（默认 0=全留）")
    parser.add_argument("--simplify-m", type=float, default=0.0,
                        help="按逐像素轮廓输出会很占体积（70×70 km 已 26 MB）；"
                             "给一个简化容差（建议 6 m ≈ 半个像元）可大幅压缩，默认 0=不简化")
    parser.add_argument("--remix-parts", action="store_true",
                        help="不重跑填洼：把分块目录里已有的分块按 --min-length-m 与 "
                             "--simplify-m 重新合成最终文件。用来在不重算的前提下改过滤/简化口径。")
    parser.add_argument("--parts-dir", type=Path, default=None,
                        help="分块目录；默认 <output>.parts/，跨输出名重合成时要显式给。")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--sweep-ha", default="",
                        help="顺便统计这些阈值下的河道长度（逗号分隔），例如 1,2,5,10,20,50,100。"
                             "填洼与汇流只算一次，所以这几乎不额外花时间——用来给阈值选型提供实测依据。")
    args = parser.parse_args()

    started = time.time()
    with rasterio.open(args.dem) as source:
        width, height, crs, transform = source.width, source.height, source.crs, source.transform
        nodata = source.nodata
        cell_m = abs(transform.a)
        cell_ha = cell_m ** 2 / 10_000.0
        if args.region:
            col, row, region_w, region_h = (int(v) for v in args.region.split(","))
        else:
            col, row, region_w, region_h = 0, 0, width, height
        size = min(args.tile_cells, region_w, region_h)
        origins = tile_origins(col, row, region_w, region_h, size)
        if args.remix_parts:
            print(f"重合成模式：不重跑填洼，只用 {args.output.stem}.parts/ 里的分块", flush=True)
        else:
            print(f"DEM {width}×{height} @ {cell_m:.4f} m | 区域 {region_w}×{region_h} @({col},{row}) "
                  f"| 块 {size}×{size} | 共 {len(origins)} 块 | 阈值 {args.threshold_ha} ha "
                  f"| 重叠 {args.overlap} 格", flush=True)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        # 全幅要跑几个小时：先按块写成小文件，最后再合并成一个。中途中断也不会让
        # 整个成果作废（shapefile 的头部只在 close 时写，直接写大文件一旦中断就不可读）。
        parts_dir = args.parts_dir or (args.output.parent / (args.output.stem + ".parts"))
        parts_dir.mkdir(parents=True, exist_ok=True)
        kept = discarded = 0
        total_cells = 0
        sweep = [float(v) for v in args.sweep_ha.split(",") if v.strip()]
        sweep_totals: dict[float, int] = {}
        pending_tiles = [] if args.remix_parts else list(enumerate(origins, start=1))
        for index, (col, row) in pending_tiles:
            read_col = max(0, col - args.overlap)
            read_row = max(0, row - args.overlap)
            read_w = min(width - read_col, size + 2 * args.overlap)
            read_h = min(height - read_row, size + 2 * args.overlap)
            window = Window(read_col, read_row, read_w, read_h)
            dem = source.read(1, window=window, masked=True).astype("float32").filled(np.nan)
            if nodata is not None:
                dem[dem == nodata] = np.nan
            valid = np.isfinite(dem)
            if not valid.any():
                continue

            t0 = time.time()
            filled, parents = fill_depressions_priority_flood(dem, valid)
            receivers, _ = build_receivers(filled, valid, parents)
            accumulation, _ = accumulation_and_order(receivers, valid)
            channel = valid & (accumulation * cell_ha >= args.threshold_ha)

            # 只取内区：把重叠环（以及超出本块的右/下边）切掉，接边像素不在本块输出。
            inner = np.zeros_like(channel)
            left = col - read_col
            top = row - read_row
            inner[top:top + size, left:left + size] = True

            for threshold in sweep:
                count = int((valid & inner
                             & (accumulation * cell_ha >= threshold)).sum())
                sweep_totals[threshold] = sweep_totals.get(threshold, 0) + count
            channel &= inner

            tile_transform = rasterio.windows.transform(window, transform)
            cells = int(channel.sum())
            total_cells += cells
            if cells:
                # 把相连的河道像素标成同一个"河段"，每个河段单独算自己的上游汇水面积与河长。
                labels, count = ndimage.label(channel, structure=np.ones((3, 3), dtype=int))
                per_label_max = np.zeros(count + 1, dtype="float64")
                np.maximum.at(per_label_max, labels[channel], accumulation[channel])
                # 河长沿 D8 路径逐格累加（斜向一步算 √2 格）。不要用多边形周长/2 估：
                # 一像素宽的斜向阶梯，周长明显大于中轴长度（实测高估 1.5 倍）。
                length_by_label = _channel_length_by_label(
                    channel, receivers, labels, count, dem.shape, cell_m)
                part_path = parts_dir / f"tile_{col:06d}_{row:06d}"
                part_writer = _open_writer(part_path)
                for geom, value in raster_shapes(labels.astype("int32"), mask=channel,
                                                 transform=tile_transform, connectivity=8):
                    label = int(value)
                    polygon = shapely_shape(geom)
                    if polygon.is_empty or polygon.geom_type != "Polygon":
                        continue
                    length_m = float(length_by_label[label])
                    if args.min_length_m and length_m < args.min_length_m:
                        discarded += 1
                        continue
                    upstream_km2 = float(per_label_max[label]) * cell_ha / 100.0
                    # Shapefile 约定外环顺时针；rasterio 输出是 GeoJSON 约定（逆时针），
                    # 与 data/重庆市_水系.shp 不一致，这里统一过来。
                    oriented = shapely_orient(polygon, sign=-1.0)
                    if args.simplify_m > 0:
                        oriented = oriented.simplify(args.simplify_m)
                    part_writer.poly([list(oriented.exterior.coords)])
                    part_writer.record("", "", "0123", round(upstream_km2, 4),
                                       round(length_m / 1000.0, 4))
                    kept += 1
                part_writer.close()
                _write_prj(part_path, crs)

            print(f"  [{index}/{len(origins)}] 块({col},{row}) {read_w}×{read_h} "
                  f"河道 {cells:,} 格 · {time.time() - t0:.0f}s "
                  f"· 累计已写 {kept:,} 个面", flush=True)

        part_count, written, merge_dropped = _merge_parts(
            parts_dir, args.output, crs,
            min_length_m=args.min_length_m, simplify_m=args.simplify_m)
        print(f"合并 {part_count} 个分块文件 → {args.output}：写出 {written:,} 个要素，"
              f"按最短长度与简化丢弃 {merge_dropped:,} 个", flush=True)

    summary = {
        "dem": str(args.dem), "output": str(args.output),
        "threshold_ha": args.threshold_ha, "cell_m": round(cell_m, 4),
        "tiles": len(origins), "tile_size": size, "overlap_cells": args.overlap,
        "channel_cells": total_cells,
        "channel_km": round(total_cells * cell_m / 1000.0, 1),
        "polygons_written": written, "polygons_dropped": discarded + merge_dropped,
        "seconds": round(time.time() - started, 1),
        "note": "字段与 data/重庆市_水系.shp 一致；Name/pyname 为空（DEM 抽的河网没有名字）。",
    }
    if sweep_totals:
        area_km2 = (region_w * region_h) * (cell_m ** 2) / 1e6
        summary["sweep"] = {
            "region_area_km2": round(area_km2, 1),
            "by_threshold_ha": {
                f"{threshold:g}": {
                    "km": round(cells * cell_m / 1000.0, 1),
                    "density_km_per_km2": round(cells * cell_m / 1000.0 / area_km2, 3),
                } for threshold, cells in sorted(sweep_totals.items())
            },
            "note": "河道长度按河道像元数×格距计；同一次填洼里顺带统计，未计入接缝影响。",
        }
    if args.report:
        args.report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
