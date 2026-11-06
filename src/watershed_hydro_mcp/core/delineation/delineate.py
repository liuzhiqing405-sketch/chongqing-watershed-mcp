from __future__ import annotations

import csv
import heapq
import json
from pathlib import Path
import sys

from watershed_hydro_mcp.core import runtime_bootstrap  # noqa: F401

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from scipy import ndimage
import shapefile
import yaml

from ..spatial_domain import landuse_mask_for_grid
from .partition_topology import coarsen_labels, partition_facts, clip_labels_to_local_drainage, filter_boundary_fragments
from watershed_exchange.paths import sha256_file
from watershed_exchange.dem_budget import DEFAULT_MAXIMUM_PROCESSING_CELLS, cell_budget, require_cells


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from watershed_hydro_mcp.core.common import LazyConfig, RUN_ROOT, RUN_ROOT as ROOT

CFG = LazyConfig()

D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def read_dem_for_delineation() -> tuple[np.ndarray, np.ndarray, Affine, rasterio.crs.CRS]:
    dem_path = Path(CFG["terrain"]["dem"])
    resolution = float(CFG["swmm"].get("delineation_resolution_m") or 20)
    with rasterio.open(dem_path) as src:
        if not src.crs:
            raise RuntimeError(f"DEM has no CRS: {dem_path}")
        scale_x = max(1, int(round(resolution / abs(src.transform.a))))
        scale_y = max(1, int(round(resolution / abs(src.transform.e))))
        out_h = max(1, src.height // scale_y)
        out_w = max(1, src.width // scale_x)
        require_cells(out_h*out_w, cell_budget(CFG.get("terrain", {}).get(
            "maximum_processing_cells", DEFAULT_MAXIMUM_PROCESSING_CELLS)), "分区DEM")
        data = src.read(1, out_shape=(out_h, out_w), masked=True).astype("float64")
        transform = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
        valid = ~np.ma.getmaskarray(data)
        arr = data.filled(np.nan)
        if CFG.get("study_area", {}).get("calculation_domain", "dem_landuse_intersection") == "dem_landuse_intersection":
            landuse_path = Path(CFG.get("study_area", {}).get("landuse_shp") or "")
            if landuse_path.exists():
                lu_mask = landuse_mask_for_grid(arr.shape, transform, src.crs, landuse_path)
                valid &= lu_mask
                print(f"Calculation domain: DEM and land-use intersection; valid cells={int(np.count_nonzero(valid))}")
            else:
                print("[WARN] Land-use layer not found; using DEM valid area only.")
    return arr, valid, transform, src.crs


def write_full_dem_grid_subcatchments(out_shp: Path) -> None:
    dem_path = Path(CFG["terrain"]["dem"])
    grid_size = float(CFG["swmm"].get("full_coverage_grid_size_m") or 2000)
    min_area_ha = float(CFG["swmm"].get("delineation_min_area_ha") or 0)
    if grid_size < 100:
        raise RuntimeError(
            f"full_coverage_grid_size_m={grid_size:g} m is too small for regional SWMM partitioning. "
            "Use 1000-3000 m for whole-area rapid simulation; the default is 2000 m."
        )
    with rasterio.open(dem_path) as src:
        if not src.crs:
            raise RuntimeError(f"DEM has no CRS: {dem_path}")
        left, bottom, right, top = src.bounds
        crs = src.crs

    out_shp.parent.mkdir(parents=True, exist_ok=True)
    base = out_shp.with_suffix("")
    writer = shapefile.Writer(str(base), shapeType=shapefile.POLYGON)
    writer.field("name", "C", 32)
    writer.field("area_ha", "F", 14, 4)
    writer.field("source", "C", 32)

    records = []
    idx = 1
    y = bottom
    while y < top:
        y2 = min(y + grid_size, top)
        x = left
        while x < right:
            x2 = min(x + grid_size, right)
            area_ha = (x2 - x) * (y2 - y) / 10000.0
            if area_ha >= min_area_ha:
                name = f"G{idx:03d}"
                ring = [(x, y), (x, y2), (x2, y2), (x2, y), (x, y)]
                writer.poly([[tuple(map(float, p)) for p in ring]])
                writer.record(name, area_ha, "full_dem_grid")
                records.append((name, area_ha))
                idx += 1
            x = x2
        y = y2

    writer.close()
    base.with_suffix(".prj").write_text(crs.to_wkt("WKT1_GDAL"), encoding="utf-8")

    summary = out_shp.with_name("dem_derived_subcatchments_summary.csv")
    with summary.open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["name", "area_ha", "source"])
        for name, area_ha in records:
            wr.writerow([name, round(area_ha, 4), "full_dem_grid"])

    print(f"Written DEM full-coverage SWMM partitions: {out_shp}")
    print(f"Grid size: {grid_size:.1f} m")
    print(f"Subcatchments: {len(records)}")
    print(f"Total DEM bbox area: {sum(area for _, area in records):.3f} ha")


def find_grass_cli() -> Path | None:
    from watershed_hydro_mcp.grass_runtime import resolve_grass_cli
    return resolve_grass_cli(CFG.get("executables", {}).get("grass_cli"), RUN_ROOT)


def write_grass_watersheds(out_shp: Path) -> None:
    from .grass_delineation import write_grass_watersheds as write
    write(out_shp)


def fill_depressions_priority_flood(dem: np.ndarray, valid: np.ndarray,
                                   *, return_parents: bool = False):
    """Flood from every valid-domain edge, retaining a flat drainage tree.

    A clipped polygon's NoData neighbours are open edges, just like the
    raster perimeter. Parents lead to previously visited cells at no greater
    elevation; they resolve flats without modifying real slopes.
    """
    h, w = dem.shape
    if valid.shape != dem.shape or not np.any(valid) or not np.all(np.isfinite(dem[valid])):
        raise ValueError("Hydrologic DEM must contain finite elevations on a nonempty valid mask.")
    filled = dem.copy()
    visited = np.zeros((h, w), dtype=bool)
    parents = np.full((h, w), -1, dtype=np.int64)
    heap: list[tuple[float, int, int]] = []

    boundary = domain_boundary(valid)
    for r, c in zip(*np.where(boundary)):
        visited[r, c] = True
        heapq.heappush(heap, (float(filled[r, c]), int(r), int(c)))

    while heap:
        z, r, c = heapq.heappop(heap)
        for dr, dc in D8:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= h or cc < 0 or cc >= w or visited[rr, cc] or not valid[rr, cc]:
                continue
            visited[rr, cc] = True
            parents[rr, cc] = r * w + c
            nz = float(filled[rr, cc])
            if nz < z:
                filled[rr, cc] = z
                nz = z
            heapq.heappush(heap, (nz, rr, cc))
    return (filled, parents) if return_parents else filled


def domain_boundary(valid: np.ndarray) -> np.ndarray:
    boundary = valid & ndimage.binary_dilation(~valid, structure=np.ones((3, 3), dtype=bool))
    boundary[0, :] |= valid[0, :]
    boundary[-1, :] |= valid[-1, :]
    boundary[:, 0] |= valid[:, 0]
    boundary[:, -1] |= valid[:, -1]
    return boundary


def build_receivers(filled: np.ndarray, valid: np.ndarray, flat_parents=None,
                    *, cell_width: float = 1.0, cell_height: float = 1.0) -> np.ndarray:
    return build_receivers_fast(filled, valid, flat_parents,
                                cell_width=cell_width, cell_height=cell_height)


def build_receivers_fast(dem: np.ndarray, valid: np.ndarray, flat_parents=None,
                         *, cell_width: float = 1.0, cell_height: float = 1.0) -> np.ndarray:
    h, w = dem.shape
    z = np.where(valid, dem, np.nan)
    rows = np.arange(h, dtype=np.int64)[:, None]
    cols = np.arange(w, dtype=np.int64)[None, :]
    current = np.where(valid, z, np.nan)
    best_drop = np.zeros((h, w), dtype=np.float64)
    receivers = np.full((h, w), -1, dtype=np.int64)

    for dr, dc in D8:
        neighbor = np.full((h, w), np.nan, dtype=np.float64)
        r_src0 = max(0, dr)
        r_src1 = h + min(0, dr)
        c_src0 = max(0, dc)
        c_src1 = w + min(0, dc)
        r_dst0 = max(0, -dr)
        r_dst1 = h - max(0, dr)
        c_dst0 = max(0, -dc)
        c_dst1 = w - max(0, dc)
        neighbor[r_dst0:r_dst1, c_dst0:c_dst1] = z[r_src0:r_src1, c_src0:c_src1]
        distance = float(np.hypot(dc * cell_width, dr * cell_height))
        drop = (current - neighbor) / distance
        candidate_rows = rows + dr
        candidate_cols = cols + dc
        candidate_idx = candidate_rows * w + candidate_cols
        ok = valid & np.isfinite(drop) & (drop > best_drop)
        receivers[ok] = candidate_idx[ok]
        best_drop[ok] = drop[ok]
    if flat_parents is not None:
        flat = valid & (receivers < 0) & (flat_parents >= 0)
        receivers[flat] = flat_parents[flat]
    return receivers.ravel()


def accumulation(receivers: np.ndarray, valid: np.ndarray) -> np.ndarray:
    flat_valid = valid.ravel()
    n = receivers.size
    indegree = np.zeros(n, dtype=np.int32)
    acc = flat_valid.astype(np.float64)
    for i, rec in enumerate(receivers):
        if flat_valid[i] and rec >= 0:
            indegree[rec] += 1
    queue = list(np.flatnonzero(flat_valid & (indegree == 0)))
    head = 0
    while head < len(queue):
        i = int(queue[head])
        head += 1
        rec = int(receivers[i])
        if rec >= 0:
            acc[rec] += acc[i]
            indegree[rec] -= 1
            if indegree[rec] == 0:
                queue.append(rec)
    if head != int(np.count_nonzero(flat_valid)):
        raise RuntimeError("Flow routing contains a cycle; subcatchments cannot be delineated.")
    return acc.reshape(valid.shape)


def donor_count(receivers: np.ndarray, valid: np.ndarray, channel: np.ndarray | None = None) -> np.ndarray:
    flat_valid = valid.ravel()
    n = receivers.size
    counts = np.zeros(n, dtype=np.int16)
    if channel is None:
        donor_mask = flat_valid
    else:
        donor_mask = flat_valid & channel.ravel()
    for i, rec in enumerate(receivers):
        if donor_mask[i] and rec >= 0:
            counts[rec] += 1
    return counts.reshape(valid.shape)


def outlet_labels(receivers: np.ndarray, valid: np.ndarray) -> np.ndarray:
    flat_valid = valid.ravel()
    outlets = np.full(receivers.size, -1, dtype=np.int64)
    valid_ids = np.flatnonzero(flat_valid)
    for start in valid_ids:
        if outlets[start] >= 0:
            continue
        path = []
        cur = int(start)
        while cur >= 0 and outlets[cur] < 0:
            path.append(cur)
            cur = int(receivers[cur])
        outlet = outlets[cur] if cur >= 0 and outlets[cur] >= 0 else path[-1]
        for idx in path:
            outlets[idx] = outlet
    return outlets.reshape(valid.shape)


def assign_to_downstream_nodes(receivers: np.ndarray, valid: np.ndarray, seed_labels: np.ndarray) -> np.ndarray:
    flat_valid = valid.ravel()
    seed_flat = seed_labels.ravel()
    out = np.zeros(receivers.size, dtype=np.int32)
    out[seed_flat > 0] = seed_flat[seed_flat > 0]
    valid_ids = np.flatnonzero(flat_valid)
    for start in valid_ids:
        if out[start] > 0:
            continue
        path = []
        cur = int(start)
        while cur >= 0 and flat_valid[cur] and out[cur] == 0:
            path.append(cur)
            cur = int(receivers[cur])
        label = out[cur] if cur >= 0 and cur < out.size and out[cur] > 0 else 0
        for idx in path:
            out[idx] = label
    return out.reshape(valid.shape)


def ring_area_ha(ring: list[tuple[float, float]]) -> float:
    area = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0 / 10000.0


def write_subcatchment_shp(
    label_grid: np.ndarray,
    transform: Affine,
    crs,
    out_shp: Path,
    min_area_ha: float,
    outlet_by_label: dict[int, tuple[float, float]] | None = None,
    boundary_labels: set[int] | None = None,
    summary_name: str = "dem_derived_subcatchments_summary.csv",
    source: str = "dem_delineation",
) -> None:
    out_shp.parent.mkdir(parents=True, exist_ok=True)
    base = out_shp.with_suffix("")
    writer = shapefile.Writer(str(base), shapeType=shapefile.POLYGON)
    writer.field("name", "C", 32)
    writer.field("area_ha", "F", 18, 8)
    writer.field("source", "C", 32)
    writer.field("out_x", "F", 18, 3)
    writer.field("out_y", "F", 18, 3)
    writer.field("boundary", "N", 1, 0)
    records = []
    grouped = {}
    for geom, value in shapes(label_grid.astype("int32"), mask=label_grid > 0, transform=transform):
        label = int(value)
        if label <= 0:
            continue
        polygons = []
        if geom["type"] == "Polygon":
            polygons = [geom["coordinates"]]
        elif geom["type"] == "MultiPolygon":
            polygons = geom["coordinates"]
        rings = grouped.setdefault(label, [])
        for polygon in polygons:
            for index, ring in enumerate(polygon):
                pts = [(float(x), float(y)) for x, y in ring]
                signed = sum(x1*y2-x2*y1 for (x1,y1),(x2,y2) in zip(pts,pts[1:]))
                # ESRI outer rings clockwise, holes counterclockwise.
                if (index == 0 and signed > 0) or (index > 0 and signed < 0):
                    pts.reverse()
                rings.append(pts)
    counts = np.bincount(label_grid.ravel())
    for label, rings in sorted(grouped.items()):
        area_ha = counts[label] * abs(transform.a * transform.e) / 10000.0
        if area_ha < min_area_ha:
            continue
        out_x, out_y = outlet_by_label.get(label, (0.0, 0.0)) if outlet_by_label else (0.0, 0.0)
        name = f"D{len(records) + 1:05d}"
        is_boundary = int(label in (boundary_labels or set()))
        writer.poly(rings)
        writer.record(name, area_ha, source, out_x, out_y, is_boundary)
        records.append((name, label, area_ha, is_boundary))
    writer.close()
    base.with_suffix(".prj").write_text(crs.to_wkt("WKT1_GDAL"), encoding="utf-8")

    summary = out_shp.with_name(summary_name)
    with summary.open("w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["name", "label", "area_ha", "source", "boundary"])
        for row in sorted(records):
            wr.writerow([row[0], row[1], round(row[2], 8), source, row[3]])


def write_channel_node_layers(
    seed_labels: np.ndarray,
    channel: np.ndarray,
    acc: np.ndarray,
    transform: Affine,
    crs,
    out_shp: Path,
    cell_area_ha: float,
) -> None:
    base = out_shp.with_suffix("")
    node_base = base.with_name(base.name + "_nodes")
    channel_base = base.with_name(base.name + "_channel_cells")

    node_writer = shapefile.Writer(str(node_base), shapeType=shapefile.POINT)
    node_writer.field("node_id", "N", 8, 0)
    node_writer.field("acc_ha", "F", 14, 3)
    rows, cols = np.where(seed_labels > 0)
    for r, c in zip(rows, cols):
        x, y = transform * (c + 0.5, r + 0.5)
        label = int(seed_labels[r, c])
        node_writer.point(float(x), float(y))
        node_writer.record(label, float(acc[r, c] * cell_area_ha))
    node_writer.close()
    node_base.with_suffix(".prj").write_text(crs.to_wkt("WKT1_GDAL"), encoding="utf-8")

    channel_writer = shapefile.Writer(str(channel_base), shapeType=shapefile.POINT)
    channel_writer.field("acc_ha", "F", 14, 3)
    ch_rows, ch_cols = np.where(channel)
    step = max(1, int(len(ch_rows) / 20000))
    for r, c in zip(ch_rows[::step], ch_cols[::step]):
        x, y = transform * (c + 0.5, r + 0.5)
        channel_writer.point(float(x), float(y))
        channel_writer.record(float(acc[r, c] * cell_area_ha))
    channel_writer.close()
    channel_base.with_suffix(".prj").write_text(crs.to_wkt("WKT1_GDAL"), encoding="utf-8")


def write_channel_subbasins(out_shp: Path) -> None:
    max_count = int(CFG["swmm"].get("delineation_max_subcatchments") or 250)
    min_area_ha = float(CFG["swmm"].get("delineation_min_area_ha") or 5)
    threshold_ha = float(CFG["swmm"].get("channel_threshold_ha") or 20)
    spacing_m = float(CFG["swmm"].get("channel_node_spacing_m") or 500)

    dem, valid, transform, crs = read_dem_for_delineation()
    cell_size = (abs(transform.a) + abs(transform.e)) / 2
    cell_area_ha = abs(transform.a * transform.e) / 10000.0
    min_spacing_cells = max(1, int(round(spacing_m / cell_size)))
    print(f"DEM hydrologic delineation grid: {dem.shape[1]} x {dem.shape[0]}; cell size ~= {cell_size:.2f} m")
    print(f"Channel threshold: {threshold_ha:.2f} ha; node spacing: {spacing_m:.1f} m; max nodes: {max_count}")

    filled, parents = fill_depressions_priority_flood(dem, valid, return_parents=True)
    receivers = build_receivers(filled, valid, parents, cell_width=abs(transform.a), cell_height=abs(transform.e))
    acc = accumulation(receivers, valid)
    channel = valid & (acc * cell_area_ha >= threshold_ha)
    if not np.any(channel):
        raise RuntimeError("No channel cells found. Reduce channel_threshold_ha or increase delineation resolution.")

    ch_donors = donor_count(receivers, valid, channel)
    flat_channel = channel.ravel()
    rec = receivers
    terminal = np.zeros(receivers.size, dtype=bool)
    channel_ids = np.flatnonzero(flat_channel)
    terminal[channel_ids] = (rec[channel_ids] < 0) | (~flat_channel[np.clip(rec[channel_ids], 0, rec.size - 1)])
    terminal_grid = terminal.reshape(valid.shape)
    confluence_grid = channel & (ch_donors >= 2)

    candidate_mask = channel.copy()
    priority_mask = terminal_grid | confluence_grid
    candidates = []
    for r, c in zip(*np.where(priority_mask)):
        candidates.append((float(acc[r, c]), int(r), int(c), 1))
    for r, c in zip(*np.where(candidate_mask & ~priority_mask)):
        candidates.append((float(acc[r, c]), int(r), int(c), 0))
    candidates.sort(reverse=True)

    seed_labels = np.zeros(valid.shape, dtype=np.int32)
    selected: list[tuple[int, int]] = []
    for _, r, c, priority in candidates:
        if len(selected) >= max_count:
            break
        if priority == 0 and selected:
            too_close = False
            for rr, cc in selected:
                if (r - rr) * (r - rr) + (c - cc) * (c - cc) < min_spacing_cells * min_spacing_cells:
                    too_close = True
                    break
            if too_close:
                continue
        selected.append((r, c))
        seed_labels[r, c] = len(selected)

    if not selected:
        raise RuntimeError("No DEM channel nodes selected. Lower channel_threshold_ha or channel_node_spacing_m.")

    label_grid = assign_to_downstream_nodes(receivers, valid, seed_labels)
    missing = valid & (label_grid <= 0)
    if np.any(missing):
        outlet_grid = outlet_labels(receivers, valid)
        next_label = int(seed_labels.max()) + 1
        rows_count, cols_count = valid.shape
        for outlet in np.unique(outlet_grid[missing]):
            if outlet < 0:
                continue
            basin = missing & (outlet_grid == outlet)
            if np.count_nonzero(basin) * cell_area_ha < min_area_ha:
                continue
            label_grid[basin] = next_label
            rr, cc = divmod(int(outlet), cols_count)
            if 0 <= rr < rows_count and 0 <= cc < cols_count:
                seed_labels[rr, cc] = next_label
            next_label += 1
        unresolved = valid & (label_grid <= 0)
        if np.any(unresolved):
            label_grid[unresolved] = next_label

    outlet_by_label = {}
    for r, c in zip(*np.where(seed_labels > 0)):
        label = int(seed_labels[r, c])
        x, y = transform * (c + 0.5, r + 0.5)
        outlet_by_label[label] = (float(x), float(y))

    write_subcatchment_shp(label_grid, transform, crs, out_shp, min_area_ha, outlet_by_label=outlet_by_label)
    write_channel_node_layers(seed_labels, channel, acc, transform, crs, out_shp, cell_area_ha)
    print(f"Written DEM channel-controlled SWMM partitions: {out_shp}")
    print(f"Selected channel/outlet nodes: {len(selected)}")
    print(f"Channel cells: {int(np.count_nonzero(channel))}")
    print(f"Total labelled area: {float(np.count_nonzero(label_grid > 0) * cell_area_ha):.3f} ha")


def stream_link_labels(receivers: np.ndarray, valid: np.ndarray, channel: np.ndarray, ch_donors: np.ndarray, acc: np.ndarray) -> np.ndarray:
    flat_valid = valid.ravel()
    flat_channel = channel.ravel()
    flat_donors = ch_donors.ravel()
    flat_acc = acc.ravel()
    labels = np.zeros(receivers.size, dtype=np.int32)
    current_label = 0
    start_ids = np.flatnonzero(flat_channel & ((flat_donors == 0) | (flat_donors >= 2)))
    start_ids = sorted(start_ids, key=lambda idx: float(flat_acc[idx]))

    for start in start_ids:
        if labels[start] > 0 and flat_donors[start] < 2:
            continue
        current_label += 1
        cur = int(start)
        first = True
        while cur >= 0 and flat_valid[cur] and flat_channel[cur]:
            if labels[cur] == 0 or (first and flat_donors[cur] >= 2):
                labels[cur] = current_label
            rec = int(receivers[cur])
            if rec < 0 or not flat_channel[rec]:
                break
            if not first and flat_donors[rec] >= 2:
                break
            if first and rec != cur and flat_donors[rec] >= 2:
                break
            first = False
            cur = rec

    for start in np.flatnonzero(flat_channel & (labels == 0)):
        current_label += 1
        cur = int(start)
        while cur >= 0 and flat_valid[cur] and flat_channel[cur] and labels[cur] == 0:
            labels[cur] = current_label
            rec = int(receivers[cur])
            if rec < 0 or not flat_channel[rec] or flat_donors[rec] >= 2:
                break
            cur = rec
    return labels.reshape(valid.shape)


def assign_cells_to_stream_links(receivers: np.ndarray, valid: np.ndarray, link_grid: np.ndarray) -> np.ndarray:
    flat_valid = valid.ravel()
    flat_link = link_grid.ravel()
    assigned = np.zeros(receivers.size, dtype=np.int32)
    assigned[flat_link > 0] = flat_link[flat_link > 0]
    valid_ids = np.flatnonzero(flat_valid)
    for start in valid_ids:
        if assigned[start] > 0:
            continue
        path = []
        cur = int(start)
        while cur >= 0 and flat_valid[cur] and assigned[cur] == 0:
            path.append(cur)
            cur = int(receivers[cur])
        label = assigned[cur] if 0 <= cur < assigned.size and assigned[cur] > 0 else 0
        for idx in path:
            assigned[idx] = label
    return assigned.reshape(valid.shape)


def merge_small_labels_to_downstream(label_grid: np.ndarray, receivers: np.ndarray, valid: np.ndarray, min_area_ha: float, cell_area_ha: float) -> np.ndarray:
    out = label_grid.copy()
    labels, counts = np.unique(out[out > 0], return_counts=True)
    small = {int(label) for label, count in zip(labels, counts) if count * cell_area_ha < min_area_ha}
    if not small:
        return out
    rows, cols = out.shape
    for label in small:
        cells = np.flatnonzero(out.ravel() == label)
        replacement = 0
        for idx in cells:
            cur = int(idx)
            seen = set()
            while cur >= 0 and cur not in seen:
                seen.add(cur)
                rec = int(receivers[cur])
                if rec < 0:
                    break
                rec_label = int(out.ravel()[rec])
                if rec_label > 0 and rec_label != label and rec_label not in small:
                    replacement = rec_label
                    break
                cur = rec
            if replacement:
                break
        if replacement:
            out.ravel()[cells] = replacement
    return out


def merge_small_labels_to_neighbors(label_grid: np.ndarray, valid: np.ndarray, min_area_ha: float, cell_area_ha: float) -> np.ndarray:
    out = label_grid.copy()
    min_cells = max(1, int(round(min_area_ha / cell_area_ha)))
    structure = np.ones((3, 3), dtype=bool)

    for _ in range(2000):
        labels, counts = np.unique(out[out > 0], return_counts=True)
        sizes = {int(label): int(count) for label, count in zip(labels, counts)}
        small_labels = [label for label, count in sizes.items() if count < min_cells]
        if not small_labels:
            break

        merged = False
        for label in small_labels:
            mask = out == int(label)
            if not np.any(mask):
                continue
            ring = ndimage.binary_dilation(mask, structure=structure) & valid & (out > 0) & (out != int(label))
            neighbors, shared = np.unique(out[ring], return_counts=True)
            candidates = [(int(n), int(s)) for n, s in zip(neighbors, shared) if int(n) > 0 and int(n) != int(label)]
            if not candidates:
                continue
            chosen = max(candidates, key=lambda item: (item[1], sizes.get(item[0], 0)))[0]
            out[mask] = chosen
            merged = True
            break
        if not merged:
            break

    relabelled = np.zeros(out.shape, dtype=np.int32)
    next_label = 1
    for label in np.unique(out[out > 0]):
        relabelled[out == int(label)] = next_label
        next_label += 1
    return relabelled


def read_drainage_context(dem, valid, transform, crs):
    """Read the frozen exterior DEM on an exactly aligned calculation lattice."""
    import math
    from rasterio.warp import reproject, Resampling
    name = CFG.get("terrain", {}).get("drainage_context_dem")
    if not name:
        return dem, valid, transform, (slice(None), slice(None)), {"used": False, "reason": "legacy_package_without_context"}
    path = Path(name)
    if not path.is_file():
        raise RuntimeError("Declared drainage context DEM is missing")
    with rasterio.open(path) as source:
        if source.crs != crs or transform.b or transform.d or transform.a <= 0 or transform.e >= 0:
            raise RuntimeError("Drainage context must use the calculation DEM's north-up projected CRS")
        def offset(value, rounding):
            # Native aligned bounds can land just beyond an integer because
            # coordinate subtraction loses a few floating-point digits.
            nearest = round(value)
            return nearest if abs(value-nearest) <= 1e-7 else rounding(value)
        col0 = offset((source.bounds.left-transform.c)/transform.a, math.floor)
        row0 = offset((source.bounds.top-transform.f)/transform.e, math.floor)
        col1 = offset((source.bounds.right-transform.c)/transform.a, math.ceil)
        row1 = offset((source.bounds.bottom-transform.f)/transform.e, math.ceil)
        height, width = row1-row0, col1-col0
        if col0 > 0 or row0 > 0 or col1 < dem.shape[1] or row1 < dem.shape[0]:
            raise RuntimeError("Drainage context does not cover the calculation DEM")
        maximum = cell_budget(CFG.get("terrain", {}).get(
            "maximum_processing_cells", DEFAULT_MAXIMUM_PROCESSING_CELLS))
        require_cells(height*width, maximum, "外围排水DEM")
        grid = np.full((height, width), np.nan)
        grid_transform = transform * Affine.translation(col0, row0)
        data = source.read(1, masked=True).astype('float64').filled(np.nan)
        reproject(data, grid, src_transform=source.transform, src_crs=source.crs,
                  dst_transform=grid_transform, dst_crs=crs,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
    window = (slice(-row0, -row0+dem.shape[0]), slice(-col0, -col0+dem.shape[1]))
    if not np.all(np.isfinite(grid[window])[valid]):
        raise RuntimeError("Drainage context has missing elevations inside the calculation domain")
    if not np.allclose(grid[window][valid], dem[valid], atol=1e-5, rtol=0):
        raise RuntimeError("Drainage context and calculation DEM elevations are inconsistent")
    return grid, np.isfinite(grid), grid_transform, window, {
        "used": True, "cells": int(np.isfinite(grid).sum()),
        "processing_cells": height*width, "maximum_cells": maximum, "sha256": sha256_file(path),
        "scope": "buffered terrain only; external upstream inflow excluded"}


def restrict_receivers(receivers, context_shape, window, valid):
    """Cut every first exit at the user's domain; do not import outside runoff."""
    full = receivers.reshape(context_shape)[window]
    rows, cols = np.divmod(np.maximum(full, 0), context_shape[1])
    rows = rows - (window[0].start or 0)
    cols = cols - (window[1].start or 0)
    inside = (full >= 0) & (rows >= 0) & (cols >= 0) & (rows < valid.shape[0]) & (cols < valid.shape[1])
    local = np.full(valid.shape, -1, dtype=np.int64)
    local[inside] = rows[inside]*valid.shape[1]+cols[inside]
    ids = np.flatnonzero(inside.ravel())
    local.ravel()[ids[~valid.ravel()[local.ravel()[ids]]]] = -1
    local[~valid] = -1
    return local.ravel()


def write_stream_link_watersheds(out_shp: Path) -> None:
    min_area_ha = float(CFG["swmm"].get("delineation_min_area_ha") or 5)
    threshold_ha = float(CFG["swmm"].get("channel_threshold_ha") or 20)
    peripheral_threshold_ha = float(CFG["swmm"].get("peripheral_channel_threshold_ha") or max(1.0, threshold_ha * 0.25))

    dem, valid, transform, crs = read_dem_for_delineation()
    cell_size = (abs(transform.a) + abs(transform.e)) / 2
    cell_area_ha = abs(transform.a * transform.e) / 10000.0
    print(f"DEM hydrologic delineation grid: {dem.shape[1]} x {dem.shape[0]}; cell size ~= {cell_size:.2f} m")
    print(
        "Engineering watershed mode: stream-link catchments; "
        f"main channel threshold={threshold_ha:.2f} ha; peripheral threshold={peripheral_threshold_ha:.2f} ha; "
        f"merge guide area={min_area_ha:.2f} ha"
    )

    local_dem, local_valid, local_transform = dem, valid, transform
    dem, valid, transform, context_window, context_facts = read_drainage_context(dem, valid, transform, crs)
    filled, parents = fill_depressions_priority_flood(dem, valid, return_parents=True)
    receivers = build_receivers(filled, valid, parents, cell_width=abs(transform.a), cell_height=abs(transform.e))
    acc = accumulation(receivers, valid)
    channel = valid & (acc * cell_area_ha >= threshold_ha)
    if not np.any(channel):
        raise RuntimeError("No channel cells found. Reduce channel_threshold_ha or increase delineation resolution.")

    ch_donors = donor_count(receivers, valid, channel)
    link_grid = stream_link_labels(receivers, valid, channel, ch_donors, acc)
    label_grid = assign_cells_to_stream_links(receivers, valid, link_grid)
    missing = valid & (label_grid <= 0)
    if np.any(missing):
        peripheral_channel = missing & (acc * cell_area_ha >= peripheral_threshold_ha)
        if np.any(peripheral_channel):
            ch2_donors = donor_count(receivers, valid, peripheral_channel)
            link2 = stream_link_labels(receivers, valid, peripheral_channel, ch2_donors, acc)
            if np.any(link2 > 0):
                offset = int(label_grid.max())
                link2 = np.where(link2 > 0, link2 + offset, 0).astype(np.int32)
                assigned2 = assign_cells_to_stream_links(receivers, missing, link2)
                label_grid[assigned2 > 0] = assigned2[assigned2 > 0]
                link_grid[link2 > 0] = link2[link2 > 0]
                missing = valid & (label_grid <= 0)
    if np.any(missing):
        next_label = int(label_grid.max()) + 1
        outlet_grid = outlet_labels(receivers, valid)
        # One label per real terminal drainage tree. D8 diagonal connections
        # stay together as multipart polygons instead of duplicate records.
        _, inverse = np.unique(outlet_grid[missing], return_inverse=True)
        label_grid[missing] = inverse + next_label
    if context_facts["used"]:
        receivers = restrict_receivers(receivers, valid.shape, context_window, local_valid)
        label_grid = clip_labels_to_local_drainage(label_grid[context_window], receivers, local_valid)
        filled = filled[context_window]
        channel = channel[context_window] & local_valid
        link_grid = np.where(local_valid, link_grid[context_window], 0)
        dem, valid, transform = local_dem, local_valid, local_transform
        acc = accumulation(receivers, valid)  # Local rain only, independent of outside accumulation.
    target_count = int(CFG["swmm"].get("delineation_target_major_subcatchments")
                       or CFG["swmm"].get("delineation_max_subcatchments") or 500)
    label_grid, terminal_labels, merge_facts = coarsen_labels(
        label_grid, receivers, valid, minimum_cells=max(1, int(np.ceil(min_area_ha/cell_area_ha))),
        target_count=target_count)
    original_labels = label_grid.copy()
    label_grid, retained, excluded, filtering = filter_boundary_fragments(
        label_grid, receivers, valid, cell_area_ha=cell_area_ha,
        minimum_area_ha=float(CFG["swmm"].get("boundary_fragment_min_area_ha", 1)),
        maximum_fraction=float(CFG["swmm"].get("boundary_fragment_max_excluded_fraction", .01)))
    original_coarsening = merge_facts.copy()
    merge_facts["partition_count"] = int(len(np.unique(label_grid[retained])))
    merge_facts["target_reached"] = merge_facts["partition_count"] <= target_count
    valid = retained
    channel &= retained
    link_grid = np.where(retained, link_grid, 0)
    terminal_labels &= set(map(int, np.unique(label_grid[valid])))
    retained_counts = np.bincount(label_grid.ravel())
    small_terminal = [k for k in terminal_labels if retained_counts[k]*cell_area_ha < min_area_ha]
    merge_facts['independent_outlet_count'] = len(terminal_labels)
    merge_facts['retained_small_terminal_count'] = len(small_terminal)
    merge_facts['retained_small_terminal_cells'] = int(sum(retained_counts[k] for k in small_terminal))
    boundary = domain_boundary(valid)
    checks = partition_facts(label_grid, receivers, valid, boundary)
    diagnostics = {"schema_version": 2, "algorithm": "buffered_d8_local_partition_boundary_filter",
                   "flat_routing": "priority_flood_parent_tree", "boundary_cells": int(boundary.sum()),
                   "valid_area_ha": float(valid.sum()*cell_area_ha), "cell_area_ha": cell_area_ha,
                   "fill_above_1m_fraction": float(np.mean((filled-dem)[valid]>1)),
                   "maximum_fill_depth_m": float(np.max((filled-dem)[valid])),
                   "coarsening": merge_facts, "topology": checks, "original_coarsening": original_coarsening,
                   "drainage_context": context_facts, "boundary_filter": filtering,
                   "limitation": "Local clipped-domain drainage; external upstream inflow excluded."}
    diagnostic_path = out_shp.with_name("dem_derived_subcatchments_diagnostics.json")
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    if not checks["passed"]:
        raise RuntimeError("Subcatchment coverage or drainage topology checks failed.")
    hard_max = int(CFG["swmm"].get("delineation_hard_max_subcatchments") or 10000)
    diagnostics["model_budget"] = {"limit": hard_max, "passed": merge_facts["partition_count"] <= hard_max}
    diagnostic_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    if merge_facts["partition_count"] > hard_max:
        raise RuntimeError(f"{merge_facts['partition_count']} independent drainage units exceed model budget {hard_max}; select a smaller domain.")
    if not merge_facts["target_reached"]:
        print(f"[WARN] Requested {target_count} units cannot be reached without joining independent outlets; retained {merge_facts['partition_count']}.")
    outlet_by_label = outlet_points_by_label(label_grid, receivers, acc, transform)
    write_subcatchment_shp(label_grid, transform, crs, out_shp, 0.0,
                          outlet_by_label=outlet_by_label, boundary_labels=terminal_labels)
    excluded_dir = out_shp.parent / "excluded"
    excluded_shp = excluded_dir / "boundary_fragments.shp"
    write_subcatchment_shp(np.where(excluded, original_labels, 0), transform, crs, excluded_shp, 0,
                          summary_name="boundary_fragments_summary.csv")
    for name, mask in [("calculation_mask.tif", retained), ("excluded_boundary_mask.tif", excluded)]:
        path = out_shp.parent / name
        with rasterio.open(path, 'w', driver='GTiff', height=mask.shape[0], width=mask.shape[1],
                           count=1, dtype='uint8', crs=crs, transform=transform, nodata=0, compress='lzw') as dst:
            dst.write(mask.astype('uint8'), 1)
        filtering[name.replace('.tif', '_sha256')] = sha256_file(path)
    filtering['excluded_shape_sha256'] = sha256_file(excluded_shp)
    filtering['excluded_summary_sha256'] = sha256_file(excluded_dir / 'boundary_fragments_summary.csv')
    print(f"Boundary filtering: {filtering['excluded_count']} units, {filtering['excluded_area_ha']:.4f} ha ({filtering['excluded_fraction']:.3%}) excluded")
    diagnostics["partition_sha256"] = sha256_file(out_shp)
    diagnostics["summary_sha256"] = sha256_file(out_shp.with_name("dem_derived_subcatchments_summary.csv"))
    diagnostic_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_channel_node_layers(link_grid, channel, acc, transform, crs, out_shp, cell_area_ha)
    print(f"Written engineering stream-link watersheds: {out_shp}")
    print(f"Stream links: {int(link_grid.max())}")
    print(f"Channel cells: {int(np.count_nonzero(channel))}")
    print(f"Total labelled area: {float(np.count_nonzero(label_grid > 0) * cell_area_ha):.3f} ha")


def label_whole_area_blocks(valid: np.ndarray, target_area_ha: float, cell_area_ha: float) -> np.ndarray:
    block_cells = max(3, int(round(np.sqrt(max(target_area_ha, cell_area_ha) / cell_area_ha))))
    label_grid = np.zeros(valid.shape, dtype=np.int32)
    next_label = 1
    structure = np.ones((3, 3), dtype=np.int8)
    rows, cols = valid.shape
    for r0 in range(0, rows, block_cells):
        r1 = min(rows, r0 + block_cells)
        for c0 in range(0, cols, block_cells):
            c1 = min(cols, c0 + block_cells)
            block = valid[r0:r1, c0:c1]
            if not np.any(block):
                continue
            components, count = ndimage.label(block, structure=structure)
            for comp_id in range(1, count + 1):
                mask = components == comp_id
                if not np.any(mask):
                    continue
                label_grid[r0:r1, c0:c1][mask] = next_label
                next_label += 1
    return label_grid


def merge_small_whole_area_units(
    label_grid: np.ndarray,
    valid: np.ndarray,
    target_area_ha: float,
    min_area_ha: float,
    cell_area_ha: float,
) -> np.ndarray:
    out = label_grid.copy()
    min_cells = max(1, int(round(min_area_ha / cell_area_ha)))
    target_cells = max(min_cells, int(round(target_area_ha / cell_area_ha)))
    structure = np.ones((3, 3), dtype=bool)

    changed = True
    while changed:
        changed = False
        labels, counts = np.unique(out[out > 0], return_counts=True)
        sizes = {int(label): int(count) for label, count in zip(labels, counts)}
        small_labels = [label for label, count in sizes.items() if count < min_cells]
        if not small_labels:
            break
        for label in small_labels:
            if label not in sizes:
                continue
            mask = out == label
            dilated = ndimage.binary_dilation(mask, structure=structure) & valid & (out != label) & (out > 0)
            neighbors, ncounts = np.unique(out[dilated], return_counts=True)
            candidates = []
            for nb, shared in zip(neighbors, ncounts):
                nb = int(nb)
                if nb <= 0:
                    continue
                combined = sizes.get(label, 0) + sizes.get(nb, 0)
                penalty = abs(combined - target_cells)
                candidates.append((penalty, -int(shared), nb))
            if not candidates:
                continue
            _, _, chosen = min(candidates)
            out[mask] = chosen
            changed = True
            break
    return out


def drop_and_renumber_small_units(label_grid: np.ndarray, min_area_ha: float, cell_area_ha: float) -> tuple[np.ndarray, int]:
    out = label_grid.copy()
    min_cells = max(1, int(round(min_area_ha / cell_area_ha)))
    labels, counts = np.unique(out[out > 0], return_counts=True)
    dropped = 0
    for label, count in zip(labels, counts):
        if int(count) < min_cells:
            out[out == int(label)] = 0
            dropped += 1

    new_grid = np.zeros(out.shape, dtype=np.int32)
    next_label = 1
    for label in np.unique(out[out > 0]):
        new_grid[out == int(label)] = next_label
        next_label += 1
    return new_grid, dropped


def outlet_points_by_label(label_grid: np.ndarray, receivers: np.ndarray, acc: np.ndarray, transform: Affine) -> dict[int, tuple[float, float]]:
    flat_labels = label_grid.ravel()
    crossing = (receivers < 0) | (flat_labels[np.clip(receivers, 0, flat_labels.size-1)] != flat_labels)
    ids = np.flatnonzero((flat_labels > 0) & crossing)
    order = np.lexsort((acc.ravel()[ids], flat_labels[ids]))
    ids = ids[order]
    last = np.r_[flat_labels[ids[:-1]] != flat_labels[ids[1:]], True] if len(ids) else []
    outlet_by_label = {}
    for outlet in ids[last]:
        label = int(flat_labels[outlet])
        r, c = divmod(int(outlet), label_grid.shape[1])
        x, y = transform * (c + 0.5, r + 0.5)
        outlet_by_label[label] = (float(x), float(y))
    return outlet_by_label


def write_whole_area_hydrological_units(out_shp: Path) -> None:
    target_area_ha = float(CFG["swmm"].get("whole_area_target_unit_ha") or 50)
    min_area_ha = float(CFG["swmm"].get("whole_area_min_unit_ha") or 5)
    threshold_ha = float(CFG["swmm"].get("channel_threshold_ha") or 20)

    dem, valid, transform, crs = read_dem_for_delineation()
    cell_size = (abs(transform.a) + abs(transform.e)) / 2
    cell_area_ha = abs(transform.a * transform.e) / 10000.0
    print(f"DEM hydrologic partition grid: {dem.shape[1]} x {dem.shape[0]}; cell size ~= {cell_size:.2f} m")
    print(
        "Whole-area hydrological response units: "
        f"target area={target_area_ha:.2f} ha; min area={min_area_ha:.2f} ha; channel threshold={threshold_ha:.2f} ha"
    )

    filled, parents = fill_depressions_priority_flood(dem, valid, return_parents=True)
    receivers = build_receivers(filled, valid, parents, cell_width=abs(transform.a), cell_height=abs(transform.e))
    acc = accumulation(receivers, valid)
    channel = valid & (acc * cell_area_ha >= threshold_ha)

    label_grid = label_whole_area_blocks(valid, target_area_ha, cell_area_ha)
    label_grid = merge_small_whole_area_units(label_grid, valid, target_area_ha, min_area_ha, cell_area_ha)
    label_grid, dropped_small = drop_and_renumber_small_units(label_grid, min_area_ha, cell_area_ha)
    outlet_by_label = outlet_points_by_label(label_grid, receivers, acc, transform)

    seed_labels = np.zeros(valid.shape, dtype=np.int32)
    for label, (x, y) in outlet_by_label.items():
        col, row = ~transform * (x, y)
        r, c = int(row), int(col)
        if 0 <= r < seed_labels.shape[0] and 0 <= c < seed_labels.shape[1]:
            seed_labels[r, c] = int(label)

    write_subcatchment_shp(label_grid, transform, crs, out_shp, 0.001, outlet_by_label=outlet_by_label)
    write_channel_node_layers(seed_labels, channel, acc, transform, crs, out_shp, cell_area_ha)
    labels, counts = np.unique(label_grid[label_grid > 0], return_counts=True)
    areas = counts * cell_area_ha
    print(f"Written whole-area hydrological response units: {out_shp}")
    print(f"Units: {len(labels)}")
    print(f"Unit area ha: min={areas.min():.3f}; median={np.median(areas):.3f}; p95={np.percentile(areas, 95):.3f}; max={areas.max():.3f}")
    if dropped_small:
        print(f"Dropped isolated fragments smaller than {min_area_ha:.2f} ha: {dropped_small}")
    print(f"Channel cells: {int(np.count_nonzero(channel))}")
    print(f"Total labelled area: {float(np.count_nonzero(label_grid > 0) * cell_area_ha):.3f} ha")


def main() -> None:
    out_shp = Path(CFG["swmm"].get("dem_derived_boundary_shp") or ROOT / "inputs/dem_derived/dem_derived_subcatchments.shp")
    if not bool(CFG["swmm"].get("auto_delineate_subcatchments", True)):
        if out_shp.exists():
            print(f"[OK] Using existing DEM-derived catchments: {out_shp}")
            print("[INFO] Set swmm.auto_delineate_subcatchments=true after replacing the DEM.")
            return
        raise RuntimeError(
            "Existing DEM-derived catchments were not found. Set "
            "swmm.auto_delineate_subcatchments=true and install GRASS GIS."
        )
    method = str(CFG["swmm"].get("delineation_method") or "full_dem_grid").lower()
    if method == "full_dem_grid":
        write_full_dem_grid_subcatchments(out_shp)
        return
    if method == "grass_watershed":
        write_grass_watersheds(out_shp)
        return
    if method == "channel_subbasin":
        write_channel_subbasins(out_shp)
        return
    if method == "stream_link_watershed":
        write_stream_link_watersheds(out_shp)
        return
    if method == "whole_area_hru":
        write_whole_area_hydrological_units(out_shp)
        return

    max_count = int(CFG["swmm"].get("delineation_max_subcatchments") or 20)
    min_area_ha = float(CFG["swmm"].get("delineation_min_area_ha") or 5)

    dem, valid, transform, crs = read_dem_for_delineation()
    print(f"DEM delineation grid: {dem.shape[1]} x {dem.shape[0]}; cell size ~= {abs(transform.a):.2f} m")
    filled, parents = fill_depressions_priority_flood(dem, valid, return_parents=True)
    receivers = build_receivers(filled, valid, parents, cell_width=abs(transform.a), cell_height=abs(transform.e))
    acc = accumulation(receivers, valid)
    outlets = outlet_labels(receivers, valid)

    cell_area_ha = abs(transform.a * transform.e) / 10000.0
    outlet_ids, counts = np.unique(outlets[valid], return_counts=True)
    rows = [(int(outlet), int(count), int(outlet)) for outlet, count in zip(outlet_ids, counts) if count * cell_area_ha >= min_area_ha]
    rows.sort(key=lambda x: x[1], reverse=True)
    selected = rows[:max_count]
    if not selected:
        raise RuntimeError("No DEM-derived subcatchments met the minimum area threshold.")

    label_grid = np.zeros(valid.shape, dtype=np.int32)
    for new_label, (outlet, _, _) in enumerate(selected, start=1):
        label_grid[outlets == outlet] = new_label

    write_subcatchment_shp(label_grid, transform, crs, out_shp, min_area_ha)
    print(f"Written DEM-derived SWMM partitions: {out_shp}")
    print(f"Subcatchments: {len(selected)}")
    print(f"Minimum area threshold: {min_area_ha:.2f} ha")
    print(f"Total selected area: {float(np.count_nonzero(label_grid) * cell_area_ha):.3f} ha")


if __name__ == "__main__":
    main()
