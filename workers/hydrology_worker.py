from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage


D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
D8_CODE = {direction: index + 1 for index, direction in enumerate(D8)}


def fill_depressions_priority_flood(dem: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = dem.shape
    filled = dem.copy()
    visited = np.zeros((height, width), dtype=bool)
    parents = np.full((height, width), -1, dtype=np.int64)
    heap: list[tuple[float, int, int]] = []
    boundary = valid & ndimage.binary_dilation(~valid, structure=np.ones((3, 3), dtype=bool))
    boundary[0, :] |= valid[0, :]
    boundary[-1, :] |= valid[-1, :]
    boundary[:, 0] |= valid[:, 0]
    boundary[:, -1] |= valid[:, -1]
    for row, col in zip(*np.where(boundary)):
        visited[row, col] = True
        heapq.heappush(heap, (float(filled[row, col]), int(row), int(col)))
    while heap:
        elevation, row, col = heapq.heappop(heap)
        for dr, dc in D8:
            rr, cc = row + dr, col + dc
            if rr < 0 or rr >= height or cc < 0 or cc >= width or visited[rr, cc] or not valid[rr, cc]:
                continue
            visited[rr, cc] = True
            parents[rr, cc] = row * width + col
            next_elevation = float(filled[rr, cc])
            if next_elevation < elevation:
                filled[rr, cc] = elevation
                next_elevation = elevation
            heapq.heappush(heap, (next_elevation, rr, cc))
    return filled, parents


def build_receivers(dem: np.ndarray, valid: np.ndarray, flat_parents: np.ndarray,
                    *, cell_width: float = 1.0, cell_height: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    height, width = dem.shape
    z = np.where(valid, dem, np.nan)
    rows = np.arange(height, dtype=np.int64)[:, None]
    cols = np.arange(width, dtype=np.int64)[None, :]
    best_drop = np.zeros((height, width), dtype=np.float32)
    receivers = np.full((height, width), -1, dtype=np.int64)
    directions = np.zeros((height, width), dtype=np.int16)
    for dr, dc in D8:
        neighbor = np.full((height, width), np.nan, dtype=np.float32)
        r_src0, r_src1 = max(0, dr), height + min(0, dr)
        c_src0, c_src1 = max(0, dc), width + min(0, dc)
        r_dst0, r_dst1 = max(0, -dr), height - max(0, dr)
        c_dst0, c_dst1 = max(0, -dc), width - max(0, dc)
        neighbor[r_dst0:r_dst1, c_dst0:c_dst1] = z[r_src0:r_src1, c_src0:c_src1]
        distance = float(np.hypot(dc * cell_width, dr * cell_height))
        drop = (z - neighbor) / distance
        candidate_idx = (rows + dr) * width + (cols + dc)
        accept = valid & np.isfinite(drop) & (drop > best_drop)
        receivers[accept] = candidate_idx[accept]
        directions[accept] = D8_CODE[(dr, dc)]
        best_drop[accept] = drop[accept]
    flat = valid & (receivers < 0) & (flat_parents >= 0)
    receivers[flat] = flat_parents[flat]
    flat_rows, flat_cols = np.where(flat)
    parent_rows, parent_cols = np.divmod(flat_parents[flat], width)
    for dr, dc in D8:
        matched = (parent_rows - flat_rows == dr) & (parent_cols - flat_cols == dc)
        directions[flat_rows[matched], flat_cols[matched]] = D8_CODE[(dr, dc)]
    return receivers.ravel(), directions


def accumulation_and_order(receivers: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[int]]:
    flat_valid = valid.ravel()
    indegree = np.zeros(receivers.size, dtype=np.int32)
    active = np.flatnonzero(flat_valid & (receivers >= 0))
    np.add.at(indegree, receivers[active], 1)
    accumulation = flat_valid.astype(np.float64)
    queue = list(np.flatnonzero(flat_valid & (indegree == 0)))
    order: list[int] = []
    head = 0
    while head < len(queue):
        index = int(queue[head])
        head += 1
        order.append(index)
        receiver = int(receivers[index])
        if receiver >= 0:
            accumulation[receiver] += accumulation[index]
            indegree[receiver] -= 1
            if indegree[receiver] == 0:
                queue.append(receiver)
    return accumulation.reshape(valid.shape), order


def write_raster(path: Path, array: np.ndarray, profile: dict, dtype: str, nodata) -> None:
    output_profile = profile.copy()
    output_profile.update(dtype=dtype, count=1, nodata=nodata, compress="lzw")
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(array.astype(dtype), 1)


def select_outlet(accumulation: np.ndarray, transform, x: float, y: float,
                  radius: int, *, river_mask: np.ndarray | None = None,
                  maximum_distance_m: float = 500.0,
                  minimum_area_km2: float = 1.8) -> tuple[int, int, dict]:
    """A named-river snap may grow, but may never move onto an unrelated river.

    If the supplied river corridor has no hydrologically usable cell, preserve
    the local diagnostic basin and explicitly fail alignment QA. Never silently
    substitute the largest nearby catchment for the requested river.
    """
    if radius < 1:
        raise ValueError("snap radius must be positive")
    col, row = map(int, np.floor((~transform) * (x, y)))
    height, width = accumulation.shape
    if row-radius < 0 or col-radius < 0 or row+radius >= height or col+radius >= width:
        raise ValueError("Requested outlet is too close to the processing-window edge")
    local = accumulation[row-radius:row+radius+1, col-radius:col+radius+1]
    offset = np.unravel_index(np.argmax(local), local.shape)
    fallback = row-radius+int(offset[0]), col-radius+int(offset[1])
    if river_mask is None:
        return *fallback, {"checked": False, "passed": None, "method": "local_accumulation"}
    if river_mask.shape != accumulation.shape:
        raise ValueError("River snap mask and accumulation grid must match")
    if not np.isfinite(maximum_distance_m) or maximum_distance_m <= 0:
        raise ValueError("Named river snap distance must be positive and finite")
    cell_area = abs(transform.a*transform.e-transform.b*transform.d)
    if not np.isfinite(minimum_area_km2) or minimum_area_km2 < 0:
        raise ValueError("Minimum river drainage area must be finite and nonnegative")
    # Distance is measured in the projected CRS, not in a fixed number of cells.
    search = int(np.ceil(maximum_distance_m / min(abs(transform.a), abs(transform.e)))) + 1
    r0, r1 = max(0,row-search), min(height,row+search+1)
    c0, c1 = max(0,col-search), min(width,col+search+1)
    rr, cc = np.indices((r1-r0,c1-c0))
    xx, yy = transform * (cc+c0+.5, rr+r0+.5)
    distance = np.hypot(xx-x, yy-y)
    values = accumulation[r0:r1,c0:c1]
    allowed = river_mask[r0:r1,c0:c1].astype(bool) & (distance <= maximum_distance_m)
    allowed &= np.isfinite(values) & (values > 0)
    best = float(np.max(values[allowed])) if allowed.any() else 0.0
    diagnostic = {"checked": True, "passed": False, "method": "named_river_corridor",
                  "maximum_snap_distance_m": maximum_distance_m,
                  "river_channel_area_km2": round(best*cell_area/1e6,4)}
    if not allowed.any() or best*cell_area/1e6 < minimum_area_km2:
        diagnostic["reason"] = "river_flow_mismatch"
        return *fallback, diagnostic
    # Among equally accumulated cells, keep the one nearest the requested point.
    index = np.unravel_index(np.argmin(np.where(allowed & (values == best),distance,np.inf)),values.shape)
    diagnostic["passed"] = True
    return r0+int(index[0]), c0+int(index[1]), diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--snap-radius", type=int, default=5)
    parser.add_argument("--river-mask", type=Path)
    parser.add_argument("--maximum-snap-distance-m", type=float, default=500.0)
    parser.add_argument("--minimum-area-km2", type=float, default=1.8)
    args = parser.parse_args()
    with rasterio.open(args.dem) as source:
        masked = source.read(1, masked=True)
        valid = ~np.ma.getmaskarray(masked)
        dem = masked.astype("float32").filled(np.nan)
        profile = source.profile.copy()
        transform = source.transform
    if not valid.any():
        raise ValueError("DEM processing window has no valid cells")
    filled, flat_parents = fill_depressions_priority_flood(dem, valid)
    receivers, directions = build_receivers(filled, valid, flat_parents)
    accumulation, order = accumulation_and_order(receivers, valid)
    river_mask = None
    if args.river_mask:
        with rasterio.open(args.river_mask) as mask_source:
            if mask_source.transform != transform or mask_source.crs != profile["crs"]:
                raise ValueError("River mask must use the DEM grid and CRS")
            river_mask = mask_source.read(1)
    snap_row, snap_col, channel_check = select_outlet(
        accumulation, transform, args.x, args.y, args.snap_radius,
        river_mask=river_mask, maximum_distance_m=args.maximum_snap_distance_m,
        minimum_area_km2=args.minimum_area_km2)
    outlet_index = snap_row * accumulation.shape[1] + snap_col
    if accumulation[snap_row, snap_col] <= 0:
        raise ValueError("No valid drainage cell was found near the outlet")
    basin_flat = np.zeros(receivers.size, dtype=bool)
    basin_flat[outlet_index] = True
    for index in reversed(order):
        receiver = int(receivers[index])
        if receiver >= 0 and basin_flat[receiver]:
            basin_flat[index] = True
    basin = basin_flat.reshape(valid.shape)
    write_raster(args.output / "dem_filled.tif", np.where(valid, filled, -9999), profile, "float32", -9999)
    write_raster(args.output / "fill_depth.tif", np.where(valid, filled - dem, -9999), profile, "float32", -9999)
    write_raster(args.output / "flow_direction.tif", np.where(valid, directions, 0), profile, "int16", 0)
    write_raster(args.output / "flow_accumulation.tif", np.where(valid, accumulation, -9999), profile, "float64", -9999)
    write_raster(args.output / "basin.tif", basin.astype("uint8"), profile, "uint8", 0)
    snapped_x, snapped_y = transform * (snap_col + 0.5, snap_row + 0.5)
    (args.output / "outlet_projected.json").write_text(json.dumps({
        "requested_x": args.x, "requested_y": args.y,
        "snapped_x": snapped_x, "snapped_y": snapped_y,
        "accumulation_cells": float(accumulation[snap_row, snap_col]),
        "shift_m": float(np.hypot(snapped_x - args.x, snapped_y - args.y)),
        "channel_check": channel_check,
    }), encoding="utf-8")
    print(json.dumps({"valid_cells": int(valid.sum()), "basin_cells": int(basin.sum())}))


if __name__ == "__main__":
    main()
