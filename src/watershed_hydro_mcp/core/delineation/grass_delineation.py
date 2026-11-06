"""Yongchuan's r.watershed -s, with auditable raster and model domains."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import rasterio
from scipy import ndimage

from watershed_exchange.paths import sha256_file
from watershed_hydro_mcp.grass_runtime import run_grass
from .partition_topology import filter_boundary_fragments, partition_facts, single_exit_facts

ALGORITHM = "grass_r_watershed_sfd_local_domain"
# GRASS angles run counterclockwise from east in multiples of 45 degrees.
DIRECTIONS = {1: (-1, 1), 2: (-1, 0), 3: (-1, -1), 4: (0, -1),
              5: (1, -1), 6: (1, 0), 7: (1, 1), 8: (0, 1)}


def drainage_receivers(drainage, valid):
    if drainage.shape != valid.shape or np.any(valid & ~np.isin(drainage, range(-8, 9))):
        raise RuntimeError("Invalid GRASS drainage raster")
    h, w = valid.shape
    result = np.full(valid.shape, -1, dtype=np.int64)
    for code, (dr, dc) in DIRECTIONS.items():
        rows, cols = np.where(valid & (drainage == code))
        rr, cc = rows + dr, cols + dc
        inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rows, cols, rr, cc = rows[inside], cols[inside], rr[inside], cc[inside]
        inside = valid[rr, cc]
        result[rows[inside], cols[inside]] = rr[inside]*w + cc[inside]
    # Negative codes explicitly leave the GRASS region; never wrap them back.
    return result.ravel()


def write_grid(path, values, transform, crs, nodata=0):
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
                       count=1, dtype=values.dtype, transform=transform, crs=crs,
                       nodata=nodata, compress="lzw") as dst:
        dst.write(values, 1)


def write_grass_watersheds(out_shp: Path):
    from . import delineate as d
    cli = d.find_grass_cli()
    if cli is None:
        raise RuntimeError("未找到 GRASS GIS；请配置 executables.grass_cli，当前任务不会回退到其他划分算法")
    dem, original, transform, crs = d.read_dem_for_delineation()
    if not original.any() or not crs.is_projected or transform.b or transform.d or transform.a <= 0 or transform.e >= 0:
        raise RuntimeError("GRASS划分需要非空的北向上投影DEM")
    cell_area = abs(transform.a*transform.e)/10000
    swmm = d.CFG["swmm"]
    threshold_ha = float(swmm.get("channel_threshold_ha", 20))
    if not np.isfinite(threshold_ha) or threshold_ha <= 0:
        raise RuntimeError("GRASS河网阈值必须是正面积")
    threshold = max(1, round(threshold_ha/cell_area))
    raw = out_shp.parent / "grass"
    raw.mkdir(parents=True, exist_ok=True)
    dem_path = raw / "dem_valid.tif"
    write_grid(dem_path, np.where(original, dem, np.nan).astype("float64"), transform, crs, np.nan)
    print(f"GRASS r.watershed -s: threshold={threshold_ha:g} ha ({threshold} cells); local DEM={dem.shape}", flush=True)
    run_grass(cli, dem_path, Path(__file__).with_name("grass_worker.py"), raw, threshold)
    grids = {}
    for name in ("basin", "drainage", "accumulation", "stream"):
        with rasterio.open(raw/(name+".tif")) as src:
            if src.shape != dem.shape or src.crs != crs or not src.transform.almost_equals(transform):
                raise RuntimeError(f"GRASS output grid is misaligned: {name}")
            grids[name] = src.read(1)
    labels = grids["basin"].astype("int32")
    if np.any(labels < 0) or np.any((labels > 0) & ~original):
        raise RuntimeError("GRASS basin labels extend outside the calculation domain")
    assigned = original & (labels > 0)
    unassigned = original & ~assigned
    maximum_loss = float(swmm.get("grass_maximum_unassigned_fraction", .2))
    if not 0 <= maximum_loss < 1 or not assigned.any() or unassigned.sum()/original.sum() > maximum_loss:
        raise RuntimeError("GRASS未分区范围过大或未形成流域，请增大选区或减小河网阈值；原始输出保留供复核")
    # Route over exactly the same local calculation DEM as the old project.
    receivers = drainage_receivers(grids["drainage"], assigned)
    d.accumulation(receivers, assigned)  # Reject cycles, including flat routing cycles.
    labels, retained, tiny, small_facts = filter_boundary_fragments(
        labels, receivers, assigned, cell_area_ha=cell_area,
        minimum_area_ha=float(swmm.get("boundary_fragment_min_area_ha", 1)),
        maximum_fraction=float(swmm.get("boundary_fragment_max_excluded_fraction", .01)))
    excluded = unassigned | tiny
    receivers = drainage_receivers(grids["drainage"], retained)
    acc = d.accumulation(receivers, retained)
    topology = partition_facts(labels, receivers, retained, d.domain_boundary(retained))
    single_exit = single_exit_facts(labels, receivers, retained)
    if not single_exit['passed']:
        raise RuntimeError(
            f"GRASS分区未汇向唯一出口：{single_exit['invalid_partition_count']} 个分区缺少出口或存在多个出流位置；"
            "请检查选区边界与DEM缺值，原始栅格已保留供复核")
    outlets = d.outlet_points_by_label(labels, receivers, acc, transform)
    count = len(np.unique(labels[labels > 0]))
    hard_max = int(swmm.get("delineation_hard_max_subcatchments", 10000))
    if not topology["passed"] or len(outlets) != count or count > hard_max:
        raise RuntimeError("GRASS分区排水拓扑或分区数量检查未通过；原始输出保留供复核")
    terminal_labels = set(map(int, np.unique(labels.ravel()[retained.ravel() & (receivers < 0)])))
    d.write_subcatchment_shp(labels, transform, crs, out_shp, 0, outlets, terminal_labels,
                            source="grass_r_watershed")
    # Exterior areas below r.watershed's threshold are NULL, not invented basins.
    # Export them together with the optional tiny-boundary policy for conservation.
    excluded_labels, excluded_count = ndimage.label(excluded, structure=np.ones((3, 3)))
    excluded_shp = out_shp.parent / "excluded/boundary_fragments.shp"
    d.write_subcatchment_shp(excluded_labels, transform, crs, excluded_shp, 0,
                            summary_name="boundary_fragments_summary.csv",
                            source="grass_unassigned_or_boundary")
    audit = {"excluded_count": int(excluded_count), "retained_cells": int(retained.sum()),
             "excluded_cells": int(excluded.sum()), "original_cells": int(original.sum()),
             "retained_area_ha": float(retained.sum()*cell_area),
             "excluded_area_ha": float(excluded.sum()*cell_area),
             "original_area_ha": float(original.sum()*cell_area),
             "excluded_fraction": float(excluded.sum()/original.sum()),
             "maximum_excluded_fraction": maximum_loss + (1-maximum_loss)*float(swmm.get("boundary_fragment_max_excluded_fraction", .01)),
             "reason": "GRASS NULL exterior areas and separately audited independent boundary fragments"}
    for name, mask in (("calculation_mask", retained), ("excluded_boundary_mask", excluded)):
        path = out_shp.parent/(name+".tif")
        write_grid(path, mask.astype("uint8"), transform, crs)
        audit[name+"_sha256"] = sha256_file(path)
    audit["excluded_shape_sha256"] = sha256_file(excluded_shp)
    audit["excluded_summary_sha256"] = sha256_file(excluded_shp.with_name("boundary_fragments_summary.csv"))
    diagnostics = {"schema_version": 3, "algorithm": ALGORITHM,
        "grass": {"flags": "s", "threshold_ha": threshold_ha, "threshold_cells": threshold,
                  "cli": str(cli), "version": (raw/"version.txt").read_text(encoding="utf-8"),
                  "rasters": {name+".tif": sha256_file(raw/(name+".tif")) for name in grids},
                  "dem_sha256": sha256_file(dem_path), "unassigned_cells": int(unassigned.sum()),
                  "source_dem_sha256": sha256_file(Path(d.CFG['terrain']['dem'])),
                  "unassigned_fraction": float(unassigned.sum()/original.sum()),
                  "maximum_unassigned_fraction": maximum_loss},
        "partitioning": {"partition_count": count, "coarsened": False,
                         "organization": "grass_basin_label", "single_exit": single_exit},
        "depression_handling": "GRASS least-cost routing on original elevations; no DEM sink filling",
        "drainage_context": {"used": False, "reason": "Yongchuan local DEM method"},
        "small_boundary_filter": small_facts, "boundary_filter": audit,
        "topology": topology, "model_budget": {"limit": hard_max, "passed": True},
        "valid_area_ha": float(retained.sum()*cell_area), "cell_area_ha": cell_area,
        "partition_sha256": sha256_file(out_shp),
        "summary_sha256": sha256_file(out_shp.with_name("dem_derived_subcatchments_summary.csv")),
        "limitation": "Local domain; external upstream runoff is excluded. GRASS NULL areas are explicitly excluded."}
    out_shp.with_name("dem_derived_subcatchments_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"GRASS partitioned {count} units, {retained.sum()*cell_area/100:.6f} km2; excluded {excluded.sum()*cell_area/100:.6f} km2", flush=True)
