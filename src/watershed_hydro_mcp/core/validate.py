from __future__ import annotations

from pathlib import Path
import sys

import yaml


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from watershed_hydro_mcp.core.common import LazyConfig, RUN_ROOT as ROOT

CFG = LazyConfig()


def resolve(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def must_exist(label: str, value: str | None) -> bool:
    if not value:
        print(f"[WARN] {label}: not set")
        return False
    path = resolve(value)
    assert path is not None
    if path.exists():
        print(f"[OK]   {label}: {path}")
        return True
    print(f"[MISS] {label}: {path}")
    return False


def main() -> None:
    swmm_value = CFG.get("executables", {}).get("swmm_cli") or "engine/runswmm.exe"
    checks = [
        ("Runoff engine", swmm_value),
        ("DEM", CFG["terrain"].get("dem")),
        ("Station rainfall zip", CFG["rainfall"].get("station_zip")),
    ]
    if not CFG["swmm"].get("auto_generate_from_gis", True):
        checks.append(("SWMM template INP", CFG["swmm"].get("template_inp")))
    else:
        print("[OK]   SWMM input: will be generated automatically from DEM-derived partitions, DEM, land use, and rainfall.")
    ok = [must_exist(label, value) for label, value in checks]
    print("[OK]   Rainfall source: measured multi-station mode; hourly IDW and subcatchment area averaging.")
    if CFG["study_area"].get("domain_source", "dem_extent") == "dem_extent":
        print("[OK]   Study-area domain: defined by DEM extent.")
    if CFG["study_area"].get("calculation_domain", "dem_landuse_intersection") == "dem_landuse_intersection":
        print("[OK]   Calculation domain: DEM pixels covered by land-use layer only.")
    if CFG["swmm"].get("auto_delineate_subcatchments", True):
        method = CFG["swmm"].get("delineation_method", "full_dem_grid")
        if method == "full_dem_grid":
            print(
                "[OK]   SWMM partitions: will cover the full DEM extent "
                f"using {CFG['swmm'].get('full_coverage_grid_size_m', 2000)} m DEM-derived grid cells."
            )
        elif method == "channel_subbasin":
            print(
                "[OK]   SWMM partitions: will be delineated from DEM flow direction, "
                f"flow accumulation, channel threshold={CFG['swmm'].get('channel_threshold_ha', 20)} ha, "
                f"and node spacing={CFG['swmm'].get('channel_node_spacing_m', 500)} m."
            )
        elif method == "stream_link_watershed":
            print(
                "[OK]   SWMM partitions: engineering stream-link watersheds will be delineated from DEM flow direction, "
                f"flow accumulation, and channel threshold={CFG['swmm'].get('channel_threshold_ha', 20)} ha."
            )
        elif method == "grass_watershed":
            from watershed_hydro_mcp.grass_runtime import resolve_grass_cli
            grass_ok = resolve_grass_cli(CFG.get("executables", {}).get("grass_cli"), ROOT) is not None
            if not grass_ok:
                print("[MISS] GRASS GIS: not configured, but catchment redelineation selects "
                      "delineation_method=grass_watershed. Set executables.grass_cli or switch "
                      "to the native stream_link_watershed method.")
                ok.append(False)
            else:
                print(
                    "[OK]   SWMM partitions: GRASS r.watershed -s will delineate local DEM-derived catchments, "
                    f"threshold={CFG['swmm'].get('channel_threshold_ha', 20)} ha."
                )
        elif method == "whole_area_hru":
            print(
                "[OK]   SWMM partitions: whole-area hydrological response units will cover all DEM pixels with land-use coverage, "
                f"target unit area={CFG['swmm'].get('whole_area_target_unit_ha', 50)} ha, "
                f"min unit area={CFG['swmm'].get('whole_area_min_unit_ha', 5)} ha."
            )
        else:
            print(f"[OK]   SWMM partitions: will be delineated automatically from DEM using method={method}.")
    else:
        boundary = CFG["swmm"].get("dem_derived_boundary_shp")
        ok.append(must_exist("Existing DEM-derived catchments", boundary))
    for label, key in [("Land-use layer", "landuse_shp"), ("River layer", "river_shp")]:
        value = CFG["study_area"].get(key)
        if value:
            must_exist(label, value)

    if not all(ok):
        raise SystemExit("Required input check failed. Fill config.yaml and rerun.")


if __name__ == "__main__":
    main()
