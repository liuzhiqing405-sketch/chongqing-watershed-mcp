from __future__ import annotations

from pathlib import Path
import sys

from .common import RUN_ROOT, event_slug, event_times, load_config, resolve_path, save_config


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    cfg = load_config()
    slug = event_slug(cfg)
    root = resolve_path(cfg.get("project", {}).get("root")) or RUN_ROOT
    cfg.setdefault("project", {})["root"] = str(root).replace("\\", "/")
    event_start, event_end, sim_end = event_times(cfg)
    rain_dir = root / "inputs" / "rainfall"
    cfg.setdefault("rainfall", {})["source_mode"] = "station"
    cfg["rainfall"]["station_aggregation"] = "idw_subcatchment_area_mean"
    cfg["rainfall"]["csv"] = str(rain_dir / f"{slug}_active_hourly.csv").replace("\\", "/")
    cfg["rainfall"]["station_csv"] = str(rain_dir / f"{slug}_station_hourly.csv").replace("\\", "/")
    cfg["rainfall"]["station_observations_csv"] = str(
        rain_dir / f"{slug}_station_observations_long.csv"
    ).replace("\\", "/")
    cfg["rainfall"]["station_summary_csv"] = str(
        rain_dir / f"{slug}_station_rainfall_summary.csv"
    ).replace("\\", "/")
    cfg["rainfall"]["station_domain_weights_csv"] = str(
        rain_dir / f"{slug}_station_domain_idw_weights.csv"
    ).replace("\\", "/")
    for obsolete in ["open_meteo_csv", "open_source_csv", "comparison_primary_source", "open_meteo"]:
        cfg["rainfall"].pop(obsolete, None)

    cfg["swmm"]["working_inp"] = str(root / "swmm" / "model" / f"{slug}.inp").replace("\\", "/")
    cfg["swmm"]["dem_derived_boundary_shp"] = str(
        root / "inputs" / "dem_derived" / f"{slug}_dem_derived_subcatchments.shp"
    ).replace("\\", "/")
    cfg["swmm"]["report"] = str(root / "swmm" / "results" / f"{slug}.rpt").replace("\\", "/")
    cfg["swmm"]["output"] = str(root / "swmm" / "results" / f"{slug}.out").replace("\\", "/")
    cfg["swmm"]["start_date"] = event_start.strftime("%m/%d/%Y")
    cfg["swmm"]["start_time"] = event_start.strftime("%H:%M:%S")
    cfg["swmm"]["report_start_date"] = event_start.strftime("%m/%d/%Y")
    cfg["swmm"]["report_start_time"] = event_start.strftime("%H:%M:%S")
    cfg["swmm"]["end_date"] = sim_end.strftime("%m/%d/%Y")
    cfg["swmm"]["end_time"] = sim_end.strftime("%H:%M:%S")

    save_config(cfg)
    print(f"[INFO] Event slug: {slug}")
    print(f"[INFO] Event window: {event_start} to {event_end}; simulation end: {sim_end}")
    print(f"[INFO] Active rainfall CSV: {cfg['rainfall']['csv']}")
    print(f"[INFO] DEM-derived SWMM partitions: {cfg['swmm']['dem_derived_boundary_shp']}")
    print(f"[INFO] SWMM input: {cfg['swmm']['working_inp']}")


if __name__ == "__main__":
    main()
