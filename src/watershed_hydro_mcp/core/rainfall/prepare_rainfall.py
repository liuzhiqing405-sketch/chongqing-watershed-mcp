from __future__ import annotations

import csv
import re
import shutil
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

from watershed_hydro_mcp.core import runtime_bootstrap  # noqa: F401

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.transform import Affine
import shapefile

from ..common import RUN_ROOT as ROOT, event_slug, event_times, load_config, prj_crs, save_config
from ..spatial_domain import landuse_mask_for_grid
from ..rainfall.station_rainfall import (
    idw_mean_weights_by_zone,
    read_station_observations,
    station_coordinates,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def event_hour_range(start: datetime, sim_end: datetime) -> list[datetime]:
    values = []
    current = start.replace(minute=0, second=0, microsecond=0)
    while current <= sim_end:
        values.append(current)
        current += timedelta(hours=1)
    return values


def extract_station_zip(zip_path: Path, out_dir: Path) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(out_dir)
    shapefiles = list(out_dir.rglob("*.shp"))
    if not shapefiles:
        raise RuntimeError(f"No shapefile found in station rainfall zip: {zip_path}")
    return shapefiles[0]


def parse_hour_fields(fields: list[str], event_date: datetime) -> dict[str, datetime]:
    parsed: dict[str, datetime] = {}
    for field in fields:
        match = re.match(r"(\d{1,2})日(\d{1,2})时$", field)
        if match:
            parsed[field] = event_date.replace(
                day=int(match.group(1)),
                hour=int(match.group(2)),
                minute=0,
                second=0,
                microsecond=0,
            )
    return parsed


def read_station_records(cfg: dict, shp_path: Path) -> tuple[list[datetime], list[dict]]:
    reader = shapefile.Reader(str(shp_path))
    fields = [field[0] for field in reader.fields[1:]]
    start, _, simulation_end = event_times(cfg)
    event_date = datetime.strptime(str(cfg["event"]["date"]), "%Y-%m-%d")
    hour_fields = parse_hour_fields(fields, event_date)
    if not hour_fields:
        raise RuntimeError("No hourly rainfall fields like '24日0时' were found in station rainfall shapefile.")

    times = event_hour_range(start, simulation_end)
    source_crs = prj_crs(shp_path)
    to_wgs84 = Transformer.from_crs(source_crs, 4326, always_xy=True) if source_crs else None
    records: list[dict] = []
    for index, shape_record in enumerate(reader.iterShapeRecords(), start=1):
        record = dict(zip(fields, shape_record.record))
        values_by_time: dict[datetime, float] = {}
        for field, stamp in hour_fields.items():
            try:
                values_by_time[stamp] = max(0.0, float(record.get(field) or 0.0))
            except (TypeError, ValueError):
                values_by_time[stamp] = 0.0
        values = [values_by_time.get(stamp, 0.0) for stamp in times]

        lon_value = record.get("LGTD")
        lat_value = record.get("LTTD")
        if lon_value not in (None, "") and lat_value not in (None, ""):
            lon, lat = float(lon_value), float(lat_value)
        else:
            x, y = shape_record.shape.points[0]
            lon, lat = to_wgs84.transform(x, y) if to_wgs84 else (float(x), float(y))
        records.append(
            {
                "station_key": f"ST{index:02d}",
                "station_id": str(record.get("STCD") or ""),
                "station_name": str(record.get("STNM") or f"Station {index}"),
                "lon": float(lon),
                "lat": float(lat),
                "series": values,
                "total": float(sum(values)),
                "peak": float(max(values) if values else 0.0),
            }
        )
    if not records:
        raise RuntimeError("At least one rainfall station is required.")
    return times, records


def write_station_tables(
    slug: str,
    rain_dir: Path,
    times: list[datetime],
    records: list[dict],
) -> tuple[Path, Path]:
    observations_csv = rain_dir / f"{slug}_station_observations_long.csv"
    with observations_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["station_key", "station_id", "station_name", "lon", "lat", "datetime_beijing", "rain_mm"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for stamp, value in zip(times, record["series"]):
                writer.writerow(
                    {
                        "station_key": record["station_key"],
                        "station_id": record["station_id"],
                        "station_name": record["station_name"],
                        "lon": f"{record['lon']:.8f}",
                        "lat": f"{record['lat']:.8f}",
                        "datetime_beijing": stamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "rain_mm": f"{float(value):.4f}",
                    }
                )

    summary_csv = rain_dir / f"{slug}_station_rainfall_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["station_key", "station_id", "station_name", "lon", "lat", "total_mm", "peak_mm_h"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "station_key": record["station_key"],
                    "station_id": record["station_id"],
                    "station_name": record["station_name"],
                    "lon": f"{record['lon']:.8f}",
                    "lat": f"{record['lat']:.8f}",
                    "total_mm": f"{record['total']:.3f}",
                    "peak_mm_h": f"{record['peak']:.3f}",
                }
            )
    return observations_csv, summary_csv


def domain_idw_series(cfg: dict, observations_csv: Path, weights_csv: Path) -> list[tuple[datetime, float]]:
    observations = read_station_observations(observations_csv)
    dem_path = Path(cfg["terrain"]["dem"])
    resolution = float(cfg["rainfall"].get("station_idw_resolution_m", 5.0))
    power = float(cfg["rainfall"].get("station_idw_power", 2.0))
    minimum_distance = float(cfg["rainfall"].get("station_idw_min_distance_m", 100.0))

    with rasterio.open(dem_path) as source:
        width = max(1, int(round(source.width * abs(source.transform.a) / resolution)))
        height = max(1, int(round(source.height * abs(source.transform.e) / resolution)))
        dem = source.read(1, out_shape=(height, width), masked=True, resampling=Resampling.nearest)
        transform = source.transform * Affine.scale(source.width / width, source.height / height)
        crs = source.crs
    valid = ~np.ma.getmaskarray(dem)
    # DEMs arrive as integer or floating point; cast before using NaN as the fill.
    valid &= np.isfinite(np.ma.getdata(dem).astype("float64"))
    if cfg.get("study_area", {}).get("calculation_domain") == "dem_landuse_intersection":
        landuse_path = Path(cfg.get("study_area", {}).get("landuse_shp") or "")
        valid &= landuse_mask_for_grid(valid.shape, transform, crs, landuse_path)
    from ..spatial_domain import apply_calculation_mask
    valid = apply_calculation_mask(valid, transform, crs, Path((cfg.get('project') or {}).get('root') or ROOT))
    if not np.any(valid):
        raise RuntimeError("DEM and land-use intersection contains no valid cells for station IDW rainfall.")

    station_x, station_y = station_coordinates(observations, crs)
    zones = np.where(valid, 1, 0).astype("int32")
    weights, counts = idw_mean_weights_by_zone(
        zones,
        valid,
        transform,
        station_x,
        station_y,
        zone_count=1,
        power=power,
        minimum_distance_m=minimum_distance,
    )
    domain_weights = weights[0]
    values = domain_weights @ observations.rainfall_mm

    with weights_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["station_key", "station_id", "station_name", "lon", "lat", "domain_idw_weight"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, key in enumerate(observations.keys):
            writer.writerow(
                {
                    "station_key": key,
                    "station_id": observations.station_ids[index],
                    "station_name": observations.names[index],
                    "lon": f"{observations.lon[index]:.8f}",
                    "lat": f"{observations.lat[index]:.8f}",
                    "domain_idw_weight": f"{domain_weights[index]:.8f}",
                }
            )
    print(
        f"[INFO] Domain IDW grid={width}x{height}; valid cells={int(counts[0])}; "
        f"power={power:.2f}; minimum distance={minimum_distance:.1f} m"
    )
    return list(zip(observations.datetimes, values.tolist()))


def write_hourly_series(path: Path, series: list[tuple[datetime, float]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["datetime_beijing", "rain_mm"])
        writer.writeheader()
        for stamp, value in series:
            writer.writerow({"datetime_beijing": stamp.strftime("%Y-%m-%d %H:%M:%S"), "rain_mm": f"{value:.4f}"})


def main() -> None:
    cfg = load_config()
    rain_cfg = cfg.setdefault("rainfall", {})
    zip_path = Path(rain_cfg.get("station_zip") or "")
    if not zip_path.is_file():
        raise RuntimeError(f"Station rainfall zip not found: {zip_path}")

    rain_dir = ROOT / "inputs" / "rainfall"
    rain_dir.mkdir(parents=True, exist_ok=True)
    slug = event_slug(cfg)
    shp_path = extract_station_zip(zip_path, rain_dir / "station_source")
    times, records = read_station_records(cfg, shp_path)
    observations_csv, summary_csv = write_station_tables(slug, rain_dir, times, records)
    weights_csv = rain_dir / f"{slug}_station_domain_idw_weights.csv"
    series = domain_idw_series(cfg, observations_csv, weights_csv)

    station_csv = rain_dir / f"{slug}_station_hourly.csv"
    active_csv = rain_dir / f"{slug}_active_hourly.csv"
    write_hourly_series(station_csv, series)
    shutil.copy2(station_csv, active_csv)

    rain_cfg["source_mode"] = "station"
    rain_cfg["station_aggregation"] = "idw_subcatchment_area_mean"
    rain_cfg["station_observations_csv"] = str(observations_csv).replace("\\", "/")
    rain_cfg["station_summary_csv"] = str(summary_csv).replace("\\", "/")
    rain_cfg["station_domain_weights_csv"] = str(weights_csv).replace("\\", "/")
    rain_cfg["station_csv"] = str(station_csv).replace("\\", "/")
    rain_cfg["csv"] = str(active_csv).replace("\\", "/")
    rain_cfg["source_note"] = "Measured multi-station rainfall; hourly IDW and subcatchment area averaging."
    save_config(cfg)

    totals = [float(record["total"]) for record in records]
    print(f"[OK] Station rainfall source: {shp_path}")
    print(f"[OK] Station observations: {observations_csv}")
    print(f"[OK] Active domain-mean rainfall: {active_csv}")
    print(f"[INFO] Stations={len(records)}; method=hourly IDW + equal-area grid mean")
    print(f"[INFO] Station event totals={min(totals):.3f}-{max(totals):.3f} mm")
    print(f"[INFO] IDW domain total={sum(value for _, value in series):.3f} mm; peak={max(value for _, value in series):.3f} mm/h")


if __name__ == "__main__":
    main()
