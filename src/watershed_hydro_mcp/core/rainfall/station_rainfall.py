from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer


@dataclass(frozen=True)
class StationObservations:
    keys: list[str]
    station_ids: list[str]
    names: list[str]
    lon: np.ndarray
    lat: np.ndarray
    datetimes: list[datetime]
    rainfall_mm: np.ndarray


def read_station_observations(path: str | Path) -> StationObservations:
    source = Path(path)
    grouped: dict[str, dict] = {}
    all_times: set[datetime] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = str(row.get("station_key") or row.get("station_id") or row.get("station_name") or "").strip()
            if not key:
                continue
            stamp = datetime.fromisoformat(str(row["datetime_beijing"]).strip())
            item = grouped.setdefault(
                key,
                {
                    "station_id": str(row.get("station_id") or ""),
                    "station_name": str(row.get("station_name") or key),
                    "lon": float(row["lon"]),
                    "lat": float(row["lat"]),
                    "values": {},
                },
            )
            item["values"][stamp] = max(0.0, float(row["rain_mm"]))
            all_times.add(stamp)
    if not grouped or not all_times:
        raise RuntimeError(f"No station observations were found in {source}")

    times = sorted(all_times)
    keys = list(grouped)
    rainfall = np.zeros((len(keys), len(times)), dtype="float64")
    for station_index, key in enumerate(keys):
        values = grouped[key]["values"]
        rainfall[station_index] = [float(values.get(stamp, 0.0)) for stamp in times]
    return StationObservations(
        keys=keys,
        station_ids=[grouped[key]["station_id"] for key in keys],
        names=[grouped[key]["station_name"] for key in keys],
        lon=np.asarray([grouped[key]["lon"] for key in keys], dtype="float64"),
        lat=np.asarray([grouped[key]["lat"] for key in keys], dtype="float64"),
        datetimes=times,
        rainfall_mm=rainfall,
    )


def station_coordinates(observations: StationObservations, target_crs) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_user_input(target_crs), always_xy=True)
    x, y = transformer.transform(observations.lon, observations.lat)
    return np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")


def idw_weights_at_points(
    point_x: np.ndarray,
    point_y: np.ndarray,
    station_x: np.ndarray,
    station_y: np.ndarray,
    power: float = 2.0,
    minimum_distance_m: float = 100.0,
) -> np.ndarray:
    px = np.asarray(point_x, dtype="float64").reshape(-1, 1)
    py = np.asarray(point_y, dtype="float64").reshape(-1, 1)
    sx = np.asarray(station_x, dtype="float64").reshape(1, -1)
    sy = np.asarray(station_y, dtype="float64").reshape(1, -1)
    distance = np.sqrt(np.square(px - sx) + np.square(py - sy))
    distance = np.maximum(distance, max(float(minimum_distance_m), 1e-6))
    raw = 1.0 / np.power(distance, max(float(power), 0.1))
    return raw / np.maximum(np.sum(raw, axis=1, keepdims=True), 1e-30)


def idw_mean_weights_by_zone(
    zone_grid: np.ndarray,
    valid: np.ndarray,
    transform,
    station_x: np.ndarray,
    station_y: np.ndarray,
    zone_count: int,
    power: float = 2.0,
    minimum_distance_m: float = 100.0,
    chunk_rows: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-area cell means of normalized station IDW weights by zone."""
    zones = np.asarray(zone_grid, dtype="int32")
    valid_mask = np.asarray(valid, dtype=bool) & (zones > 0) & (zones <= int(zone_count))
    station_x = np.asarray(station_x, dtype="float64")
    station_y = np.asarray(station_y, dtype="float64")
    sums = np.zeros((zone_count + 1, station_x.size), dtype="float64")
    counts = np.bincount(zones[valid_mask], minlength=zone_count + 1).astype("int64")

    width = zones.shape[1]
    columns = np.arange(width, dtype="float64") + 0.5
    for row_start in range(0, zones.shape[0], max(1, int(chunk_rows))):
        row_stop = min(zones.shape[0], row_start + max(1, int(chunk_rows)))
        rows = np.arange(row_start, row_stop, dtype="float64") + 0.5
        col_grid, row_grid = np.meshgrid(columns, rows)
        x_grid = transform.c + transform.a * col_grid + transform.b * row_grid
        y_grid = transform.f + transform.d * col_grid + transform.e * row_grid
        local_valid = valid_mask[row_start:row_stop]
        if not np.any(local_valid):
            continue
        local_zones = zones[row_start:row_stop][local_valid]
        point_weights = idw_weights_at_points(
            x_grid[local_valid],
            y_grid[local_valid],
            station_x,
            station_y,
            power=power,
            minimum_distance_m=minimum_distance_m,
        )
        for station_index in range(station_x.size):
            sums[:, station_index] += np.bincount(
                local_zones,
                weights=point_weights[:, station_index],
                minlength=zone_count + 1,
            )

    means = np.zeros((zone_count, station_x.size), dtype="float64")
    populated = counts[1:] > 0
    means[populated] = sums[1:][populated] / counts[1:][populated, None]
    return means, counts[1:]
