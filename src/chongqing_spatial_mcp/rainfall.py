from __future__ import annotations

import csv
import json
import math
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import shapefile
import numpy as np
from shapely.geometry import Point, box, shape as shapely_shape
from shapely.ops import unary_union

from .catalog import rainfall_files
from .models import RainfallRequest
from .settings import SETTINGS, Settings
from .state import rainfall_dir, require_confirmed, utc_now, write_json


def parse_grid_column(name: str) -> tuple[float, float] | None:
    try:
        lon, lat = name.split("_", 1)
        return float(lon), float(lat)
    except (ValueError, TypeError):
        return None


def _rings(geometry: dict) -> list[list[list[float]]]:
    if geometry.get("type") == "Polygon":
        return geometry.get("coordinates", [])[:1]
    if geometry.get("type") == "MultiPolygon":
        return [polygon[0] for polygon in geometry.get("coordinates", []) if polygon]
    return []


def _point_in_ring(x: float, y: float, ring: list[list[float]]) -> bool:
    inside = False
    if len(ring) < 3:
        return False
    j = len(ring) - 1
    for i, point in enumerate(ring):
        xi, yi = point[:2]
        xj, yj = ring[j][:2]
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def _load_basin(path: Path) -> tuple[list[list[list[float]]], tuple[float, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rings: list[list[list[float]]] = []
    for feature in data.get("features", []):
        rings.extend(_rings(feature.get("geometry") or {}))
    if not rings:
        raise ValueError("流域边界没有有效Polygon几何")
    points = [point for ring in rings for point in ring]
    return rings, (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _load_basin_geometry(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    geometries = [
        shapely_shape(feature["geometry"])
        for feature in payload.get("features", [])
        if feature.get("geometry")
    ]
    if not geometries:
        raise ValueError("流域边界没有有效Polygon几何")
    geometry = unary_union(geometries)
    if geometry.is_empty or geometry.area <= 0:
        raise ValueError("流域边界面积无效")
    return geometry


def _distributed_interior_points(geometry, count: int) -> list[Point]:
    """Generate deterministic, well-spread station locations inside a basin."""
    candidates: list[Point] = []
    centroid = geometry.centroid
    if geometry.contains(centroid):
        candidates.append(centroid)
    representative = geometry.representative_point()
    if not any(representative.equals(item) for item in candidates):
        candidates.append(representative)
    parts = list(getattr(geometry, "geoms", [geometry]))
    for part in parts:
        point = part.representative_point()
        if geometry.contains(point) and not any(point.equals(item) for item in candidates):
            candidates.append(point)

    min_x, min_y, max_x, max_y = geometry.bounds
    # Cell-centre grids avoid systematically putting stations on the boundary.
    for side in (8, 16, 32, 64, 128):
        for row in range(side):
            y = min_y + (row + 0.5) * (max_y - min_y) / side
            for column in range(side):
                x = min_x + (column + 0.5) * (max_x - min_x) / side
                point = Point(x, y)
                if geometry.contains(point):
                    candidates.append(point)
        if len(candidates) >= max(count * 20, count):
            break
    if len(candidates) < count:
        raise ValueError(f"流域几何过窄，无法在内部布置{count}个不同虚拟站")

    centre = geometry.centroid
    coordinates = np.asarray([(p.x, p.y) for p in candidates])
    first_index = int(np.argmin(np.sum((coordinates - [centre.x, centre.y]) ** 2, axis=1)))
    selected = [candidates[first_index]]
    distances = np.sum((coordinates - coordinates[first_index]) ** 2, axis=1)
    while len(selected) < count:
        index = int(np.argmax(distances))
        if distances[index] <= 1e-24:
            raise ValueError(f"流域内部可用站点不足，无法布置{count}个虚拟站")
        selected.append(candidates[index])
        distances = np.minimum(distances, np.sum((coordinates - coordinates[index]) ** 2, axis=1))
    return selected


def _source_weights(
    lon: float,
    lat: float,
    grid_points: list[dict],
    fallback_pool_multiplier: int,
) -> list[dict]:
    """Bilinear weights on the ERA5 grid, with inverse-distance fallback."""
    lookup = {(item["lon"], item["lat"]): item for item in grid_points}
    longitudes = sorted({item["lon"] for item in grid_points})
    latitudes = sorted({item["lat"] for item in grid_points})

    def bracket(values: list[float], value: float) -> tuple[float, float]:
        lower = max((item for item in values if item <= value), default=values[0])
        upper = min((item for item in values if item >= value), default=values[-1])
        return lower, upper

    x0, x1 = bracket(longitudes, lon)
    y0, y1 = bracket(latitudes, lat)
    x_weights = [(x0, 1.0)] if x0 == x1 else [
        (x0, (x1 - lon) / (x1 - x0)),
        (x1, (lon - x0) / (x1 - x0)),
    ]
    y_weights = [(y0, 1.0)] if y0 == y1 else [
        (y0, (y1 - lat) / (y1 - y0)),
        (y1, (lat - y0) / (y1 - y0)),
    ]
    weighted: list[dict] = []
    for x, x_weight in x_weights:
        for y, y_weight in y_weights:
            source = lookup.get((x, y))
            weight = x_weight * y_weight
            if source is not None and weight > 0:
                weighted.append({**source, "weight": weight})
    if weighted and abs(sum(item["weight"] for item in weighted) - 1.0) < 1e-8:
        return weighted

    nearest = sorted(
        grid_points,
        key=lambda item: math.hypot(item["lon"] - lon, item["lat"] - lat),
    )[:max(4, fallback_pool_multiplier)]
    if not nearest:
        raise ValueError("降雨表没有可解析的ERA5网格列")
    distances = [math.hypot(item["lon"] - lon, item["lat"] - lat) for item in nearest]
    if distances[0] <= 1e-12:
        return [{**nearest[0], "weight": 1.0}]
    inverses = [1.0 / max(distance, 1e-12) for distance in distances]
    total = sum(inverses)
    return [{**item, "weight": weight / total} for item, weight in zip(nearest, inverses)]


def select_virtual_stations(
    boundary: Path,
    header: list[str],
    count: int,
    fallback_pool_multiplier: int = 4,
) -> list[dict]:
    geometry = _load_basin_geometry(boundary)
    centre = geometry.centroid
    grid_points = []
    for index, name in enumerate(header[1:], start=1):
        coordinate = parse_grid_column(name)
        if coordinate is None:
            continue
        lon, lat = coordinate
        grid_points.append({"column": index, "grid_name": name, "lon": lon, "lat": lat})
    if not grid_points:
        raise ValueError("降雨表没有可解析的ERA5网格列")
    if (len({(p['lon'],p['lat']) for p in grid_points}) != len(grid_points)
            or not all(math.isfinite(p['lon']) and math.isfinite(p['lat']) for p in grid_points)):
        raise ValueError('ERA5源格点坐标重复或无效，不能构建不同的雨量源')

    # Prefer actual source columns. CSV coordinates are grid-point centres;
    # half-spacing supports explicitly determine whether an outside centre
    # covers the drawn study area. This does not increase ERA5 resolution.
    steps = []
    for axis in ('lon','lat'):
        values = sorted({p[axis] for p in grid_points})
        differences = np.diff(values)
        step = float(np.min(differences)) if differences.size else .25
        if not math.isfinite(step) or step <= 0 or (differences.size and not np.allclose(differences/step,np.round(differences/step),atol=1e-7)):
            raise ValueError('ERA5格点坐标不是可确定代表范围的规则格网')
        steps.append(step)
    available = []
    for source in grid_points:
        x,y = source['lon'],source['lat']
        bounds = [x-steps[0]/2,y-steps[1]/2,x+steps[0]/2,y+steps[1]/2]
        overlap = geometry.intersection(box(*bounds)).area
        if overlap <= 1e-15:
            continue
        available.append(dict(**source, support_bounds=bounds, overlap=overlap,
                              inside=bool(geometry.covers(Point(x,y)))))
    interiors = [p for p in available if p['inside']]
    # With two or more interior source centres, do not add outside edge ones.
    if len(interiors) >= 2:
        available = interiors
    available.sort(key=lambda p:(not p['inside'],-p['overlap'],math.hypot(p['lon']-centre.x,p['lat']-centre.y)))
    selected = []
    for source in available[:count]:
        point = Point(source['lon'],source['lat'])
        placement = 'original_grid_centre'
        if not geometry.contains(point):
            overlap = geometry.intersection(box(*source['support_bounds']))
            point = overlap.centroid
            if not overlap.contains(point) or not geometry.contains(point):
                point = overlap.representative_point()
            if not geometry.contains(point):
                raise ValueError('ERA5覆盖交集无法生成严格位于研究区内的代理点')
            placement = 'inside_overlap_proxy'
        selected.append(dict(lon=point.x,lat=point.y,name=source['grid_name'],
            inside=True,source_inside=source['inside'],source_lon=source['lon'],source_lat=source['lat'],
            placement_basis=placement,center_distance=point.distance(centre),
            source_weights=[{k:source[k] for k in ('column','grid_name','lon','lat')} | {'weight':1.}],
            source_grid_distance=point.distance(Point(source['lon'],source['lat'])),
            selection_basis='existing_era5_grid',rainfall_value_method='source_column',support_bounds=source['support_bounds']))
    if len(selected) < count:
        pool = _distributed_interior_points(geometry,count+len(selected))
        while len(selected) < count:
            if selected:
                point = max(pool,key=lambda p:min((p.x-s['lon'])**2+(p.y-s['lat'])**2 for s in selected))
                if min((point.x-s['lon'])**2+(point.y-s['lat'])**2 for s in selected) <= 1e-24:
                    raise ValueError('区内可用的不同雨量位置不足')
            else:
                point = pool[0]
            pool.remove(point)
            selected.append(_interpolated_station(point,geometry,centre,grid_points,fallback_pool_multiplier))
    for index,item in enumerate(selected,1):
        item['station_key'] = f'VS{index:02d}'
        item['station_name'] = (f'ERA5格网代理点{index:02d}' if item['selection_basis']=='existing_era5_grid' else f'ERA5虚拟站{index:02d}')
    return selected


def _interpolated_station(point,geometry,centre,grid_points,fallback_pool_multiplier):
    sources = _source_weights(
        point.x, point.y, grid_points, max(1, int(fallback_pool_multiplier))
    )
    item = {
        "lon": float(point.x),
        "lat": float(point.y),
        "name": f"virtual_{point.x:.6f}_{point.y:.6f}",
        "inside": bool(geometry.covers(point)),
        "center_distance": float(point.distance(centre)),
        "source_weights": sources,
        "source_grid_distance": max(
            math.hypot(source["lon"] - point.x, source["lat"] - point.y)
            for source in sources
        ),
        "selection_basis": "inside_basin_bilinear_interpolation",
        'rainfall_value_method':'era5_interpolation',
    }
    return item

def _station_value(row: list[str], station: dict) -> float:
    total = 0.0
    for source in station.get("source_weights", []):
        column = int(source["column"])
        raw = row[column] if column < len(row) else ""
        total += max(0.0, float(raw or 0.0)) * float(source["weight"])
    return total


def _parse_stamp(text: str) -> datetime:
    return datetime.fromisoformat(text.strip())


def _explicit_range(request: RainfallRequest) -> tuple[datetime, datetime]:
    if request.date:
        start = datetime.strptime(request.date, "%Y-%m-%d")
        return start, start + timedelta(hours=23)
    return datetime.fromisoformat(str(request.start_datetime)), datetime.fromisoformat(str(request.end_datetime))


def _read_explicit(files: dict[int, Path], stations: list[dict], start: datetime, end: datetime) -> dict[str, list[float]]:
    if (start.year, start.month) != (end.year, end.month):
        raise ValueError("事件必须位于同一自然月")
    path = files.get(start.year)
    if path is None:
        raise ValueError(f"没有{start.year}年的降雨数据")
    expected = int((end - start).total_seconds() // 3600) + 1
    result = {item["station_key"]: [] for item in stations}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            try:
                stamp = _parse_stamp(row[0])
            except (ValueError, IndexError):
                # Some legacy annual tables contain a repeated header row.
                continue
            if stamp < start:
                continue
            if stamp > end:
                break
            for item in stations:
                result[item["station_key"]].append(_station_value(row, item))
    if any(len(values) != expected for values in result.values()):
        raise ValueError("指定时段的逐时记录不完整")
    return result


def _read_historical_max(
    files: dict[int, Path], stations: list[dict], years: Iterable[int]
) -> tuple[datetime, dict[str, list[float]], float]:
    daily: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for year in years:
        path = files.get(year)
        if path is None:
            raise ValueError(f"缺少历史统计期内的{year}年降雨数据")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            next(reader)
            for row in reader:
                try:
                    stamp = _parse_stamp(row[0])
                except (ValueError, IndexError):
                    continue
                day = stamp.strftime("%Y-%m-%d")
                for item in stations:
                    daily[day][item["station_key"]].append(_station_value(row, item))
    best_day = None
    best_score = -1.0
    for day, station_rows in daily.items():
        if any(len(station_rows[item["station_key"]]) != 24 for item in stations):
            continue
        score = sum(sum(station_rows[item["station_key"]]) for item in stations) / len(stations)
        if score > best_score:
            best_day, best_score = day, score
    if best_day is None:
        raise ValueError("统计期内没有完整的24小时日期")
    return datetime.strptime(best_day, "%Y-%m-%d"), dict(daily[best_day]), best_score


def _write_package(
    output: Path,
    stations: list[dict],
    start: datetime,
    series: dict[str, list[float]],
    metadata: dict,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    station_root = output / "station_rainfall"
    shp_base = station_root / "虚拟雨站降雨过程"
    station_root.mkdir()
    count = len(next(iter(series.values())))
    times = [start + timedelta(hours=index) for index in range(count)]
    writer = shapefile.Writer(str(shp_base), shapeType=shapefile.POINT, encoding="utf-8")
    writer.field("STCD", "C", size=32)
    writer.field("STNM", "C", size=80)
    writer.field("LGTD", "F", size=19, decimal=8)
    writer.field("LTTD", "F", size=19, decimal=8)
    for stamp in times:
        writer.field(f"{stamp.day}日{stamp.hour}时", "F", size=12, decimal=4)
    for item in stations:
        writer.point(item["lon"], item["lat"])
        writer.record(item["station_key"], item["station_name"], item["lon"], item["lat"],
                      *series[item["station_key"]])
    writer.close()
    shp_base.with_suffix(".prj").write_text(
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
        'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]', encoding="utf-8"
    )
    shp_base.with_suffix(".cpg").write_text("UTF-8\n", encoding="ascii")
    long_csv = output / "virtual_station_observations_long.csv"
    with long_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["station_key", "station_name", "lon", "lat", "datetime_beijing", "rain_mm"]
        csv_writer = csv.DictWriter(stream, fieldnames=fields)
        csv_writer.writeheader()
        for item in stations:
            for stamp, value in zip(times, series[item["station_key"]]):
                csv_writer.writerow({
                    "station_key": item["station_key"], "station_name": item["station_name"],
                    "lon": item["lon"], "lat": item["lat"],
                    "datetime_beijing": stamp.strftime("%Y-%m-%d %H:%M:%S"), "rain_mm": f"{value:.4f}",
                })
    zip_path = output / "station_rainfall.zip"
    selection_file = output/'rainfall_source_selection.json'
    selection_payload = {'policy':'existing_era5_grid_first_then_inside_virtual',
        'support_convention':'CSV centres +/- half regular spacing; explicit midpoint support assumption',
        'stations':[{k:s[k] for k in ('station_key','lon','lat','inside','selection_basis','support_bounds',
                                    'source_lon','source_lat','source_inside','placement_basis','rainfall_value_method') if k in s} for s in stations]}
    write_json(selection_file,selection_payload)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in station_root.iterdir():
            archive.write(path, arcname=f"虚拟雨站降雨过程/{path.name}")
        archive.write(selection_file,arcname=selection_file.name)
    metadata.update({
        "status": "ready", "created_at": utc_now(), "event_start": start.isoformat(" "),
        "event_end": times[-1].isoformat(" "), "timestamp_count": len(times),
        "stations": [{key: item[key] for key in ("station_key", "station_name", "lon", "lat", "name", "inside",
                      'source_lon','source_lat','source_inside','placement_basis','rainfall_value_method')
                      if key in item} | {"center_distance_deg": round(item.get("center_distance") or 0.0, 4),
                                         "source_grid_distance_deg": round(item.get("source_grid_distance") or 0.0, 4),
                                         "selection_basis": item.get("selection_basis"),
                                         "source_grid_weights": [
                                             {"grid_name": source["grid_name"],
                                              "lon": source["lon"], "lat": source["lat"],
                                              "weight": round(float(source["weight"]), 8)}
                                             for source in item.get("source_weights", [])
                                         ]}
                     for item in stations],
        "station_zip": str(zip_path), "observations_csv": str(long_csv),
        'existing_grid_count':sum(s.get('selection_basis')=='existing_era5_grid' for s in stations),
        'interpolated_station_count':sum(s.get('selection_basis')!='existing_era5_grid' for s in stations),
        'outside_grid_centre_count':sum(not s.get('source_inside',s['inside']) for s in stations),
        'relocated_source_count':sum(s.get('placement_basis')=='inside_overlap_proxy' for s in stations),
        'all_model_stations_inside':all(s['inside'] for s in stations),
        'source_selection_policy':selection_payload['policy'],
        "source_note": ('Existing ERA5 grid columns are preferred; deficits are filled with in-basin '
                        'virtual stations using the legacy interpolation. Every model station is inside the AOI. '
                        'An outside source centre supplies its original column to an interior proxy in its '
                        'midpoint-support/AOI overlap; source coordinates are preserved separately. '
                        'None are measured gauges; downstream subcatchment IDW is retained.'),
    })
    write_json(output / "rainfall_manifest.json", metadata)
    return metadata


def resolve_station_count(request: RainfallRequest, boundary: Path, settings: Settings = SETTINGS) -> dict:
    """Use the submitted polygon area, not its bbox or a clipped DEM window."""
    from pyproj import Geod
    from shapely.geometry.polygon import orient
    geometry = _load_basin_geometry(boundary)
    # Orient each polygon so holes subtract and multipart pieces cannot cancel.
    polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    geod = Geod(ellps="WGS84")
    area = sum(abs(geod.geometry_area_perimeter(orient(p, sign=1))[0]) for p in polygons) / 1e6
    if not math.isfinite(area) or area <= 0:
        raise ValueError("选区测地面积无效，无法确定虚拟雨站数量")
    if request.station_count is not None:
        return {"mode": "manual", "station_count": request.station_count,
                "area_km2": area, "reason": "用户明确指定虚拟雨站数量"}
    options = settings.rainfall_defaults
    unit = float(options.get("automatic_station_area_km2", 80.0))
    minimum = options.get("automatic_station_minimum", 1)
    maximum = options.get("automatic_station_maximum", 20)
    if (not math.isfinite(unit) or unit <= 0 or type(minimum) is not int or type(maximum) is not int
            or not 1 <= minimum <= maximum <= 100):
        raise ValueError("自动虚拟雨站数量配置无效")
    raw = math.ceil(area / unit)
    count = max(minimum, min(maximum, raw))
    return {"mode": "auto", "station_count": count, "area_km2": area,
            "area_per_station_km2": unit, "minimum": minimum, "maximum": maximum,
            "stations_per_unit": 1,
            "unclamped_count": raw, "capped": raw > maximum,
            "reason": "按选区测地面积向上取整，并应用配置的最少和最多站数",
            "limitation": "虚拟采样点数量增加不代表增加了独立实测降雨信息"}


def build_package(request: RainfallRequest, settings: Settings = SETTINGS) -> dict:
    review = require_confirmed(request.review_id, settings)
    files = rainfall_files(settings)
    if not files:
        raise ValueError("没有可用的逐时降雨文件")
    sample = files[max(files)]
    with sample.open("r", encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream))
    rainfall_defaults = settings.rainfall_defaults
    boundary = Path(review["artifacts"]["boundary_geojson"])
    station_selection = resolve_station_count(request, boundary, settings)
    stations = select_virtual_stations(
        boundary, header, station_selection["station_count"],
        fallback_pool_multiplier=int(rainfall_defaults.get("fallback_pool_multiplier", 4)),
    )
    farthest = max(item["source_grid_distance"] for item in stations) if stations else 0.0
    metadata = {
        "package_id": f"rain-{uuid4().hex[:12]}", "review_id": request.review_id,
        "station_count": len(stations), "station_selection": station_selection,
        "selection_mode": request.mode, "source_timezone": request.source_timezone,
        "time_semantics": "hourly timestamps are interpreted in the supplied source_timezone",
        "station_selection_basis": sorted({item.get("selection_basis") for item in stations}),
        "farthest_station_distance_deg": round(farthest, 4),
        "farthest_source_grid_distance_deg": round(farthest, 4),
    }
    maximum_distance = float(rainfall_defaults.get("maximum_station_distance_deg", 1.0))
    if farthest > maximum_distance:
        metadata["warning_station_distance"] = (
            f"虚拟站插值所用ERA5源网格最远距离为 {farthest:.2f}°，"
            f"超过配置阈值 {maximum_distance:.2f}°，雨量代表性有限"
        )
    if request.mode == "explicit_period":
        start, end = _explicit_range(request)
        values = _read_explicit(files, stations, start, end)
        metadata["selection_rule"] = "user-specified inclusive hourly period"
    else:
        last = request.end_year or max(files)
        first = last - request.lookback_years + 1
        years = list(range(first, last + 1))
        start, values, score = _read_historical_max(files, stations, years)
        metadata.update({
            "selection_rule": "maximum complete calendar-day mean total across selected virtual stations",
            "statistics_years": years, "selected_day_basin_station_mean_total_mm": round(score, 4),
            "warning": "sample historical maximum; not a return-period or design-storm estimate",
        })
    output = rainfall_dir(metadata["package_id"], settings)
    return _write_package(output, stations, start, values, metadata)
