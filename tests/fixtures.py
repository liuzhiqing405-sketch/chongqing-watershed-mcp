"""Small builders shared by the test suites.

These create genuinely valid inputs (a real GeoTIFF, real shapefiles, a real
station archive) rather than mocks, so the safety checks are exercised against
the same file types the service meets in production.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio
import shapefile
from rasterio.transform import Affine


def _wgs84_wkt() -> str:
    from pyproj import CRS

    return CRS.from_epsg(4326).to_wkt("WKT1_GDAL")


def _utm49_wkt() -> str:
    """The exact EPSG:32649 definition, so the sidecar cannot misplace coordinates."""
    from pyproj import CRS

    return CRS.from_epsg(32649).to_wkt("WKT1_GDAL")

#: Everything in the fixture sits in this box, so the DEM, the boundary, the
#: land use and the river network all describe the same place.
UTM_BOX = (74_500.0, 3_304_600.0, 76_600.0, 3_307_100.0)   # left, bottom, right, top
WGS_BOX = (106.60, 29.80, 106.62, 29.82)                    # west, south, east, north
DEM_SHAPE = (84, 70)                                        # rows, cols


def write_dem(path: Path, *, shape: tuple[int, int] = DEM_SHAPE,
              nodata: float = -9999.0) -> Path:
    """A small UTM valley that drains southward, inside :data:`UTM_BOX`."""
    left, bottom, right, top = UTM_BOX
    height, width = shape
    transform = Affine.translation(left, top) @ Affine.scale(
        (right - left) / width, -(top - bottom) / height
    )
    rows, cols = np.mgrid[0:height, 0:width]
    x = cols / max(width - 1, 1)
    y = rows / max(height - 1, 1)
    # V-shaped valley: high on both flanks, low along the centre line, falling
    # towards the southern edge so the outlet sits on the south boundary.
    data = (200.0 + 120.0 * np.abs(x - 0.5) + 80.0 * (1.0 - y)).astype("float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:32649", transform=transform, nodata=nodata, compress="lzw",
    ) as target:
        target.write(data, 1)
    return path


def write_boundary(path: Path, *, box: tuple[float, float, float, float] = WGS_BOX) -> Path:
    west, south, east, north = box
    ring = [[west, south], [east, south], [east, north], [west, north], [west, south]]
    payload = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _utm_ring(inset: float = 0.0) -> list[list[float]]:
    left, bottom, right, top = UTM_BOX
    return [
        [left + inset, bottom + inset], [right - inset, bottom + inset],
        [right - inset, top - inset], [left + inset, top - inset],
        [left + inset, bottom + inset],
    ]


def write_polygon_shapefile(path: Path, name: str = "测试区", *,
                            encoding: str = "gbk", ring: list[list[float]] | None = None) -> Path:
    return write_multi_polygon_shapefile(
        path, [(name, ring if ring is not None else _utm_ring())], encoding=encoding)


def write_multi_polygon_shapefile(path: Path, features: list[tuple[str, list[list[float]]]],
                                  *, encoding: str = "gbk", name_field: str = "Name") -> Path:
    base = path.with_suffix("")
    writer = shapefile.Writer(str(base), shapeType=shapefile.POLYGON, encoding=encoding)
    writer.field(name_field, "C", 60)
    writer.field("area_ha", "F", 14, 4)
    for name, ring in features:
        writer.poly([ring])
        writer.record(name, 9.0)
    writer.close()
    base.with_suffix(".prj").write_text(_utm49_wkt(), encoding="utf-8")
    base.with_suffix(".cpg").write_text(encoding.upper() + "\n", encoding="ascii")
    return path


def _utm_subring(x0: float, x1: float, y0: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def landuse_features() -> list[tuple[str, list[list[float]]]]:
    """Two land-use zones covering the DEM box, split on the valley centre."""
    left, bottom, right, top = UTM_BOX
    middle = (left + right) / 2.0
    return [
        ("耕地", _utm_subring(left, middle, bottom, top)),
        ("林地", _utm_subring(middle, right, bottom, top)),
    ]


def river_feature() -> tuple[str, list[list[float]]]:
    """A narrow strip down the valley centre, where the DEM drains."""
    left, bottom, right, top = UTM_BOX
    middle = (left + right) / 2.0
    half = (right - left) * 0.01
    return ("测试河", _utm_subring(middle - half, middle + half, bottom, top))


def write_station_archive(path: Path, *, stations: int = 4) -> Path:
    """A station rainfall package shaped like the one the spatial MCP exports."""
    base = path.parent / "虚拟雨站降雨过程"
    writer = shapefile.Writer(str(base), shapeType=shapefile.POINT, encoding="utf-8")
    writer.field("STCD", "C", 32)
    writer.field("STNM", "C", 80)
    writer.field("LGTD", "F", 19, 8)
    writer.field("LTTD", "F", 19, 8)
    for hour in range(24):
        writer.field(f"{hour + 1}日{hour}时", "F", 12, 4)
    for index in range(stations):
        lon = 106.600 + index * 0.005
        lat = 29.800 + index * 0.005
        writer.point(lon, lat)
        writer.record(f"VS{index + 1:02d}", f"测试雨站{index + 1:02d}", lon, lat,
                      *[1.0] * 24)
    writer.close()
    base.with_suffix(".prj").write_text(_wgs84_wkt(), encoding="utf-8")
    base.with_suffix(".cpg").write_text("UTF-8\n", encoding="ascii")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member in sorted(path.parent.glob("虚拟雨站降雨过程.*")):
            archive.write(member, arcname=f"虚拟雨站降雨过程/{member.name}")
        for member in sorted((path.parent / "虚拟雨站降雨过程").glob("*")):
            archive.write(member, arcname=f"虚拟雨站降雨过程/{member.name}")
    for member in list(path.parent.glob("虚拟雨站降雨过程.*")):
        member.unlink()
    folder = path.parent / "虚拟雨站降雨过程"
    if folder.is_dir():
        for member in folder.iterdir():
            member.unlink()
        folder.rmdir()
    return path


def build_package(directory: Path, *, package_id: str = "package-test000001") -> Path:
    """Assemble a complete, hash-correct Model Input Package v1."""
    from watershed_exchange.paths import sha256_file
    from watershed_exchange.schemas import (
        EventSpec, ModelInputPackage, PackageFile, StudyArea,
    )

    directory.mkdir(parents=True, exist_ok=True)
    write_dem(directory / "dem.tif")
    write_boundary(directory / "basin.geojson")
    write_station_archive(directory / "station_rainfall.zip")
    write_multi_polygon_shapefile(directory / "landuse.shp", landuse_features(), name_field="DLMC")
    write_multi_polygon_shapefile(directory / "river.shp", [river_feature()])

    files: list[PackageFile] = []
    for role, members in (
        ("dem", ["dem.tif"]),
        ("boundary", ["basin.geojson"]),
        ("rainfall", ["station_rainfall.zip"]),
        ("landuse", ["landuse.shp", "landuse.shx", "landuse.dbf", "landuse.prj", "landuse.cpg"]),
        ("river", ["river.shp", "river.shx", "river.dbf", "river.prj", "river.cpg"]),
    ):
        for member in members:
            path = directory / member
            files.append(PackageFile(role=role, relative_path=member,
                                     sha256=sha256_file(path),
                                     size_bytes=path.stat().st_size))
    package = ModelInputPackage(
        package_id=package_id,
        created_at="2026-01-01T00:00:00+00:00",
        event=EventSpec(start_datetime="2024-06-01 00:00", end_datetime="2024-06-01 05:00",
                        rainfall_source="ERA5 virtual stations"),
        study_area=StudyArea(crs="OGC:CRS84", bbox=[106.60, 29.80, 106.62, 29.82],
                             area_km2=4.2),
        files=files,
        qa={"spatial": {"passed": True}},
        provenance={"producer": "test"},
        limitations=["ERA5 grid points are virtual stations, not measured gauges."],
    )
    (directory / "manifest.json").write_text(
        json.dumps(package.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return directory


def temp_workspace(root: Path, name: str) -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory
