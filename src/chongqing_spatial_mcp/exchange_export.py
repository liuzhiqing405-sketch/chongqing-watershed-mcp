"""Build a self-contained Model Input Package v1 from a prepared input run.

The package is written so that the hydro MCP can copy it away and never look
back at this project's run directory: only relative paths, a SHA-256 for every
file, and the ERA5-virtual-station caveat travelling with the data.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import rasterio
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union

from watershed_exchange.paths import sha256_file
from watershed_exchange.schemas import (
    SCHEMA_VERSION,
    EventSpec,
    ModelInputPackage,
    PackageFile,
    StudyArea,
)

from .settings import SETTINGS, Settings
from .state import utc_now, write_json


#: role -> member file names inside the package directory.
PACKAGE_MEMBERS = {
    "dem": ["dem.tif"],
    "boundary": ["basin.geojson"],
    "rainfall": ["station_rainfall.zip"],
    "landuse": ["landuse.shp", "landuse.shx", "landuse.dbf", "landuse.prj", "landuse.cpg"],
    "river": ["river.shp", "river.shx", "river.dbf", "river.prj", "river.cpg"],
}

LIMITATIONS = [
    "Virtual stations are placed inside the basin and interpolated from ERA5 grid points; they are not measured gauges.",
    "Candidates and boundaries were produced by automatic processing and were confirmed by a human reviewer before export.",
]


def _study_area(boundary_path: Path) -> StudyArea:
    payload = json.loads(boundary_path.read_text(encoding="utf-8"))
    geometries = [
        shapely_shape(feature["geometry"])
        for feature in (payload.get("features") or [])
        if feature.get("geometry")
    ]
    if not geometries:
        raise ValueError("流域边界没有有效几何")
    merged = unary_union(geometries)
    min_x, min_y, max_x, max_y = merged.bounds
    return StudyArea(
        crs="OGC:CRS84",
        bbox=[round(min_x, 7), round(min_y, 7), round(max_x, 7), round(max_y, 7)],
        area_km2=_geodesic_area_km2(merged),
    )


def _geodesic_area_km2(geometry) -> float:
    """Approximate the WGS84 area of a geographic polygon in square kilometres."""
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    area, _ = geod.geometry_area_perimeter(geometry)
    return abs(area) / 1_000_000


def _crs_of(path: Path) -> str:
    with rasterio.open(path) as source:
        return source.crs.to_wkt() if source.crs is not None else ""


def build_manifest(
    package_dir: Path,
    *,
    review: dict[str, Any],
    rain_manifest: dict[str, Any],
    qa: dict[str, Any],
    terrain_budget: dict[str, Any] | None = None,
) -> ModelInputPackage:
    files: list[PackageFile] = []
    for role, members in PACKAGE_MEMBERS.items():
        for member in members:
            path = package_dir / member
            if not path.is_file():
                raise ValueError(f"输入包缺少 {role} 文件：{member}")
            files.append(PackageFile(
                role=role,
                relative_path=member,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            ))
    boundary = package_dir / "basin.geojson"
    context = package_dir / "dem_context.tif"
    if context.is_file():
        files.append(PackageFile(role="drainage_context", relative_path="dem_context.tif",
                                 sha256=sha256_file(context), size_bytes=context.stat().st_size))
    event = EventSpec(
        start_datetime=str(rain_manifest["event_start"]),
        end_datetime=str(rain_manifest["event_end"]),
        timezone=str(rain_manifest.get("source_timezone") or "Asia/Shanghai"),
        rainfall_source="existing ERA5 source grids preferred; in-basin virtual stations supplement deficits; legacy subcatchment IDW",
    )
    limitations = list(LIMITATIONS)
    if rain_manifest.get('source_selection_policy'):
        limitations[0] = str(rain_manifest['source_note'])
    if len(rain_manifest.get("stations") or []) == 1:
        limitations.append("A single virtual station supplies spatially uniform rainfall and cannot represent spatial rainfall differences.")
    if review.get("selection_mode")=="drawn_study_area":
        limitations[1]="The evaluation boundary was drawn by the user and confirmed by a human reviewer before export."
        policy=(review.get("qa") or {}).get("modeling_check") or {}
        limitations[0:0]=policy.get("boundary_assumptions") or []
        limitations.append("The study domain is user-drawn, not a verified complete natural watershed. External upstream inflow is not supplied by this package.")
        diagnostic=(review.get("qa") or {}).get("hydrologic_check") or {}
        limitations.extend(policy.get("warnings") or diagnostic.get("warnings") or [])
        limitations.extend(diagnostic.get("limitations") or [])
    if rain_manifest.get("selection_mode") == "historical_max":
        limitations.append(
            "The historical maximum day is the sample maximum within the data period and is not a return-period design storm."
        )
    if rain_manifest.get("warning"):
        limitations.append(str(rain_manifest["warning"]))
    return ModelInputPackage(
        schema_version=SCHEMA_VERSION,
        package_id=package_dir.name,
        created_at=utc_now(),
        event=event,
        study_area=_study_area(boundary),
        files=files,
        qa=qa,
        provenance={
            "producer": "chongqing-spatial-mcp",
            "review_id": review.get("review_id"),
            "rainfall_package_id": rain_manifest.get("package_id"),
            "rainfall_station_count": len(rain_manifest.get("stations") or []),
            "rainfall_station_selection": rain_manifest.get("station_selection") or {},
            'rainfall_existing_grid_count':rain_manifest.get('existing_grid_count'),
            'rainfall_interpolated_station_count':rain_manifest.get('interpolated_station_count'),
            "selection_mode": rain_manifest.get("selection_mode"),
            "selection_rule": rain_manifest.get("selection_rule"),
            "source_note": rain_manifest.get("source_note"),
            "dem_crs": _crs_of(package_dir / "dem.tif"),
            "dem_processing": terrain_budget or {},
            "outlet": review.get("outlet"),
            "spatial_selection_mode": review.get("selection_mode") or "outlet_watershed",
            "domain_policy":(review.get("qa") or {}).get("modeling_check") or {},
            "administrative_area": review.get("administrative_area"),
            "watershed_qa": (review.get("qa") or {}),
        },
        limitations=limitations,
    )


def export_package(
    package_dir: Path,
    *,
    review: dict[str, Any],
    rain_manifest: dict[str, Any],
    settings: Settings = SETTINGS,
    terrain_budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``manifest.json`` beside the package files and archive a copy.

    The directory itself already holds only relative members, so it can be
    handed over as an ``exchange/exports`` package or zipped for transport.
    """
    package = build_manifest(
        package_dir,
        review=review,
        rain_manifest=rain_manifest,
        terrain_budget=terrain_budget,
        qa={"spatial": review.get("qa"), "rainfall": {
            "timestamp_count": rain_manifest.get("timestamp_count"),
            "stations": [item.get("station_key") for item in rain_manifest.get("stations", [])],
        }},
    )
    write_json(package_dir / "manifest.json", package.model_dump(mode="json"))

    exports_root = settings.exports_root
    exports_root.mkdir(parents=True, exist_ok=True)
    export_dir = exports_root / package.package_id
    if export_dir.exists():
        shutil.rmtree(export_dir)
    shutil.copytree(package_dir, export_dir)
    archive = exports_root / f"{package.package_id}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(export_dir.rglob("*")):
            if path.is_file():
                bundle.write(path, arcname=path.relative_to(export_dir).as_posix())
    return {
        "package_id": package.package_id,
        "schema_version": package.schema_version,
        "directory": str(export_dir),
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "files": [entry.relative_path for entry in package.files],
        "limitations": package.limitations,
    }
