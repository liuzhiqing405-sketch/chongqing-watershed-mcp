from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import rasterio
import shapefile
from pyproj import CRS, Transformer
from rasterio.mask import mask as raster_mask
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union

from watershed_exchange.paths import safe_id
from watershed_exchange.vector_io import attribute_encoding

from .exchange_export import build_manifest, export_package
from .dem_budget import drainage_window, preflight_geometry
from .settings import SETTINGS, Settings
from .state import read_json, review_dir, require_confirmed, require_spatial_quality, utc_now, write_json
from .visualization.catalog import requirements


#: Attributes of the managed vector layers are GBK encoded without a .cpg file.
VECTOR_ATTRIBUTE_ENCODING = "gbk"


def _run(command: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[-4000:])
    return result


def recommend(river_name: str, max_candidates: int = 5, settings: Settings = SETTINGS) -> dict[str, Any]:
    worker = settings.root / "workers" / "gis_worker.py"
    result = _run([
        sys.executable, str(worker), "recommend", "--dem", str(settings.dem),
        "--rivers", str(settings.rivers), "--name", river_name, "--limit", str(max_candidates),
        "--config", str(settings.config_path),
    ], timeout=300)
    return json.loads(result.stdout)


def outlet_map(river_name: str, max_candidates: int = 0,
               settings: Settings = SETTINGS, *, longitude: float | None = None,
               latitude: float | None = None) -> dict[str, Any]:
    """推荐出水口候选 + 一张可点选的选址底图（PNG 与地理范围一并返回）。

    与 recommend 共用同一套候选排序，只是多带一张图；前端据此把点击位置换算成经纬度。
    """
    worker = settings.root / "workers" / "gis_worker.py"
    if (longitude is None) != (latitude is None):
        raise ValueError("地图中心必须同时提供经度和纬度")
    centre_args = [] if longitude is None else ["--lon", str(longitude), "--lat", str(latitude)]
    overlay_args = [] if settings.display_rivers is None else [
        "--display-rivers",str(settings.display_rivers),
        "--display-rivers-label",settings.display_rivers_label]
    result = _run([
        sys.executable, str(worker), "outlet_map", "--dem", str(settings.dem),
        "--rivers", str(settings.rivers), "--name", river_name,
        "--limit", str(max_candidates),
        "--config", str(settings.config_path), *centre_args, *overlay_args,
    ], timeout=300)
    return json.loads(result.stdout)


def delineate(
    longitude: float,
    latitude: float,
    snap_radius_cells: int = 5,
    half_window_km: float = 20.0,
    settings: Settings = SETTINGS,
    *, river_name: str | None = None,
) -> dict[str, Any]:
    review_id = f"basin-{uuid4().hex[:12]}"
    output = review_dir(review_id, settings)
    output.mkdir(parents=True, exist_ok=False)
    worker = settings.root / "workers" / "gis_worker.py"
    try:
        result = _run([
            sys.executable, str(worker), "delineate", "--dem", str(settings.dem),
            "--config", str(settings.config_path),
            "--output", str(output),
            "--lon", str(longitude), "--lat", str(latitude),
            "--snap-radius", str(snap_radius_cells), "--half-window-km", str(half_window_km),
            *(["--rivers", str(settings.rivers), "--name", river_name] if river_name else []),
        ], timeout=1200)
        record = json.loads(result.stdout)
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    record.update({
        "review_id": review_id, "status": "awaiting_human_confirmation"
        if (record.get("qa") or {}).get("passed") else "qa_failed", "created_at": utc_now(),
        "confirmation_required": bool((record.get("qa") or {}).get("passed")),
        "review_instruction": "检查吸附点、偏移距离、流域边界、水系叠加及QA后再确认。",
    })
    write_json(output / "review.json", record)
    return record


def get_review(review_id: str, settings: Settings = SETTINGS) -> dict[str, Any]:
    return read_json(review_dir(review_id, settings) / "review.json")


def confirm(review_id: str, confirmed: bool, reviewer_note: str, settings: Settings = SETTINGS) -> dict[str, Any]:
    path = review_dir(review_id, settings) / "review.json"
    record = read_json(path)
    if record.get("status") not in {"awaiting_human_confirmation", "confirmed", "rejected", "qa_failed"}:
        raise ValueError("流域当前状态不能确认")
    if confirmed and not (record.get("qa") or {}).get("passed", False):
        raise ValueError("空间QA未通过，不能确认该范围；请核对检查结果，调整边界或补齐空间数据后重新提交")
    if confirmed:
        require_spatial_quality(record)
    record.update({
        "status": "confirmed" if confirmed else "rejected",
        "confirmed": bool(confirmed), "reviewer_note": reviewer_note.strip(), "reviewed_at": utc_now(),
    })
    write_json(path, record)
    return record


def render_artifact(
    review_id: str,
    artifact: str,
    settings: Settings = SETTINGS,
    rainfall_package_id: str | None = None,
    model_input_package_id: str | None = None,
) -> Path:
    folder = review_dir(review_id, settings)
    review = read_json(folder / "review.json")
    if review.get("selection_mode") == "drawn_study_area" and artifact == "watershed":
        preview = folder / "study_area.png"
        if not preview.is_file():
            raise ValueError("评价范围预览不存在，请重新提交边界")
        return preview
    requires = requirements(artifact)

    command = [sys.executable, str(settings.root / "workers" / "gis_worker.py"), "render",
               "--review-dir", str(folder), "--artifact", artifact,
               "--config", str(settings.config_path),
               "--visual-dir", str(settings.spatial_visualizations_root / safe_id(review_id))]
    if "rainfall" in requires:
        if not rainfall_package_id:
            raise ValueError(f"图件 {artifact} 需要降雨包，请提供 rainfall_package_id")
        rain_dir = settings.rainfall_runs_root / safe_id(rainfall_package_id)
        manifest = read_json(rain_dir / "rainfall_manifest.json")
        if manifest.get("review_id") != review_id:
            raise ValueError("降雨包与流域审核编号不一致")
        command += ["--rainfall-package", str(rain_dir)]
    if "model_input" in requires:
        if not model_input_package_id:
            raise ValueError(f"图件 {artifact} 需要模型输入包，请提供 model_input_package_id")
        input_dir = settings.model_inputs_root / safe_id(model_input_package_id)
        if not input_dir.is_dir():
            raise ValueError(f"模型输入包不存在：{model_input_package_id}")
        command += ["--model-input", str(input_dir)]

    result = _run(command, timeout=300)
    payload = json.loads(result.stdout)
    image = Path(payload["image"])
    if not image.is_file():
        raise RuntimeError("可视化工具未生成PNG")
    return image


def _basin_geometry(boundary_path: Path):
    data = json.loads(boundary_path.read_text(encoding="utf-8"))
    geometries = [
        shapely_shape(feature["geometry"])
        for feature in (data.get("features") or [])
        if feature.get("geometry")
    ]
    if not geometries:
        raise ValueError("流域边界没有有效几何")
    return unary_union(geometries)


def _reproject(geometry, target_crs, source_crs: CRS | None = None):
    """Move a WGS84 boundary into the layer's own CRS.

    The boundary is always exchanged in WGS84 while the raster and vector
    layers are projected, so every clip has to reproject the cutline first.
    """
    source = source_crs or CRS.from_epsg(4326)
    if source == target_crs:
        return geometry
    from shapely.ops import transform as shapely_transform

    transformer = Transformer.from_crs(source, target_crs, always_xy=True)
    return shapely_transform(lambda x, y, z=None: transformer.transform(x, y), geometry)


def _layer_crs(path: Path) -> CRS | None:
    """CRS of a shapefile or raster layer, read from its sidecar or header."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(path) as source:
            return CRS.from_user_input(source.crs) if source.crs else None
    prj = path.with_suffix(".prj")
    if not prj.is_file():
        return None
    try:
        return CRS.from_wkt(prj.read_text(encoding="utf-8", errors="ignore"))
    except Exception:  # noqa: BLE001 - an unreadable .prj means "unknown CRS"
        return None


def _clip_raster(source_path: Path, boundary_path: Path, destination: Path) -> None:
    geometry = _basin_geometry(boundary_path)
    with rasterio.open(source_path) as source:
        target_crs = CRS.from_user_input(source.crs) if source.crs else CRS.from_epsg(4326)
        clipped, transform = raster_mask(
            source, [_reproject(geometry, target_crs)], crop=True, nodata=-9999.0
        )
        profile = source.profile.copy()
    profile.update({
        "height": clipped.shape[1], "width": clipped.shape[2], "count": 1,
        "transform": transform, "nodata": -9999.0, "compress": "lzw",
        "dtype": clipped.dtype.name,
    })
    with rasterio.open(destination, "w", **profile) as target:
        target.write(clipped[0], 1)


def _clip_vector(source_path: Path, boundary_path: Path, destination: Path) -> dict[str, int]:
    target_crs = _layer_crs(source_path)
    geometry = _basin_geometry(boundary_path)
    if target_crs is not None:
        geometry = _reproject(geometry, target_crs)
    encoding = attribute_encoding(source_path)
    reader = shapefile.Reader(str(source_path), encoding=encoding)
    writer = shapefile.Writer(
        str(destination.with_suffix("")), encoding=encoding,
        shapeType=reader.shapeType,
    )
    for field in reader.fields[1:]:
        writer.field(*field)
    kept = 0
    read_count = 0
    try:
        for record in reader.iterShapeRecords():
            read_count += 1
            feature = shapely_shape(record.shape.__geo_interface__)
            if not feature.intersects(geometry):
                continue
            clipped = feature.intersection(geometry)
            if clipped.is_empty or clipped.geom_type not in {"Polygon", "MultiPolygon"}:
                continue
            writer.shape(clipped.__geo_interface__)
            writer.record(*record.record)
            kept += 1
    finally:
        reader.close()
        writer.close()
    source_crs = source_path.with_suffix(".prj")
    if source_crs.is_file():
        shutil.copy2(source_crs, destination.with_suffix(".prj"))
    # Declare the encoding so the exported package is unambiguous downstream.
    destination.with_suffix(".cpg").write_text(f"{encoding}\n", encoding="ascii")
    return {"read": read_count, "kept": kept}


def prepare_inputs(
    review_id: str,
    rainfall_package_id: str,
    settings: Settings = SETTINGS,
    export: bool = True,
) -> dict[str, Any]:
    review = require_confirmed(review_id, settings)
    rain_dir = settings.rainfall_runs_root / safe_id(rainfall_package_id)
    rain_manifest = read_json(rain_dir / "rainfall_manifest.json")
    if rain_manifest.get("review_id") != review_id:
        raise ValueError("降雨包与流域审核编号不一致")
    boundary = Path(review["artifacts"]["boundary_geojson"])
    with rasterio.open(settings.dem) as source:
        geometry = _reproject(_basin_geometry(boundary), source.crs)
        budget = preflight_geometry(source, geometry, settings)
    package_id = f"package-{uuid4().hex[:12]}"
    output = settings.model_inputs_root / package_id
    output.mkdir(parents=True, exist_ok=False)
    dem_out = output / "dem.tif"
    landuse_out = output / "landuse.shp"
    river_out = output / "river.shp"
    _clip_raster(settings.dem, boundary, dem_out)
    context = _export_drainage_context(settings.dem, dem_out, output / "dem_context.tif", settings)
    landuse_counts = _clip_vector(settings.landuse, boundary, landuse_out)
    river_counts = _clip_vector(settings.rivers, boundary, river_out)
    shutil.copy2(boundary, output / "basin.geojson")
    rain_zip = Path(rain_manifest["station_zip"])
    shutil.copy2(rain_zip, output / "station_rainfall.zip")
    result: dict[str, Any] = {
        "package_id": package_id, "status": "ready_for_hydro_preflight",
        "review_id": review_id, "rainfall_package_id": rainfall_package_id,
        "directory": str(output),
        "clipping": {"landuse": landuse_counts, "river": river_counts, "drainage_context": context},
        "dem_preflight": budget,
        "hydrology_artifacts": review["artifacts"],
        "limitations": [
            "In-basin virtual stations are interpolated from ERA5 grid points and are not measured gauges.",
            "Ready status means preprocessing completed; the hydro/model QA still has to run.",
        ],
    }
    if export:
        package = build_manifest(
            output,
            review=review,
            rain_manifest=rain_manifest,
            qa={"spatial": review.get("qa")},
            terrain_budget=context,
        )
        write_json(output / "manifest.json", package.model_dump(mode="json"))
        result["exchange"] = export_package(
            output, review=review, rain_manifest=rain_manifest, settings=settings, terrain_budget=context
        )
    write_json(output / "input_manifest.json", result)
    return result


def _export_drainage_context(source_path: Path, clipped_path: Path, destination: Path,
                             settings: Settings) -> dict[str, Any]:
    """Freeze aligned unmasked terrain; context is never rainfall area."""
    from rasterio.windows import Window
    with rasterio.open(clipped_path) as clipped, rasterio.open(source_path) as source:
        window, facts = drainage_window(source, clipped.bounds, settings)
        profile = source.profile.copy()
        profile.update(width=int(window.width), height=int(window.height), count=1,
                       transform=source.window_transform(window), compress="lzw")
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        with rasterio.open(destination, "w", **profile) as out:
            for row in range(0, int(window.height), 512):
                for col in range(0, int(window.width), 512):
                    tile = Window(col, row, min(512, int(window.width)-col), min(512, int(window.height)-row))
                    source_tile = Window(window.col_off+col, window.row_off+row, tile.width, tile.height)
                    out.write(source.read(1, window=source_tile), 1, window=tile)
                    out.write_mask(source.read_masks(1, window=source_tile), window=tile)
    return facts
