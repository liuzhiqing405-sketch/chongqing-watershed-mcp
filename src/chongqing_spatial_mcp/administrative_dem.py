"""Native-grid administrative cutlines, cached independently of model inputs."""
from __future__ import annotations

import hashlib
import json
from uuid import uuid4

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask, geometry_window
from rasterio.errors import WindowError
from rasterio.windows import Window
from shapely.geometry import mapping, shape
from shapely.ops import transform as project

from .settings import Settings
from .state import utc_now, write_json


def boundary_data(settings: Settings):
    path=settings.data_root/settings.datasets.get("administrative_boundaries","administrative_districts.geojson")
    if not path.is_file():
        raise ValueError("缺少区县边界数据，无法加载行政区 DEM")
    return json.loads(path.read_text(encoding="utf-8"))


def district_feature(name: str, settings: Settings):
    data=boundary_data(settings)
    if data.get("coordinate_system")!="EPSG:4326":
        raise ValueError("区县边界尚未转换为 WGS84，不能用于 DEM 裁切")
    matched=[f for f in data["features"] if name.strip() in {
        f["properties"]["name"], f["properties"]["name"].removesuffix("区").removesuffix("县")}]
    if len(matched)!=1:
        raise ValueError("请从可用区县列表选择一个行政区")
    feature=matched[0]
    geometry=shape(feature["geometry"])
    if geometry.geom_type not in {"Polygon","MultiPolygon"} or not geometry.is_valid or geometry.is_empty:
        raise ValueError("区县行政边界不是有效多边形，请更新边界数据")
    return feature,geometry


def clip_district_dem(feature: dict, geometry, settings: Settings):
    """Keep the native grid and values; write NoData outside the polygon.

    Block processing limits memory for large counties. The cache fingerprint
    changes with the actual cutline and the source DEM. A manifest is published
    only after the entire GeoTIFF has been written successfully.
    """
    stat=settings.dem.stat()
    fingerprint={"version":1,"source":str(settings.dem.resolve()),
                 "size":stat.st_size,"mtime_ns":stat.st_mtime_ns,"feature":feature,
                 "cutline":mapping(geometry)}
    digest=hashlib.sha256(json.dumps(fingerprint,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:20]
    output=settings.runs_root/"administrative_dem"/digest
    manifest=output/"manifest.json"
    if manifest.is_file() and (output/"dem.tif").is_file() and (output/"boundary.geojson").is_file():
        return json.loads(manifest.read_text(encoding="utf-8"))
    with rasterio.open(settings.dem) as source:
        if source.count!=1 or source.crs is None or not source.crs.is_projected:
            raise ValueError("区县 DEM 裁切要求单波段且具有投影坐标系的源 DEM")
        tx=Transformer.from_crs(4326,source.crs,always_xy=True)
        projected=project(tx.transform,geometry)
        try:
            window=geometry_window(source,[mapping(projected)])
        except WindowError as exc:
            raise ValueError("该区县不在现有 DEM 数据范围内") from exc
        transform=source.window_transform(window)
        width,height=int(window.width),int(window.height)
        profile=source.profile.copy()
        nodata=source.nodata
        if nodata is None:
            # A float destination represents NoData without sacrificing any
            # valid integer source value as a made-up sentinel.
            profile.update(dtype="float64")
            nodata=float("nan")
        profile.update(driver="GTiff",width=width,height=height,transform=transform,
                       nodata=nodata,compress="lzw",tiled=True,blockxsize=512,blockysize=512,
                       BIGTIFF="IF_SAFER")
        output.mkdir(parents=True,exist_ok=True)
        temporary=output/("dem-"+uuid4().hex+".tmp.tif")
        selected_count=valid_count=0
        try:
            with rasterio.open(temporary,"w",**profile) as dst:
                for _,block in dst.block_windows(1):
                    source_window=Window(window.col_off+block.col_off,window.row_off+block.row_off,
                                         block.width,block.height)
                    values=source.read(1,window=source_window,masked=True).astype(profile["dtype"])
                    inside=geometry_mask([mapping(projected)],values.shape,
                                         dst.window_transform(block),invert=True)
                    valid=inside & ~np.ma.getmaskarray(values) & np.isfinite(values.data)
                    selected_count+=int(inside.sum());valid_count+=int(valid.sum())
                    dst.write(np.where(valid,values.data,nodata),1,window=block)
                dst.update_tags(selection_method="administrative_polygon",
                                administrative_area=feature["properties"]["name"],
                                boundary_crs="EPSG:4326",source_dem=str(settings.dem))
            if not valid_count:
                raise ValueError("该区县内没有有效 DEM 数据")
            temporary.replace(output/"dem.tif")
        finally:
            temporary.unlink(missing_ok=True)
        write_json(output/"boundary.geojson",{"type":"FeatureCollection","features":[feature]})
        area=projected.area/1e6
        valid_area=valid_count*abs(transform.determinant)/1e6
        result={"administrative_area":feature["properties"]["name"],
                "code":str(feature["properties"]["adcode"]),"created_at":utc_now(),
                "method":"administrative_polygon","boundary_crs":"EPSG:4326",
                "dem_crs":source.crs.to_string(),"resolution_m":list(source.res),
                "size":[width,height],"administrative_area_km2":round(area,4),
                "valid_dem_area_km2":round(valid_area,4),"selected_cells":selected_count,
                "valid_cells":valid_count,"missing_cells":selected_count-valid_count,
                "dem_coverage_fraction":valid_count/selected_count,
                "includes_full_administrative_extent":window==geometry_window(source,[mapping(projected)],boundless=True),
                "artifacts":{"dem":str(output/"dem.tif"),"boundary_geojson":str(output/"boundary.geojson"),
                             "manifest":str(manifest)}}
        pending=output/("manifest-"+uuid4().hex+".tmp.json")
        write_json(pending,result)
        pending.replace(manifest)
        return result
