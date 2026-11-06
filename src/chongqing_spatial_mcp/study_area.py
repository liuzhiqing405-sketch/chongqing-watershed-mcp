"""Administrative DEM maps and user-drawn study domains."""
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
import rasterio
import shapefile
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize
from rasterio.mask import mask
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import transform as project

from .settings import SETTINGS, Settings
from .state import review_dir, utc_now, write_json
from .administrative_dem import boundary_data, district_feature, clip_district_dem


def districts(settings: Settings = SETTINGS) -> dict:
    data = boundary_data(settings)
    return {"areas": [{"name": f["properties"]["name"],
                       "code": str(f["properties"]["adcode"])} for f in data["features"]],
            "reference_only": True,"dem_clip_method":"administrative_polygon"}


def _district(name: str, settings: Settings):
    feature,geometry=district_feature(name,settings)
    return feature["properties"]["name"],geometry


def _map(extent, label: str, settings: Settings, boundary=None, *, dem_path=None, administrative_boundary=None,
         hydrologic_check=None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from workers.gis_worker import _configure_fonts, _all_river_rings, _river_layer_crs
    _configure_fonts(plt)
    west,south,east,north = extent
    width = 1200
    height = max(200,min(1400,round(width*(north-south)/((east-west)*math.cos(math.radians((north+south)/2))))))
    grid = from_bounds(west,south,east,north,width,height)
    dem = np.full((height,width),np.nan,dtype="float32")
    with rasterio.open(dem_path or settings.dem) as source:
        if source.crs is None:
            raise ValueError("DEM 缺少坐标系")
        reproject(source=rasterio.band(source,1),destination=dem,src_transform=source.transform,
                  src_crs=source.crs,src_nodata=source.nodata,dst_transform=grid,
                  dst_crs="EPSG:4326",dst_nodata=np.nan,resampling=Resampling.bilinear)
    if administrative_boundary is not None:
        selected=geometry_mask([mapping(administrative_boundary)],dem.shape,grid,invert=True)
        dem[~selected]=np.nan
    if not np.isfinite(dem).any():
        raise ValueError("该区县图幅没有有效 DEM 数据")
    figure,axis = plt.subplots(figsize=(10,8),layout="constrained")
    image = axis.imshow(dem,extent=(west,east,south,north),origin="upper",cmap="terrain")
    figure.colorbar(image,ax=axis,shrink=.75,label="高程 (m)")
    layer = _river_layer_crs(settings.rivers)
    to_layer = Transformer.from_crs(4326,layer,always_xy=True)
    corners = [to_layer.transform(x,y) for x in (west,east) for y in (south,north)]
    to_wgs = Transformer.from_crs(layer,4326,always_xy=True)
    for ring in _all_river_rings(settings.rivers,min(x for x,y in corners),min(y for x,y in corners),
                                 max(x for x,y in corners),max(y for x,y in corners)):
        xs,ys = zip(*[to_wgs.transform(x,y) for x,y in ring])
        axis.fill(xs,ys,color="#2477a8",alpha=.6,linewidth=0)
    legend_handles=[]
    if settings.display_rivers is not None:
        from .map_hydrography import draw_reference
        legend_handles.append(draw_reference(axis,settings.display_rivers,extent,
                                             settings.display_rivers_label,width,height))
    if administrative_boundary is not None:
        from matplotlib.path import Path as PlotPath
        from matplotlib.patches import PathPatch
        from matplotlib.lines import Line2D
        from shapely.geometry.polygon import orient
        polygons=([administrative_boundary] if administrative_boundary.geom_type=="Polygon"
                  else administrative_boundary.geoms)
        paths=[]
        for polygon in polygons:
            polygon=orient(polygon,sign=1.)
            for ring in [polygon.exterior,*polygon.interiors]:
                coordinates=np.asarray(ring.coords)
                codes=np.full(len(coordinates),PlotPath.LINETO,dtype="uint8")
                codes[0]=PlotPath.MOVETO;codes[-1]=PlotPath.CLOSEPOLY
                paths.append(PlotPath(coordinates,codes))
        patch=PathPatch(PlotPath.make_compound_path(*paths),transform=axis.transData,
                        facecolor="none",edgecolor="#27364b",linewidth=1.2,zorder=5)
        # Both the water polygons and the HydroRIVERS collections stop at the
        # cutline, including islands and holes; the legend remains intact.
        for artist in [*axis.patches,*axis.collections]:
            artist.set_clip_path(patch)
        axis.add_patch(patch)
        legend_handles.append(Line2D([],[],color="#27364b",linewidth=1.2,label="行政区边界"))
    if boundary is not None:
        xs,ys = boundary.exterior.xy
        boundary_line,=axis.plot(xs,ys,color="#ed5e20",linewidth=2,
                                 label="用户绘制评价边界",zorder=6)
        legend_handles.append(boundary_line)
    if hydrologic_check:
        for key, marker, color, caption in [("outlets", "*", "#111827", "DEM 主要出流位置（待核对）"),
                                            ("inlets", "^", "#d91b16", "框外汇水进入位置")]:
            points = hydrologic_check.get(key) or []
            if points:
                artist=axis.scatter([p["longitude"] for p in points], [p["latitude"] for p in points],
                                    marker=marker,c=color,s=100,edgecolors="white",linewidths=.8,
                                    label=caption,zorder=8)
                legend_handles.append(artist)
    if legend_handles:
        if hydrologic_check:
            axis.legend(handles=legend_handles,loc="upper center",bbox_to_anchor=(.5,-.12),ncol=2,framealpha=.95)
        else:
            axis.legend(handles=legend_handles,loc="upper right",framealpha=.95)
    axis.set(xlim=(west,east),ylim=(south,north),xlabel="经度 (°E)",ylabel="纬度 (°N)",title=label)
    axis.ticklabel_format(useOffset=False,style="plain")
    figure.set_dpi(135)
    figure.canvas.draw()
    position=axis.get_position()
    stream=io.BytesIO();figure.savefig(stream,format="png",dpi=135)
    result={"image_base64":base64.b64encode(stream.getvalue()).decode("ascii"),
            "media_type":"image/png","bbox_wgs84":list(extent),"display_crs":"EPSG:4326",
            "axes_rect":[position.x0,position.y0,position.width,position.height],
            "image_width":int(figure.bbox.width),"image_height":int(figure.bbox.height)}
    plt.close(figure)
    return result


def area_map(administrative_area: str, settings: Settings = SETTINGS) -> dict:
    feature,geometry=district_feature(administrative_area,settings)
    name=feature["properties"]["name"]
    clipped=clip_district_dem(feature,geometry,settings)
    west,south,east,north=geometry.bounds
    padx,pady=(east-west)*.03,(north-south)*.03
    message="DEM 已按行政区边界裁切，区外留白。请沿山脊或分水岭逐点绘制闭合评价边界。"
    if clipped["missing_cells"] or not clipped["includes_full_administrative_extent"]:
        message+="当前源 DEM 在区内有数据缺口，空白区域没有可用地形，提交选区时会核对覆盖。"
    return {"administrative_area":name,"selection_mode":"drawn_study_area",
            "dem_clip":clipped,
            "message":message,
            "map":_map((west-padx,south-pady,east+padx,north+pady),name+" · 行政边界裁切 DEM",settings,
                       dem_path=clipped["artifacts"]["dem"],administrative_boundary=geometry)}


def _landuse_mask(settings: Settings, transform, crs, size, bounds):
    from workers.gis_worker import _layer_crs
    layer_crs = _layer_crs(settings.landuse)
    if layer_crs is None:
        raise ValueError("土地利用图层缺少坐标系，无法核对覆盖范围")
    to_layer=Transformer.from_crs(crs,layer_crs,always_xy=True)
    corners=[to_layer.transform(x,y) for x in (bounds.left,bounds.right) for y in (bounds.bottom,bounds.top)]
    bbox=(min(x for x,y in corners),min(y for x,y in corners),max(x for x,y in corners),max(y for x,y in corners))
    tx=Transformer.from_crs(layer_crs,crs,always_xy=True)
    geometries=[]
    reader=shapefile.Reader(str(settings.landuse))
    try:
        for item in reader.iterShapes(bbox=bbox):
            geometry=shape(item.__geo_interface__)
            if layer_crs != crs:
                geometry=project(tx.transform,geometry)
            if not geometry.is_empty:
                geometries.append((mapping(geometry),1))
    finally:
        reader.close()
    return (rasterize(geometries,out_shape=size,transform=transform,dtype="uint8").astype(bool)
            if geometries else np.zeros(size,dtype=bool))


def create_area_review(administrative_area: str | None, coordinates: list[list[float]],
                       settings: Settings = SETTINGS, *, location_label: str | None = None,
                       assessment_mode: str = "regional_screening") -> dict:
    from watershed_exchange.domain_policy import ASSESSMENT_MODES, modeling_check
    if assessment_mode not in ASSESSMENT_MODES:
        raise ValueError("未知评价方式")
    name=(_district(administrative_area,settings)[0] if administrative_area else location_label or "自选评价区域")
    if not 3 <= len(coordinates) <= 500:
        raise ValueError("评价边界需包含 3 到 500 个顶点")
    for point in coordinates:
        if (len(point)!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in point)
                or not -180<=point[0]<=180 or not -90<=point[1]<=90):
            raise ValueError("边界顶点必须是有效的 WGS84 经度、纬度")
    polygon=Polygon(coordinates)
    if not polygon.is_valid or polygon.is_empty or polygon.area==0:
        raise ValueError("边界不能自交或退化，请撤销交叉线段后重新闭合")
    review_id="basin-"+uuid4().hex[:12]
    output=review_dir(review_id,settings)
    with rasterio.open(settings.dem) as source:
        crs=source.crs
        if crs is None or not crs.is_projected:
            raise ValueError("计算 DEM 必须使用投影坐标系")
        tx=Transformer.from_crs(4326,crs,always_xy=True)
        projected=project(tx.transform,polygon)
        from shapely.geometry import box
        if not box(*source.bounds).covers(projected):
            raise ValueError("所画范围超出 DEM 覆盖范围，请缩小选区或补齐地形数据")
        area_km2=projected.area/1e6
        from .dem_budget import preflight_geometry
        budget = preflight_geometry(source, projected, settings)
        data,transform=mask(source,[mapping(projected)],crop=True,pad=True,pad_width=1,filled=False)
        profile=source.profile.copy()
    values=data[0].astype("float32")
    selected=geometry_mask([mapping(projected)],values.shape,transform,invert=True)
    valid=selected & ~np.ma.getmaskarray(values) & np.isfinite(values.filled(np.nan))
    from rasterio.coords import BoundingBox
    bounds=BoundingBox(*rasterio.transform.array_bounds(*values.shape,transform))
    landuse=_landuse_mask(settings,transform,crs,values.shape,bounds)
    covered=valid & landuse
    count=int(selected.sum()); valid_count=int(valid.sum()); covered_count=int(covered.sum())
    if count==0:
        raise ValueError("选区不足一个 DEM 栅格，请重新绘制")
    errors=[]
    if area_km2 < float(settings.delineation.get("minimum_area_km2",1.8)):
        errors.append("选区面积低于当前区域建模下限，请扩大边界")
    if valid_count != count:
        errors.append("所画区域存在 DEM 无数据缺口，请调整边界或补齐数据")
    if covered_count != valid_count:
        errors.append("所画区域的土地利用覆盖不完整，请调整边界或补齐数据")
    from .hydrologic_check import check_polygon, unavailable_check
    hydrology=(check_polygon(settings.dem,projected,settings.defaults.get("drawn_area_hydrology") or {})
               if valid_count == count else unavailable_check("选区地形不完整，未进行水文检查。",fatal=True))
    policy=modeling_check(hydrology,assessment_mode)
    errors.extend(policy["errors"])
    output.mkdir(parents=True,exist_ok=False)
    profile.update(height=values.shape[0],width=values.shape[1],transform=transform,
                   count=1,dtype="float32",nodata=-9999,compress="lzw")
    # The source's full-width strip layout is invalid for a small crop. Let
    # GDAL choose a layout for this raster instead of carrying those sizes over.
    profile.pop("blockxsize",None)
    profile.pop("blockysize",None)
    dem_path=output/"dem_processing_window.tif"
    with rasterio.open(dem_path,"w",**profile) as dst:
        dst.write(values.filled(-9999),1)
    profile.update(dtype="uint8",nodata=0)
    basin_path=output/"basin.tif"
    with rasterio.open(basin_path,"w",**profile) as dst:
        dst.write(selected.astype("uint8"),1)
    boundary_path=output/"basin.geojson"
    write_json(boundary_path,{"type":"FeatureCollection","features":[{
        "type":"Feature","properties":{"selection_mode":"drawn_study_area","assessment_mode":assessment_mode},"geometry":mapping(polygon)}]})
    diagnostic_path=output/"hydrologic_check.json"
    write_json(diagnostic_path,hydrology)
    locations_path=output/"hydrologic_locations.geojson"
    write_json(locations_path,{"type":"FeatureCollection","features":[
        {"type":"Feature","properties":{"kind":kind,**point},
         "geometry":{"type":"Point","coordinates":[point["longitude"],point["latitude"]]}}
        for kind in ("outlets","inlets") for point in hydrology[kind]]})
    west,south,east,north=polygon.bounds
    preview=_map((west-(east-west)*.04,south-(north-south)*.04,
                  east+(east-west)*.04,north+(north-south)*.04),"用户绘制评价区域 · "+name,settings,polygon,
                  hydrologic_check=hydrology)
    preview_path=output/"study_area.png"
    preview_path.write_bytes(base64.b64decode(preview["image_base64"]))
    record={"review_id":review_id,"selection_mode":"drawn_study_area",
            "assessment_mode":assessment_mode,
            "administrative_area":name if administrative_area else None,"location_label":name,
            "created_at":utc_now(),"status":"awaiting_human_confirmation" if not errors else "qa_failed",
            "confirmation_required":not errors,"confirmed":False,"outlet":{},
            "qa":{"passed":not errors,"errors":errors,"hydrologic_check":hydrology,"modeling_check":policy,"notes":[
                "这是用户绘制的评价范围，尚未证明它是完整的自然流域；区域内部将在建模阶段按 DEM 划分子汇水区。",
                *policy["boundary_assumptions"],*policy["warnings"],*hydrology.get("limitations",[])],
                "area_km2":round(area_km2,4),"selected_cells":count,
                "calculation_area_km2":round(covered_count*abs(transform.a*transform.e)/1e6,4),
                "dem_coverage_fraction":valid_count/count,"landuse_coverage_fraction":covered_count/count},
            "processing":{"method":"user_drawn_polygon","resolution_m":abs(transform.a),
                          "dem_preflight":budget},
            "artifacts":{"original_dem":str(dem_path),"basin_raster":str(basin_path),
                         "boundary_geojson":str(boundary_path),"study_area_preview":str(preview_path),
                         "hydrologic_check":str(diagnostic_path),"hydrologic_locations":str(locations_path)}}
    write_json(output/"review.json",record)
    return record
