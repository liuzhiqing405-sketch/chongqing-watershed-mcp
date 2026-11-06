import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely import contains_xy
from shapely.geometry import Polygon, MultiPolygon, box, mapping
from shapely.ops import transform as project

from chongqing_spatial_mcp.administrative_dem import district_feature, clip_district_dem
from chongqing_spatial_mcp.coordinate_conversion import gcj02_to_wgs84, wgs84_to_gcj02
from chongqing_spatial_mcp.settings import Settings
from chongqing_spatial_mcp.study_area import area_map, _map


@pytest.fixture
def district_inputs(tmp_path):
    data=tmp_path/"data";data.mkdir()
    grid=from_origin(500000,3200000,30,30)
    values=np.arange(144,dtype="int16").reshape(12,12)+1
    values[2,2]=-9999
    with rasterio.open(data/"dem.tif","w",driver="GTiff",height=12,width=12,count=1,
                       dtype="int16",nodata=-9999,crs="EPSG:32649",transform=grid) as dst:
        dst.write(values,1)
    native=MultiPolygon([
        Polygon(box(500030,3199670,500240,3199970).exterior.coords,
                [box(500090,3199760,500150,3199850).exterior.coords]),
        box(500270,3199700,500330,3199790)])
    tx=Transformer.from_crs(32649,4326,always_xy=True)
    geometry=project(tx.transform,native)
    feature={"type":"Feature","properties":{"name":"Test区","adcode":123},"geometry":mapping(geometry)}
    (data/"administrative_districts.geojson").write_text(json.dumps({
        "coordinate_system":"EPSG:4326","features":[feature]}),encoding="utf-8")
    settings=Settings(root=tmp_path,data_root=data,runs_root=tmp_path/"runs",
                      exchange_root=tmp_path/"exchange",native_root=tmp_path/"native",
                      defaults={"datasets":{"dem":"dem.tif","rivers":"rivers.shp"}})
    return settings,feature,geometry,native


def test_native_cutline_preserves_islands_holes_values_resolution_and_nodata(district_inputs):
    settings,feature,geometry,native=district_inputs
    result=clip_district_dem(feature,geometry,settings)
    assert result["resolution_m"]==[30,30]
    assert result["missing_cells"]==1
    assert result["includes_full_administrative_extent"]
    with rasterio.open(result["artifacts"]["dem"]) as clipped, rasterio.open(settings.dem) as source:
        assert clipped.crs==source.crs and clipped.dtypes==source.dtypes
        assert clipped.res==source.res and clipped.nodata==source.nodata
        rows,cols=np.indices((clipped.height,clipped.width))
        x=clipped.transform.c+(cols+.5)*30;y=clipped.transform.f-(rows+.5)*30
        inside=contains_xy(native,x,y)
        values=clipped.read(1)
        original=source.read(1)[((3200000-y)/30).astype(int),((x-500000)/30).astype(int)]
        assert np.array_equal(values[inside],original[inside])
        assert np.all(values[~inside]==source.nodata)
        assert result["selected_cells"]==int(inside.sum())
        assert clipped.tags()["selection_method"]=="administrative_polygon"
    saved=json.loads(Path(result["artifacts"]["boundary_geojson"]).read_text(encoding="utf-8"))["features"][0]
    assert saved==json.loads(json.dumps(feature))


def test_cached_cutline_changes_when_source_dem_changes(district_inputs):
    settings,feature,geometry,_=district_inputs
    first=clip_district_dem(feature,geometry,settings)
    path=Path(first["artifacts"]["dem"]);mtime=path.stat().st_mtime_ns
    assert clip_district_dem(feature,geometry,settings)==first
    assert path.stat().st_mtime_ns==mtime
    with rasterio.open(settings.dem,"r+") as dst:
        values=dst.read(1);values[3,3]=1000;dst.write(values,1)
    second=clip_district_dem(feature,geometry,settings)
    assert second["artifacts"]["dem"]!=str(path)
    changed=feature.copy();changed["geometry"]=mapping(geometry.buffer(-.0001))
    third=clip_district_dem(changed,geometry.buffer(-.0001),settings)
    assert third["artifacts"]["dem"]!=second["artifacts"]["dem"]


def test_area_map_uses_the_saved_cutline_dem(district_inputs):
    settings,feature,geometry,_=district_inputs
    with patch("chongqing_spatial_mcp.study_area._map",return_value={"image_base64":"preview"}) as render:
        result=area_map("Test",settings)
    assert result["dem_clip"]["method"]=="administrative_polygon"
    assert render.call_args.kwargs["dem_path"]==result["dem_clip"]["artifacts"]["dem"]
    assert render.call_args.kwargs["administrative_boundary"].equals(geometry)
    assert "区外留白" in result["message"]


def test_rendered_dem_has_no_bilinear_bleed_outside_the_boundary(district_inputs):
    from matplotlib.axes import Axes
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds
    settings,feature,geometry,_=district_inputs
    result=clip_district_dem(feature,geometry,settings)
    captured=[]
    original=Axes.imshow
    def capture(axis,values,*args,**kwargs):
        captured.append(values.copy())
        return original(axis,values,*args,**kwargs)
    extent=geometry.bounds
    with patch.object(Axes,"imshow",capture),patch("workers.gis_worker._river_layer_crs",return_value="EPSG:4326"),patch("workers.gis_worker._all_river_rings",return_value=[]):
        rendered=_map(extent,"Test",settings,dem_path=result["artifacts"]["dem"],administrative_boundary=geometry)
    values=captured[0];height,width=values.shape
    outside=geometry_mask([mapping(geometry)],values.shape,from_bounds(*extent,width,height))
    assert np.isnan(values[outside]).all()
    assert np.isfinite(values[~outside]).any()
    assert rendered["display_crs"]=="EPSG:4326"


def test_unknown_coordinate_system_is_not_used_as_a_cutline(district_inputs):
    settings,_,_,_=district_inputs
    path=settings.data_root/"administrative_districts.geojson"
    data=json.loads(path.read_text());data.pop("coordinate_system")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="WGS84"):
        district_feature("Test区",settings)


def test_dem_outside_district_is_rejected_and_partial_coverage_is_reported(district_inputs):
    settings,feature,_,_=district_inputs
    with pytest.raises(ValueError,match="DEM 数据范围"):
        clip_district_dem(feature,box(100,20,101,21),settings)
    # Fingerprint follows the actual feature used for the cutline.
    tx=Transformer.from_crs(32649,4326,always_xy=True)
    geometry=project(tx.transform,box(499900,3199700,500150,3200100))
    feature=dict(feature,geometry=mapping(geometry))
    result=clip_district_dem(feature,geometry,settings)
    assert not result["includes_full_administrative_extent"]
    assert result["valid_dem_area_km2"]<result["administrative_area_km2"]


def test_coordinate_conversion_known_point_and_round_trip():
    lon,lat=wgs84_to_gcj02(116.397128,39.916527)
    assert float(lon)==pytest.approx(116.403372494,abs=1e-9)
    assert float(lat)==pytest.approx(39.917930749,abs=1e-9)
    x,y=gcj02_to_wgs84(lon,lat)
    assert float(x)==pytest.approx(116.397128,abs=1e-9)
    assert float(y)==pytest.approx(39.916527,abs=1e-9)
    assert tuple(float(v) for v in gcj02_to_wgs84(-.1,51.5))==(-.1,51.5)
