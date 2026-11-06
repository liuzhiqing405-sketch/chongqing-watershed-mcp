import base64
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
import shapefile
from pyproj import CRS, Transformer
from rasterio.transform import from_origin

from chongqing_spatial_mcp.settings import Settings
from chongqing_spatial_mcp.state import require_confirmed, require_spatial_quality, write_json
from chongqing_spatial_mcp.study_area import create_area_review
from chongqing_spatial_mcp.workflow import confirm, prepare_inputs


@pytest.fixture
def area_inputs(tmp_path):
    data=tmp_path/"data";data.mkdir()
    crs=CRS.from_epsg(32649);transform=from_origin(500000,3200000,30,30)
    with rasterio.open(data/"dem.tif","w",driver="GTiff",height=120,width=120,count=1,
                       dtype="float32",nodata=-9999,crs=crs,transform=transform) as dst:
        # Ridges around the top and sides drain outside terrain away from the
        # selection; the valley inside slopes toward its southern cutline.
        rows,cols=np.indices((120,120))
        cross=np.where(cols<10,cols+31,np.where(cols>=90,130-cols,abs(cols-49.5)))
        along=np.where(rows<10,80+rows,100-rows)
        dst.write((4*cross+along).astype("float32"),1)
    for filename in ["landuse","rivers"]:
        with shapefile.Writer(str(data/(filename+".shp"))) as writer:
            writer.field("Name","C")
            writer.poly([[(500000,3196400),(500000,3200000),(503600,3200000),
                          (503600,3196400),(500000,3196400)]])
            writer.record("Test")
        (data/(filename+".prj")).write_text(crs.to_wkt(),encoding="utf-8")
    tx=Transformer.from_crs(crs,4326,always_xy=True)
    coordinates=[list(tx.transform(x,y)) for x,y in
                 [(500300,3197300),(502700,3197300),(502700,3199700),(500300,3199700)]]
    (data/"administrative_districts.geojson").write_text(json.dumps({"coordinate_system":"EPSG:4326","features":[{
        "properties":{"name":"Test区","adcode":"test"},
        "geometry":{"type":"Polygon","coordinates":[coordinates+[coordinates[0]]]}}]}),encoding="utf-8")
    settings=Settings(root=tmp_path,data_root=data,runs_root=tmp_path/"runs",exchange_root=tmp_path/"exchange",
                      native_root=tmp_path/"native",defaults={"datasets":{"dem":"dem.tif",
                      "landuse":"landuse.shp","rivers":"rivers.shp"},"delineation":{"minimum_area_km2":1.8}})
    with patch("chongqing_spatial_mcp.study_area._map",return_value={"image_base64":base64.b64encode(b"preview").decode()}):
        yield settings,coordinates


def test_user_boundary_is_preserved_and_confirmation_is_required(area_inputs):
    settings,coordinates=area_inputs
    record=create_area_review("Test区",coordinates,settings)
    assert record["qa"]["passed"] and record["confirmation_required"]
    assert record["qa"]["area_km2"]==pytest.approx(5.76,abs=.001)
    assert record["outlet"]=={} and record["selection_mode"]=="drawn_study_area"
    assert record["qa"]["hydrologic_check"]["checked"]
    assert record["qa"]["hydrologic_check"]["passed"]
    geometry=json.loads(Path(record["artifacts"]["boundary_geojson"]).read_text())["features"][0]["geometry"]
    assert geometry["coordinates"][0]==coordinates+[coordinates[0]]
    with pytest.raises(ValueError,match="尚未人工确认"):
        require_confirmed(record["review_id"],settings)
    confirm(record["review_id"],True,"confirm",settings)
    assert require_confirmed(record["review_id"],settings)["confirmed"]


def test_self_intersection_does_not_become_a_different_polygon(area_inputs):
    settings,coordinates=area_inputs
    with pytest.raises(ValueError,match="自交"):
        create_area_review("Test区",[coordinates[i] for i in [0,2,1,3]],settings)
    assert not settings.watersheds_root.exists()


@pytest.mark.parametrize("point",[[float("nan"),29],[105,91],["bad",29],[105]])
def test_invalid_coordinates_rejected_before_writing(area_inputs,point):
    settings,coordinates=area_inputs
    with pytest.raises(ValueError,match="顶点"):
        create_area_review("Test区",[point,*coordinates[1:]],settings)


def test_dem_gap_blocks_confirmation(area_inputs):
    settings,coordinates=area_inputs
    with rasterio.open(settings.dem,"r+") as dst:
        data=dst.read(1);data[40,40]=-9999;dst.write(data,1)
    record=create_area_review("Test区",coordinates,settings)
    assert record["status"]=="qa_failed"
    assert record["qa"]["dem_coverage_fraction"]<1
    with pytest.raises(ValueError,match="空间QA"):
        confirm(record["review_id"],True,"confirm",settings)


def test_landuse_gap_blocks_confirmation(area_inputs):
    settings,coordinates=area_inputs
    with shapefile.Writer(str(settings.landuse)) as writer:
        writer.field("Name","C")
    record=create_area_review("Test区",coordinates,settings)
    assert not record["qa"]["passed"]
    assert any("土地利用" in error for error in record["qa"]["errors"])


def test_prepared_dem_uses_only_the_user_boundary(area_inputs):
    settings,coordinates=area_inputs
    record=create_area_review("Test区",coordinates,settings)
    confirm(record["review_id"],True,"confirm",settings)
    rain_dir=settings.rainfall_runs_root/"rain-test123";rain_dir.mkdir(parents=True)
    archive=rain_dir/"stations.zip";archive.write_bytes(b"test")
    write_json(rain_dir/"rainfall_manifest.json",{"review_id":record["review_id"],"station_zip":str(archive)})
    result=prepare_inputs(record["review_id"],"rain-test123",settings,export=False)
    with rasterio.open(Path(result["directory"])/"dem.tif") as dem:
        assert np.count_nonzero(~np.ma.getmaskarray(dem.read(1,masked=True)))==record["qa"]["selected_cells"]
        assert dem.width<120 and dem.height<120
    assert result["dem_preflight"]["cells"] == result["clipping"]["drainage_context"]["cells"]
    assert record["processing"]["dem_preflight"] == result["dem_preflight"]


def test_boundary_budget_includes_context_before_terrain_read_or_artifacts(area_inputs):
    settings, coordinates = area_inputs
    # The ~6,500-cell crop fits; its 14,400-cell context does not.
    settings.defaults["delineation"]["maximum_processing_cells"] = 10_000
    with patch("chongqing_spatial_mcp.study_area.mask") as read_terrain:
        with pytest.raises(ValueError, match="14,400.*10,000.*预算"):
            create_area_review("Test区", coordinates, settings)
        read_terrain.assert_not_called()
    assert not settings.watersheds_root.exists()


def test_existing_review_rechecks_changed_budget_before_creating_package(area_inputs):
    settings, coordinates = area_inputs
    record = create_area_review("Test区", coordinates, settings)
    confirm(record["review_id"], True, "confirm", settings)
    rain_dir = settings.rainfall_runs_root/"rain-budget"
    rain_dir.mkdir(parents=True)
    write_json(rain_dir/"rainfall_manifest.json", {"review_id": record["review_id"]})
    settings.defaults["model_inputs"] = {"drainage_context_maximum_cells": 10_000}
    with patch("chongqing_spatial_mcp.workflow._clip_raster") as clip:
        with pytest.raises(ValueError, match="预算"):
            prepare_inputs(record["review_id"], "rain-budget", settings, export=False)
        clip.assert_not_called()
    assert not settings.model_inputs_root.exists()


def test_historical_proximity_only_river_check_cannot_unlock_more_calculations():
    with pytest.raises(ValueError,match="旧划分"):
        require_spatial_quality({"qa":{"passed":True,"river_check":{"checked":True,"passed":True}}})


def test_river_or_coordinate_domain_does_not_require_an_administrative_name(area_inputs):
    settings,coordinates=area_inputs
    record=create_area_review(None,coordinates,settings,location_label="测试河")
    assert record["administrative_area"] is None
    assert record["location_label"]=="测试河"
    assert record["qa"]["passed"] and record["outlet"]=={}
    saved=json.loads(Path(record["artifacts"]["boundary_geojson"]).read_text(encoding="utf-8"))
    assert saved["features"][0]["geometry"]["coordinates"][0]==coordinates+[coordinates[0]]


def test_partial_region_keeps_diagnostic_failure_but_can_prepare_local_model(area_inputs):
    settings,coordinates=area_inputs
    tx=Transformer.from_crs(32649,4326,always_xy=True)
    cut=[list(tx.transform(x,y)) for x,y in
         [(500300,3197300),(502700,3197300),(502700,3198500),(500300,3198500)]]
    regional=create_area_review("Test区",cut,settings)
    strict=create_area_review("Test区",cut,settings,assessment_mode="catchment_check")
    for review in [regional,strict]:
        diagnostic=review["qa"]["hydrologic_check"]
        assert diagnostic["checked"] and not diagnostic["passed"]
        assert diagnostic["classification"]=="upstream_truncated"
        assert diagnostic["external_upstream_area_km2"]>0
        geometry=json.loads(Path(review["artifacts"]["boundary_geojson"]).read_text())["features"][0]["geometry"]
        assert geometry["coordinates"][0]==cut+[cut[0]]
        assert review["processing"]["resolution_m"]==30
    assert regional["qa"]["passed"] and regional["qa"]["modeling_check"]["passed"]
    assert not strict["qa"]["passed"]
    with pytest.raises(ValueError,match="空间QA"):
        confirm(strict["review_id"],True,"confirm",settings)
    confirm(regional["review_id"],True,"accept local rainfall assumption",settings)
    rain_dir=settings.rainfall_runs_root/"rain-partial";rain_dir.mkdir(parents=True)
    archive=rain_dir/"stations.zip";archive.write_bytes(b"test")
    write_json(rain_dir/"rainfall_manifest.json",{"review_id":regional["review_id"],"station_zip":str(archive)})
    result=prepare_inputs(regional["review_id"],"rain-partial",settings,export=False)
    with rasterio.open(Path(result["directory"])/"dem.tif") as dem:
        assert np.count_nonzero(~np.ma.getmaskarray(dem.read(1,masked=True)))==regional["qa"]["selected_cells"]


@pytest.mark.parametrize("mode",["regional_screening","catchment_check"])
def test_model_budget_still_blocks_both_modes(area_inputs,mode):
    settings,coordinates=area_inputs
    settings.defaults["delineation"]["maximum_processing_cells"]=10
    with pytest.raises(ValueError,match="预算"):
        create_area_review("Test区",coordinates,settings,assessment_mode=mode)


def test_unfinished_diagnostic_is_a_regional_warning_but_not_a_strict_pass(area_inputs):
    settings,coordinates=area_inputs
    settings.defaults["drawn_area_hydrology"]={"maximum_cells":10}
    regional=create_area_review("Test区",coordinates,settings)
    strict=create_area_review("Test区",coordinates,settings,assessment_mode="catchment_check")
    assert not regional["qa"]["hydrologic_check"]["checked"]
    assert regional["qa"]["passed"] and regional["qa"]["modeling_check"]["warnings"]
    assert not strict["qa"]["passed"]
