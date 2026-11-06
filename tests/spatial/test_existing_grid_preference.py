"""Source-grid preference, edge policy, deficits and evidence stay explicit."""
import json
from pathlib import Path

import pytest
from shapely.geometry import box,mapping,Point,Polygon,MultiPolygon

from chongqing_spatial_mcp import rainfall
from chongqing_spatial_mcp.models import RainfallRequest
from chongqing_spatial_mcp.settings import Settings
from slope_evaluation_agent.response_writer import describe_rainfall_package
from slope_evaluation_agent.scientific_policy import numeric_audit


def boundary(tmp_path,geometry):
    path=tmp_path/'basin.geojson'
    path.write_text(json.dumps(dict(type='FeatureCollection',features=[dict(type='Feature',geometry=mapping(geometry))])),encoding='utf-8')
    return path


def test_four_existing_grids_then_one_virtual_without_duplicate_positions(tmp_path):
    area=box(105.8444,29.3454,106.039,29.5362)
    path=boundary(tmp_path,area)
    header=['time','105.75_29.5','106_29.5','105.75_29.25','106_29.25']
    selected=rainfall.select_virtual_stations(path,header,5)
    original=[s for s in selected if s['selection_basis']=='existing_era5_grid']
    assert len(original)==4 and sum(not s['source_inside'] for s in original)==3
    assert sum(s['placement_basis']=='inside_overlap_proxy' for s in original)==3
    assert all(area.contains(Point(s['lon'],s['lat'])) for s in selected)
    assert len({(s['lon'],s['lat']) for s in selected})==5
    assert selected[-1]['inside'] and selected[-1]['selection_basis']=='inside_basin_bilinear_interpolation'
    assert selected[-1]['rainfall_value_method']=='era5_interpolation'
    assert len(selected[-1]['source_weights'])>1
    assert rainfall._station_value(['time','1','2','3','4'],selected[-1])==pytest.approx(
        sum(w['column']*w['weight'] for w in selected[-1]['source_weights']))
    for source in original:
        assert len(source['source_weights'])==1 and source['source_weights'][0]['weight']==1
        assert source['rainfall_value_method']=='source_column'
        column=source['source_weights'][0]['column']
        assert rainfall._station_value(['time','1','2','3','4'],source)==column
    options=Settings(root=tmp_path,data_root=tmp_path,runs_root=tmp_path/'runs',exchange_root=tmp_path/'exchange',native_root=tmp_path/'native')
    policy=rainfall.resolve_station_count(RainfallRequest(review_id='basin-abcdef123456',mode='historical_max'),path,options)
    assert policy['area_per_station_km2']==80 and policy['station_count']==5


def test_two_interior_sources_exclude_outside_edge_sources(tmp_path):
    path=boundary(tmp_path,box(-.1,-.1,1.1,.1))
    selected=rainfall.select_virtual_stations(path,['time','0_0','1_0','2_0','0_1','1_1','2_1'],5)
    assert sum(s['selection_basis']=='existing_era5_grid' for s in selected)==2
    assert all(s['inside'] for s in selected)


@pytest.mark.parametrize('headers',[
    ['time','0_0','0_0','1_1'],['time','nan_0','1_1']])
def test_invalid_or_duplicated_source_coordinates_are_rejected(tmp_path,headers):
    with pytest.raises(ValueError,match='重复或无效'):
        rainfall.select_virtual_stations(boundary(tmp_path,box(.1,.1,.2,.2)),headers,3)


@pytest.mark.parametrize('area',[
    Polygon(box(.1,.1,.4,.4).exterior.coords,[box(.2,.2,.3,.3).exterior.coords]),
    MultiPolygon([box(.1,.1,.2,.2),box(.6,.6,.7,.7)])])
def test_global_rule_places_all_points_inside_holes_and_multipart_domains(tmp_path,area):
    selected=rainfall.select_virtual_stations(boundary(tmp_path,area),['time','0_0','1_0','0_1','1_1'],5)
    assert len(selected)==5 and all(area.contains(Point(s['lon'],s['lat'])) for s in selected)
    assert len({(s['lon'],s['lat']) for s in selected})==5
    for station in selected:
        if station['selection_basis']=='existing_era5_grid':
            source=station['source_weights'][0]
            assert (source['lon'],source['lat'])==(station['source_lon'],station['source_lat'])
            assert source['weight']==1
        else:
            assert station['rainfall_value_method']=='era5_interpolation'


def test_package_reply_is_audited_with_source_counts(tmp_path,monkeypatch):
    path=boundary(tmp_path,box(105.8444,29.3454,106.039,29.5362))
    rain=tmp_path/'rain.csv'
    rain.write_text('time,105.75_29.5,106_29.5,105.75_29.25,106_29.25\n2024-07-01 00:00,1,2,3,4\n2024-07-01 01:00,2,3,4,5\n',encoding='utf-8')
    monkeypatch.setattr(rainfall,'rainfall_files',lambda s:{2024:rain})
    monkeypatch.setattr(rainfall,'require_confirmed',lambda *a:{'artifacts':{'boundary_geojson':str(path)}})
    options=Settings(root=tmp_path,data_root=tmp_path,runs_root=tmp_path/'runs',exchange_root=tmp_path/'exchange',native_root=tmp_path/'native')
    package=rainfall.build_package(RainfallRequest(review_id='basin-abcdef123456',mode='explicit_period',
        start_datetime='2024-07-01 00:00',end_datetime='2024-07-01 01:00'),options)
    assert package['existing_grid_count']==4 and package['interpolated_station_count']==1
    reply=describe_rainfall_package(package)
    assert numeric_audit(reply,{'rainfall_package':package})==[]
