import csv

import matplotlib.pyplot as plt
import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.geometry.polygon import orient
import shapefile

from watershed_hydro_mcp.core.swmm import build_inp as build
from watershed_hydro_mcp.core.stability.safety_factor import shape_geometry
from watershed_hydro_mcp.visualization.renderer import RenderContext, render_model_structure, render_rainfall_station_map
from watershed_hydro_mcp.qa import spatial_gate
from test_partition_qa import partition_project, save


def polygon_file(path):
    outer = orient(Polygon([(0, 0), (6, 0), (6, 6), (0, 6)],
                          [[(2, 2), (4, 2), (4, 4), (2, 4)]]), sign=-1)
    other = orient(Polygon([(8, 0), (10, 0), (10, 2), (8, 2)]), sign=-1)
    with shapefile.Writer(str(path), shapeType=shapefile.POLYGON) as writer:
        writer.field('name', 'C', 20)
        writer.poly([list(outer.exterior.coords), list(outer.interiors[0].coords), list(other.exterior.coords)])
        writer.record('D1')
    path.with_suffix('.prj').write_text(CRS.from_epsg(32649).to_wkt(), encoding='utf-8')
    return MultiPolygon([outer, other])


def test_one_hydrologic_record_remains_one_model_unit_with_holes_and_parts(tmp_path):
    path = tmp_path / 'subcatchments.shp'
    expected = polygon_file(path)
    records = build.read_polygon_records(path, CRS.from_epsg(32649))
    assert len(records) == 1
    assert shape(records[0]['geometry']).equals(expected)
    assert shape(records[0]['geometry']).area == 36
    with shapefile.Reader(str(path)) as reader:
        assert shape(shape_geometry(reader.shape(0))).equals(expected)


def test_raster_properties_include_all_parts_and_exclude_hole(tmp_path, monkeypatch):
    path = tmp_path / 'subcatchments.shp'
    expected = polygon_file(path)
    item = build.read_polygon_records(path, CRS.from_epsg(32649))[0]
    transform = from_origin(0, 6, 1, 1)
    mask = rasterize([(mapping(expected), 1)], out_shape=(6, 10), transform=transform).astype(bool)
    elevations = np.full((6, 10), 1000, dtype='float32')
    elevations[mask] = 10
    dem = tmp_path / 'dem.tif'
    with rasterio.open(dem, 'w', driver='GTiff', width=10, height=6, count=1,
                       dtype='float32', transform=transform, crs='EPSG:32649') as dst:
        dst.write(elevations, 1)
    monkeypatch.setattr(build, 'CFG', {'swmm': {'delineation_resolution_m': 1}})
    stats = build.raster_stats(item['points'], dem, item['geometry'])
    assert stats['mean_elev'] == 10
    assert mask.sum() == 36
    grid = np.where(mask, 1, 2).astype('int16')
    name, fraction = build.dominant_landuse_from_grid(item['points'], grid,
        {1: 'correct', 2: 'outside'}, transform, item['geometry'])
    # all_touched also touches the surrounding ring; it must never lose the other part.
    assert name == 'correct' and fraction > .5


@pytest.mark.parametrize('count,areas,sources,passed', [
    (3, [100, 100, 100], ['D1', 'D2', 'D3'], True),
    (4, [100, 100, 100, 1], ['D1', 'D2', 'D3', 'D3'], False),
    (3, [101, 100, 100], ['D1', 'D2', 'D3'], False),
    (3, [100, 100, 100], ['D1', 'D2', 'D2'], False),
])
def test_qa_rejects_duplicate_model_parts_area_and_parent_mapping(tmp_path, count, areas, sources, passed):
    directory, diagnostics = partition_project(tmp_path)
    save(directory, diagnostics)
    model = tmp_path / 'swmm/model'; model.mkdir(parents=True)
    (model / 'model.inp').write_text('[SUBCATCHMENTS]\n' + '\n'.join(
        f'S{i} G{i} O{i} {area:.6f} 5 10 1 0' for i, area in enumerate(areas)), encoding='utf-8')
    with (model / 'auto_swmm_parameter_review.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['source_name']); writer.writeheader()
        writer.writerows({'source_name': source} for source in sources)
    assert spatial_gate(tmp_path, {})['passed'] is passed


def test_structure_distinguishes_named_outfalls_from_rain_sources(tmp_path):
    model = tmp_path / 'swmm/model'; model.mkdir(parents=True)
    (model / 'model.inp').write_text('[SUBCATCHMENTS]\nS1 G1 exit 1 0 1 1 0\n'
        '[OUTFALLS]\nexit 10 FREE NO\n[COORDINATES]\nexit 500000 3300000\n'
        '[RAINGAGES]\nG1 VOLUME 1:00 1 TIMESERIES TS1\n', encoding='utf-8')
    dem_dir = tmp_path / 'inputs/user_data'; dem_dir.mkdir(parents=True)
    with rasterio.open(dem_dir / 'dem.tif', 'w', driver='GTiff', width=2, height=2, count=1,
                       dtype='float32', crs='EPSG:32649', transform=from_origin(500000, 3300000, 1, 1)) as dst:
        dst.write(np.ones((2, 2), dtype='float32'), 1)
    rain = tmp_path / 'inputs/rainfall'; rain.mkdir(parents=True)
    (rain / 'event_station_rainfall_summary.csv').write_text(
        'station_key,station_name,lon,lat,total_mm\nST01,ERA5 proxy,111,29.8,100\n', encoding='utf-8')
    result = render_model_structure(RenderContext(tmp_path, tmp_path, 'test', {}), {})
    axis = result['figure'].axes[0]
    labels = [text.get_text() for text in axis.get_legend().get_texts()]
    assert any('出水口' in label and '倒三角' in label for label in labels)
    assert any('降雨来源点' in label and '紫色圆点' in label for label in labels)
    assert len(axis.collections) == 2  # one named outlet, one source, no gage pretending to be a station
    assert '1 个雨源点' in axis.get_title() and '1 个分区插值雨量计' in axis.get_title()
    plt.close(result['figure'])


def test_rain_source_map_uses_event_total_and_same_marker(tmp_path):
    rain = tmp_path / 'inputs/rainfall'; rain.mkdir(parents=True)
    (rain / 'event_station_rainfall_summary.csv').write_text(
        'station_key,lon,lat,total_mm\nST01,111,29.8,100\n', encoding='utf-8')
    (rain / 'event_station_observations_long.csv').write_text(
        'station_key,lon,lat,rain_mm\nST01,111,29.8,0\n', encoding='utf-8')
    result = render_rainfall_station_map(RenderContext(tmp_path, tmp_path, 'test', {}), {})
    axis = result['figure'].axes[0]
    assert any('100.0 mm' in text.get_text() for text in axis.texts)
    assert any('紫色圆点' in text.get_text() for text in axis.get_legend().get_texts())
    plt.close(result['figure'])
