import csv
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.geometry.polygon import orient
import shapefile
from pyproj import CRS

from watershed_hydro_mcp.core.swmm import build_inp as build
from watershed_hydro_mcp.qa import model_outlet_check


def dem_file(path, values, nodata=-9999):
    values = np.asarray(values, dtype='float32')
    with rasterio.open(path, 'w', driver='GTiff', width=values.shape[1], height=values.shape[0],
                       count=1, dtype='float32', crs='EPSG:32649', nodata=nodata,
                       transform=from_origin(0, values.shape[0], 1, 1)) as dem:
        dem.write(values, 1)


def test_native_minimum_ignores_hole_nodata_and_includes_other_part(tmp_path):
    path = tmp_path / 'dem.tif'
    values = np.full((6, 10), 99, dtype='float32')
    values[2:4, 2:4] = -500  # hole is lower, but excluded
    values[0, 0] = -9999
    values[0, 1] = np.nan
    values[5, 9] = 2
    dem_file(path, values)
    geom = MultiPolygon([Polygon([(0,0),(6,0),(6,6),(0,6)], [[(2,2),(4,2),(4,4),(2,4)]]),
                         Polygon([(8,0),(10,0),(10,2),(8,2)])])
    with rasterio.open(path) as dem:
        actual = build.minimum_dem_outlet(mapping(geom), dem)
    assert actual['outlet'] == (9.5, .5)
    assert actual['min_elev'] == 2
    assert (actual['outlet_dem_row'], actual['outlet_dem_col']) == (5, 9)


@pytest.mark.parametrize('values,point,elevation', [
    ([[7]], (.5,.5), 7),
    ([[9,3,8]], (1.5,.5), 3),
    ([[2,2],[3,2]], (.5,1.5), 2),
])
def test_single_pixel_narrow_and_tied_minima_use_native_centres(tmp_path,monkeypatch,values,point,elevation):
    path=tmp_path/'dem.tif';dem_file(path,values)
    height,width=np.asarray(values).shape
    geom=Polygon([(0,0),(width,0),(width,height),(0,height)])
    monkeypatch.setattr(build,'CFG',{'swmm':{'delineation_resolution_m':100}})
    stats=build.raster_stats(list(geom.exterior.coords),path,mapping(geom))
    assert stats['outlet']==point and stats['min_elev']==elevation


def test_no_valid_dem_cell_is_refused_instead_of_fabricating_outlet(tmp_path):
    path=tmp_path/'dem.tif';dem_file(path,[[-9999,np.nan]])
    geom=mapping(Polygon([(0,0),(2,0),(2,1),(0,1)]))
    with rasterio.open(path) as dem:
        with pytest.raises(ValueError,match='没有有效DEM像元'):
            build.minimum_dem_outlet(geom,dem)


@pytest.mark.parametrize('method,expected_point,expected_elevation', [
    ('drainage_exit', (2.5,2.5), 7), ('minimum_dem_elevation', (.5,.5), 1)])
def test_build_selection_and_qa_detects_tampering(tmp_path,monkeypatch,method,expected_point,expected_elevation):
    inputs=tmp_path/'inputs/user_data';inputs.mkdir(parents=True)
    dem=inputs/'dem.tif';dem_file(dem,[[9,8,7],[6,5,4],[1,3,2]])
    partitions=tmp_path/'inputs/dem_derived';partitions.mkdir()
    shp=partitions/'test_subcatchments.shp'
    geom=orient(Polygon([(0,0),(3,0),(3,3),(0,3)]),sign=-1)
    with shapefile.Writer(str(shp),shapeType=shapefile.POLYGON) as writer:
        writer.field('name','C',20);writer.field('out_x','F',14,3);writer.field('out_y','F',14,3)
        writer.poly([list(geom.exterior.coords)]);writer.record('D1',2.5,2.5)
    shp.with_suffix('.prj').write_text(CRS.from_epsg(32649).to_wkt(),encoding='utf-8')
    inp=tmp_path/'swmm/model/test.inp'
    monkeypatch.setattr(build,'CFG',{'terrain':{'dem':str(dem)},'study_area':{'calculation_domain':'dem_extent'},
        'swmm':{'working_inp':str(inp),'dem_derived_boundary_shp':str(shp),'delineation_resolution_m':5,'outlet_selection':method},
        'rainfall':{'csv':str(tmp_path/'rainfall.csv')}})
    monkeypatch.setattr(build,'section_options',lambda:['[OPTIONS]','FLOW_UNITS CMS'])
    def rainfall(rows,*args):
        for row in rows:row['rain_gage']='G1'
        return ['[RAINGAGES]','G1 VOLUME 1:00 1 TIMESERIES TS1'],np.array([10])
    monkeypatch.setattr(build,'spatial_station_rainfall',rainfall)
    build.main()
    with (inp.parent/'auto_swmm_parameter_review.csv').open(encoding='utf-8-sig') as stream:
        rows=list(csv.DictReader(stream))
    assert (float(rows[0]['outlet_x']),float(rows[0]['outlet_y']))==expected_point
    assert float(rows[0]['outfall_elev_m'])==expected_elevation
    assert rows[0]['outlet_method']==method
    assert float(rows[0]['drainage_outlet_x'])==2.5 and float(rows[0]['drainage_outlet_y'])==2.5
    text=inp.read_text(encoding='utf-8')
    errors,facts=model_outlet_check(tmp_path,shp,rows,text)
    assert not errors and facts['verified_count']==1
    correct=f'{expected_point[0]:.3f} {expected_point[1]:.3f}'
    corrupted=text.replace(correct,'1.500 1.500')
    assert corrupted!=text
    assert model_outlet_check(tmp_path,shp,rows,corrupted)[0]
