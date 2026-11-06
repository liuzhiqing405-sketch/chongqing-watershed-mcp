import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from watershed_hydro_mcp.result_distribution import safety_distribution


def write_grid(path, data, crs='EPSG:4544'):
    with rasterio.open(path, 'w', driver='GTiff', width=data.shape[1], height=data.shape[0],
                       count=1, dtype='float32', nodata=-9999, crs=crs,
                       transform=from_origin(500000, 3260000, 5, 5)) as dst:
        dst.write(data.astype('float32'), 1)


def test_distribution_counts_native_cells_and_does_not_count_nodata(tmp_path):
    path = tmp_path/'minimum_safety_factor.tif'
    data = np.full((6,6), 2.)
    data[:2,:2] = .5; data[:2,2:4] = 1.2; data[2:4,2:4] = 1.4; data[-1,-1] = -9999
    write_grid(path, data)
    result = safety_distribution(path)
    assert result['valid_cells'] == 35
    assert result['below_one_fraction'] == pytest.approx(4/35)
    assert result['below_one_area_km2'] == pytest.approx(4*25/1e6)
    assert result['calculation_area_km2'] == pytest.approx(35*25/1e6)
    assert result['largest_below_one_sector'] == '西北部'
    assert result['sectors']['西北部']['share_of_all_below_one'] == 1
    assert sum(b['cells'] for b in result['bands'].values()) == 35
    assert sum(b['valid_cells'] for b in result['sectors'].values()) == 35
    assert result['basis'] == 'event_minimum_per_pixel'


def test_no_finite_cells_and_geographic_units_are_not_invented_areas(tmp_path):
    path = tmp_path/'empty.tif'
    write_grid(path, np.full((2,2), -9999))
    assert not safety_distribution(path)['available']
    geographic = tmp_path/'geographic.tif'
    write_grid(geographic, np.ones((2,2)), 'EPSG:4326')
    assert not safety_distribution(geographic)['available']


def test_cached_result_cannot_be_mutated_by_another_call(tmp_path):
    path = tmp_path/'fs.tif'
    write_grid(path, np.full((3,3), .8))
    first = safety_distribution(path)
    first['bands'].clear()
    assert len(safety_distribution(path)['bands']) == 4
