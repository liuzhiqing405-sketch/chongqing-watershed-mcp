from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from chongqing_spatial_mcp.workflow import _export_drainage_context
from chongqing_spatial_mcp.dem_budget import preflight_geometry


def test_export_keeps_exterior_terrain_and_native_alignment(tmp_path):
    full = np.arange(400, dtype='float32').reshape(20, 20)
    profile = dict(driver='GTiff', count=1, dtype='float32', crs='EPSG:32649', nodata=-9999)
    src, clip, dst = [tmp_path/name for name in ['source.tif', 'clip.tif', 'context.tif']]
    with rasterio.open(src, 'w', height=20, width=20, transform=from_origin(0, 200, 10, 10), **profile) as out:
        out.write(full, 1)
    with rasterio.open(clip, 'w', height=4, width=4, transform=from_origin(80, 120, 10, 10), **profile) as out:
        cut = full[8:12, 8:12].copy(); cut[0, 0] = -9999
        out.write(cut, 1)
    settings = SimpleNamespace(defaults={'model_inputs': {'drainage_context_buffer_m': 20}})
    facts = _export_drainage_context(src, clip, dst, settings)
    with rasterio.open(dst) as result:
        assert result.transform == from_origin(60, 140, 10, 10)
        assert np.array_equal(result.read(1), full[6:14, 6:14])
        assert facts['cells'] == 64 and not facts['source_extent_reached']
    settings.defaults['model_inputs']['drainage_context_maximum_cells'] = 10
    with pytest.raises(ValueError, match='预算'):
        _export_drainage_context(src, clip, tmp_path/'too_big.tif', settings)


def test_preflight_matches_fractional_crop_and_budget_edge(tmp_path):
    from rasterio.mask import mask
    from shapely.geometry import mapping
    src, clip = tmp_path/'source.tif', tmp_path/'clip.tif'
    profile = dict(driver='GTiff', count=1, dtype='float32', crs='EPSG:32649', nodata=-9999,
                   height=20, width=20, transform=from_origin(0, 200, 10, 10))
    with rasterio.open(src, 'w', **profile) as out:
        out.write(np.ones((20, 20), dtype='float32'), 1)
    settings = SimpleNamespace(defaults={'model_inputs': {'drainage_context_buffer_m': 20},
                                         'delineation': {'maximum_processing_cells': 72}})
    geometry = box(81.3, 83.9, 113.8, 127.2)
    with rasterio.open(src) as source:
        facts = preflight_geometry(source, geometry, settings)
        data, transform = mask(source, [mapping(geometry)], crop=True)
    profile.update(height=data.shape[1], width=data.shape[2], transform=transform)
    with rasterio.open(clip, 'w', **profile) as out:
        out.write(data)
    actual = _export_drainage_context(src, clip, tmp_path/'context.tif', settings)
    assert facts['cells'] == actual['cells'] == 72
    assert facts['clipped_cells'] == data.shape[1]*data.shape[2] == 20
    settings.defaults['delineation']['maximum_processing_cells'] = 71
    with rasterio.open(src) as source, pytest.raises(ValueError, match='预算'):
        preflight_geometry(source, geometry, settings)
    with pytest.raises(ValueError, match='预算'):
        _export_drainage_context(src, clip, tmp_path/'refused.tif', settings)
    assert not (tmp_path/'refused.tif').exists()


def test_block_export_preserves_data_and_mask_across_tile_seams(tmp_path):
    src, clip, dst = [tmp_path/name for name in ['source.tif', 'clip.tif', 'context.tif']]
    data = np.arange(1031*1027, dtype='float32').reshape(1031, 1027)
    valid = np.full(data.shape, 255, dtype='uint8')
    valid[510:515, 510:515] = 0
    valid[-1, -1] = 0
    profile = dict(driver='GTiff', count=1, dtype='float32', crs='EPSG:32649',
                   height=1031, width=1027, transform=from_origin(0, 10310, 10, 10))
    with rasterio.open(src, 'w', **profile) as out:
        out.write(data, 1)
        out.write_mask(valid)
    profile.update(height=1, width=1, transform=from_origin(5000, 5310, 10, 10))
    with rasterio.open(clip, 'w', **profile) as out:
        out.write(np.ones((1, 1), dtype='float32'), 1)
    settings = SimpleNamespace(defaults={'model_inputs': {'drainage_context_buffer_m': 20_000}})
    facts = _export_drainage_context(src, clip, dst, settings)
    with rasterio.open(dst) as result:
        assert np.array_equal(result.read(1), data)
        assert np.array_equal(result.read_masks(1), valid)
        assert result.transform == from_origin(0, 10310, 10, 10)
    assert facts['source_extent_reached']
