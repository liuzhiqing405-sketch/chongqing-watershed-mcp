import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from watershed_hydro_mcp.core.delineation import delineate as d
from watershed_hydro_mcp.core.delineation.partition_topology import (
    clip_labels_to_local_drainage, filter_boundary_fragments,
)
from watershed_hydro_mcp.core.spatial_domain import apply_calculation_mask


def test_filter_only_independent_boundary_trees_and_account_area():
    labels = np.array([[1, 2, 3], [4, 5, 3], [4, 6, 3]], np.int32)
    valid = np.ones(labels.shape, bool)
    # 1 -> 2: protect both sides of a real connection. 5 is an interior unit.
    receivers = np.array([1, -1, 5, 6, -1, 8, -1, -1, -1])
    kept, mask, excluded, facts = filter_boundary_fragments(labels, receivers, valid,
        cell_area_ha=.4, minimum_area_ha=1, maximum_fraction=.5)
    assert facts['excluded_count'] == 2
    assert {r['label'] for r in facts['excluded_units']} == {4, 6}
    assert np.all(kept[labels == 5] == 5)  # Small interior catchment is never dropped.
    assert np.all(kept[np.isin(labels, [1, 2])] > 0)
    assert np.array_equal(mask | excluded, valid)
    assert not np.any(mask & excluded)
    assert facts['original_area_ha'] == pytest.approx(facts['retained_area_ha']+facts['excluded_area_ha'])


def test_threshold_is_strict_and_zero_disables_filter():
    labels = np.array([[1, 2]], np.int32)
    rec = np.array([-1, -1])
    for threshold in (0, 1):
        kept, _, excluded, _ = filter_boundary_fragments(labels, rec, labels > 0,
            cell_area_ha=1, minimum_area_ha=threshold, maximum_fraction=1)
        assert np.array_equal(kept, labels) and not excluded.any()


def test_filter_refuses_excessive_loss_or_empty_domain():
    labels = np.array([[1, 2]], np.int32)
    with pytest.raises(RuntimeError, match='above'):
        filter_boundary_fragments(labels, np.array([-1, -1]), labels > 0,
            cell_area_ha=.1, minimum_area_ha=1, maximum_fraction=.01)
    with pytest.raises(RuntimeError, match='entire'):
        filter_boundary_fragments(labels, np.array([-1, -1]), labels > 0,
            cell_area_ha=.1, minimum_area_ha=1, maximum_fraction=1)


def test_buffered_identity_splits_at_different_local_exits():
    labels = np.ones((1, 4), np.int32)
    rec = np.array([1, -1, 3, -1])
    result = clip_labels_to_local_drainage(labels, rec, labels > 0)
    assert result[0, 0] == result[0, 1]
    assert result[0, 2] == result[0, 3]
    assert result[0, 0] != result[0, 2]


def test_context_receivers_do_not_import_or_jump_over_outside_cells():
    full = np.arange(1, 13)
    full[[3, 7, 11]] = -1
    valid = np.array([[True, False], [True, True]])
    local = d.restrict_receivers(full, (3, 4), (slice(1, 3), slice(1, 3)), valid)
    assert np.array_equal(local, [-1, -1, 3, -1])


def write_raster(path, values, transform, nodata=-9999):
    with rasterio.open(path, 'w', driver='GTiff', height=values.shape[0], width=values.shape[1],
                       count=1, dtype=values.dtype, crs='EPSG:32649', transform=transform, nodata=nodata) as dst:
        dst.write(values, 1)


def test_frozen_context_aligns_and_checks_interior_elevations(tmp_path, monkeypatch):
    full = np.arange(36, dtype='float64').reshape(6, 6)
    path = tmp_path/'dem_context.tif'
    write_raster(path, full, from_origin(0, 60, 10, 10))
    monkeypatch.setattr(d, 'CFG', {'terrain': {'drainage_context_dem': str(path)}})
    local = full[2:4, 2:4].copy()
    valid = np.ones((2, 2), bool)
    with rasterio.open(path) as src:
        grid, _, transform, window, facts = d.read_drainage_context(local, valid, from_origin(20, 40, 10, 10), src.crs)
        assert np.array_equal(grid[window], local) and facts['used']
        local[0, 0] += 1
        with pytest.raises(RuntimeError, match='inconsistent'):
            d.read_drainage_context(local, valid, from_origin(20, 40, 10, 10), src.crs)


def test_downstream_mask_hides_excluded_cells_and_missing_mask_fails(tmp_path):
    root = tmp_path/'inputs/dem_derived'
    root.mkdir(parents=True)
    (root/'dem_derived_subcatchments_diagnostics.json').write_text('{"schema_version":2}')
    with pytest.raises(RuntimeError, match='missing'):
        apply_calculation_mask(np.ones((2, 2), bool), from_origin(0, 20, 10, 10), 'EPSG:32649', tmp_path)
    write_raster(root/'calculation_mask.tif', np.array([[1, 0], [1, 1]], dtype='uint8'), from_origin(0, 20, 10, 10), nodata=0)
    mask = apply_calculation_mask(np.ones((2, 2), bool), from_origin(0, 20, 10, 10), 'EPSG:32649', tmp_path)
    assert np.array_equal(mask, [[True, False], [True, True]])


def test_qa_rejects_forged_area_and_geometry_even_with_matching_hash(tmp_path):
    from watershed_hydro_mcp.qa import boundary_filter_check
    from watershed_exchange.paths import sha256_file
    root = tmp_path/'inputs/dem_derived'
    user = tmp_path/'inputs/user_data'
    root.mkdir(parents=True); user.mkdir(parents=True)
    transform = from_origin(0, 200, 100, 100)
    write_raster(user/'dem.tif', np.arange(4, dtype='float64').reshape(2, 2), transform)
    with rasterio.open(user/'dem.tif') as src:
        crs = src.crs
    retained = np.array([[1, 0], [1, 1]], np.int32)
    excluded = np.array([[0, 2], [0, 0]], np.int32)
    d.write_subcatchment_shp(retained, transform, crs, root/'event_subcatchments.shp', 0)
    d.write_subcatchment_shp(excluded, transform, crs, root/'excluded/boundary_fragments.shp', 0,
                            summary_name='boundary_fragments_summary.csv')
    for name, grid in [('calculation_mask.tif', retained), ('excluded_boundary_mask.tif', excluded > 0)]:
        write_raster(root/name, grid.astype('uint8'), transform, nodata=0)
    (tmp_path/'config.yaml').write_text('study_area:\n  calculation_domain: dem_extent\n')
    facts = dict(original_cells=4, retained_cells=3, excluded_cells=1,
                 original_area_ha=4, retained_area_ha=3, excluded_area_ha=1,
                 excluded_fraction=.25, maximum_excluded_fraction=.3, excluded_count=1)
    for name, key in [('calculation_mask.tif', 'calculation_mask_sha256'),
                      ('excluded_boundary_mask.tif', 'excluded_boundary_mask_sha256'),
                      ('excluded/boundary_fragments.shp', 'excluded_shape_sha256'),
                      ('excluded/boundary_fragments_summary.csv', 'excluded_summary_sha256')]:
        facts[key] = sha256_file(root/name)
    diagnostics = dict(boundary_filter=facts, valid_area_ha=3)
    assert not boundary_filter_check(tmp_path, diagnostics)[0]
    facts['original_area_ha'] = 5
    assert boundary_filter_check(tmp_path, diagnostics)[0]
    facts['original_area_ha'] = 4
    # Correct count, disjoint masks, and recomputed hash still cannot hide a wrong spatial domain.
    d.write_subcatchment_shp(np.rot90(retained).copy(), transform, crs, root/'event_subcatchments.shp', 0)
    assert any('几何' in e for e in boundary_filter_check(tmp_path, diagnostics)[0])
