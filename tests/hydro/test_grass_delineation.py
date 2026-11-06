from pathlib import Path
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import yaml

from watershed_hydro_mcp.core.delineation import delineate as d
from watershed_hydro_mcp.core.delineation import grass_delineation as g
from watershed_hydro_mcp.grass_runtime import resolve_grass_cli
from watershed_hydro_mcp.qa import spatial_gate
from watershed_hydro_mcp.core.delineation.partition_topology import single_exit_facts


def test_diagonal_multipart_basin_has_one_common_exit():
    labels = np.eye(3, dtype='int32')*2
    valid = labels > 0
    drainage = np.zeros(labels.shape, dtype='int32')
    drainage[valid] = 7  # Diagonal south-east links; only the last cell exits.
    rec = g.drainage_receivers(drainage, valid)
    d.accumulation(rec, valid)
    facts = single_exit_facts(labels, rec, valid)
    assert facts['passed']
    assert facts['single_exit_partition_count'] == 1
    assert facts['exit_cell_count'] == 1


def test_multiple_exits_to_same_downstream_basin_are_rejected():
    labels = np.array([[2, 4], [2, 4]], dtype='int32')
    valid = np.ones(labels.shape, bool)
    drainage = np.array([[8, 6], [8, -6]], dtype='int32')
    rec = g.drainage_receivers(drainage, valid)
    d.accumulation(rec, valid)
    facts = single_exit_facts(labels, rec, valid)
    assert not facts['passed']
    assert facts['multiple_exit_partition_count'] == 1
    assert facts['invalid_partitions'] == [{'label': 2, 'exit_cells': 2}]


def test_basin_without_exit_is_rejected():
    labels = np.array([[2, 2]], dtype='int32')
    facts = single_exit_facts(labels, np.array([1, 0]), np.ones(labels.shape, bool))
    assert not facts['passed']
    assert facts['missing_exit_partition_count'] == 1


def test_default_grass_organization_keeps_flow_outlets():
    config = yaml.safe_load((Path(__file__).resolve().parents[2]/'config/hydro.default.yaml').read_text(encoding='utf-8'))
    assert config['swmm']['delineation_method'] == 'grass_watershed'
    assert config['swmm']['outlet_selection'] == 'drainage_exit'
    assert config['swmm']['channel_threshold_ha'] == 20


@pytest.mark.parametrize('code,offset', list(g.DIRECTIONS.items()))
def test_grass_angles_are_not_native_d8_codes(code, offset):
    grid = np.full((3, 3), -6, dtype='int32')
    grid[1, 1] = code
    rec = g.drainage_receivers(grid, np.ones(grid.shape, bool))
    assert rec[4] == (1+offset[0])*3 + 1+offset[1]
    grid[1, 1] = -code
    assert g.drainage_receivers(grid, np.ones(grid.shape, bool))[4] == -1


def test_nodata_holes_do_not_receive_runoff_and_invalid_codes_fail():
    grid = np.full((3, 3), 8, dtype='int32')
    valid = np.ones(grid.shape, bool)
    valid[1, 2] = False
    assert g.drainage_receivers(grid, valid)[4] == -1
    grid[1, 1] = 9
    with pytest.raises(RuntimeError, match='Invalid'):
        g.drainage_receivers(grid, valid)


def test_explicit_missing_grass_does_not_autofallback(tmp_path):
    assert resolve_grass_cli(str(tmp_path/'missing-grass.exe')) is None


def test_missing_grass_blocks_plan_before_copying_inputs(tmp_path, monkeypatch):
    from watershed_hydro_mcp import planner
    monkeypatch.setattr(planner, '_load_yaml', lambda _: {'swmm': {'delineation_method': 'grass_watershed'}})
    monkeypatch.setattr('watershed_hydro_mcp.grass_runtime.resolve_grass_cli', lambda *args: None)
    with pytest.raises(planner.PlanningError, match='GRASS'):
        planner.terrain_preflight({'input_id': 'not-copied'})


def test_grass_masks_holes_outlets_and_qa_are_bound_to_original_rasters(tmp_path, monkeypatch):
    transform = from_origin(500000, 3300000, 200, 200)
    crs = rasterio.crs.CRS.from_epsg(32649)
    valid = np.ones((10, 10), bool)
    valid[3:5, 3:5] = False
    dem = np.arange(100, dtype='float64').reshape(10, 10)
    source = tmp_path/'inputs/user_data/dem.tif'
    source.parent.mkdir(parents=True)
    g.write_grid(source, np.where(valid, dem, np.nan), transform, crs, np.nan)
    config = {'terrain': {'dem': str(source)}, 'study_area': {'calculation_domain': 'dem_extent'},
              'swmm': {'channel_threshold_ha': 20, 'boundary_fragment_min_area_ha': 0}}
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    monkeypatch.setattr(d, 'CFG', config)
    monkeypatch.setattr(d, 'find_grass_cli', lambda: Path('grass'))
    monkeypatch.setattr(d, 'read_dem_for_delineation', lambda: (dem, valid, transform, crs))

    def fake_grass(cli, source_dem, worker, output, threshold):
        assert threshold == 5
        labels = np.where(np.indices(valid.shape)[1] < 5, 2, 4).astype('int32')
        labels[~valid] = 0
        labels[0, :2] = 0  # Exterior areas below the threshold must stay excluded.
        direction = np.full(valid.shape, 6, dtype='int32')
        direction[2, 3:5] = 4  # Route around the hole, within basin 2.
        direction[-1] = 8
        direction[-1, [4, 9]] = -6  # One actual exit per basin.
        direction[~valid] = 0
        for name, values in [('basin', labels), ('drainage', direction),
                             ('accumulation', valid.astype('float64')), ('stream', valid.astype('int32'))]:
            g.write_grid(output/(name+'.tif'), values, transform, crs)
        (output/'version.txt').write_text('GRASS test fixture', encoding='utf-8')

    monkeypatch.setattr(g, 'run_grass', fake_grass)
    out = tmp_path/'inputs/dem_derived/dem_derived_subcatchments.shp'
    g.write_grass_watersheds(out)
    report = spatial_gate(tmp_path, {})
    assert report['passed'], report['errors']
    assert report['facts']['subcatchment_count'] == 2  # Hole never becomes another unit.
    assert report['facts']['total_area_ha'] == 94*4
    facts = report['facts']['grass_partition']
    assert facts['single_exit']['passed']
    assert facts['single_exit']['exit_cell_count'] == 2
    assert facts['excluded_cells'] == 2
    assert facts['original_cells'] == facts['retained_cells'] + facts['excluded_cells']
    # Matching hashes alone cannot hide a second independent outlet.
    import json
    from watershed_exchange.paths import sha256_file
    path = out.parent/'grass/drainage.tif'
    diagnostic_path = out.parent/'dem_derived_subcatchments_diagnostics.json'
    with rasterio.open(path, 'r+') as dst:
        values = dst.read(1); values[-1, 0] = -6; dst.write(values, 1)
    diagnostic = json.loads(diagnostic_path.read_text(encoding='utf-8'))
    diagnostic['grass']['rasters']['drainage.tif'] = sha256_file(path)
    diagnostic_path.write_text(json.dumps(diagnostic), encoding='utf-8')
    report = spatial_gate(tmp_path, {})
    assert not report['passed']
    assert any('唯一出口' in message for message in report['errors'])
    assert report['facts']['grass_partition']['single_exit']['multiple_exit_partition_count'] == 1
    # Corrupting a flow raster must also fail checksum verification.
    with rasterio.open(path, 'r+') as dst:
        values = dst.read(1); values[0, 0] = 8; dst.write(values, 1)
    report = spatial_gate(tmp_path, {})
    assert not report['passed']
    assert any('drainage.tif' in message for message in report['errors'])

    def multiple_exit_grass(*args):
        fake_grass(*args)
        path = args[3]/'drainage.tif'
        with rasterio.open(path, 'r+') as dst:
            values = dst.read(1); values[-1, 0] = -6; dst.write(values, 1)

    monkeypatch.setattr(g, 'run_grass', multiple_exit_grass)
    with pytest.raises(RuntimeError, match='唯一出口'):
        g.write_grass_watersheds(tmp_path/'invalid/dem_derived_subcatchments.shp')
