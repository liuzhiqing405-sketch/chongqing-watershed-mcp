from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import yaml

from watershed_exchange.dem_budget import DEFAULT_MAXIMUM_PROCESSING_CELLS, cell_budget
from watershed_hydro_mcp import planner
from watershed_hydro_mcp.core.delineation import delineate as d
from watershed_hydro_mcp.models import HydroRunRequest
from watershed_hydro_mcp.settings import Settings


def raster(path, height=3, width=3):
    with rasterio.open(path, 'w', driver='GTiff', height=height, width=width,
                       count=1, dtype='float32', crs='EPSG:32649',
                       transform=from_origin(0, height*10, 10, 10), compress='lzw') as out:
        out.write(np.ones((height, width), dtype='float32'), 1)


def test_legacy_context_above_four_million_keeps_native_grid(tmp_path, monkeypatch):
    path = tmp_path/'context.tif'
    raster(path, height=2001, width=2000)
    transform = from_origin(0, 20010, 10, 10)
    monkeypatch.setattr(d, 'CFG', {'terrain': {'drainage_context_dem': str(path)}})
    dem = np.ones((2, 2)); valid = np.ones((2, 2), bool)
    grid, mask, actual, window, facts = d.read_drainage_context(
        dem, valid, transform, rasterio.crs.CRS.from_epsg(32649))
    assert grid.shape == (2001, 2000) and actual == transform
    assert np.array_equal(grid[window], dem) and mask.all()
    assert facts['processing_cells'] == 4_002_000
    assert facts['maximum_cells'] == DEFAULT_MAXIMUM_PROCESSING_CELLS


@pytest.mark.parametrize('value', [True, 0, -1, 4.5, '8000000'])
def test_invalid_limits_are_rejected(value):
    with pytest.raises(ValueError, match='正整数'):
        cell_budget(value)


def test_worker_honors_explicit_limit_before_reprojecting(tmp_path, monkeypatch):
    path = tmp_path/'context.tif'; raster(path)
    monkeypatch.setattr(d, 'CFG', {'terrain': {'drainage_context_dem': str(path),
                                               'maximum_processing_cells': 8}})
    with pytest.raises(ValueError, match='9.*8.*预算'):
        d.read_drainage_context(np.ones((2, 2)), np.ones((2, 2), bool),
                                from_origin(0, 30, 10, 10), rasterio.crs.CRS.from_epsg(32649))


def test_fractional_native_alignment_does_not_add_row_at_budget_edge(tmp_path, monkeypatch):
    tx = from_origin(125898.11806820835, 3260014.652130228, 12.49181777365578, 12.491825566969831)
    from rasterio.transform import Affine
    context_tx = tx * Affine.translation(-81, -81)
    path = tmp_path/'context.tif'
    with rasterio.open(path, 'w', driver='GTiff', height=165, width=165, count=1,
                       dtype='float32', crs='EPSG:32649', transform=context_tx) as out:
        out.write(np.ones((165, 165), dtype='float32'), 1)
    monkeypatch.setattr(d, 'CFG', {'terrain': {'drainage_context_dem': str(path),
                                               'maximum_processing_cells': 165*165}})
    grid, _, actual, window, facts = d.read_drainage_context(
        np.ones((3, 3)), np.ones((3, 3), bool), tx, rasterio.crs.CRS.from_epsg(32649))
    assert grid.shape == (165, 165) and actual == context_tx
    assert window == (slice(81, 84), slice(81, 84)) and facts['processing_cells'] == 165*165


def test_planner_passes_package_budget_to_worker_and_refuses_stricter_local_cap(tmp_path):
    settings = Settings(tmp_path, tmp_path/'runs', tmp_path/'hydro', tmp_path/'native',
                        Path(__file__).resolve().parents[2]/'config')
    record = {'input_id': 'hydro-input-testbudget',
              'provenance': {'dem_processing': {'maximum_cells': 9}}}
    source = settings.input_dir(record['input_id']); source.mkdir(parents=True)
    raster(source/'dem.tif', height=2, width=2); raster(source/'dem_context.tif')
    for name, _ in planner.PROJECT_INPUTS:
        if not (source/name).exists(): (source/name).touch()
    request = HydroRunRequest(input_id=record['input_id'], mode='build_run_qa')
    plan = planner.build_project('hydro-run-budgettest', record, request, settings)
    config = yaml.safe_load(Path(plan['config']).read_text(encoding='utf-8'))
    assert config['terrain']['maximum_processing_cells'] == 9
    assert plan['dem_preflight']['grids']['dem_context.tif']['cells'] == 9
    record['provenance']['dem_processing']['maximum_cells'] = 8
    with pytest.raises(ValueError, match='9.*8.*预算'):
        planner.build_project('hydro-run-refusedbudget', record, request, settings)
    assert not (settings.job_dir('hydro-run-refusedbudget')/'project').exists()


def test_explicit_local_cap_is_stricter_than_package_cap(tmp_path, monkeypatch):
    settings = SimpleNamespace(input_dir=lambda _: tmp_path,
                               preset_file=lambda _: tmp_path/'default.yaml')
    (tmp_path/'default.yaml').write_text('terrain:\n  maximum_processing_cells: 8\n', encoding='utf-8')
    raster(tmp_path/'dem.tif')
    with pytest.raises(ValueError, match='9.*8.*预算'):
        planner.terrain_preflight({'input_id': 'input',
                                  'provenance': {'dem_processing': {'maximum_cells': 10}}}, settings)
