import json

import numpy as np
import pytest
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import shapefile

from watershed_hydro_mcp.core.delineation import delineate as d
from watershed_hydro_mcp.core.delineation.partition_topology import coarsen_labels, partition_facts


def test_irregular_and_disconnected_domains_use_every_open_edge():
    valid=np.zeros((10,12),bool);valid[2:5,2:5]=True;valid[6:8,8:10]=True
    dem=np.full(valid.shape,np.nan);dem[valid]=10
    dem[3,3]=2
    filled,parents=d.fill_depressions_priority_flood(dem,valid,return_parents=True)
    assert filled[3,3]==10
    assert np.all(filled[valid]==10)
    rec=d.build_receivers(filled,valid,parents)
    acc=d.accumulation(rec,valid)
    assert acc.ravel()[valid.ravel() & (rec<0)].sum()==valid.sum()
    assert np.all(d.domain_boundary(valid).ravel()[valid.ravel() & (rec<0)])


def test_clipped_valley_not_filled_to_rectangle_corner_height():
    valid=np.zeros((7,7),bool);valid[1:6,1:6]=True;valid[0,3]=True
    dem=np.full((7,7),np.nan);dem[valid]=100
    dem[1:6,3]=np.arange(5,0,-1)
    filled=d.fill_depressions_priority_flood(dem,valid)
    assert np.array_equal(filled[valid],dem[valid])


def test_flats_drain_to_real_spill_and_real_slopes_do_not_change():
    valid=np.ones((7,7),bool);dem=np.full((7,7),20.)
    dem[1:6,1:6]=5;dem[6,3]=5
    filled,parents=d.fill_depressions_priority_flood(dem,valid,return_parents=True)
    rec=d.build_receivers(filled,valid,parents)
    outlets=d.outlet_labels(rec,valid)
    assert np.all(outlets[1:6,1:6]==6*7+3)
    assert np.array_equal(filled,dem)
    slope=np.tile(np.arange(7,0,-1,dtype=float),(7,1))
    filled,parents=d.fill_depressions_priority_flood(slope,valid,return_parents=True)
    assert np.array_equal(d.build_receivers(slope,valid),d.build_receivers(filled,valid,parents))


def test_rectangular_cell_metric_changes_steepest_direction():
    dem=np.array([[30.,30.,30.],[30.,10.,8.],[30.,7.,30.]])
    valid=np.ones(dem.shape,bool)
    assert d.build_receivers(dem,valid)[4]==7
    assert d.build_receivers(dem,valid,cell_height=10)[4]==5


def test_cycle_is_rejected_instead_of_hanging_path_assignment():
    with pytest.raises(RuntimeError,match='cycle'):
        d.accumulation(np.array([1,0]),np.ones((1,2),bool))


def test_graph_merge_exhausts_more_than_2000_fragments_without_losing_area():
    n=3005;grid=np.arange(1,n+1,dtype=np.int32)[None,:]
    rec=np.r_[np.arange(1,n),-1];valid=np.ones(grid.shape,bool)
    out,terminals,facts=coarsen_labels(grid,rec,valid,minimum_cells=3,target_count=1)
    assert np.all(out>0)
    assert facts['major_partition_count']==1 and facts['small_merges']>2000
    assert np.count_nonzero(out)==n
    assert partition_facts(out,rec,valid,d.domain_boundary(valid))['passed']
    json.dumps(facts)  # All persisted counters must be JSON primitives.


def test_independent_boundary_outlets_are_not_merged_across_divide():
    grid=np.array([[1,2,3,4]],np.int32);valid=np.ones(grid.shape,bool)
    rec=np.full(4,-1,dtype=np.int64)
    out,terminals,facts=coarsen_labels(grid,rec,valid,minimum_cells=10,target_count=1)
    assert len(np.unique(out))==4 and len(terminals)==4
    assert not facts['target_reached'] and facts['retained_small_terminal_count']==4


def test_topology_does_not_accept_internal_terminal():
    grid=np.ones((3,3),np.int32);valid=np.ones(grid.shape,bool)
    rec=np.full(9,-1,dtype=np.int64)
    assert not partition_facts(grid,rec,valid,d.domain_boundary(valid))['passed']


def test_shapefile_preserves_holes_and_diagonal_multipart_identity(tmp_path):
    grid=np.zeros((8,8),np.int32);grid[1:6,1:6]=1;grid[3,3]=0
    grid[6,6]=2;grid[7,7]=2
    tx=from_origin(0,80,10,10);path=tmp_path/'subcatchments.shp'
    d.write_subcatchment_shp(grid,tx,rasterio.crs.CRS.from_epsg(32648),path,0,
                            boundary_labels={2})
    with shapefile.Reader(str(path)) as reader:
        assert len(reader)==2
        records=[dict(zip([f[0] for f in reader.fields[1:]],r)) for r in reader.records()]
        assert records[0]['area_ha']==.24 and records[1]['area_ha']==.02
        geoms=[(shape.__geo_interface__,i+1) for i,shape in enumerate(reader.shapes())]
    restored=rasterize(geoms,out_shape=grid.shape,transform=tx)
    assert np.array_equal(restored,grid)
