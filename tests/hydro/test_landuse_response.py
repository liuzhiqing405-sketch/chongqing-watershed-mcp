import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from watershed_hydro_mcp.core.swmm import landuse_response as response
from watershed_hydro_mcp.core.swmm.build_inp import LANDUSE_PARAMS, DEFAULT_PARAM
from watershed_hydro_mcp.core.swmm.output_reader import SwmmRunoffSeries, read_subcatchment_runoff
from watershed_hydro_mcp.core.stability import safety_factor as stability


def fixture_units(tmp_path, monkeypatch):
    dem = tmp_path/'dem.tif'
    transform = from_origin(500000, 3300000, 10, 10)
    with rasterio.open(dem, 'w', driver='GTiff', count=1, dtype='float32',
                       width=4, height=2, crs='EPSG:32649', transform=transform) as dst:
        dst.write(np.full((2, 4), 100, dtype='float32'), 1)
    classes = np.array([[1, 2, 1, 1], [1, 0, 1, 1]], dtype='int16')
    (tmp_path/'landuse.shp').touch()
    monkeypatch.setattr(response, 'landuse_class_grid', lambda *a: (classes, {1:'有林地', 2:'耕地'}))
    parents = []
    for i in range(2):
        parents.append(dict(subcatchment=f'S{i+1}', outfall=f'O{i+1}', source_name=f'D{i+1}',
            area_ha=.04, width_m=20., flow_length_m=20., slope_pct=10., rain_gage=f'G{i+1}',
            imperv_pct=3., n_perv=.35, maxrate_mm_h=70., minrate_mm_h=10., decay_h=3.5))
    items = [{'geometry':mapping(box(500000+i*20,3299980,500020+i*20,3300000))} for i in range(2)]
    units = response.build_units(parents, items, dem, tmp_path/'landuse.shp', tmp_path,
                                 tmp_path, LANDUSE_PARAMS, DEFAULT_PARAM)
    tokens = [[u['subcatchment'],u['rain_gage'],u['outfall'],f"{u['area_ha']:.6f}",
               f"{u['imperv_pct']:.6f}"] for u in units]
    inp = tmp_path/'model.inp'
    inp.write_text('[SUBCATCHMENTS]\n'+'\n'.join(' '.join(t) for t in tokens),encoding='utf-8')
    response.seal_files(tmp_path,inp)
    return parents,units,tokens,dem,transform


def test_native_classes_unknown_area_and_width_are_preserved(tmp_path, monkeypatch):
    parents, units, tokens, _, _ = fixture_units(tmp_path,monkeypatch)
    assert response.validate_units(tmp_path,tokens,parents)['passed']
    assert len(units)==4 and units[-1]['subcatchment']=='S2'
    for parent in parents:
        children=[u for u in units if u['parent_subcatchment']==parent['subcatchment']]
        assert sum(u['area_ha'] for u in children)==pytest.approx(parent['area_ha'])
        assert sum(u['width_m'] for u in children)==pytest.approx(parent['width_m'])
        assert all(u['area_ha']*10000/u['width_m']==pytest.approx(parent['flow_length_m']) for u in children)
        assert all(u['rain_gage']==parent['rain_gage'] and u['outfall']==parent['outfall'] for u in children)
    assert parents[0]['maxrate_mm_h']==''  # mixed Horton curves are never averaged
    assert {u['landuse']:u['minrate_mm_h'] for u in units if u['parent_subcatchment']=='S1'}=={
        'unknown':6.,'有林地':10.,'耕地':6.}
    facts=json.loads((tmp_path/response.SUMMARY).read_text(encoding='utf-8'))
    assert facts['unknown_area_ha']==pytest.approx(.01)
    assert all(r['native_percent']==pytest.approx(r['modeled_percent']) for r in facts['class_areas'])


def test_validation_rejects_tampering_and_wrong_connections(tmp_path,monkeypatch):
    parents,units,tokens,_,_=fixture_units(tmp_path,monkeypatch)
    broken=[row.copy() for row in tokens];broken[0][2]='wrong_exit'
    with pytest.raises(ValueError,match='出口'):
        response.validate_units(tmp_path,broken,parents)
    with (tmp_path/response.REVIEW).open('a',encoding='utf-8') as stream:stream.write('\n')
    with pytest.raises(ValueError,match='校验失败'):
        response.validate_units(tmp_path,tokens,parents)


def test_cell_mapping_retains_each_class_and_zero_hydrograph(tmp_path,monkeypatch):
    _,units,_,dem,transform=fixture_units(tmp_path,monkeypatch)
    stats={u['subcatchment']:dict(runoff_coeff=0.,peak_runoff_cms=0.,runoff_mm=0.,precip_mm=10.) for u in units}
    series=SwmmRunoffSeries(list(stats),[datetime(2024,7,12,1)],np.zeros((1,len(units))),3600)
    monkeypatch.setattr(stability,'CFG',{'terrain':{'dem':str(dem)},'swmm':{'output':str(tmp_path/'model.out'),'report':'model.rpt'}})
    monkeypatch.setattr(stability,'read_subcatchment_runoff',lambda p:series)
    result=stability.rasterize_landuse_response(tmp_path,transform,(2,4),np.ones((2,4),bool),
        [('2024-07-12 00:00:00',10.)],stats)
    assert result['dynamic_available'] is True
    assert not result['hourly_runoff_depth_m'].any()
    assert not result['runoff_coeff'].any()
    with rasterio.open(tmp_path/response.GRID) as src:expected=src.read(1)-1
    np.testing.assert_array_equal(result['subcatchment_index'],expected)
    # Supply different runoff to each land class; integrating mapped depths
    # over actual native cells must not duplicate the parent's volume.
    q=np.arange(1,len(units)+1,dtype='float64')
    monkeypatch.setattr(stability,'read_subcatchment_runoff',lambda p:SwmmRunoffSeries(
        list(stats),series.datetimes,q[None,:],3600))
    result=stability.rasterize_landuse_response(tmp_path,transform,(2,4),np.ones((2,4),bool),
        [('2024-07-12 00:00:00',10.)],stats)
    mapped=result['hourly_runoff_depth_m'][0,result['subcatchment_index']]
    assert float(mapped.sum()*100)==pytest.approx(q.sum()*3600,rel=1e-6)


def test_parent_peaks_are_computed_after_synchronous_sum(tmp_path,monkeypatch):
    _,units,_,_,_=fixture_units(tmp_path,monkeypatch)
    rates=np.zeros((2,len(units)));rates[0,0]=4;rates[1,1]=5
    series=SwmmRunoffSeries([u['subcatchment'] for u in units],
        [datetime(2024,7,12),datetime(2024,7,12,1)],rates,3600)
    parent=response.aggregate_series(series,tmp_path)
    assert parent.subcatchment_names==['S1','S2']
    assert parent.runoff_cms[:,0].max()==5  # not 4+5
    np.testing.assert_allclose(parent.runoff_cms.sum(axis=1),rates.sum(axis=1))


def test_model_inspection_distinguishes_parents_from_internal_units(tmp_path,monkeypatch):
    parents,units,_,_,_=fixture_units(tmp_path,monkeypatch)
    model=tmp_path/'swmm/model';model.mkdir(parents=True)
    text='; LANDUSE_REPRESENTATION response_units\n[SUBCATCHMENTS]\n'+'\n'.join(
        f"{u['subcatchment']} {u['rain_gage']} {u['outfall']} {u['area_ha']} {u['imperv_pct']} 1 1 0" for u in units)
    (model/'model.inp').write_text(text,encoding='utf-8')
    response.write_csv(model/'auto_swmm_parameter_review.csv',parents)
    from watershed_hydro_mcp.result_builder import inspect_swmm_model
    result=inspect_swmm_model(tmp_path)
    assert result['subcatchment_count']==2 and result['internal_response_unit_count']==4
    assert result['area_ha']['min']==.04


def test_real_swmm_homogeneous_split_preserves_runoff(tmp_path):
    engine=Path(__file__).resolve().parents[2]/'native/swmm/runswmm.exe'
    if not engine.exists():pytest.skip('bundled SWMM engine unavailable')
    def run(name,areas):
        folder=tmp_path/name;folder.mkdir()
        lines=['[OPTIONS]','FLOW_UNITS CMS','INFILTRATION HORTON','START_DATE 07/12/2024',
               'START_TIME 00:00','END_DATE 07/12/2024','END_TIME 03:00','REPORT_STEP 0:01:00',
               'WET_STEP 0:00:30','DRY_STEP 0:01:00','[RAINGAGES]','G1 VOLUME 1:00 1 TIMESERIES TS1',
               '[TIMESERIES]','TS1 07/12/2024 00:00 30','TS1 07/12/2024 01:00 20','TS1 07/12/2024 02:00 0',
               '[SUBCATCHMENTS]']
        for i,a in enumerate(areas):lines.append(f'S{i} G1 O1 {a} 5 {100*a} 10 0')
        lines+=['[SUBAREAS]']+[f'S{i} .015 .18 1.5 5 25 OUTLET' for i in range(len(areas))]
        lines+=['[INFILTRATION]']+[f'S{i} 30 2 3.5 7 0' for i in range(len(areas))]
        lines+=['[OUTFALLS]','O1 0 FREE NO','[REPORT]','SUBCATCHMENTS ALL']
        inp=folder/'model.inp';inp.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        result=subprocess.run([str(engine),str(inp),str(folder/'model.rpt'),str(folder/'model.out')],capture_output=True,timeout=40)
        assert result.returncode==0,(folder/'model.rpt').read_text(errors='replace')
        return read_subcatchment_runoff(folder/'model.out').runoff_cms.sum(axis=1)
    original=run('one',[1.]);split=run('split',[.4,.6])
    np.testing.assert_allclose(original,split,rtol=2e-5,atol=2e-7)
