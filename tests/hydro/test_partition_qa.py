import json
from pathlib import Path

import pytest

from watershed_hydro_mcp.qa import spatial_gate
from watershed_exchange.paths import sha256_file
from watershed_hydro_mcp.job_manager import JobStore, JobError
from watershed_hydro_mcp.state import write_json, read_json
from test_units import temporary_settings


def partition_project(tmp_path):
    directory=tmp_path/'inputs/dem_derived';directory.mkdir(parents=True)
    summary=directory/'dem_derived_subcatchments_summary.csv'
    summary.write_text('name,area_ha,source\nD1,100,dem_delineation\nD2,100,dem_delineation\nD3,100,dem_delineation\n',encoding='utf-8')
    shape=directory/'event_subcatchments.shp';shape.write_bytes(b'bound-partition')
    record={'schema_version':1,'algorithm':'mask_boundary_priority_flood_d8_parent_flats',
            'topology':{'passed':True,'valid_cells':30000,'labelled_cells':30000,'labels_outside_domain':0,'internal_terminal_cells':0},
            'model_budget':{'passed':True,'limit':10000},'coarsening':{'partition_count':3,'major_target_reached':True,'target_reached':False},
            'fill_above_1m_fraction':.04,'maximum_fill_depth_m':15,'valid_area_ha':300,
            'partition_sha256':sha256_file(shape),'summary_sha256':sha256_file(summary)}
    return directory,record


def save(directory,record):
    (directory/'dem_derived_subcatchments_diagnostics.json').write_text(json.dumps(record),encoding='utf-8')


def test_bound_verified_partition_passes_and_explains_boundary_units(tmp_path):
    directory,record=partition_project(tmp_path);save(directory,record)
    result=spatial_gate(tmp_path,{})
    assert result['passed'] and result['warnings']


@pytest.mark.parametrize('field,value',[
    ('fill_above_1m_fraction',.79),('maximum_fill_depth_m',190),
    ('valid_area_ha',250),('partition_sha256','old-geometry'),
    ('algorithm','distance_to_mask_edge_gradient')])
def test_distorted_or_stale_partition_is_rejected(tmp_path,field,value):
    directory,record=partition_project(tmp_path);record[field]=value;save(directory,record)
    assert not spatial_gate(tmp_path,{})['passed']


def test_unverified_legacy_dem_partition_cannot_keep_claiming_qa_passed(tmp_path):
    directory,_=partition_project(tmp_path)
    result=spatial_gate(tmp_path,{})
    assert not result['passed'] and any('重新划分' in e for e in result['errors'])


def test_changed_summary_and_internal_terminals_are_rejected(tmp_path):
    directory,record=partition_project(tmp_path);save(directory,record)
    (directory/'dem_derived_subcatchments_summary.csv').write_text('name,area_ha,source\nD1,300,dem_delineation\n',encoding='utf-8')
    assert not spatial_gate(tmp_path,{})['passed']
    record['topology']['internal_terminal_cells']=1;save(directory,record)
    assert not spatial_gate(tmp_path,{})['passed']


@pytest.mark.parametrize('prior,expected',[('succeeded','qa_failed'),('qa_failed','qa_failed'),('cancelled','cancelled')])
def test_rechecking_failure_updates_terminal_status_without_restarting(tmp_path,prior,expected):
    store=JobStore(temporary_settings(tmp_path));run_id='hydro-run-abcdef123456'
    write_json(store.status_path(run_id),{'run_id':run_id,'status':prior,'completed_stages':['delineate']})
    report={'checked_at':'2026-10-05T00:00:00+00:00','passed':False,'gates_applied':True,'failed':['spatial']}
    store.record_quality_report(run_id,report)
    assert store.status(run_id)['status']==expected
    assert store.status(run_id)['qa_passed'] is False
    assert store.status(run_id)['qa_gates_applied'] is True
    assert read_json(store.job_dir(run_id)/'qa.json')==report
    assert not store._processes


def test_qa_recheck_does_not_change_queued_job(tmp_path):
    store=JobStore(temporary_settings(tmp_path));run_id='hydro-run-abcdef123456'
    write_json(store.status_path(run_id),{'run_id':run_id,'status':'queued'})
    with pytest.raises(JobError):
        store.record_quality_report(run_id,{'checked_at':'now','passed':False,'gates_applied':True})
    assert not (store.job_dir(run_id)/'qa.json').exists()
