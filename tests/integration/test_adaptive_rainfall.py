"""Actual MCP output is imported and used by a real SWMM model build."""
import csv
import json
import logging
import shutil
import sys
from pathlib import Path

import pytest
import yaml
import shapefile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures import build_package, landuse_features
from test_integration import make_settings, wait_for
from slope_evaluation_agent.config import ServerSettings
from slope_evaluation_agent.mcp_clients import SpatialMCPClient
from slope_evaluation_agent.models import RainfallRequestPayload
from watershed_hydro_mcp.input_registry import import_package
from watershed_hydro_mcp.job_manager import JobStore
from watershed_hydro_mcp.models import HydroRunRequest
from watershed_exchange.paths import sha256_file


@pytest.mark.asyncio
@pytest.mark.parametrize("count,expected", [(None, 1), (1, 1), (6, 6), (100, 100)])
async def test_real_mcp_rainfall_reaches_import_and_swmm_build(tmp_path,monkeypatch,count,expected):
    settings=make_settings(tmp_path)
    source=build_package(tmp_path/'exchange/exports/package-abcdef123456',package_id='package-abcdef123456')
    # The general exchange fixture has a display Name field; the scientific
    # land-use classifier reads DLMC. Supply a modelling-ready synthetic layer.
    with shapefile.Writer(str(source/'landuse'),shapeType=shapefile.POLYGON,encoding='gbk') as writer:
        writer.field('DLMC','C',60)
        for name,ring in landuse_features():
            writer.poly([list(reversed(ring))]);writer.record(name)
    data=tmp_path/'data/rain';data.mkdir(parents=True)
    rainfile=data/'rain_2024.csv'
    rainfile.write_text('time,106.5_29.7,106.7_29.7,106.5_29.9,106.7_29.9\n'+
        '\n'.join(f'2024-06-01 {hour:02d}:00,1,2,3,4' for hour in range(6))+'\n',encoding='utf-8')
    config=tmp_path/'spatial.yaml'
    config.write_text(yaml.safe_dump({'datasets':{'rainfall_directory':'rain','rainfall_pattern':'rain_*.csv'}}),encoding='utf-8')
    reviewdir=tmp_path/'runs/watersheds/basin-abcdef123456';reviewdir.mkdir(parents=True)
    (reviewdir/'review.json').write_text(json.dumps({'review_id':reviewdir.name,'status':'confirmed',
        'qa':{'passed':True},'artifacts':{'boundary_geojson':str(source/'basin.geojson')}}),encoding='utf-8')
    for name,value in {'CQ_DATA_ROOT':tmp_path/'data','CQ_CONFIG':config,'CQ_RUNS_ROOT':tmp_path/'runs',
                       'CQ_EXCHANGE_ROOT':tmp_path/'exchange'}.items():
        monkeypatch.setenv(name,str(value))
    client=SpatialMCPClient(ServerSettings(args=['-m','chongqing_spatial_mcp.server']),logging.getLogger('adaptive-test'))
    async with client:
        reply=await client.build_virtual_rainfall(RainfallRequestPayload(review_id=reviewdir.name,
            mode='explicit_period',start_datetime='2024-06-01 00:00',
            end_datetime='2024-06-01 05:00',station_count=count))
    package=reply['data']
    assert package['station_count']==len(package['stations'])==expected
    assert package['station_selection']['mode']==('auto' if count is None else 'manual')
    shutil.copy2(package['station_zip'],source/'station_rainfall.zip')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        member=source/entry['relative_path']
        entry['sha256']=sha256_file(member)
        entry['size_bytes']=member.stat().st_size
    (source/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    record=import_package(source.name,settings)
    assert record['temporal']['station_count']==expected
    store=JobStore(settings)
    status=store.start(HydroRunRequest(input_id=record['input_id'],through='build'))
    final=wait_for(store,status['run_id'])
    assert final['status']=='succeeded',final
    project=settings.job_dir(status['run_id'])/'project'
    models=list((project/'swmm/model').glob('*.inp'))
    assert models and '[RAINGAGES]' in models[0].read_text(encoding='utf-8')
    summary=next((project/'inputs/rainfall').glob('*_station_rainfall_summary.csv'))
    with summary.open(encoding='utf-8-sig',newline='') as stream:
        assert len(list(csv.DictReader(stream)))==expected
    from watershed_hydro_mcp.qa import rainfall_gate
    from watershed_hydro_mcp.planner import load_gates
    check=rainfall_gate(project,load_gates()['rainfall'])
    assert check['passed'],check
    if count==1:
        assert check['facts']['spatial_rainfall_assumption']=='single_station_uniform'
