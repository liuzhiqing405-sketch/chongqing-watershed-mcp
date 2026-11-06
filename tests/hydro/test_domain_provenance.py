from unittest.mock import patch

from watershed_exchange.domain_policy import LOCAL_SCOPE, modeling_check
from watershed_hydro_mcp.job_manager import JobStore
from watershed_hydro_mcp.models import HydroRunRequest
from watershed_hydro_mcp.result_builder import summarize
from watershed_hydro_mcp.settings import Settings
from watershed_hydro_mcp.state import read_json


def test_submitted_job_and_result_keep_original_imported_domain_assumptions(tmp_path):
    settings=Settings(tmp_path,tmp_path/'runs',tmp_path/'hydro',tmp_path/'native',tmp_path/'config')
    settings.engine.parent.mkdir(parents=True);settings.engine.touch()
    (settings.engine.parent/'swmm5.dll').touch()
    store=JobStore(settings)
    policy=modeling_check({'checked':True,'passed':False,'errors':['框外入流']},'regional_screening')
    record={'provenance':{'domain_policy':policy},'limitations':[LOCAL_SCOPE,'框外汇水未计入']}
    with patch('watershed_hydro_mcp.job_manager.load_record',return_value=record), \
         patch('watershed_hydro_mcp.job_manager.validate_input',return_value={'ok':True,'errors':[],'warnings':[]}), \
         patch('watershed_hydro_mcp.job_manager.planner.build_project',return_value={'project':str(tmp_path/'project')}), \
         patch.object(store,'_ensure_scheduler'),patch.object(store,'_pump'):
        status=store.start(HydroRunRequest(input_id='hydro-input-abcdef123456',mode='build_run_qa'))
    job_dir=store.job_dir(status['run_id'])
    assert read_json(job_dir/'job.json')['domain_policy']==policy
    record['provenance']['domain_policy']={}
    result=summarize(job_dir,tmp_path/'project')
    assert result['domain_policy']==policy
    assert result['limitations'][0]==LOCAL_SCOPE
    assert '框外汇水未计入' in result['limitations']
