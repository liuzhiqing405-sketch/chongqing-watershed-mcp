import pytest

from stubs import envelope, run_result_payload
from test_task_workspace import POINTS, restorable_tools
from test_web_conversation import web_for, a_client, a_session, say
from watershed_exchange.domain_policy import LOCAL_SCOPE, modeling_check
from slope_evaluation_agent.task_workspace import revision_rows
from slope_evaluation_agent.evaluation import build_assessment, describe_assessment
from slope_evaluation_agent.models import AgentSession


def partial_tools():
    tools=restorable_tools()
    review=tools['get_watershed_review']()['data']
    review['assessment_mode']='regional_screening'
    diagnostic={'checked':True,'passed':False,'classification':'upstream_truncated',
                'external_upstream_area_km2':3.4,'major_outlet_count':2,'errors':['边界切断上游。'],'warnings':[]}
    review['qa'].update(hydrologic_check=diagnostic,modeling_check=modeling_check(diagnostic,'regional_screening'))
    tools['create_drawn_study_area']=lambda:envelope(data=review)
    return tools


@pytest.mark.asyncio
async def test_partial_region_can_confirm_plan_and_restore_without_hiding_diagnostic(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,spatial=partial_tools())
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        loaded=(await client.post(f'/api/v1/sessions/{sid}/study-area-map',json={'administrative_area':'永川区'})).json()
        submitted=(await client.post(f'/api/v1/sessions/{sid}/study-area',json={
            'map_id':loaded['map_id'],'administrative_area':'永川区','coordinates':POINTS,'assessment_mode':'regional_screening'})).json()
        assert submitted['session']['state']=='awaiting_watershed_confirmation'
        checked=submitted['session']['workspace']['spatial_qa']
        assert LOCAL_SCOPE in checked['modeling_check']['boundary_assumptions']
        assert not checked['hydrologic_check']['passed']
        assert '当前范围不能确认' not in submitted['reply']['text']
        await say(client,sid,'确认该边界')
        planned=await say(client,sid,'2024-07-30 全天')
        assert LOCAL_SCOPE not in planned['reply']['text']  # explain on request, retain evidence below
        current=agent.store.load(sid)
        assert current.hydro_planned and current.watershed_confirmed
        assert not current.study_area_qa['hydrologic_check']['passed']
        workspace=(await client.get(f'/api/v1/sessions/{sid}/task')).json()['session']['workspace']
        assert workspace['background']['boundary']['checks']['modeling_check']['boundary_assumptions']==[LOCAL_SCOPE]
        old_input=current.hydro_input_id
        await client.post(f'/api/v1/sessions/{sid}/task/edit',json={'scope':'boundary'})
        revision=next(row for row in reversed(revision_rows(agent.store,sid)) if row['snapshot']['hydro_input_id']==old_input)
        await client.post(f'/api/v1/sessions/{sid}/task/restore',json={'revision_id':revision['revision_id']})
        current=agent.store.load(sid)
        assert current.watershed_confirmed and current.hydro_input_id==old_input
        assert current.assessment_mode=='regional_screening' and not current.hydro_planned
        assert not current.study_area_qa['hydrologic_check']['passed']
        assert 'start_hydro_run' not in calls.hydro


@pytest.mark.asyncio
async def test_mode_only_change_blocks_old_plan_and_policy_mismatch_is_rejected(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,spatial=partial_tools())
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        loaded=(await client.post(f'/api/v1/sessions/{sid}/study-area-map',json={'administrative_area':'永川区'})).json()
        path=f'/api/v1/sessions/{sid}/study-area'
        body={'map_id':loaded['map_id'],'administrative_area':'永川区','coordinates':POINTS,'assessment_mode':'regional_screening'}
        await client.post(path,json=body);await say(client,sid,'确认该边界');await say(client,sid,'2024-07-30 全天')
        changed=(await client.put(f'/api/v1/sessions/{sid}/task/draft',json={
            'map_id':loaded['map_id'],'coordinates':POINTS,'closed':True,'assessment_mode':'catchment_check'})).json()
        assert not changed['session']['workspace']['actions']['start']
        assert not (await say(client,sid,'开始运行'))['reply']['ok']
        assert 'start_hydro_run' not in calls.hydro
        strict=(await client.post(path,json={**body,'assessment_mode':'catchment_check'})).json()
        assert strict['session']['state']=='watershed_qa_failed'
        assert not agent.store.load(sid).watershed_confirmed


def test_final_assessment_uses_run_snapshot_scope_instead_of_current_task_mode():
    policy=modeling_check({'checked':True,'passed':False,'errors':['外部汇水']},'regional_screening')
    result={**run_result_payload(),'domain_policy':policy,'limitations':[LOCAL_SCOPE]}
    session=AgentSession(session_id='session-abcdef123456',state='hydro_succeeded',hydro_run_id='hydro-run-abcdef123456',assessment_mode='catchment_check')
    assessment=build_assessment(session,result)
    assert assessment['domain_policy']==policy
    assert assessment['limitations'][0]==LOCAL_SCOPE
    assert LOCAL_SCOPE in describe_assessment(assessment)
