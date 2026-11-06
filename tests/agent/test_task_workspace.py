"""Exercise reversible task state through the same API used by the browser."""
import json

import pytest

from helpers import PRELUDE, step
from stubs import StubLLM, envelope, run_status_payload
from test_web_conversation import web_for, a_client, a_session, say
from test_web_study_area import area_tools
from slope_evaluation_agent.conversation_responder import ConversationResponder
from slope_evaluation_agent.task_workspace import revision_rows

pytestmark=pytest.mark.unit
POINTS=[[105.9,29.4],[105.93,29.4],[105.93,29.43],[105.9,29.4]]


async def drawn(client,sid):
    loaded=(await client.post(f'/api/v1/sessions/{sid}/study-area-map',json={'administrative_area':'永川区'})).json()
    response=await client.post(f'/api/v1/sessions/{sid}/study-area',json={
        'map_id':loaded['map_id'],'administrative_area':'永川区','coordinates':POINTS})
    assert response.status_code==200,response.text
    await say(client,sid,'确认该边界')
    return loaded


def restorable_tools():
    tools=area_tools()
    record=tools['create_drawn_study_area']()['data']
    record['status']='confirmed'
    tools['get_watershed_review']=lambda:envelope(data=record)
    return tools


@pytest.mark.asyncio
async def test_boundary_edit_and_restore_keep_session_and_revalidate_review(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,spatial=restorable_tools())
    async with agent,a_client(app) as client:
        sid=await a_session(client); loaded=await drawn(client,sid)
        await say(client,sid,'2024-07-30 全天')
        old=agent.store.load(sid);old_input=old.hydro_input_id
        agent.store.save_evidence(sid,{'old_metric':.275})
        agent.store.save_assessment(sid,{'old_metric':.275})
        changed=await client.post(f'/api/v1/sessions/{sid}/task/edit',json={'scope':'boundary'})
        assert changed.status_code==200,changed.text
        session=agent.store.load(sid)
        assert session.session_id==sid and not session.watershed_confirmed
        assert session.study_area_coordinates==POINTS and not session.hydro_input_id
        assert not session.hydro_planned and not agent.store.load_evidence(sid)
        rows=revision_rows(agent.store,sid)
        revision=next(row for row in reversed(rows) if row['snapshot']['hydro_input_id']==old_input)
        result=await client.post(f'/api/v1/sessions/{sid}/task/restore',json={'revision_id':revision['revision_id']})
        assert result.status_code==200,result.text
        restored=agent.store.load(sid)
        assert restored.watershed_confirmed and restored.hydro_input_id==old_input
        assert restored.selection_context['map_id']!=loaded['map_id']
        assert restored.state=='hydro_input_validated' and not restored.hydro_planned
        assert not agent.store.load_evidence(sid), 'Input revision must not revive uncomputed results.'
        assert 'get_watershed_review' in calls.spatial
        assert 'start_hydro_run' not in calls.hydro
        assert len(restored.conversation)>len(old.conversation)
        public=(await client.get(f'/api/v1/sessions/{sid}/task')).json()
        assert public['session']['workspace']['location']['map']['url']
        assert 'snapshot' not in public['revisions'][0]


@pytest.mark.asyncio
async def test_rain_edit_is_validated_before_invalidating_and_prepare_never_runs(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,spatial=restorable_tools())
    async with agent,a_client(app) as client:
        sid=await a_session(client);await drawn(client,sid)
        await say(client,sid,'2024-07-30 全天')
        before=agent.store.load(sid).model_dump()
        bad=await client.post(f'/api/v1/sessions/{sid}/task/edit',json={
            'scope':'rainfall','rainfall':{'mode':'explicit_period','date':'不是日期'}})
        assert bad.status_code==409
        assert agent.store.load(sid).model_dump()==before
        edited=await client.post(f'/api/v1/sessions/{sid}/task/edit',json={
            'scope':'rainfall','rainfall':{'mode':'explicit_period','date':'2024-08-01','station_count':6}})
        assert edited.status_code==200,edited.text
        current=agent.store.load(sid)
        assert current.watershed_confirmed and not current.hydro_input_id and not current.hydro_planned
        assert not current.rainfall_context
        assert edited.json()['session']['workspace']['actions']['prepare']
        prepared=await client.post(f'/api/v1/sessions/{sid}/task/prepare')
        assert prepared.status_code==200,prepared.text
        current=agent.store.load(sid)
        assert current.hydro_planned and current.hydro_input_validated
        assert current.rainfall_request['station_count']==6
        assert current.rainfall_request['date']=='2024-08-01'
        assert 'start_hydro_run' not in calls.hydro
        assert (await client.post(f'/api/v1/sessions/{sid}/task/prepare')).status_code==409


@pytest.mark.asyncio
async def test_station_only_api_edit_retains_event_and_can_restore_auto(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,spatial=restorable_tools())
    async with agent,a_client(app) as client:
        sid=await a_session(client);await drawn(client,sid)
        await say(client,sid,'2024-07-30 全天')
        for count in (6, None):
            before=agent.store.load(sid).model_dump()
            bad=await client.post(f'/api/v1/sessions/{sid}/task/edit',json={
                'scope':'rainfall','rainfall':{'station_count':101}})
            assert bad.status_code==409
            assert agent.store.load(sid).model_dump()==before
            edited=await client.post(f'/api/v1/sessions/{sid}/task/edit',json={
                'scope':'rainfall','rainfall':{'station_count':count}})
            assert edited.status_code==200,edited.text
            current=agent.store.load(sid)
            assert current.rainfall_request.get('station_count')==count
            assert current.rainfall_request['date']=='2024-07-30'
            assert current.watershed_confirmed and not current.hydro_input_id
            prepared=await client.post(f'/api/v1/sessions/{sid}/task/prepare')
            assert prepared.status_code==200,prepared.text
        assert 'start_hydro_run' not in calls.hydro


@pytest.mark.asyncio
async def test_parameter_edit_keeps_inputs_requires_new_plan_and_rejects_bad_values(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,spatial=restorable_tools())
    async with agent,a_client(app) as client:
        sid=await a_session(client);await drawn(client,sid);await say(client,sid,'2024-07-30 全天')
        old=agent.store.load(sid)
        bad=(await client.post(f'/api/v1/sessions/{sid}/task/edit',json={
            'scope':'parameters','parameter_overrides':{'saturated_cohesion_kpa':999}})).json()
        assert not bad['reply']['ok']
        assert agent.store.load(sid).hydro_planned
        edited=await client.post(f'/api/v1/sessions/{sid}/task/edit',json={
            'scope':'parameters','parameter_preset':'high','parameter_overrides':{'saturated_cohesion_kpa':15}})
        assert edited.status_code==200,edited.text
        current=agent.store.load(sid)
        assert current.hydro_input_id==old.hydro_input_id and current.hydro_input_validated
        assert current.parameter_preset=='high' and not current.hydro_planned
        reply=await say(client,sid,'开始运行')
        assert not reply['reply']['ok'] and 'start_hydro_run' not in calls.hydro
        planned=await client.post(f'/api/v1/sessions/{sid}/task/plan')
        assert planned.status_code==200,planned.text
        assert agent.store.load(sid).hydro_planned


@pytest.mark.asyncio
async def test_drafts_survive_reload_and_stale_maps_cannot_write_or_restore_other_session(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,spatial=restorable_tools())
    async with agent,a_client(app) as client:
        sid=await a_session(client);loaded=await drawn(client,sid)
        path=f'/api/v1/sessions/{sid}/task/draft'
        body={'map_id':loaded['map_id'],'coordinates':POINTS[:-1],'closed':True}
        saved=await client.put(path,json=body)
        assert saved.status_code==200,saved.text
        assert agent.store.load(sid).study_area_draft['coordinates']==POINTS[:-1]
        view=(await client.get(f'/api/v1/sessions/{sid}')).json()['workspace']
        assert view['draft']['closed']
        changed=await client.post(f'/api/v1/sessions/{sid}/task/edit',json={'scope':'location'})
        assert changed.status_code==200
        assert (await client.put(path,json=body)).status_code==409
        assert not agent.store.load(sid).study_area_draft
        row=revision_rows(agent.store,sid)[-1]
        other=await a_session(client)
        assert (await client.post(f'/api/v1/sessions/{other}/task/restore',json={'revision_id':row['revision_id']})).status_code==409
        assert not agent.store.load(other).review_id


@pytest.mark.asyncio
async def test_changed_draft_blocks_run_but_identical_open_ring_does_not(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,spatial=restorable_tools())
    async with agent,a_client(app) as client:
        sid=await a_session(client);loaded=await drawn(client,sid);await say(client,sid,'2024-07-30 全天')
        path=f'/api/v1/sessions/{sid}/task/draft'
        points=[p[:] for p in POINTS[:-1]];points[1][0]+=.01
        changed=(await client.put(path,json={'map_id':loaded['map_id'],'coordinates':points,'closed':True})).json()
        assert not changed['session']['workspace']['actions']['start']
        assert not (await say(client,sid,'开始运行'))['reply']['ok']
        assert not calls.hydro.count('start_hydro_run')
        await client.put(path,json={'map_id':loaded['map_id'],'coordinates':POINTS[:-1],'closed':True})
        assert (await say(client,sid,'开始运行'))['reply']['ok']
        assert calls.hydro.count('start_hydro_run')==1


@pytest.mark.asyncio
@pytest.mark.parametrize('scope',['location','boundary','parameters','rainfall'])
async def test_active_run_prevents_edits_without_losing_run(tmp_path,calls,scope):
    app,agent,_=web_for(tmp_path,calls,spatial=restorable_tools())
    async with agent,a_client(app) as client:
        sid=await a_session(client);await drawn(client,sid);await say(client,sid,'2024-07-30 全天');await say(client,sid,'开始运行')
        body={'scope':scope}
        if scope=='rainfall':body['rainfall']={'mode':'explicit_period','date':'2024-08-01'}
        response=await client.post(f'/api/v1/sessions/{sid}/task/edit',json=body)
        assert response.status_code==409,response.text
        assert agent.store.load(sid).hydro_run_id and agent.store.load(sid).watershed_confirmed
        assert 'get_hydro_run_status' in calls.hydro


@pytest.mark.asyncio
async def test_restore_rechecks_failed_hydrology_and_revokes_inputs(tmp_path,calls):
    tools=restorable_tools();record=tools['get_watershed_review']()['data']
    app,agent,_=web_for(tmp_path,calls,spatial=tools)
    async with agent,a_client(app) as client:
        sid=await a_session(client);await drawn(client,sid);await say(client,sid,'2024-07-30 全天')
        await client.post(f'/api/v1/sessions/{sid}/task/edit',json={'scope':'location'})
        revision=revision_rows(agent.store,sid)[-1]
        record['qa']['hydrologic_check']['passed']=False
        response=await client.post(f'/api/v1/sessions/{sid}/task/restore',json={'revision_id':revision['revision_id']})
        assert response.status_code==200,response.text
        current=agent.store.load(sid)
        assert current.state=='watershed_qa_failed' and not current.watershed_confirmed
        assert not current.hydro_input_id and not current.hydro_planned
        assert not current.study_area_qa['hydrologic_check']['passed']


@pytest.mark.asyncio
async def test_restore_finished_run_reconciles_actual_status_and_preserves_evidence(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,spatial=restorable_tools(),hydro={
        'get_hydro_run_status':envelope(data=run_status_payload(status='succeeded'))})
    async with agent,a_client(app) as client:
        sid=await a_session(client);await drawn(client,sid);await say(client,sid,'2024-07-30 全天');await say(client,sid,'开始运行')
        session=agent.store.load(sid);session.state='hydro_succeeded';session.hydro_run_status='succeeded';agent.store.save(session)
        agent.store.save_evidence(sid,{'marker':'verified old result'})
        await client.post(f'/api/v1/sessions/{sid}/task/edit',json={'scope':'parameters','parameter_overrides':{}})
        revision=revision_rows(agent.store,sid)[-1]
        restored=await client.post(f'/api/v1/sessions/{sid}/task/restore',json={'revision_id':revision['revision_id']})
        assert restored.status_code==200,restored.text
        assert agent.store.load(sid).state=='hydro_succeeded'
        assert agent.store.load_evidence(sid)['marker']=='verified old result'
        assert calls.hydro.count('start_hydro_run')==1


@pytest.mark.asyncio
async def test_knowledge_uses_readonly_task_context_without_mcp_or_state_change(build_agent,calls):
    agent=build_agent();session=agent.new_session()
    model=StubLLM({'answer':'你当前边界存在框外上游汇水，可参考分水岭重新绘制；尚不能据此判断稳定性。'})
    async with agent:
        for text in PRELUDE:
            await step(agent,session,text)
        session.study_area_qa={'passed':False,'hydrologic_check':{'external_area_fraction':.2,'errors':['框外上游汇水过多']}}
        agent.responder=ConversationResponder(llm=model)
        agent.router.llm=StubLLM({'route':'knowledge_chat','confidence':.98,'topic':'engineering_general'})
        before=(session.state,session.review_id,session.rainfall_request.copy(),session.hydro_input_id)
        calls.spatial.clear();calls.hydro.clear()
        reply=await agent.handle(session,'我选的区域为什么需要考虑框外的水？')
    assert reply.answer_source=='general_model_knowledge'
    assert before==(session.state,session.review_id,session.rainfall_request,session.hydro_input_id)
    assert not calls.spatial and not calls.hydro
    encoded=json.dumps(model.calls,ensure_ascii=False)
    assert 'task_background' in encoded and '2024-07-30' in encoded
    assert 'external_area_fraction' in encoded and '只读' in encoded
    assert 'prepared_rainfall' in encoded and 'event_start' in encoded


@pytest.mark.asyncio
async def test_public_status_is_real_stage_data_and_excludes_internal_paths(tmp_path,calls):
    status=run_status_payload();status['progress_percent']=42
    app,agent,_=web_for(tmp_path,calls,spatial=restorable_tools(),hydro={'get_hydro_run_status':envelope(data=status)})
    async with agent,a_client(app) as client:
        sid=await a_session(client);await drawn(client,sid);await say(client,sid,'2024-07-30 全天');await say(client,sid,'开始运行')
        response=await say(client,sid,'查看当前任务进度')
        detail=response['session']['workspace']['run_detail']
        assert detail['progress_percent']==42 and detail['current_stage']=='delineate'
        assert detail['completed_stages']==['sync','validate']
        assert 'project_directory' not in detail


@pytest.mark.asyncio
async def test_unsubmitted_draft_cannot_confirm_old_review(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,spatial=restorable_tools())
    async with agent,a_client(app) as client:
        sid=await a_session(client);await drawn(client,sid)
        current=agent.store.load(sid);current.watershed_confirmed=False
        current.state='awaiting_watershed_confirmation';agent.store.save(current)
        points=[p[:] for p in POINTS[:-1]];points[1][0]+=.01
        changed=await client.put(f'/api/v1/sessions/{sid}/task/draft',json={
            'map_id':current.selection_context['map_id'],'coordinates':points,'closed':True})
        assert not changed.json()['session']['workspace']['actions']['confirm']
        calls.spatial.clear()
        reply=await say(client,sid,'确认该边界')
        assert not reply['reply']['ok'] and 'confirm_watershed' not in calls.spatial


@pytest.mark.asyncio
async def test_failed_run_can_be_replanned_in_same_task_without_autostart(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,spatial=restorable_tools(),hydro={
        'get_hydro_run_status':envelope(data=run_status_payload(status='failed'))})
    async with agent,a_client(app) as client:
        sid=await a_session(client);await drawn(client,sid);await say(client,sid,'2024-07-30 全天');await say(client,sid,'开始运行')
        status=await say(client,sid,'查看当前任务进度')
        assert status['session']['workspace']['actions']['plan']
        response=await client.post(f'/api/v1/sessions/{sid}/task/plan')
        assert response.status_code==200,response.text
        current=agent.store.load(sid)
        assert current.hydro_planned and not current.hydro_run_id
        assert calls.hydro.count('start_hydro_run')==1
        assert any(row['snapshot']['hydro_run_status']=='failed' for row in revision_rows(agent.store,sid))
