import pytest

from slope_evaluation_agent.dialogue_router import DialogueRouter
from slope_evaluation_agent.intent import Intent
from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.state_machine import WorkflowState
from stubs import StubLLM, envelope, run_status_payload
from test_web_conversation import web_for, a_client, say


def session(state='hydro_succeeded'):
    return AgentSession(session_id='session-abcdef123456',state=state,
                        hydro_run_id='hydro-run-abcdef123456',hydro_run_status='succeeded' if state=='hydro_succeeded' else 'running')


def decision(text,confidence=.75,kind='delineate_subcatchments'):
    return {'route':'workflow_action','intent':'show_visualization','confidence':confidence,
            'candidates':[{'field':'visualization_kind','value':kind,'evidence_text':text}]}


@pytest.mark.asyncio
@pytest.mark.parametrize('text',['我想看一下子汇水区','查看汇水区','可视化汇水区'])
async def test_model_recognized_view_is_not_rejected_at_execution_threshold(text):
    router=DialogueRouter(llm=StubLLM(decision(text)))
    result=await router.decide(text,session())
    assert result.route=='workflow_action' and result.intent==Intent.SHOW_VISUALIZATION
    assert result.confidence==.75
    assert router.last_source=='model'


@pytest.mark.asyncio
async def test_calculation_threshold_and_knowledge_judgment_stay_separate():
    model=StubLLM({'route':'workflow_action','intent':'start_hydro_run','confidence':.75})
    assert (await DialogueRouter(llm=model).decide('运行吧',session())).route=='clarification'
    model=StubLLM({'route':'knowledge_chat','intent':'ask_explanation','confidence':.75,'topic':'子汇水区定义'})
    assert (await DialogueRouter(llm=model).decide('子汇水区怎么理解',session())).route=='knowledge_chat'


@pytest.mark.asyncio
async def test_existing_subcatchment_figure_is_returned_without_restarting_model(tmp_path,calls):
    text='我想看一下子汇水区'
    app,agent,_=web_for(tmp_path,calls,model_reply=decision(text),hydro={
        'list_hydro_visualization_options':envelope(data={'options':[{'kind':'delineate_subcatchments','stage_id':'delineate','title':'子汇水区划分'}]})})
    async with agent,a_client(app) as client:
        current=session();current.administrative_area='永川区';current.review_id='basin-abcdef123456'
        current.watershed_confirmed=True;current.rainfall_request={'mode':'historical_max','lookback_years':30}
        agent.store.save(current)
        reply=await say(client,current.session_id,text)
        assert reply['reply']['images'] and reply['reply']['ok']
        assert 'create_hydro_visualization' in calls.hydro
        assert 'start_hydro_run' not in calls.hydro and not calls.spatial
        restored=agent.store.load(current.session_id)
        assert restored.administrative_area=='永川区' and restored.hydro_run_id==current.hydro_run_id
        assert restored.review_id==current.review_id and restored.watershed_confirmed
        assert restored.rainfall_request==current.rainfall_request


@pytest.mark.asyncio
@pytest.mark.parametrize('finished',[True,False])
async def test_running_job_can_show_completed_stage_only(tmp_path,calls,finished):
    text='查看子汇水区划分图'
    status=run_status_payload(status='running');status['completed_stages']=['sync','validate']+(['delineate'] if finished else [])
    status['current_stage']='swmm' if finished else 'delineate'
    app,agent,_=web_for(tmp_path,calls,model_reply=decision(text),hydro={
        'get_hydro_run_status':envelope(data=status),
        'list_hydro_visualization_options':envelope(data={'options':[{'kind':'delineate_subcatchments','stage_id':'delineate','title':'子汇水区划分'}]})})
    async with agent,a_client(app) as client:
        current=session('hydro_running');agent.store.save(current)
        reply=await say(client,current.session_id,text)
        assert reply['session']['state']=='hydro_running'
        assert ('create_hydro_visualization' in calls.hydro)==finished
        assert 'start_hydro_run' not in calls.hydro
        if not finished:
            assert '阶段还未完成' in reply['reply']['text']


@pytest.mark.asyncio
async def test_explicit_visualization_ambiguity_asks_specific_question(tmp_path,calls):
    text='看看分区'
    model={**decision(text),'ambiguity':'你想看子汇水区划分图还是评价边界图？'}
    app,agent,_=web_for(tmp_path,calls,model_reply=model)
    async with agent,a_client(app) as client:
        current=session();agent.store.save(current)
        reply=await say(client,current.session_id,text)
        assert '子汇水区' in reply['reply']['text'] and '评价边界' in reply['reply']['text']
        assert '分析大宁河' not in reply['reply']['text']
        assert not calls.hydro and not calls.spatial


@pytest.mark.asyncio
async def test_boundary_figure_remains_accessible_before_run(tmp_path,calls):
    text='查看已画的评价边界图'
    app,agent,_=web_for(tmp_path,calls,model_reply=decision(text,kind='watershed'))
    async with agent,a_client(app) as client:
        current=AgentSession(session_id='session-abcdef123456',state='watershed_confirmed',review_id='basin-abcdef123456',watershed_confirmed=True)
        agent.store.save(current)
        reply=await say(client,current.session_id,text)
        assert reply['reply']['images'] and '评价边界图' in reply['reply']['text']
        assert 'render_process_artifact' in calls.spatial and not calls.hydro


def test_model_gets_available_semantic_figure_targets_and_actual_stage_context():
    router=DialogueRouter()
    assert router.workflow_reference()['visualization_targets']['delineate_subcatchments']
    current=session('hydro_running');current.run_status_detail={'completed_stages':['delineate'],'current_stage':'safety'}
    assert router.context(current)['known']['completed_stages']==['delineate']
