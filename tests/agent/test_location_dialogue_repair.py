"""Regression for the user's district task and subsequent spelling corrections."""
import json

import pytest

from slope_evaluation_agent.dialogue_router import DialogueRouter, RouteKind
from slope_evaluation_agent.intent import Intent, RuleParser
from slope_evaluation_agent.llm import LLMRequestFailed
from slope_evaluation_agent.models import AgentSession
from stubs import StubLLM
from test_web_conversation import a_client, a_session, say, web_for
from test_web_study_area import area_tools


@pytest.mark.parametrize('text,name', [
    ('我想评价永川区的小流域','永川区'),
    ('评价江北区的小流域','江北区'),
    ('帮我分析巫溪县','巫溪县'),
    ('选择重庆市九龙坡区','九龙坡区'),
    ('永定区','永定区'),('永安区','永安区')])
def test_district_tasks_are_not_rivers_or_existing_results(text,name):
    parsed=RuleParser().parse(text,AgentSession(session_id='session-abcd1234',state='new'))
    assert parsed.intent is Intent.START_ASSESSMENT
    assert parsed.administrative_area==name and parsed.river_name is None


@pytest.mark.asyncio
async def test_the_exact_three_turns_keep_the_task_and_clarify_changed_names(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,spatial=area_tools())
    # Simulate an unavailable router. The user must still be able to locate a
    # valid district, and an invalid name must never silently select another.
    agent.router.llm=StubLLM(LLMRequestFailed('HTTP 503',reason='http_503'))
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        first=await say(client,sid,'我想评价永川区的小流域')
        assert first['reply']['outlet_map']['administrative_area']=='永川区'
        assert agent.store.load(sid).river_name is None
        for name in ('永定区','永安区'):
            answer=(await say(client,sid,name))['reply']['text']
            assert name in answer and '永川区' in answer
            assert '我没能确定你想做什么' not in answer
            assert agent.store.load(sid).intent=='start_assessment'
            assert agent.store.load(sid).administrative_area=='永川区'
        assert calls.spatial.count('render_administrative_dem')==1
        assert 'delineate_from_outlet' not in calls.spatial and not calls.hydro
        diag=[e for e in agent.store.read_events(sid) if e['event']=='dialogue_route_diagnostic']
        assert len(diag)==3 and all(e['reason']=='http_503' for e in diag)


@pytest.mark.asyncio
async def test_model_format_error_is_repaired_before_falling_back_to_rules():
    class RepairingModel:
        def __init__(self): self.requests=[]
        async def complete_json(self,*,system,user):
            self.requests.append(json.loads(user))
            if len(self.requests)==1:
                return {'route':'clarification','intent':None,'confidence':.8,
                        'missing_fields':['district_name']}
            return {'route':'workflow_action','intent':'start_assessment','confidence':.95,
                    'candidates':[{'field':'administrative_area','value':'永川区',
                                   'evidence_text':'永川区'}]}
    model=RepairingModel()
    router=DialogueRouter(llm=model)
    decision=await router.decide('我想评价永川区的小流域',AgentSession(session_id='session-abcd1234',state='new'))
    assert decision.route is RouteKind.WORKFLOW_ACTION and decision.intent is Intent.START_ASSESSMENT
    assert len(model.requests)==2 and 'schema_repair' in model.requests[1]
    assert router.last_diagnostic['recovered'] is True


@pytest.mark.asyncio
async def test_targeted_model_question_replaces_generic_capability_boilerplate(tmp_path,calls):
    question='你说的是前面提到的永川区，还是其他地区的永定区？'
    app,agent,_=web_for(tmp_path,calls,model_reply={
        'route':'clarification','intent':'start_assessment','confidence':.8,
        'missing_fields':['administrative_area'],'ambiguity':question})
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        response=(await say(client,sid,'永定区'))['reply']
        assert response['text']==question
        assert not calls.spatial and not calls.hydro


@pytest.mark.asyncio
async def test_district_geography_question_remains_model_knowledge(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,model_reply={
        'route':'knowledge_chat','intent':'ask_explanation','confidence':.97,'topic':'geography'})
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        await say(client,sid,'永川区有哪些河流')
        assert not calls.spatial and not calls.hydro
        assert agent.store.load(sid).administrative_area is None


@pytest.mark.parametrize('text',['对，就是永川','不对，是永川区'])
def test_district_correction_is_not_boundary_confirmation(text):
    session=AgentSession(session_id='session-abcd1234',state='new',administrative_area='永川区')
    parsed=RuleParser().parse(text,session)
    assert parsed.intent is Intent.START_ASSESSMENT and parsed.administrative_area=='永川区'


def test_named_river_with_district_hint_is_preserved():
    parsed=RuleParser().parse('分析永川区的临江河',AgentSession(session_id='session-abcd1234',state='new'))
    assert parsed.administrative_area=='永川区' and parsed.river_name=='临江河'
    assert parsed.intent is Intent.START_ASSESSMENT


@pytest.mark.asyncio
@pytest.mark.parametrize('text',['永川区在哪里','永川区面积多大','永川区有哪些河流'])
async def test_model_failure_does_not_turn_district_fact_questions_into_map_tasks(tmp_path,calls,text):
    app,agent,_=web_for(tmp_path,calls,spatial=area_tools())
    agent.router.llm=StubLLM(LLMRequestFailed('HTTP 503',reason='http_503'))
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        response=await say(client,sid,text)
        assert response['reply']['route']=='knowledge_chat'
        assert not calls.spatial and not calls.hydro
