"""The model decides speech intent; validated workflows retain execution gates."""
import json

import pytest

from slope_evaluation_agent.dialogue_router import DialogueRouter, RouteKind
from slope_evaluation_agent.intent import Intent
from slope_evaluation_agent.models import AgentSession
from stubs import StubLLM
from test_web_conversation import a_client, a_session, say, web_for
from test_web_study_area import area_tools


@pytest.mark.asyncio
@pytest.mark.parametrize('message',[
    '帮我分析水流怎样影响边坡',
    '分析嘉陵江为什么水流会造成侵蚀',
    '不对，我想问的是水流作用的机理',
    '我不是这个意思，我想了解建模方法',
    '这个说法是对的，给我讲讲孔隙水压力',
    '取消入渗项对计算结果有哪些影响',
])
async def test_question_or_correction_is_always_given_to_the_model(message):
    model=StubLLM({'route':'knowledge_chat','intent':'ask_explanation','confidence':.96})
    router=DialogueRouter(llm=model)
    session=AgentSession(session_id='session-abcdef12',state='awaiting_watershed_confirmation',
                         review_id='basin-abcdef123456',watershed_qa_passed=True)
    decision=await router.decide(message,session)
    parsed,_=router.to_parsed_intent(decision,message,session)
    assert len(model.calls)==1 and decision.route is RouteKind.KNOWLEDGE_CHAT
    assert parsed.intent is Intent.ASK_EXPLANATION
    assert parsed.river_name is None and parsed.administrative_area is None
    assert session.has_pending_watershed() and not session.watershed_confirmed


@pytest.mark.asyncio
async def test_a_question_form_can_be_a_real_location_task_at_moderate_confidence(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,spatial=area_tools(),model_reply={
        'route':'workflow_action','intent':'start_assessment','confidence':.72,
        'candidates':[{'field':'administrative_area','value':'永川区','evidence_text':'永川区'}]})
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        result=await say(client,sid,'能否评价永川区的小流域？')
        assert result['reply']['outlet_map']['administrative_area']=='永川区'
        assert not agent.store.load(sid).review_id and not calls.hydro
        route=next(e for e in agent.store.read_events(sid) if e['event']=='dialogue_route')
        assert route['source']=='model'


@pytest.mark.asyncio
async def test_model_clarification_is_not_changed_to_a_task_by_river_keywords():
    question='你想了解嘉陵江的水文特点，还是评价某段小流域？'
    router=DialogueRouter(llm=StubLLM({'route':'clarification','intent':'unknown','confidence':.9,
                                    'missing_fields':['river_name'],'ambiguity':question}))
    session=AgentSession(session_id='session-abcdef12',state='new')
    decision=await router.decide('帮我分析嘉陵江',session)
    parsed,_=router.to_parsed_intent(decision,'帮我分析嘉陵江',session)
    assert decision.route is RouteKind.CLARIFICATION and parsed.intent is Intent.UNKNOWN
    assert decision.ambiguity==question


@pytest.mark.asyncio
@pytest.mark.parametrize('resume',['按我前面说的地区开始吧','是的，先评价'])
async def test_task_survives_long_knowledge_detour_and_model_can_resume_it(tmp_path,calls,resume):
    question='你希望直接做小流域评价，还是先了解建模方法？'
    app,agent,_=web_for(tmp_path,calls,spatial=area_tools(),model_reply={
        'route':'clarification','intent':'start_assessment','confidence':.65,
        'candidates':[{'field':'administrative_area','value':'永川区','evidence_text':'永川区'}],
        'missing_fields':['longitude','latitude'],'ambiguity':question})
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        first=await say(client,sid,'我想评价永川区的小流域')
        assert first['reply']['text']==question and not calls.spatial
        pending=agent.store.load(sid).pending_workflow
        assert pending['entities']['administrative_area']=='永川区'
        agent.router.llm.reply={'route':'knowledge_chat','intent':'ask_explanation','confidence':.98}
        agent.responder.llm=StubLLM({'answer':'入渗可增加孔隙水压力，并改变土体有效应力。'})
        for _ in range(10):
            await say(client,sid,'水流如何影响边坡稳定性')
        assert agent.store.load(sid).pending_workflow==pending
        assert not calls.spatial and not calls.hydro
        agent.router.llm.reply={'route':'workflow_action','intent':'start_assessment','confidence':.96}
        resumed=await say(client,sid,resume)
        assert resumed['reply']['outlet_map']['administrative_area']=='永川区'
        context=json.loads(agent.router.llm.calls[-1]['user'])['context']
        assert context['pending_workflow']['entities']['administrative_area']=='永川区'
        assert all('我想评价永川区' not in t['text'] for t in context['recent_conversation'])
        assert not agent.store.load(sid).pending_workflow
        assert not calls.hydro and 'delineate_from_outlet' not in calls.spatial


@pytest.mark.asyncio
async def test_new_river_does_not_inherit_previous_coordinate_center():
    model=StubLLM({'route':'workflow_action','intent':'start_assessment','confidence':.96,
                  'candidates':[{'field':'river_name','value':'嘉陵江','evidence_text':'嘉陵江'}]})
    router=DialogueRouter(llm=model)
    session=AgentSession(session_id='session-abcdef12',state='outlet_candidates_ready',
        selection_context={'map_center':{'longitude':105.9,'latitude':29.4}},
        river_name='大宁河')
    decision=await router.decide('这次改成嘉陵江',session)
    parsed,_=router.to_parsed_intent(decision,'这次改成嘉陵江',session)
    assert parsed.river_name=='嘉陵江' and parsed.longitude is None and parsed.latitude is None


@pytest.mark.asyncio
async def test_calculation_threshold_and_explicit_confirmation_gates_are_preserved():
    router=DialogueRouter(llm=StubLLM({'route':'workflow_action','intent':'start_hydro_run','confidence':.72}))
    session=AgentSession(session_id='session-abcdef12',state='new')
    decision=await router.decide('能不能现在计算',session)
    assert decision.route is RouteKind.CLARIFICATION
    router.llm.reply={'route':'workflow_action','intent':'confirm_watershed','confidence':1}
    decision=await router.decide('你自己确认边界',session)
    assert decision.intent is not Intent.CONFIRM_WATERSHED


@pytest.mark.asyncio
async def test_coordinate_correction_uses_the_previous_user_latitude():
    router=DialogueRouter(llm=StubLLM({'route':'workflow_action','intent':'start_assessment','confidence':.96,
        'candidates':[{'field':'longitude','value':106.1,'evidence_text':'106.1'}]}))
    session=AgentSession(session_id='session-abcdef12',state='outlet_candidates_ready',
        selection_context={'location_mode':'coordinates','map_center':{'longitude':105.9,'latitude':29.4}})
    decision=await router.decide('经度改为106.1，纬度不变',session)
    parsed,_=router.to_parsed_intent(decision,'经度改为106.1，纬度不变',session)
    assert parsed.longitude==106.1 and parsed.latitude==29.4


@pytest.mark.asyncio
async def test_conflicting_values_do_not_become_chosen_task_inputs(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,model_reply={
        'route':'workflow_action','intent':'start_assessment','confidence':.96,
        'candidates':[{'field':'longitude','value':106.1,'evidence_text':'106.1'},
                      {'field':'longitude','value':107.1,'evidence_text':'107.1'},
                      {'field':'latitude','value':29.4,'evidence_text':'29.4'}]})
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        await say(client,sid,'经度可能是106.1或107.1，纬度29.4')
        pending=agent.store.load(sid).pending_workflow
        assert 'longitude' not in pending['entities']
        assert 'longitude' in pending['unresolved_fields']
        assert not calls.spatial and not calls.hydro


@pytest.mark.asyncio
@pytest.mark.parametrize('text',['对的','没问题','不对','换一个'])
async def test_reply_to_knowledge_is_not_a_boundary_decision(text):
    model=StubLLM({'route':'knowledge_chat','intent':'ask_explanation','confidence':.96})
    router=DialogueRouter(llm=model)
    session=AgentSession(session_id='session-abcdef12',state='awaiting_watershed_confirmation',
                         review_id='basin-abcdef123456',watershed_qa_passed=True)
    session.record_turn('agent','孔隙水压力可以改变土体有效应力。',answer_source='general_model_knowledge')
    decision=await router.decide(text,session)
    assert decision.route is RouteKind.KNOWLEDGE_CHAT and len(model.calls)==1
    assert not session.watershed_confirmed


@pytest.mark.asyncio
async def test_explicit_boundary_confirmation_still_works_after_knowledge():
    model=StubLLM({'route':'knowledge_chat','confidence':.9})
    router=DialogueRouter(llm=model)
    session=AgentSession(session_id='session-abcdef12',state='awaiting_watershed_confirmation',
                         review_id='basin-abcdef123456',watershed_qa_passed=True)
    session.record_turn('agent','孔隙水压力可以改变土体有效应力。',answer_source='general_model_knowledge')
    decision=await router.decide('确认该边界',session)
    assert decision.intent is Intent.CONFIRM_WATERSHED and not model.calls


@pytest.mark.asyncio
async def test_model_can_ask_a_natural_request_without_a_question_mark(tmp_path,calls):
    question='请补充这条河所在的乡镇，便于确定你希望评价的河段。'
    app,agent,_=web_for(tmp_path,calls,model_reply={
        'route':'clarification','intent':'start_assessment','confidence':.8,
        'missing_fields':['administrative_area'],'ambiguity':question})
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        result=await say(client,sid,'我想评价朝阳河')
        assert result['reply']['text'].endswith(question)
        assert '我没能确定你想做什么' not in result['reply']['text']
        assert not calls.spatial and not calls.hydro
