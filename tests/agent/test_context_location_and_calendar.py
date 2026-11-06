"""Regression for context references, calendar validation, and draft-only edits."""
import copy
import json

import pytest

from slope_evaluation_agent.dialogue_router import DialogueRouter, RouteKind
from slope_evaluation_agent.intent import Intent, RuleParser
from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.temporal_inputs import resolve_relative_rainfall
from stubs import StubLLM
from test_web_conversation import a_client, a_session, say, web_for
from test_web_study_area import area_tools


def current(date="2025-01-06", **kwargs):
    return AgentSession(session_id="session-calendar123",state="outlet_candidates_ready",
        administrative_area="永川区",selection_mode="drawn_study_area",
        selection_context={"location_mode":"district"},
        rainfall_request={"mode":"explicit_period","date":date},**kwargs)


@pytest.mark.parametrize("text",[
    "在前面那个区继续评价", "先显示选区地图", "调出选区底图", "希望判断原定区域的稳定性",
    "接着处理这项区域评价", "把上面那个县的底图打开", "仍在这个区开展评价", "评价该区域",
])
@pytest.mark.asyncio
async def test_drawing_objects_and_references_are_not_new_administrative_names(text):
    router=DialogueRouter(llm=StubLLM({"route":"workflow_action","intent":"start_assessment","confidence":.92}))
    session=current()
    original=router.rules.parse(text,session).model_dump()
    decision=await router.decide(text,session)
    parsed,_=router.to_parsed_intent(decision,text,session)
    assert parsed.administrative_area=="永川区"
    assert router.rules.parse(text,session).model_dump()==original


@pytest.mark.parametrize("text,expected",[
    ("我想评价永川区的小流域","永川区"), ("换成九龙坡区","九龙坡区"),
    ("请加载永定区地图","永定区"), ("分析巫溪县","巫溪县"),
])
@pytest.mark.asyncio
async def test_explicit_new_names_are_still_checked_as_new_names(text,expected):
    router=DialogueRouter(llm=StubLLM({"route":"workflow_action","intent":"start_assessment","confidence":.92,
        "candidates":[{"field":"administrative_area","value":expected,"evidence_text":expected}]}))
    session=current()
    decision=await router.decide(text,session)
    parsed,_=router.to_parsed_intent(decision,text,session)
    assert parsed.administrative_area==expected


@pytest.mark.asyncio
@pytest.mark.parametrize("model_place",["那个区","调出选区","原定区"])
async def test_model_generic_place_candidate_cannot_replace_known_location(model_place):
    text=f"沿用原地点，{model_place}，打开地图"
    router=DialogueRouter(llm=StubLLM({"route":"workflow_action","intent":"start_assessment","confidence":.92,
        "candidates":[{"field":"administrative_area","value":model_place,"evidence_text":model_place}]}))
    session=current()
    decision=await router.decide(text,session)
    parsed,_=router.to_parsed_intent(decision,text,session)
    assert parsed.administrative_area=="永川区"


@pytest.mark.asyncio
async def test_model_reference_without_a_known_place_asks_for_location():
    text="打开那个区的选区地图"
    router=DialogueRouter(llm=StubLLM({"route":"workflow_action","intent":"start_assessment","confidence":.92}))
    session=AgentSession(session_id="session-no-location",state="new")
    decision=await router.decide(text,session)
    assert decision.route is RouteKind.CLARIFICATION
    parsed,_=router.to_parsed_intent(decision,text,session)
    assert parsed.administrative_area is None


@pytest.mark.asyncio
@pytest.mark.parametrize("text",["沿用之前那个区，打开选区地图", "调出选区底图，原来的设置不变"])
async def test_model_resume_with_no_new_name_uses_existing_location(text):
    router=DialogueRouter(llm=StubLLM({"route":"workflow_action","intent":"start_assessment","confidence":.92}))
    session=current()
    decision=await router.decide(text,session)
    parsed,_=router.to_parsed_intent(decision,text,session)
    assert decision.route is RouteKind.WORKFLOW_ACTION
    assert parsed.administrative_area=="永川区" and parsed.river_name is None


@pytest.mark.parametrize("anchor,text,expected",[
    ("2025-01-06","降雨往后顺延两天","2025-01-08"),
    ("2025-01-06","原日期后的第三天","2025-01-09"),
    ("2025-01-06","事件提前一天","2025-01-05"),
    ("2025-01-06","换成次日","2025-01-07"),
    ("2025-01-06","后移十二天","2025-01-18"),
    ("2024-02-28","顺延一天","2024-02-29"),
    ("2025-02-28","顺延一天","2025-03-01"),
    ("2025-12-31","改用下一天","2026-01-01"),
    ("2025-01-01","提前两天","2024-12-30"),
    ("2025-01-06","不要顺延两天，只提前一天","2025-01-05"),
])
def test_calendar_arithmetic_is_independent_of_model_guesses(anchor,text,expected):
    request={"mode":"explicit_period","date":anchor}
    before=copy.deepcopy(request)
    result=resolve_relative_rainfall(text,request)
    assert result.values=={"date":expected} and not result.error
    assert request==before


def test_relative_window_preserves_duration_across_year_boundary():
    result=resolve_relative_rainfall("时段往后移动两天",{
        "mode":"explicit_period","start_datetime":"2025-12-30 18:00","end_datetime":"2025-12-31 06:00"})
    assert result.values=={"start_datetime":"2026-01-01 18:00","end_datetime":"2026-01-02 06:00"}


@pytest.mark.parametrize("rainfall,text",[
    ({},"改为次日"), ({"mode":"historical_max"},"顺延两天"),
    ({"date":"2025-01-06"},"后移一天或提前一天"),
    ({"date":"2025-01-06"},"顺延两天，2025年1月9日"),
    ({"date":"2025-02-30"},"顺延一天"),
    ({"date":"2025-01-06"},"不要把日期往后顺延两天"),
    ({"date":"2025-01-06"},"顺延一天半"),
])
def test_missing_conflicting_and_invalid_anchors_require_clarification(rainfall,text):
    result=resolve_relative_rainfall(text,rainfall)
    assert result.error and not result.values


@pytest.mark.asyncio
async def test_locally_verified_date_edit_can_be_saved_without_authorising_preparation(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,model_reply={"route":"workflow_action",
        "intent":"set_rainfall_period","confidence":.82,
        "candidates":[{"field":"date","value":"2025-01-08","evidence_text":"后的第三天"}]})
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        session=agent.store.load(sid)
        session.rainfall_request={"mode":"explicit_period","date":"2025-01-06"}
        agent.store.save(session)
        result=await say(client,sid,"降雨事件改到原日期后的第三天，其他输入不变")
        saved=agent.store.load(sid)
        assert saved.rainfall_request["date"]=="2025-01-09"
        assert result['reply']['route']=="workflow_action" and result['reply']['ok']
        assert "2025-01-09" in result['reply']['text']
        assert not calls.spatial and not calls.hydro and not saved.watershed_confirmed
        shown=json.loads(agent.router.llm.calls[0]['user'])
        assert shown['context']['resolved_temporal_inputs']['date']=="2025-01-09"


@pytest.mark.asyncio
async def test_relative_date_question_never_changes_rainfall_inputs(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,model_reply={"route":"knowledge_chat","intent":"ask_explanation","confidence":.98})
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        session=agent.store.load(sid);session.rainfall_request={"mode":"explicit_period","date":"2025-01-06"}
        agent.store.save(session)
        await say(client,sid,"如果顺延两天会有什么影响？只解释，不修改设置")
        assert agent.store.load(sid).rainfall_request['date']=="2025-01-06"
        assert not calls.spatial and not calls.hydro


@pytest.mark.asyncio
async def test_unknown_anchor_does_not_commit_a_model_invented_date(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,model_reply={"route":"workflow_action","intent":"set_rainfall_period","confidence":.98,
        "candidates":[{"field":"date","value":"2025-01-08","evidence_text":"顺延两天"}]})
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        result=await say(client,sid,"降雨顺延两天")
        assert not agent.store.load(sid).rainfall_request
        assert result['reply']['route']=='clarification' and '基准' in result['reply']['text']
        assert not calls.spatial and not calls.hydro


@pytest.mark.asyncio
async def test_uncertain_confirmed_event_edit_preserves_active_inputs(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,model_reply={"route":"workflow_action","intent":"set_rainfall_period","confidence":.6,
        "candidates":[{"field":"date","value":"2025-01-08","evidence_text":"2025年1月8日"}]})
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        session=agent.store.load(sid);session.state='watershed_confirmed';session.watershed_confirmed=True
        session.rainfall_request={"mode":"explicit_period","date":"2025-01-06"}
        agent.store.save(session)
        result=await say(client,sid,"降雨日期改为2025年1月8日")
        saved=agent.store.load(sid)
        assert result['reply']['route']=='clarification'
        assert saved.rainfall_request['date']=='2025-01-06'
        assert saved.pending_workflow['entities']['date']=='2025-01-08'
        assert not calls.spatial and not calls.hydro


@pytest.mark.asyncio
async def test_low_confidence_compute_is_not_promoted_by_date_arithmetic():
    router=DialogueRouter(llm=StubLLM({"route":"workflow_action","intent":"start_hydro_run","confidence":.82}))
    decision=await router.decide("将日期顺延两天然后直接计算",current())
    assert decision.route is RouteKind.CLARIFICATION
