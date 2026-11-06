"""Geographic guesses clarify the map; they cannot become calculation inputs."""

import json

import pytest

from slope_evaluation_agent.geolocation import GeographicLocator
from slope_evaluation_agent.models import AgentSession
from helpers import converse, run, spatial_handlers, spatial_server
from stubs import StubLLM, outlet_map_payload, review_payload

pytestmark = pytest.mark.unit


def location(**changes):
    return {"status": "resolved", "region": "测试区域", "longitude": 107.4,
            "latitude": 30.0, "confidence": 0.9, "description": "粗略河段位置", **changes}


@pytest.mark.asyncio
async def test_failed_name_uses_model_context_to_open_one_map_only(build_agent, calls):
    replies = iter([outlet_map_payload(matched=False), outlet_map_payload()])
    agent = build_agent(spatial=spatial_server(spatial_handlers(calls,
        render_outlet_candidates=lambda: next(replies))))
    model = StubLLM(location())
    agent.locator = GeographicLocator(model)
    session = agent.new_session()
    reply = await run(agent, session, "分析测试区的测试河")
    assert reply.ok and reply.outlet_map
    assert "模型" in reply.text and "尚未核实" in reply.text
    assert session.state == "outlet_candidates_ready"
    assert session.requested_outlet is None and session.review_id is None
    assert session.location_hint["source"] == "general_model_knowledge"
    assert session.location_hint["longitude"] == 107.4
    assert session.outlet_candidates == []
    assert calls.spatial == ["render_outlet_candidates", "render_outlet_candidates"]
    assert calls.hydro == []
    assert json.loads(model.calls[0]["user"])["user_context"] == ["分析测试区的测试河"]
    args = [e["arguments"] for e in agent.store.read_events(session.session_id)
            if e["event"] == "tool_call"]
    assert args[-1] == {"river_name": session.river_name, "max_candidates": 0,
                        "longitude": 107.4, "latitude": 30.0}


@pytest.mark.asyncio
async def test_ambiguous_location_keeps_name_and_asks_only_useful_context(build_agent, calls):
    agent = build_agent(spatial=spatial_server(spatial_handlers(calls,
        render_outlet_candidates=outlet_map_payload(matched=False))))
    agent.locator = GeographicLocator(StubLLM(location(status="ambiguous", longitude=None, latitude=None)))
    session = agent.new_session()
    reply = await run(agent, session, "分析朝阳河")
    assert session.river_name == "朝阳河"
    assert "区县" in reply.text and "朝阳河" in reply.text
    assert "换一个河名" not in reply.text
    assert reply.outlet_map is None and reply.images == []
    assert session.state == "needs_location"
    assert calls.spatial == ["render_outlet_candidates"] and calls.hydro == []


@pytest.mark.parametrize("payload", [
    location(confidence=0.4), location(longitude=999), location(latitude=float("nan")),
    location(status="ambiguous"), {"status": "resolved"}, RuntimeError("provider unavailable"),
])
@pytest.mark.asyncio
async def test_unreliable_model_coordinates_are_never_used(payload):
    result = await GeographicLocator(StubLLM(payload)).resolve(
        "未知河", AgentSession(session_id="session-abcdef123456", state="new"))
    assert result.status != "resolved"
    assert result.longitude is None and result.latitude is None


@pytest.mark.asyncio
async def test_named_task_and_references_do_not_repeat_missing_location(build_agent, calls):
    agent = build_agent()
    agent.router.llm = StubLLM({"route": "clarification", "intent": "start_assessment",
                               "confidence": 0.72, "missing_fields": ["longitude", "latitude"]})
    session = agent.new_session()
    async with agent:
        first = await agent.handle(session, "分析大宁河")
        agent.router.llm.reply["intent"] = "unknown"
        agent.router.llm.reply["confidence"] = 0.9
        second = await agent.handle(session, "分析大宁河")
        agent.router.llm.reply["intent"] = "start_assessment"
        third = await agent.handle(session, "就看前面说的那条河")
    assert first.outlet_map and third.outlet_map
    # A model that chose clarification/unknown must not be overridden by the
    # heuristic river match just because the sentence contains a known name.
    assert second.outlet_map is None and second.kind == "question"
    assert session.river_name == "大宁河"
    assert calls.spatial == ["render_outlet_candidates"] * 2 and calls.hydro == []


@pytest.mark.asyncio
async def test_failed_basin_can_be_reselected_without_rejection(build_agent, calls):
    from helpers import DRAW_AREA,step
    agent = build_agent(spatial=spatial_server(spatial_handlers(calls,
        create_drawn_study_area=dict(review_payload(qa_passed=False),selection_mode="drawn_study_area",outlet={}))))
    session = agent.new_session()
    async with agent:
        await agent.handle(session, "经度 109.5、纬度 31.4")
        reply = await step(agent,session,DRAW_AREA)
        assert session.state == "watershed_qa_failed"
        assert "确认」或「拒绝" not in reply.text
        assert "不能确认" in reply.text
        assert "调整边界" in session.pending_action()
        rejected = await agent.handle(session, "确认")
        assert not rejected.ok and not session.watershed_confirmed
        opened = await agent.handle(session, "分析大宁河")
        assert opened.outlet_map
    assert session.state == "outlet_candidates_ready"
    assert "confirm_watershed" not in calls.spatial and calls.hydro == []


@pytest.mark.asyncio
async def test_restored_failed_basin_reopens_map(build_agent, calls):
    agent = build_agent()
    session = agent.new_session()
    session.state = "awaiting_watershed_confirmation"
    session.review_id = "basin-abcdef123456"
    session.watershed_qa_passed = False
    reply = await run(agent, session, "分析大宁河")
    assert reply.outlet_map and session.state == "outlet_candidates_ready"
    assert calls.spatial == ["render_outlet_candidates"] and calls.hydro == []


@pytest.mark.asyncio
async def test_coordinate_location_preserves_the_named_river_without_spatial_delineation(build_agent, calls):
    agent = build_agent()
    session = agent.new_session()
    await converse(agent, session, ["分析大宁河", "经度 109.5、纬度 31.4"])
    event = [e for e in agent.store.read_events(session.session_id)
                 if e["event"] == "tool_call" and e["tool"] == "render_outlet_candidates"][-1]
    assert event["arguments"]["river_name"] == "大宁河"
    assert not session.watershed_confirmed and calls.hydro == []
    assert not session.review_id and "delineate_from_outlet" not in calls.spatial


@pytest.mark.asyncio
async def test_river_mismatch_reply_stays_a_failed_result_even_with_large_area(build_agent, calls):
    from slope_evaluation_agent.scientific_policy import audit_reply
    record = review_payload(qa_passed=False)
    record["qa"].update(area_km2=15.5323, river_check={"passed": False, "river_name": "大宁河",
                        "river_distance_m": 3977.0}, errors=["吸附点距指定河流3977.0 m，需重新选点"])
    from stubs import envelope
    agent = build_agent(spatial=spatial_server(spatial_handlers(calls,get_watershed_review=envelope(data=record))))
    session = agent.new_session()
    session.river_name = "大宁河"
    session.state="watershed_qa_failed";session.review_id=record["review_id"];session.watershed_qa_passed=False
    reply = await run(agent, session, "这个结果对吗")
    assert session.state == "watershed_qa_failed"
    assert "出水口与指定河流不一致" in reply.text
    assert "确认」或「拒绝" not in reply.text
    assert audit_reply(reply.text, {"review": record}) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", [True, False])
async def test_current_spatial_result_question_reads_its_own_review(build_agent, calls, configured):
    from stubs import envelope
    record = review_payload(qa_passed=False)
    agent = build_agent(spatial=spatial_server(spatial_handlers(calls,
        get_watershed_review=envelope(data=record))))
    if configured:
        agent.router.llm = StubLLM({"route": "knowledge_chat", "intent": "ask_explanation",
                                   "confidence": 0.95})
    session = agent.new_session()
    session.state = "watershed_qa_failed"
    session.review_id = record["review_id"]
    session.watershed_qa_passed = False
    reply = await run(agent, session, "这个结果对吗")
    assert "未通过" in reply.text and "不能" in reply.text
    assert calls.spatial == ["get_watershed_review"] and calls.hydro == []
