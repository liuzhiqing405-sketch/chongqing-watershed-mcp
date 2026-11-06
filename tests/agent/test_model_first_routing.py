"""Model semantics precede keyword heuristics; execution remains validated."""

import json

import pytest

from slope_evaluation_agent.conversation_responder import ConversationResponder
from slope_evaluation_agent.dialogue_router import DialogueRouter, RouteKind
from slope_evaluation_agent.intent import Intent
from slope_evaluation_agent.models import AgentSession
from helpers import run
from stubs import StubLLM

pytestmark = pytest.mark.unit


def knowledge_route(**extra):
    return {"route": "knowledge_chat", "intent": "ask_explanation",
            "confidence": 0.95, "topic": "engineering_general", **extra}


@pytest.mark.parametrize("message", [
    "重庆有哪些河", "含水层的储水能力", "分析嘉陵江的地理特征",
    "帮我讲讲稳定性计算的假设", "2024-07-30 的降雨是什么类型",
    "经度106.5、纬度29.6所在区域的地貌特点", "开始计算前需要理解哪些理论",
    "展示结果通常采用哪些指标", "地下水埋深与季节变化", "如何取消任务",
    "流域边界为什么需要确认", "你好",
])
@pytest.mark.asyncio
async def test_model_decides_questions_despite_river_date_coordinate_or_action_words(build_agent, calls, message):
    agent = build_agent()
    route_model = StubLLM(knowledge_route())
    agent.router.llm = route_model
    answer_model = StubLLM({"answer": "这是一条知识说明；具体场地评价仍需相应资料。"})
    agent.responder = ConversationResponder(llm=answer_model)
    session = agent.new_session()
    before = (session.state, session.river_name, session.requested_outlet, session.rainfall_request)
    reply = await run(agent, session, message)
    assert len(route_model.calls) == 1
    assert json.loads(route_model.calls[0]["user"])["user_text"] == message
    assert session.last_route == RouteKind.KNOWLEDGE_CHAT.value
    assert reply.outlet_map is None
    assert calls.spatial == calls.hydro == []
    assert (session.state, session.river_name, session.requested_outlet, session.rainfall_request) == before


@pytest.mark.parametrize("route, confidence", [("clarification", 0.95), ("workflow_action", 0.6)])
@pytest.mark.asyncio
async def test_model_can_ask_instead_of_executing_a_confident_rules_match(build_agent, calls, route, confidence):
    agent = build_agent()
    agent.router.llm = StubLLM({"route": route, "intent": "start_assessment",
                               "confidence": confidence, "ambiguity": "请明确评价目的"})
    session = agent.new_session()
    reply = await run(agent, session, "分析嘉陵江")
    assert session.last_route == RouteKind.CLARIFICATION.value
    assert reply.outlet_map is None
    assert session.state == "new"
    assert calls.spatial == calls.hydro == []


@pytest.mark.asyncio
async def test_model_task_refines_the_river_name_and_keeps_manual_map(build_agent, calls):
    agent = build_agent()
    route_model = StubLLM({"route": "workflow_action", "intent": "start_assessment",
                          "confidence": 0.95, "candidates": [
                              {"field": "river_name", "value": "嘉陵江", "evidence_text": "嘉陵江"}]})
    agent.router.llm = route_model
    session = agent.new_session()
    session.river_name = "大宁河"
    reply = await run(agent, session, "请对重庆的嘉陵江做一次稳定性分析")
    assert len(route_model.calls) == 1
    assert session.last_route == RouteKind.WORKFLOW_ACTION.value
    assert session.river_name == "嘉陵江"
    assert reply.outlet_map is not None
    assert calls.spatial == ["render_outlet_candidates"]
    assert calls.hydro == []
    assert session.outlet_candidates == []
    event = next(item for item in agent.store.read_events(session.session_id)
                 if item["event"] == "tool_call")
    assert event["arguments"] == {"river_name": "嘉陵江", "max_candidates": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize("confidence", [0.72, 0.9])
async def test_task_location_is_clarified_on_the_map_when_a_river_is_known(build_agent, calls, confidence):
    agent = build_agent()
    model = StubLLM({"route": "clarification", "intent": "start_assessment",
                    "confidence": confidence, "missing_fields": ["longitude", "latitude"],
                    "candidates": [{"field": "river_name", "value": "嘉陵江", "evidence_text": "嘉陵江"}]})
    agent.router.llm = model
    session = agent.new_session()
    reply = await run(agent, session, "帮我对嘉陵江做坡面稳定性分析")
    assert reply.outlet_map is not None
    assert session.state == "outlet_candidates_ready"
    assert calls.spatial == ["render_outlet_candidates"]
    assert calls.hydro == []
    sent = json.loads(model.calls[0]["user"])
    assert sent["workflow_reference"]["required_inputs"]["start_assessment"] == [
        ["administrative_area"], ["river_name"], ["longitude", "latitude"]]


@pytest.mark.asyncio
async def test_borrowed_evidence_cannot_invent_a_river():
    model = StubLLM({"route": "workflow_action", "intent": "start_assessment", "confidence": 0.99,
                    "candidates": [{"field": "river_name", "value": "嘉陵江", "evidence_text": "这个地方"}]})
    router = DialogueRouter(llm=model)
    session = AgentSession(session_id="session-abcdef123456", state="new")
    decision = await router.decide("评估这个地方", session)
    assert decision.route is RouteKind.CLARIFICATION
    parsed, _ = router.to_parsed_intent(decision, "评估这个地方", session)
    assert parsed.river_name is None


@pytest.mark.asyncio
async def test_knowledge_route_discards_accidental_entities_even_when_they_conflict():
    router = DialogueRouter(llm=StubLLM(knowledge_route(candidates=[
        {"field": "river_name", "value": "嘉陵江", "evidence_text": "嘉陵江"},
        {"field": "river_name", "value": "长江", "evidence_text": "长江"},
    ])))
    decision = await router.decide("嘉陵江和长江有什么关系", AgentSession(session_id="session-abcdef123456", state="new"))
    assert decision.route is RouteKind.KNOWLEDGE_CHAT
    assert decision.candidates == []
    assert decision.missing_fields == []


@pytest.mark.asyncio
async def test_model_uses_active_task_context_without_cancelling_when_asked_how(build_agent, calls):
    agent = build_agent()
    route_model = StubLLM(knowledge_route())
    agent.router.llm = route_model
    agent.responder = ConversationResponder(llm=StubLLM({"answer": "可以明确发出取消指令。"}))
    session = agent.new_session()
    session.state = "hydro_running"
    session.hydro_run_id = "hydro-run-abcdef123456"
    reply = await run(agent, session, "如何取消任务")
    assert len(route_model.calls) == 1
    assert json.loads(route_model.calls[0]["user"])["context"]["current_state"] == "hydro_running"
    assert session.last_route == "knowledge_chat"
    assert session.state == "hydro_running"
    assert session.hydro_run_id == "hydro-run-abcdef123456"
    assert session.intent != Intent.CANCEL_RUN.value
    assert calls.spatial == calls.hydro == []
