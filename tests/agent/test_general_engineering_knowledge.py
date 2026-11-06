"""Knowledge generation and deterministic workflow execution have distinct authority."""

import json

import pytest

from slope_evaluation_agent.conversation_responder import ConversationResponder
from slope_evaluation_agent.dialogue_router import DialogueRouter, RouteKind
from slope_evaluation_agent.engineering_knowledge import is_engineering_question
from slope_evaluation_agent.agent import SlopeEvaluationAgent
from slope_evaluation_agent.intent import RuleParser
from helpers import run
from stubs import StubLLM
from test_web_conversation import web_for, a_client, a_session, say

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.asyncio
async def test_agent_wires_configured_model_and_respects_knowledge_switch(build_agent, calls, monkeypatch, enabled):
    model = StubLLM({"answer": "含水层能储存并传导地下水。"})
    monkeypatch.setattr("slope_evaluation_agent.agent.build_llm_client", lambda settings, logger=None: model)
    base = build_agent()
    settings = base.settings.model_copy(deep=True)
    settings.dialogue.allow_general_llm = enabled
    settings.dialogue.enabled = False  # The knowledge switch is independent of routing.
    agent = SlopeEvaluationAgent(settings, spatial=base.spatial, hydro=base.hydro, store=base.store)
    reply = await run(agent, agent.new_session(), "什么是含水层？")
    assert reply.answer_source == ("general_model_knowledge" if enabled else "builtin_knowledge")
    assert bool(model.calls) == enabled
    assert calls.spatial == calls.hydro == []


@pytest.mark.asyncio
async def test_general_definition_after_assessment_uses_model_without_run_evidence(build_agent, calls):
    from helpers import PRELUDE, prepared_agent

    agent, session = await prepared_agent(build_agent, calls, run_status="succeeded")
    model = StubLLM({"answer": "理论上的安全系数通常表示抗力与滑动力之比。"})
    agent.responder = ConversationResponder(llm=model)
    async with agent:
        for message in PRELUDE + ["开始运行", "结果怎么样"]:
            await agent.handle(session, message)
        before = (session.state, session.hydro_run_id)
        calls.spatial.clear()
        calls.hydro.clear()
        reply = await agent.handle(session, "安全系数一般如何定义？")
    assert reply.answer_source == "general_model_knowledge"
    assert (session.state, session.hydro_run_id) == before
    assert "0.275" not in json.dumps(model.calls, ensure_ascii=False)
    assert calls.spatial == calls.hydro == []


@pytest.mark.parametrize("question", [
    "地下水如何影响边坡稳定性？", "达西定律是什么？", "泥石流形成需要哪些条件？",
    "滑坡与崩塌的区别是什么？", "边坡支护设计有哪些方法？", "河流侵蚀与地质灾害有什么关系？",
    "风险分级的原理是什么？", "地下水补给、径流和排泄之间是什么关系？",
    "人工降雨影响径流的原理是什么？",
    "水流如何影响坡面稳定性",
])
@pytest.mark.asyncio
async def test_domain_questions_use_model_knowledge_without_mcp(build_agent, calls, question):
    agent = build_agent()
    model = StubLLM({"answer": "地下水可通过孔隙水压力改变有效应力，具体影响取决于地层与排水条件。"})
    agent.responder = ConversationResponder(llm=model)
    session = agent.new_session()
    before = session.state
    reply = await run(agent, session, question)
    assert reply.answer_source == "general_model_knowledge"
    assert reply.audit_scope == "general_knowledge"
    assert model.calls
    assert calls.spatial == calls.hydro == []
    assert session.state == before
    assert session.last_route == RouteKind.KNOWLEDGE_CHAT.value
    assert session.conversation[-1].answer_source == "general_model_knowledge"


@pytest.mark.parametrize("question", [
    "重庆有哪些河", "重庆有什么江河", "重庆有哪些支流", "介绍重庆的水系",
    "嘉陵江流经哪些地方", "长江和嘉陵江有什么关系", "嘉陵江有多长",
    "嘉陵江汇入哪条河", "重庆主要河流是什么", "重庆江河分布有什么特点",
    "四川哪些河经过重庆", "重庆的地形有什么特点", "重庆有几条河", "嘉陵江的源头",
])
@pytest.mark.asyncio
async def test_geography_questions_are_answered_without_river_lookup(build_agent, calls, question):
    agent = build_agent()
    route_model = StubLLM({"route": "knowledge_chat", "intent": "ask_explanation",
                          "confidence": 0.95, "topic": "engineering_general"})
    agent.router.llm = route_model
    agent.responder = ConversationResponder(llm=StubLLM({"answer": "重庆的主要河流包括长江、嘉陵江和乌江。"}))
    session = agent.new_session()
    before = (session.state, session.river_name, session.intent, session.review_id)
    reply = await run(agent, session, question)
    assert reply.answer_source == "general_model_knowledge"
    assert reply.outlet_map is None
    assert "长江" in reply.text
    assert calls.spatial == calls.hydro == []
    assert (session.state, session.river_name, session.intent, session.review_id) == before
    assert len(route_model.calls) == 1


@pytest.mark.parametrize("question", ["重庆有哪些河", "四川哪些河经过重庆", "嘉陵江汇入哪条河"])
def test_question_fragments_are_not_extracted_as_river_names(build_agent, question):
    parsed = RuleParser().parse(question, build_agent().new_session())
    assert parsed.river_name is None or parsed.river_name == "嘉陵江"
    assert parsed.river_name != question


@pytest.mark.parametrize("question, river_name", [
    ("分析嘉陵江", "嘉陵江"), ("分析大宁河好吗？", "大宁河"),
    ("能不能分析嘉陵江？", "嘉陵江"), ("大宁河稳定吗？", "大宁河"),
    ("我想看看大宁河那边怎么样。", "大宁河"),
])
@pytest.mark.asyncio
async def test_named_river_actions_still_open_the_map(build_agent, calls, question, river_name):
    agent = build_agent()
    model = StubLLM({"answer": "这只是一个泛工程回答。"})
    agent.responder = ConversationResponder(llm=model)
    session = agent.new_session()
    reply = await run(agent, session, question)
    assert "render_outlet_candidates" in calls.spatial
    assert not model.calls
    assert reply.outlet_map is not None
    assert session.river_name == river_name


@pytest.mark.asyncio
async def test_theory_numbers_and_risk_terms_are_not_mcp_metrics(build_agent, calls):
    agent = build_agent()
    answer = "教学示例：假设水深为 5 m，取水的重度为 9.81 kN/m³，静水压力为 49.05 kPa。风险等级取决于采用的评价方法。"
    agent.responder = ConversationResponder(llm=StubLLM({"answer": answer}))
    reply = await run(agent, agent.new_session(), "地下水静水压力怎么算？")
    assert reply.text == answer
    assert reply.ok and reply.audit == []
    assert reply.audit_scope == "general_knowledge"
    assert calls.spatial == calls.hydro == []


@pytest.mark.parametrize("question", ["边坡支护设计有哪些方法？", "重庆有哪些河"])
@pytest.mark.asyncio
async def test_knowledge_during_a_run_does_not_change_pending_workflow(build_agent, calls, question):
    agent = build_agent()
    agent.responder = ConversationResponder(llm=StubLLM({"answer": "可讨论排水、抗滑与防护措施的适用条件。"}))
    session = agent.new_session()
    async with agent:
        await agent.handle(session, "经度 109.5、纬度 31.4")
        before = (session.state, session.review_id, session.watershed_confirmed, session.intent,
                  session.rainfall_request)
        calls.spatial.clear()
        agent.router.llm = StubLLM({"route": "knowledge_chat", "confidence": 0.95,
                                   "topic": "engineering_general"})
        reply = await agent.handle(session, question)
    after = (session.state, session.review_id, session.watershed_confirmed, session.intent,
             session.rainfall_request)
    assert before == after
    assert reply.answer_source == "general_model_knowledge"
    assert calls.spatial == calls.hydro == []


@pytest.mark.parametrize("payload", [
    {"answer": "本次流域安全系数为 1.5。"},
    {"answer": "已经调用 MCP 完成流域计算。"},
    {"answer": "保证不会失稳。"},
    {"answer": "说明", "tool": "confirm_watershed"},
    {"answer": " "}, TimeoutError("offline"),
])
@pytest.mark.asyncio
async def test_failed_or_overreaching_answers_fall_back_without_tools(build_agent, calls, payload):
    agent = build_agent()
    agent.responder = ConversationResponder(llm=StubLLM(payload))
    session = agent.new_session()
    reply = await run(agent, session, "地下水如何影响边坡稳定？")
    assert reply.answer_source == "builtin_knowledge"
    assert reply.facts["knowledge_fallback"]
    assert "本次流域安全系数为" not in reply.text
    assert calls.spatial == calls.hydro == []
    assert session.state == "new"


@pytest.mark.asyncio
async def test_general_model_receives_knowledge_history_but_no_run_evidence(build_agent):
    agent = build_agent()
    model = StubLLM({"answer": "它的适用条件取决于所讨论的含水层。"})
    agent.responder = ConversationResponder(llm=model)
    session = agent.new_session()
    session.record_turn("user", "展示结果")
    session.record_turn("agent", "本次计算 FS=0.2752", answer_source="mcp_evidence")
    session.record_turn("user", "什么是含水层？")
    session.record_turn("agent", "含水层可以储存并传导地下水。", answer_source="general_model_knowledge")
    reply = await run(agent, session, "地下水补给有什么条件？")
    sent = json.loads(model.calls[-1]["user"])
    assert "0.2752" not in json.dumps(sent, ensure_ascii=False)
    assert "含水层" in json.dumps(sent, ensure_ascii=False)
    assert "assessment" not in sent
    assert reply.answer_source == "general_model_knowledge"


@pytest.mark.parametrize("question", ["本次安全系数是多少，为什么这么低？", "为什么最小安全系数这么低"])
@pytest.mark.asyncio
async def test_current_result_question_cannot_use_generic_model_invented_metrics(build_agent, calls, question):
    agent = build_agent()
    model = StubLLM({"answer": "本次安全系数是 9.99。"})
    agent.responder = ConversationResponder(llm=model)
    reply = await run(agent, agent.new_session(), question)
    assert not model.calls
    assert "9.99" not in reply.text
    assert reply.audit_scope == "computation"
    assert calls.spatial == calls.hydro == []


@pytest.mark.parametrize("question", ["最近一次比较大的雨会不会不稳定？", "帮我分析大宁河是否稳定？",
                                      "大宁河边坡稳定吗？", "边坡稳不稳定？", "这个流域会不会失稳？"])
def test_scenario_requests_are_not_shortcut_into_general_knowledge(question):
    assert not is_engineering_question(question)


@pytest.mark.asyncio
async def test_named_river_stability_question_enters_workflow_not_generic_answer(build_agent, calls):
    agent = build_agent()
    model = StubLLM({"answer": "没有场地资料也可以判定稳定。"})
    agent.responder = ConversationResponder(llm=model)
    session = agent.new_session()
    reply = await run(agent, session, "大宁河边坡稳定吗？")
    assert session.last_route == RouteKind.WORKFLOW_ACTION.value
    assert "render_outlet_candidates" in calls.spatial
    assert not model.calls
    assert reply.audit_scope == "computation"


@pytest.mark.parametrize("question", ["达西定律是什么？", "重庆有哪些河"])
@pytest.mark.asyncio
async def test_web_reports_knowledge_provenance_and_restores_it(tmp_path, calls, question):
    app, agent, _ = web_for(tmp_path, calls)
    agent.router.llm = StubLLM({"route": "knowledge_chat", "confidence": 0.95,
                               "topic": "engineering_general"})
    agent.responder = ConversationResponder(llm=StubLLM({"answer": "达西定律描述一定条件下的多孔介质渗流。"}))
    async with agent, a_client(app) as client:
        identifier = await a_session(client)
        reply = (await say(client, identifier, question))["reply"]
        assert reply["answer_source"] == "general_model_knowledge"
        assert reply["audit_scope"] == "general_knowledge"
        assert reply["audit_passed"] is None
        assert reply["route"] == "knowledge_chat"
        assert reply["outlet_map"] is None
        restored = (await client.get(f"/api/v1/sessions/{identifier}")).json()
        assert restored["conversation"][-1]["answer_source"] == "general_model_knowledge"
    assert calls.spatial == calls.hydro == []


@pytest.mark.asyncio
async def test_deepseek_answer_channel_recovers_from_token_truncation_without_tools(build_agent, calls, monkeypatch):
    from test_llm import FakeHTTPX, FakeResponse, install, reply, settings
    from slope_evaluation_agent.llm import OpenAICompatibleClient

    initial = FakeResponse({"choices": [{"finish_reason": "length", "message": {"content": None}}]})
    fake = FakeHTTPX(initial)
    responses = iter([initial._payload, reply('{"answer":"水流可增加孔隙水压力、侵蚀坡脚，并改变坡体的受力条件。"}')._payload])
    monkeypatch.setattr(initial, "json", lambda: next(responses))
    install(monkeypatch, fake)
    client = OpenAICompatibleClient(settings(provider="deepseek", maximum_tokens=4096, max_retries=1))
    agent = build_agent()
    agent.responder = ConversationResponder(llm=client)
    session = agent.new_session()
    reply = await run(agent, session, "水流如何影响坡面稳定性")
    assert reply.answer_source == "general_model_knowledge"
    assert "孔隙水压力" in reply.text
    assert session.state == "new"
    assert len(fake.requests) == 2
    assert fake.requests[0]["json"]["thinking"] == {"type": "disabled"}
    assert calls.spatial == calls.hydro == []
