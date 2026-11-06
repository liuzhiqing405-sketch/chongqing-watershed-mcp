"""Invalid model answers get one checked repair, with no workflow authority."""
import copy
import json

import pytest

from slope_evaluation_agent.conversation_responder import ConversationResponder
from slope_evaluation_agent.models import AgentSession
from stubs import StubLLM
from helpers import run


class AnswerSequence:
    def __init__(self, *answers):
        self.answers = iter(answers)
        self.calls = []

    async def complete_json(self, *, system, user):
        self.calls.append(json.loads(user))
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    {"text": "字段名称错误"},
    {"answer": "原理说明", "route": "workflow_action"},
    {"answer": "说明", "tool": "confirm_watershed"},
    {"answer": " "},
    {"answer": "本次流域安全系数为 0.8。"},
    {"answer": "我已经调用MCP完成计算。"},
])
async def test_repair_rechecks_schema_and_claims_without_changing_session(bad):
    model = AnswerSequence(bad, {"answer": "降雨可通过入渗、孔隙水压力和坡面水流影响稳定性，具体响应需结合模型假设。"})
    responder = ConversationResponder(llm=model)
    session = AgentSession(session_id="session-answer-repair", state="new")
    before = copy.deepcopy(session.model_dump())
    reply = await responder.respond_general(question="换一场雨会怎样影响安全系数？", session=session)
    assert reply.answer_source == "general_model_knowledge" and not reply.fallback_reason
    assert reply.repair_attempted and reply.repair_reasons
    assert len(model.calls) == 2 and "answer_repair" in model.calls[1]
    assert model.calls[1]["schema"]["additionalProperties"] is False
    assert session.model_dump() == before
    assert bad not in model.calls[1].values()


@pytest.mark.asyncio
async def test_persistent_invalid_answer_is_rejected_after_one_repair():
    bad = {"answer": "我已经调用MCP完成计算。"}
    model = AnswerSequence(bad, bad)
    session = AgentSession(session_id="session-answer-failure", state="new")
    reply = await ConversationResponder(llm=model).respond_general(question="什么是孔隙水压力？", session=session)
    assert reply.answer_source == "builtin_knowledge"
    assert reply.fallback_reason == "knowledge_policy_rejected"
    assert reply.repair_attempted and len(model.calls) == 2
    assert bad["answer"] not in reply.text and session.state == "new"


@pytest.mark.asyncio
async def test_transport_failure_is_not_retried_by_the_answer_repair_layer():
    model = AnswerSequence(TimeoutError("offline"))
    session = AgentSession(session_id="session-answer-offline", state="new")
    reply = await ConversationResponder(llm=model).respond_general(question="什么是孔隙水压力？", session=session)
    assert reply.fallback_reason and not reply.repair_attempted and len(model.calls) == 1


@pytest.mark.asyncio
async def test_repaired_model_answer_has_no_tools_or_input_changes(build_agent, calls):
    agent = build_agent()
    agent.router.llm = StubLLM({"route": "knowledge_chat", "intent": "ask_explanation",
                               "confidence": .95, "topic": "boundary_confirmation_rationale"})
    model = AnswerSequence({"answer": "我已经确认边界并运行计算。"},
                           {"answer": "确认边界是为了核对评价范围是否符合研究目的和数据覆盖条件。"})
    agent.responder = ConversationResponder(llm=model)
    session = agent.new_session()
    session.rainfall_request = {"mode": "explicit_period", "date": "2025-01-06"}
    before = (session.state, session.watershed_confirmed, copy.deepcopy(session.rainfall_request))
    reply = await run(agent, session, "为什么需要确认边界？只解释，不操作。")
    assert reply.answer_source == "general_model_knowledge"
    assert before == (session.state, session.watershed_confirmed, session.rainfall_request)
    assert calls.spatial == calls.hydro == []
    event = next(e for e in agent.store.read_events(session.session_id) if e["event"] == "knowledge_reply")
    assert event["repair_attempted"] and event["repair_reasons"]
