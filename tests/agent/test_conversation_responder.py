"""Acceptance for the constrained conversation channel (stage D).

The conversational answer is assembled from text a person approved. These tests
pin that down: which topic a question reaches, that the answer carries no number
it cannot support, and that the current result is quoted only from the evidence
ledger — never from the model's imagination.
"""

from __future__ import annotations

import pytest

from slope_evaluation_agent import domain_knowledge
from slope_evaluation_agent.conversation_responder import ConversationResponder
from slope_evaluation_agent.dialogue_router import RouteKind
from slope_evaluation_agent.evaluation import TOPICS as TERM_TOPICS
from slope_evaluation_agent.intent import ParsedIntent
from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.scientific_policy import numeric_audit
from slope_evaluation_agent.state_machine import WorkflowState


pytestmark = [pytest.mark.unit]


def unsupported_numbers(text: str) -> list[str]:
    """What the *production* audit would say about this text with no evidence.

    Using the real function rather than a regex is what keeps this honest: the
    audit already knows that ``FS<1`` is a threshold and not a quantity, and a
    hand-written pattern would only re-litigate that.
    """
    return numeric_audit(text, {})


def a_session(state: WorkflowState = WorkflowState.NEW, **facts) -> AgentSession:
    return AgentSession.model_validate({
        "session_id": "session-abcdef123456", "state": state.value, **facts})


def an_assessment(minimum: float = 0.2752, mean: float = 1.4295) -> dict:
    return {
        "assessment_status": "valid_preliminary",
        "qa_passed": True,
        "hydro_run_id": "hydro-run-abcdef123456",
        "factors": {
            "rainfall": {"total_rain_mm": 79.288, "peak_hourly_rain_mm": 6.9361},
            "safety_factor": {"minimum": minimum, "mean": mean, "p95": 2.0, "maximum": 2.0},
            "instability_extent_duration": {"max_unstable_fraction": 0.150457,
                                            "max_unstable_duration_h": 3.25},
            "hydraulic_pore_pressure": {"max_flow_depth_m": 0.0225,
                                        "max_velocity_ms": 0.345,
                                        "max_pore_water_pressure_kpa": 12.5},
        },
        "model_context": {"parameter_preset": "medium"},
        "interpretations": [],
        "warnings": [],
        "limitations": [],
    }


class TestThePlanQuestionsReachTheRightTopic:
    """The seven conversational inputs the plan names, verbatim."""

    @pytest.mark.parametrize("question,topic", [
        ("你好。", "greeting"),
        ("你是什么智能体？", "capability"),
        ("你能分析哪些东西？", "capability"),
        ("安全系数越大越好吗？", "safety_factor"),
        ("为什么要我确认边界？", "boundary"),
        ("没有降雨日期怎么办？", "no_date"),
        ("这个结果可以直接用于工程设计吗？", "engineering"),
    ])
    def test_the_topic_is_recognised(self, question, topic):
        assert domain_knowledge.match_topic(question) == topic

    @pytest.mark.parametrize("question,topic", [
        ("你好。", "greeting"),
        ("你是什么智能体？", "capability"),
        ("你能分析哪些东西？", "capability"),
        ("安全系数越大越好吗？", "safety_factor"),
        ("为什么要我确认边界？", "boundary"),
        ("没有降雨日期怎么办？", "no_date"),
        ("这个结果可以直接用于工程设计吗？", "engineering"),
    ])
    def test_the_answer_is_non_empty_and_on_topic(self, question, topic):
        reply = ConversationResponder().respond(question=question, session=a_session())
        assert reply.topic == topic
        assert reply.text.strip()
        assert reply.route == RouteKind.KNOWLEDGE_CHAT.value


class TestTheAnswersAreAuditSafe:
    @pytest.mark.parametrize("topic", sorted(domain_knowledge.CONCEPTS) + ["limitations"])
    def test_no_answer_prints_a_number_it_cannot_support(self, topic):
        """The reply audit reads a bare digit as a quantity. None of these is one."""
        for part in domain_knowledge.snippets(topic):
            assert not unsupported_numbers(part), part[:60]

    @pytest.mark.parametrize("question", [
        "你好。", "你是什么智能体？", "你能分析哪些东西？", "安全系数越大越好吗？",
        "为什么要我确认边界？", "没有降雨日期怎么办？", "这个结果可以直接用于工程设计吗？",
    ])
    def test_the_rendered_answer_carries_no_bare_number(self, question):
        reply = ConversationResponder().respond(question=question, session=a_session())
        assert not unsupported_numbers(reply.text), reply.text[:120]

    @pytest.mark.parametrize("question", ["帮我写首诗", "今天天气怎么样", "帮我看看股票"])
    def test_an_out_of_scope_answer_is_also_clean(self, question):
        reply = ConversationResponder().out_of_scope(question=question, session=a_session())
        assert not unsupported_numbers(reply.text)


class TestTheResultIsQuotedOnlyFromEvidence:
    def test_a_question_about_this_run_quotes_the_ledger(self):
        session = a_session(WorkflowState.HYDRO_SUCCEEDED,
                            hydro_run_id="hydro-run-abcdef123456")
        reply = ConversationResponder().respond(
            question="这次的最小安全系数是多少？", session=session,
            parsed=ParsedIntent(topic="安全系数"), assessment=an_assessment())
        assert "0.275" in reply.text
        assert reply.used_knowledge == ["current_assessment"]

    def test_a_concept_question_with_no_run_stays_general(self):
        reply = ConversationResponder().respond(
            question="安全系数是什么意思？", session=a_session(),
            parsed=ParsedIntent(topic="安全系数"), assessment=None)
        assert "0.275" not in reply.text
        assert "抗滑力" in reply.text or "力学模型" in reply.text

    def test_a_result_question_with_no_assessment_does_not_invent_one(self):
        session = a_session(WorkflowState.HYDRO_SUCCEEDED,
                            hydro_run_id="hydro-run-abcdef123456")
        reply = ConversationResponder().respond(
            question="这次的最小安全系数是多少？", session=session, assessment=None)
        assert "0.275" not in reply.text

    def test_two_sessions_do_not_borrow_each_others_numbers(self):
        responder = ConversationResponder()
        first = a_session(WorkflowState.HYDRO_SUCCEEDED, hydro_run_id="hydro-run-abcdef123456")
        quotted = responder.respond(question="这次的结果怎么样？", session=first,
                                    parsed=ParsedIntent(topic="安全系数"),
                                    assessment=an_assessment(minimum=0.1111))
        second = a_session(WorkflowState.HYDRO_SUCCEEDED, hydro_run_id="hydro-run-abcdef999999")
        other = responder.respond(question="这次的结果怎么样？", session=second,
                                  parsed=ParsedIntent(topic="安全系数"),
                                  assessment=an_assessment(minimum=0.2222))
        assert "0.111" in quotted.text and "0.222" in other.text


class TestOutOfScope:
    def test_it_says_what_this_agent_is_for(self):
        reply = ConversationResponder().out_of_scope(question="帮我写首诗",
                                                     session=a_session())
        assert reply.route == RouteKind.OUT_OF_SCOPE.value
        assert "坡面稳定性" in reply.text

    def test_it_offers_a_way_back(self):
        reply = ConversationResponder().out_of_scope(question="帮我写首诗",
                                                     session=a_session())
        assert "河" in reply.text or "经纬度" in reply.text

    def test_it_is_not_the_same_wording_as_the_capability_answer(self):
        responder = ConversationResponder()
        scope = responder.out_of_scope(question="帮我写首诗", session=a_session()).text
        capability = responder.respond(question="你能做什么？", session=a_session()).text
        assert scope != capability

    def test_it_does_not_claim_numeric_or_engineering_authority(self):
        reply = ConversationResponder().out_of_scope(question="这里会不会滑坡",
                                                     session=a_session())
        for claim in ("会失稳", "已经稳定", "风险等级"):
            assert claim not in reply.text


class TestTheKnowledgeBaseItself:
    def test_capabilities_and_non_capabilities_are_both_stated(self):
        assert domain_knowledge.CAPABILITIES
        assert domain_knowledge.NOT_CAPABILITIES

    def test_the_non_capabilities_name_the_forbidden_claims(self):
        joined = "".join(domain_knowledge.NOT_CAPABILITIES)
        for phrase in ("工程级边坡安全鉴定", "灾害预报", "泥石流易发性", "风险等级",
                       "综合风险分数"):
            assert phrase in joined, phrase

    def test_the_two_services_are_described_separately(self):
        assert set(domain_knowledge.SERVICES) == {"spatial", "hydro"}

    def test_the_term_explanations_are_the_ones_results_use(self):
        """One source of truth: a concept answer and a result explanation agree."""
        for topic in ("safety_factor", "pore_pressure", "unstable", "qa", "rainfall", "model"):
            assert domain_knowledge.snippets(topic) == [TERM_TOPICS[topic]]

    def test_the_limitations_include_the_scientific_ones(self):
        joined = "".join(domain_knowledge.LIMITATIONS)
        for phrase in ("ERA5", "虚拟雨站", "重现期设计暴雨", "路由流量过程", "局部像元"):
            assert phrase in joined, phrase

    def test_the_workflow_steps_are_named(self):
        assert len(domain_knowledge.WORKFLOW_STEPS) >= 5

    def test_an_unknown_topic_yields_no_text(self):
        assert domain_knowledge.snippets("nowhere") == []

    def test_the_catalogue_is_serialisable(self):
        import json

        assert json.dumps(domain_knowledge.catalogue(), ensure_ascii=False)


class TestNoModelNeeded:
    def test_answering_a_concept_question_needs_no_model(self):
        """The deterministic text is the answer; a model may only rephrase it."""
        responder = ConversationResponder()
        first = responder.respond(question="为什么要我确认边界？", session=a_session())
        second = responder.respond(question="为什么要我确认边界？", session=a_session())
        assert first.text == second.text

    def test_the_responder_has_no_workflow_clients_and_model_is_optional(self):
        responder = ConversationResponder()
        for attribute in ("spatial", "hydro", "agent"):
            assert not hasattr(responder, attribute), attribute
        assert responder.llm is None

    def test_a_question_that_matches_nothing_still_gets_an_answer(self):
        reply = ConversationResponder().respond(question="嗯那这个呢", session=a_session())
        assert reply.text.strip()
        assert reply.topic == domain_knowledge.DEFAULT_TOPIC
