"""Phase 7 acceptance: the evidence object built from an MCP result.

Every field must be traceable to what ``get_hydro_run_result`` returned, and the
status must follow the quality gates rather than how complete the numbers look.
"""

from __future__ import annotations

import json

import pytest

from slope_evaluation_agent.evaluation import (
    audit,
    build_assessment,
    describe_assessment,
    describe_diagnostics,
    explain,
)
from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.scientific_policy import AssessmentStatus
from slope_evaluation_agent.state_machine import WorkflowState

from stubs import run_result_payload


def a_session(**overrides) -> AgentSession:
    return AgentSession.model_validate({
        "session_id": "session-abcdef123456", "state": "hydro_succeeded",
        "hydro_run_id": "hydro-run-abcdef123456", **overrides})


def quality(passed: bool) -> dict:
    return run_result_payload(qa_passed=passed)["quality"]


class TestBuildAssessment:
    def test_a_passed_run_yields_a_valid_preliminary_assessment(self):
        assessment = build_assessment(a_session(), run_result_payload(), quality(True))
        assert assessment["assessment_status"] == AssessmentStatus.VALID_PRELIMINARY.value
        assert assessment["qa_passed"] is True

    def test_a_failed_gate_yields_a_diagnostic_only_assessment(self):
        assessment = build_assessment(a_session(), run_result_payload(qa_passed=False),
                                      quality(False))
        assert assessment["assessment_status"] == AssessmentStatus.DIAGNOSTIC_ONLY.value

    def test_a_failed_run_yields_an_unavailable_assessment(self):
        assessment = build_assessment(
            a_session(hydro_run_status="failed"),
            run_result_payload(status="failed", qa_passed=False), quality(False))
        assert assessment["assessment_status"] == AssessmentStatus.UNAVAILABLE.value

    def test_a_missing_safety_summary_yields_an_unavailable_assessment(self):
        payload = run_result_payload()
        payload["safety"] = {"available": False}
        assessment = build_assessment(a_session(), payload, quality(True))
        assert assessment["assessment_status"] == AssessmentStatus.UNAVAILABLE.value

    def test_the_documented_structure_is_present(self):
        assessment = build_assessment(a_session(), run_result_payload(), quality(True))
        for key in ("assessment_status", "qa_passed", "event", "model_context",
                    "factors", "interpretations", "warnings", "limitations",
                    "evidence_sources"):
            assert key in assessment, key

    def test_the_four_factor_groups_are_present(self):
        factors = build_assessment(a_session(), run_result_payload(), quality(True))["factors"]
        assert set(factors) == {"rainfall", "safety_factor",
                                "instability_extent_duration", "hydraulic_pore_pressure"}

    def test_no_composite_score_is_produced(self):
        """Plan section 10.4: this release yields no single score."""
        payload = build_assessment(a_session(), run_result_payload(), quality(True))
        blob = json.dumps(payload, ensure_ascii=False).lower()
        for key in ("risk_score", "composite", "risk_level", "susceptibility", "总分"):
            assert key not in blob, key


class TestFactorFaithfulness:
    """Plan section 12 phase 7: every value traces to a response field."""

    def test_safety_factors_are_copied_verbatim(self):
        result = run_result_payload()
        factors = build_assessment(a_session(), result, quality(True))["factors"]
        safety = factors["safety_factor"]
        assert safety["minimum"] == result["safety"]["minimum_safety_factor"]
        assert safety["mean"] == result["safety"]["mean_safety_factor"]
        assert safety["maximum"] == result["safety"]["maximum_safety_factor"]

    def test_extent_and_duration_are_not_confused(self):
        result = run_result_payload()
        extent = build_assessment(a_session(), result, quality(True))["factors"][
            "instability_extent_duration"]
        assert extent["max_unstable_fraction"] == result["safety"]["max_unstable_fraction"]
        assert extent["max_unstable_duration_h"] == result["safety"]["max_unstable_duration_h"]

    def test_rainfall_is_copied_verbatim(self):
        result = run_result_payload()
        rainfall = build_assessment(a_session(), result, quality(True))["factors"]["rainfall"]
        assert rainfall["total_rain_mm"] == result["rainfall"]["total_rain_mm"]
        assert rainfall["peak_hourly_rain_mm"] == result["rainfall"]["peak_hourly_rain_mm"]
        assert rainfall["source"] == "ERA5 virtual stations"

    def test_the_routing_limitation_is_carried_through(self):
        assessment = build_assessment(a_session(), run_result_payload(), quality(True))
        assert any("路由流量过程" in item for item in assessment["limitations"])

    def test_server_limitations_are_not_dropped(self):
        result = run_result_payload()
        result["limitations"] = ["服务自己记录的一条限制"]
        assessment = build_assessment(a_session(), result, quality(True))
        assert "服务自己记录的一条限制" in assessment["limitations"]

    def test_model_context_records_what_produced_the_result(self):
        context = build_assessment(a_session(), run_result_payload(),
                                   quality(True))["model_context"]
        assert context["parameter_preset"] == "medium"
        assert context["mechanics_model"] == "reference_equation_9"
        assert context["routing_network_present"] is False

    def test_the_run_mode_comes_from_the_session_not_the_plan(self):
        """The plan payload has no ``mode`` field, so reading it there is a
        silent null: the assessment would not say which pipeline ran."""
        plan_without_mode = {"stages": [{"stage_id": "sync"}], "engine": {}}
        assessment = build_assessment(
            a_session(run_mode="build_run_qa", last_plan=plan_without_mode),
            run_result_payload(), quality(True))
        assert assessment["model_context"]["run_mode"] == "build_run_qa"

    def test_a_session_with_no_recorded_mode_says_so_rather_than_guessing(self):
        assert build_assessment(a_session(), run_result_payload(),
                                quality(True))["model_context"]["run_mode"] is None

    def test_evidence_sources_name_their_fields(self):
        sources = build_assessment(a_session(), run_result_payload(),
                                   quality(True))["evidence_sources"]
        assert sources
        for item in sources:
            assert item["source"] == "get_hydro_run_result"
            assert item["field"]


class TestRendering:
    def test_a_valid_assessment_is_described_with_its_numbers(self):
        assessment = build_assessment(a_session(), run_result_payload(), quality(True))
        text = describe_assessment(assessment)
        assert "初步评价" in text
        assert "0.275" in text
        assert "1.429" in text  # the mean, rounded from 1.4295

    def test_a_diagnostic_assessment_refuses_a_verdict(self):
        assessment = build_assessment(a_session(), run_result_payload(qa_passed=False),
                                      quality(False))
        text = describe_assessment(assessment)
        assert "诊断" in text
        assert "不构成" in text or "不能" in text
        # It still shows the numbers, so the failure can be investigated.
        assert "0.275" in text

    def test_an_unavailable_assessment_says_why(self):
        assessment = build_assessment(
            a_session(hydro_run_status="failed"),
            run_result_payload(status="failed", qa_passed=False), quality(False))
        text = describe_assessment(assessment)
        assert "没有可用于评价的结果" in text

    def test_a_valid_assessment_carries_its_limitations(self):
        text = describe_assessment(build_assessment(a_session(), run_result_payload(),
                                                    quality(True)))
        assert "限制" in text
        assert "初步筛查与诊断" in text

    def test_a_local_minimum_is_never_stretched_to_the_watershed(self):
        text = describe_assessment(build_assessment(a_session(), run_result_payload(),
                                                    quality(True)))
        assert "局部" in text
        for phrase in ("整个流域不稳定", "全流域失稳", "流域已失稳"):
            assert phrase not in text

    def test_the_diagnostic_form_is_available(self):
        assessment = build_assessment(a_session(), run_result_payload(qa_passed=False),
                                      quality(False))
        assert describe_diagnostics(assessment) == describe_assessment(assessment)

    def test_the_rendered_reply_survives_its_own_audit(self):
        assessment = build_assessment(a_session(), run_result_payload(), quality(True))
        text = describe_assessment(assessment)
        assert audit(assessment, text) == []


class TestExplain:
    def test_a_safety_factor_question_is_answered(self):
        session = a_session()
        from slope_evaluation_agent.intent import ParsedIntent

        text = explain(session, ParsedIntent(topic="为什么安全系数这么低"), "为什么安全系数这么低")
        assert "抗滑力" in text or "局部" in text

    def test_a_qa_question_is_answered(self):
        from slope_evaluation_agent.intent import ParsedIntent

        text = explain(a_session(), ParsedIntent(topic="qa"), "质量门槛是什么")
        assert "门槛" in text

    def test_an_unmatched_question_gets_the_general_answer(self):
        from slope_evaluation_agent.intent import ParsedIntent

        text = explain(a_session(), ParsedIntent(topic="随便问问"), "随便问问")
        assert "解释" in text or "含义" in text

    def test_the_answer_carries_the_current_numbers_when_available(self):
        from slope_evaluation_agent.intent import ParsedIntent

        assessment = build_assessment(a_session(), run_result_payload(), quality(True))
        text = explain(a_session(), ParsedIntent(topic="安全系数"), "安全系数是什么", assessment)
        assert "0.275" in text

    def test_a_diagnostic_result_is_labelled_in_the_explanation(self):
        from slope_evaluation_agent.intent import ParsedIntent

        assessment = build_assessment(a_session(), run_result_payload(qa_passed=False),
                                      quality(False))
        text = explain(a_session(), ParsedIntent(topic="安全系数"), "安全系数", assessment)
        assert "诊断" in text

    def test_explanations_never_assert_a_forbidden_claim(self):
        from slope_evaluation_agent.intent import ParsedIntent
        from slope_evaluation_agent.scientific_policy import forbidden_reasons

        for question in ("安全系数", "孔压", "失稳", "qa", "降雨", "模型"):
            text = explain(a_session(), ParsedIntent(topic=question), question)
            # A negated phrase ("not measured gauges") is the correct caveat;
            # forbidden_reasons distinguishes it from an assertion.
            assert forbidden_reasons(text) == [], question
