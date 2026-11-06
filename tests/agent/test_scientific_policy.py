"""Phase 7 acceptance: the quality-gate decision and the wording rules."""

from __future__ import annotations

import pytest

from slope_evaluation_agent.scientific_policy import (
    AssessmentStatus,
    FORBIDDEN_CLAIMS,
    audit_reply,
    decide_status,
    forbidden_reasons,
    interpretations,
    limitations,
    numeric_audit,
)
from slope_evaluation_agent.response_writer import sanitize


class TestStatusDecision:
    def test_a_passed_run_is_a_valid_preliminary_assessment(self):
        decision = decide_status(qa_passed=True, run_state="succeeded", safety_available=True)
        assert decision.status is AssessmentStatus.VALID_PRELIMINARY
        assert decision.is_formal is True

    def test_a_qa_failed_run_is_diagnostic_only(self):
        """Plan section 12 phase 7: qa_passed false must mean diagnostic_only."""
        decision = decide_status(qa_passed=False, run_state="qa_failed", safety_available=True)
        assert decision.status is AssessmentStatus.DIAGNOSTIC_ONLY
        assert decision.is_formal is False
        assert any("门槛" in reason for reason in decision.reasons)

    def test_a_failed_run_is_unavailable(self):
        decision = decide_status(qa_passed=None, run_state="failed", safety_available=True)
        assert decision.status is AssessmentStatus.UNAVAILABLE

    def test_a_cancelled_run_is_unavailable(self):
        decision = decide_status(qa_passed=None, run_state="cancelled", safety_available=False)
        assert decision.status is AssessmentStatus.UNAVAILABLE

    def test_a_running_run_is_unavailable(self):
        decision = decide_status(qa_passed=None, run_state="running", safety_available=True)
        assert decision.status is AssessmentStatus.UNAVAILABLE
        assert "尚未结束" in decision.reasons[0]

    def test_missing_safety_summary_is_unavailable(self):
        """Plan section 12 phase 7: no safety factor summary means no assessment."""
        decision = decide_status(qa_passed=True, run_state="succeeded", safety_available=False)
        assert decision.status is AssessmentStatus.UNAVAILABLE
        assert any("安全系数" in reason for reason in decision.reasons)

    def test_an_unknown_gate_verdict_is_treated_as_not_passed(self):
        decision = decide_status(qa_passed=None, run_state="succeeded", safety_available=True)
        assert decision.status is AssessmentStatus.DIAGNOSTIC_ONLY

    def test_only_a_passed_run_can_be_formal(self):
        formal = [
            decide_status(qa_passed=qa, run_state=state, safety_available=safety).status
            for qa, state, safety in (
                (True, "succeeded", True), (False, "qa_failed", True),
                (None, "failed", True), (None, "running", True),
                (True, "succeeded", False), (None, "cancelled", False),
            )
        ]
        assert formal.count(AssessmentStatus.VALID_PRELIMINARY) == 1


class TestInterpretations:
    SAFETY = {"minimum_safety_factor": 0.2752, "mean_safety_factor": 1.4295,
              "max_unstable_fraction": 0.150457, "max_unstable_duration_h": 3.25,
              "max_flow_depth_m": 0.0225, "max_velocity_ms": 0.345,
              "max_pore_water_pressure_kpa": 12.5}
    RAINFALL = {"total_rain_mm": 79.288, "peak_hourly_rain_mm": 6.9361}

    def _statements(self):
        return interpretations(self.SAFETY, self.RAINFALL, self.SAFETY)

    def test_a_minimum_below_one_is_scoped_to_local_cells(self):
        statements = self._statements()
        minimum = next(item for item in statements if "minimum" in item["basis"])
        assert "局部" in minimum["text"]
        assert minimum["scope"] == "local"
        # It must not read as a statement about the whole watershed.
        assert "整个流域" not in minimum["text"]

    def test_the_mean_is_reported_separately_from_the_minimum(self):
        """Plan section 12 phase 7: min, mean and extent must not be confused."""
        statements = self._statements()
        bases = {item["basis"] for item in statements}
        assert "safety.minimum_safety_factor" in bases
        assert "safety.mean_safety_factor" in bases

    def test_the_unstable_fraction_is_marked_instantaneous(self):
        statements = self._statements()
        extent = next(item for item in statements if item["basis"].endswith("max_unstable_fraction"))
        assert "瞬时" in extent["text"]

    def test_every_statement_names_its_source_field(self):
        for statement in self._statements():
            assert statement["basis"]
            assert statement["text"]

    def test_a_safe_minimum_is_not_called_engineering_safety(self):
        statements = interpretations({"minimum_safety_factor": 1.4}, {}, {})
        text = " ".join(item["text"] for item in statements)
        assert "稳定" in text or "不低于" in text
        for phrase in ("工程安全", "绝对安全", "保证"):
            assert phrase not in text

    def test_absent_metrics_produce_no_statement(self):
        assert interpretations({}, {}, {}) == []

    def test_no_composite_score_is_produced(self):
        """Plan section 10.4: this release produces no single risk score."""
        statements = self._statements()
        for statement in statements:
            assert "综合" not in statement["text"]
            assert "风险等级" not in statement["text"]
            assert "评分" not in statement["text"]


class TestLimitations:
    def test_the_four_standing_caveats_are_always_present(self):
        items = limitations()
        joined = " ".join(items)
        assert "初步筛查与诊断" in joined
        assert "虚拟雨站" in joined
        assert "路由流量过程" in joined

    def test_a_preset_is_named_when_known(self):
        assert any("medium" in item for item in limitations(preset="medium"))

    def test_a_sample_maximum_is_flagged_as_such(self):
        items = limitations(rainfall_source="historical_max")
        assert any("样本最大日" in item for item in items)
        assert any("不是重现期设计暴雨" in item for item in items)

    def test_extra_limitations_from_the_server_are_carried_through(self):
        items = limitations(extra=["服务记录的限制"])
        assert "服务记录的限制" in items


class TestForbiddenClaims:
    def test_a_negated_phrase_is_not_a_violation(self):
        """「不是地面实测雨量站」 is the correct statement, not a violation."""
        assert forbidden_reasons("这些站点是 ERA5 网格代理点，不是地面实测雨量站。") == []

    def test_a_bare_forbidden_phrase_is_a_violation(self):
        assert forbidden_reasons("本次使用实测雨量驱动模型。")

    def test_sanitize_repairs_a_bare_phrase(self):
        repaired = sanitize("本次使用实测雨量驱动模型。")
        assert "实测雨量" not in repaired
        assert "虚拟雨站雨量" in repaired

    def test_sanitize_leaves_a_negation_alone(self):
        text = "这些站点不是地面实测雨量站。"
        assert sanitize(text) == text

    def test_sanitize_handles_several_occurrences(self):
        text = "实测雨量不是实测雨量，但降雨是实测降雨。"
        repaired = sanitize(text)
        # The negated one survives; the asserted ones are repaired.
        assert repaired.count("实测雨量") == 1
        assert "实测降雨" not in repaired

    def test_every_listed_claim_is_detectable(self):
        for phrase in FORBIDDEN_CLAIMS:
            assert forbidden_reasons(f"结论：{phrase}。"), phrase


class TestNumericAudit:
    EVIDENCE = {"safety": {"minimum_safety_factor": 0.2752, "mean_safety_factor": 1.4295,
                           "max_unstable_fraction": 0.150457},
                "rainfall": {"total_rain_mm": 79.288}}

    def test_a_faithful_reply_passes(self):
        text = ("最小安全系数 0.28，均值 1.43，"
                "最大失稳面积比例 15.05%，累计降雨 79.29 mm。")
        assert numeric_audit(text, self.EVIDENCE) == []

    def test_an_invented_number_is_caught(self):
        """Plan section 12 phase 7: every shown number must be traceable."""
        unexplained = numeric_audit("最小安全系数 0.2752，风险等级 7 级。", self.EVIDENCE)
        assert any("7" in item for item in unexplained), unexplained

    def test_times_and_dates_are_exempt(self):
        text = "事件窗口 2024-07-30 00:00 至 2024-07-30 23:00。"
        assert numeric_audit(text, self.EVIDENCE) == []

    def test_a_percentage_matches_its_fraction(self):
        assert numeric_audit("失稳面积 15.05%", self.EVIDENCE) == []

    def test_a_percentage_is_recognised_by_its_sign(self):
        assert numeric_audit("失稳面积 15.05%", {"x": 0.150457}) == []

    def test_a_bare_number_is_not_silently_read_as_a_percentage(self):
        """R3 requires the conversion to be declared, i.e. written with a %."""
        unexplained = numeric_audit("失稳面积 15.0457", {"x": 0.150457})
        assert any("15.0457" in item for item in unexplained), unexplained

    def test_numbers_inside_server_text_count_as_evidence(self):
        assert numeric_audit("共 168 项成果", {"artifact_count": 168}) == []

    def test_rounding_is_tolerated(self):
        assert numeric_audit("最小值 0.2752", {"v": 0.27521}) == []

    def test_audit_reply_reports_both_classes_of_problem(self):
        problems = audit_reply("保证不会失稳，风险等级 9 级。", self.EVIDENCE)
        assert any("不允许" in item for item in problems)
        assert any("证据中找不到" in item for item in problems)


class TestNoCompositeScore:
    def test_the_module_exposes_no_scoring_function(self):
        import slope_evaluation_agent.scientific_policy as policy

        public = [name for name in dir(policy) if not name.startswith("_")]
        for name in public:
            assert "score" not in name.lower()
            assert "rank" not in name.lower()
            assert "risk_level" not in name.lower()
