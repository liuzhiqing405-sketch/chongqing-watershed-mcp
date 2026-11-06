"""R0 regression tests for R3: the scientific numeric audit.

The current audit exempts every value at or below 1, which is exactly where the
numbers that matter live — a safety factor of 0.2752, an unstable-area fraction
of 0.42. These tests fail against that behaviour and must pass once it is gone.

They also check that the audit is *enforced on the way out*: a reply the agent
returns to a user goes through it, rather than the helper being callable but
unused.
"""

from __future__ import annotations

import re

import pytest

from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.scientific_policy import audit_reply, numeric_audit
from slope_evaluation_agent.state_machine import WorkflowState


pytestmark = [pytest.mark.unit]


#: Evidence shaped like a real ``get_hydro_run_result`` safety block.
EVIDENCE = {
    "safety": {
        "minimum_safety_factor": 0.2752,
        "mean_safety_factor": 1.4295,
        "max_unstable_fraction": 0.42,
        "max_unstable_duration_h": 3.25,
        "max_flow_depth_m": 0.0225,
        "max_velocity_ms": 0.345,
    },
    "rainfall": {"total_rain_mm": 79.288, "peak_hourly_rain_mm": 6.9361},
}


class TestSmallDecimalsAreAudited:
    """No value is exempt for being small."""

    def test_a_wrong_safety_factor_below_one_is_caught(self):
        unexplained = numeric_audit("最小安全系数为 0.9。", EVIDENCE)
        assert any("0.9" in item for item in unexplained), "小于 1 的安全系数被错误地豁免了"

    def test_the_true_safety_factor_passes(self):
        assert numeric_audit("最小安全系数为 0.2752。", EVIDENCE) == []

    def test_a_wrong_ratio_below_one_is_caught(self):
        unexplained = numeric_audit("最大失稳面积比例为 0.43。", EVIDENCE)
        assert any("0.43" in item for item in unexplained), "小于 1 的比例被错误地豁免了"

    def test_the_true_ratio_passes(self):
        assert numeric_audit("最大失稳面积比例为 0.42。", EVIDENCE) == []

    def test_one_point_zero_is_audited(self):
        unexplained = numeric_audit("流速为 1.0 m/s。", EVIDENCE)
        assert any("1.0" in item for item in unexplained)

    def test_a_small_wrong_velocity_is_caught(self):
        unexplained = numeric_audit("最大流速 0.9 m/s。", EVIDENCE)
        assert any("0.9" in item for item in unexplained)


class TestPercentEquivalence:
    def test_a_fraction_may_be_shown_as_a_percentage(self):
        assert numeric_audit("最大失稳面积比例为 42%。", EVIDENCE) == []

    def test_a_fraction_may_be_shown_as_a_percentage_with_decimals(self):
        assert numeric_audit("最大失稳面积比例为 42.00%。", EVIDENCE) == []

    def test_a_wrong_percentage_is_caught(self):
        unexplained = numeric_audit("最大失稳面积比例为 43%。", EVIDENCE)
        assert any("43" in item for item in unexplained)

    def test_a_fraction_suffixed_with_a_percent_sign_is_wrong(self):
        """0.42 written as 0.42% is not the same value."""
        unexplained = numeric_audit("最大失稳面积比例为 0.42%。", EVIDENCE)
        assert any("0.42" in item for item in unexplained)

    def test_a_safety_factor_is_not_a_percentage(self):
        """A safety factor has no percent form, so 27.52% is not its value."""
        unexplained = numeric_audit("最小安全系数为 27.52%。", EVIDENCE)
        assert any("27.52" in item for item in unexplained)


class TestRoundingTolerance:
    def test_three_decimal_rounding_passes(self):
        assert numeric_audit("最小安全系数为 0.275。", EVIDENCE) == []

    def test_two_decimal_rounding_passes(self):
        assert numeric_audit("最小安全系数为 0.28。", EVIDENCE) == []

    def test_a_different_second_decimal_fails(self):
        unexplained = numeric_audit("最小安全系数为 0.29。", EVIDENCE)
        assert any("0.29" in item for item in unexplained)

    def test_a_different_tenth_fails(self):
        unexplained = numeric_audit("最小安全系数为 0.2。", EVIDENCE)
        assert any("0.2" in item for item in unexplained)

    def test_rounding_to_an_integer_is_too_coarse(self):
        unexplained = numeric_audit("最小安全系数为 0。", EVIDENCE)
        assert any("0" in item for item in unexplained)

    def test_rounding_to_one_decimal_is_too_coarse(self):
        unexplained = numeric_audit("最小安全系数为 0.3。", EVIDENCE)
        assert any("0.3" in item for item in unexplained)


class TestFieldsAndUnitsStayBound:
    def test_a_rainfall_value_cannot_be_relabelled_as_safety_factor(self):
        problems = audit_reply("最小安全系数为 79.288 mm。", EVIDENCE)
        assert problems

    def test_a_safety_factor_cannot_be_relabelled_as_water_depth(self):
        problems = audit_reply("最大水深为 0.2752 m。", EVIDENCE)
        assert problems

    def test_a_dimensionless_factor_cannot_gain_a_unit(self):
        problems = audit_reply("最小安全系数为 0.2752 mm。", EVIDENCE)
        assert problems

    def test_a_depth_with_its_correct_unit_passes(self):
        assert audit_reply("最大水深为 0.0225 m。", EVIDENCE) == []


class TestFieldTypesAreNotConfused:
    def test_dates_and_times_are_not_evidence(self):
        assert numeric_audit("事件窗口 2024-07-30 00:00 至 2024-07-30 23:00。", EVIDENCE) == []

    def test_a_leading_list_marker_is_not_a_value(self):
        text = "1. 最小安全系数为 0.2752。\n2. 均值为 1.4295。"
        assert numeric_audit(text, EVIDENCE) == []

    def test_a_digit_inside_a_word_is_not_a_value(self):
        assert numeric_audit("降雨来自 ERA5 网格。", EVIDENCE) == []

    def test_an_identifier_is_not_a_value(self):
        assert numeric_audit("任务 hydro-run-abcdef123456 已完成。", EVIDENCE) == []

    def test_a_thousands_separator_is_one_number(self):
        assert numeric_audit("共 2,389 个像元。", {"selected_cells": 2389}) == []

    def test_negative_values_are_handled(self):
        assert numeric_audit("连续性误差 -0.002%。", {"error": -0.00002}) == []

    def test_scientific_notation_is_handled(self):
        assert numeric_audit("量级为 2.5e-3。", {"value": 0.0025}) == []


class TestForbiddenConclusions:
    def test_a_safety_claim_is_flagged(self):
        problems = audit_reply("已经证明坡体安全。", EVIDENCE)
        assert any("安全" in item for item in problems)

    def test_a_risk_grade_is_flagged(self):
        problems = audit_reply("综合风险等级为中风险。", EVIDENCE)
        assert any("风险" in item for item in problems)

    def test_a_causal_claim_is_flagged(self):
        problems = audit_reply("降雨导致该坡体失稳。", EVIDENCE)
        assert any("导致" in item for item in problems)

    def test_a_debris_flow_claim_is_flagged(self):
        problems = audit_reply("该区域泥石流易发性高。", EVIDENCE)
        assert any("泥石流" in item for item in problems)

    def test_a_correct_caveat_is_not_flagged(self):
        assert audit_reply("这些站点不是地面实测雨量站。", EVIDENCE) == []


class TestAuditIsEnforcedOnTheReplyPath:
    """The audit must run on what the agent returns, not only when called."""

    @pytest.mark.asyncio
    async def test_the_agents_own_replies_pass_its_own_audit(self, build_agent, calls):
        """A deterministic template must not print a number its ledger cannot explain.

        The audit exists to catch a mismatch between a reply and its evidence —
        which includes the agent's own templates. When the intake chain was a
        concatenation of five steps' text but carried only the first step's
        facts, every number in the rainfall, package, import, validation and
        plan sections was reported as unsupported, and any rewrite of that
        reply would have been discarded. Clean templates are what keep the
        audit's verdicts meaningful.
        """
        from helpers import PRELUDE, converse, prepared_agent

        agent, session = await prepared_agent(build_agent, calls, run_status="succeeded")
        replies = await converse(agent, session, PRELUDE + ["开始运行", "结果怎么样"])
        dirty = {reply.text.splitlines()[0][:60]: reply.audit
                 for reply in replies if reply.audit}
        assert not dirty, f"智能体自己的回复未通过审计：{dirty}"

    @pytest.mark.asyncio
    async def test_the_agents_own_replies_pass_its_own_audit_when_a_gate_failed(
            self, build_agent, calls):
        """The diagnostic branch prints its own numbers and must be as clean."""
        from helpers import PRELUDE, converse, prepared_agent

        agent, session = await prepared_agent(build_agent, calls, run_status="qa_failed")
        replies = await converse(agent, session, PRELUDE + ["开始运行", "结果怎么样"])
        dirty = {reply.text.splitlines()[0][:60]: reply.audit
                 for reply in replies if reply.audit}
        assert not dirty, f"诊断分支未通过审计：{dirty}"

    @pytest.mark.asyncio
    async def test_the_agent_exposes_an_audited_reply_path(self, build_agent, calls):
        agent = build_agent()
        session = agent.new_session()
        async with agent:
            reply = await agent.handle(session, "今天天气不错")
        # A reply the agent produced carries its audit outcome.
        assert hasattr(reply, "audit") or hasattr(reply, "audit_problems"), (
            "回复对象没有携带审计结果，无法证明审计被强制执行")

    @pytest.mark.asyncio
    async def test_a_rewritten_reply_with_a_wrong_number_falls_back(
            self, build_agent, calls, monkeypatch):
        """A rewrite that invents a number must be replaced by the canonical text."""
        from helpers import PRELUDE, converse, prepared_agent

        agent, session = await prepared_agent(build_agent, calls, run_status="succeeded")

        async def lying_rewrite(*, canonical, evidence, **kwargs):
            return "最小安全系数为 0.9，坡体安全。"

        monkeypatch.setattr(agent, "_rewrite_reply", lying_rewrite, raising=False)
        replies = await converse(agent, session, PRELUDE + ["开始运行", "结果怎么样"])
        text = replies[-1].text
        assert "0.9" not in text, "未经审计的改写文本被返回给了用户"
        # A negated caveat ("不是工程级边坡安全鉴定") is correct and must survive;
        # what the audit forbids is an asserted conclusion.
        from slope_evaluation_agent.scientific_policy import stability_claim_reasons

        assert stability_claim_reasons(text) == [], "禁止性结论被返回给了用户"
        assert "0.275" in text, "应当回退到确定性文本"
        assert replies[-1].rewritten is False, "回退后不应标记为已改写"


class TestAuditDoesNotBreakTheSession:
    @pytest.mark.asyncio
    async def test_a_failed_audit_keeps_the_session_usable(self, build_agent, calls):
        from helpers import converse

        agent = build_agent()
        session = agent.new_session()
        replies = await converse(agent, session, ["今天天气不错", "结果怎么样"])
        assert session.session_id
        assert all(reply.text for reply in replies)
