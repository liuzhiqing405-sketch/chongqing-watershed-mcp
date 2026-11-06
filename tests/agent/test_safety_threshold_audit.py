"""Distinguish Fs=1 decision boundaries from reported result values."""
from __future__ import annotations

import pytest

from slope_evaluation_agent.result_responder import answer_result
from slope_evaluation_agent.scientific_policy import EvidenceLedger, audit_reply


ASSESSMENT = {
    "assessment_status": "valid_preliminary",
    "factors": {
        "safety_factor": {"minimum": 0.2498, "mean": 1.7956},
        "hydraulic_pore_pressure": {"max_flow_depth_m": 0.4466},
        "instability_extent_duration": {"max_unstable_fraction": 0.073017},
        "spatial_distribution": {"available": True, "below_one": {"area_km2": 28.709895260551896}},
    },
}

THRESHOLDS = [
    "安全系数小于1", "安全系数小于 1.0", "安全系数低于1.00",
    "安全系数不大于1", "安全系数不高于1", "安全系数小于等于1",
    "安全系数小于或等于1", "安全系数大于1", "安全系数高于1",
    "安全系数不低于1", "安全系数不小于1", "安全系数大于等于1",
    "Fs<1", "fs < 1.0", "Fs<=1", "Fs ≤ 1.00", "Fs≥1", "Fs>1",
    "Fs＜1", "Fs＞＝1", "F_s < 1", "Fₛ < 1", "当Fs<1",
    "安全系数（Fs）小于1", "安全系数(F_s)低于1",
    "安全系数阈值为1", "安全系数的临界值为1.0",
    "安全系数判别阈值取1", "Fs的判别阈值=1.00",
    "以Fs=1作为判别阈值", "以安全系数等于1为界",
    "Fs < 1 is the decision boundary", "Fs<1.",
]


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_canonical_decision_thresholds_do_not_require_result_evidence(threshold):
    text = f"{threshold}是判别条件；最小安全系数为0.2498。"
    assert audit_reply(text, EvidenceLedger.from_assessment(ASSESSMENT)) == []


@pytest.mark.parametrize("threshold", ["安全系数小于1", "安全系数低于1.0", "Fs<1", "F_s≤1"])
def test_correct_area_with_chinese_or_symbolic_threshold_passes(threshold):
    text = f"{threshold}的区域面积为28.710 km²。"
    assert audit_reply(text, EvidenceLedger.from_assessment(ASSESSMENT)) == []


@pytest.mark.parametrize("prefix", ["事件最小", "模拟期间最小", "逐像元最小"])
@pytest.mark.parametrize("area,correct", [("28.710", True), ("28.821", False)])
def test_threshold_field_description_does_not_relabel_the_reported_area(prefix, area, correct):
    text = f"{prefix}安全系数小于1的区域面积为{area} km²。"
    problems = audit_reply(text, EvidenceLedger.from_assessment(ASSESSMENT))
    assert (problems == []) == correct


@pytest.mark.parametrize("threshold", ["安全系数小于1", "Fs<1", "F_s≤1"])
@pytest.mark.parametrize("claim,wrong", [
    ("区域面积为28.821 km²", "28.821"),
    ("区域面积为28.710 m²", "28.710"),
    ("最小安全系数为1.7956", "1.7956"),
    ("最大水深为0.4466 mm", "0.4466"),
    ("最大失稳面积比例为0.073017%", "0.073017"),
])
def test_threshold_exemption_does_not_hide_wrong_results(threshold, claim, wrong):
    problems = audit_reply(f"{threshold}作为判别条件；{claim}。", EvidenceLedger.from_assessment(ASSESSMENT))
    assert len(problems) == 1, problems
    assert wrong in problems[0]


@pytest.mark.parametrize("text", [
    "最小安全系数为1.0。", "平均安全系数为1.00。", "最小安全系数=1。",
    "Fs=1。", "安全系数等于1。", "最大水深为1.0 m。",
    "Fs<1.5。", "Fs<1.0001。", "Fs<10。", "Fs<100。",
    "Fs<1e1。", "Fs<1.0e1。", "Fs<1,000。", "Fs<1.0%。",
    "Fs<1 m。", "安全系数小于1 km²。", "安全系数低于1.0 mm。",
    "安全系数阈值为1.5。", "notFs<1。",
    "安全系数小于1小时。", "Fs<1 米。", "Fs<1 kg。",
])
def test_only_exact_unitless_threshold_one_can_be_exempted(text):
    assert audit_reply(text, EvidenceLedger.from_assessment(ASSESSMENT)), text


def test_threshold_does_not_hide_another_one_in_the_same_reply():
    problems = audit_reply("安全系数小于1作为判别条件；最大水深为1.0 m。", EvidenceLedger.from_assessment(ASSESSMENT))
    assert len(problems) == 1 and "1.0" in problems[0]


def test_repeated_thresholds_across_lines_remain_independent():
    assert audit_reply("Fs<1\nFs<1作为判别条件。", EvidenceLedger.from_assessment(ASSESSMENT)) == []


class Candidate:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    async def complete_json(self, **kwargs):
        self.calls += 1
        return {"reads": [], "answer": self.text}


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate,expected_calls", [
    ("安全系数小于1的区域面积为28.710 km²。", 1),
    ("安全系数小于1的区域面积为28.821 km²。", 5),
    ("安全系数小于1的区域面积为28.710 m²。", 5),
])
async def test_response_and_final_audit_accept_good_area_and_replace_bad_area(build_agent, candidate, expected_calls):
    correct = "安全系数小于1的区域面积为28.710 km²。"
    model = Candidate(candidate)
    answer = await answer_result(llm=model, question="低安全系数区域面积是多少？",
                                 assessment=ASSESSMENT, background={}, catalog=[],
                                 read_artifact=None, fallback=correct, recent_turns=[])
    agent = build_agent()
    session = agent.new_session()
    reply = await agent._finish(session, answer.text, kind="result", ledger=answer.ledger,
                                rewrite_response=False, answer_source="controlled_regression")
    assert reply.ok and reply.text == correct and reply.audit == []
    assert model.calls == expected_calls
    assert answer.used_model == (expected_calls == 1)
