"""Acceptance for entity candidates: grounded, bounded, locally validated.

A candidate is what a model *thinks* it read out of the user's message. Nothing
here trusts it: it must quote the user, it must survive the workflow's own
validators, and a value that fails is reported rather than dropped in silence.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from slope_evaluation_agent.dialogue_router import (
    DialogueDecision,
    EntityCandidate,
    validated_payload,
)


pytestmark = [pytest.mark.unit]


MESSAGE = "帮我看看大宁河 2024-07-30 全天的情况，坐标大概是东经106.5北纬29.6"


def a_candidate(**overrides) -> EntityCandidate:
    payload = {"field": "river_name", "value": "大宁河", "evidence_text": "大宁河"}
    payload.update(overrides)
    return EntityCandidate.model_validate(payload)


class TestCandidatesMustBeGrounded:
    def test_evidence_must_be_a_verbatim_fragment(self):
        assert a_candidate().is_grounded_in(MESSAGE)

    def test_invented_evidence_is_not_grounded(self):
        invented = a_candidate(value="嘉陵江", evidence_text="嘉陵江")
        assert not invented.is_grounded_in(MESSAGE)

    def test_a_value_seen_only_in_the_reply_is_not_grounded(self):
        """The evidence has to be in the *user's* words, not somewhere nearby."""
        smuggled = a_candidate(value="长江", evidence_text="大宁河")
        assert not smuggled.is_grounded_in(MESSAGE)
        decision = DialogueDecision(route="workflow_action", intent="start_assessment",
                                    confidence=0.9, candidates=[smuggled])
        assert decision.grounded_candidates(MESSAGE) == []
        assert decision.ungrounded_candidates(MESSAGE) == [smuggled]
        # A real quote cannot support a different river name.

    def test_the_grounded_filter_keeps_only_what_the_user_wrote(self):
        decision = DialogueDecision(
            route="workflow_action", intent="start_assessment", confidence=0.9,
            candidates=[a_candidate(),
                        a_candidate(field="date", value="2024-07-30",
                                    evidence_text="2024-07-30"),
                        a_candidate(field="river_name", value="嘉陵江",
                                    evidence_text="嘉陵江")])
        assert {item.value for item in decision.grounded_candidates(MESSAGE)} == {
            "大宁河", "2024-07-30"}

    def test_the_dropped_ones_are_reportable(self):
        decision = DialogueDecision(
            route="clarification", confidence=0.5,
            candidates=[a_candidate(),
                        a_candidate(field="river_name", value="嘉陵江",
                                    evidence_text="嘉陵江")])
        assert [item.value for item in decision.ungrounded_candidates(MESSAGE)] == ["嘉陵江"]

    def test_empty_evidence_is_refused_at_construction(self):
        with pytest.raises(ValidationError):
            a_candidate(evidence_text="   ")

    def test_an_over_long_evidence_is_refused(self):
        with pytest.raises(ValidationError):
            a_candidate(evidence_text="证" * 401)


class TestCandidateValues:
    def test_a_boolean_is_never_an_entity(self):
        with pytest.raises(ValidationError):
            a_candidate(value=True)

    def test_an_empty_value_is_refused(self):
        with pytest.raises(ValidationError):
            a_candidate(value="   ")

    def test_an_over_long_value_is_refused(self):
        with pytest.raises(ValidationError):
            a_candidate(value="河" * 201)

    def test_numbers_are_kept_as_numbers(self):
        candidate = a_candidate(field="longitude", value=108.03, evidence_text="108.03")
        assert candidate.value == 108.03
        assert candidate.as_text() == "108.03"

    def test_values_are_stripped(self):
        assert a_candidate(value="  大宁河  ").value == "大宁河"


class TestConflictingCandidates:
    def test_two_values_for_one_field_are_a_conflict(self):
        assert DialogueDecision.conflicts([
            a_candidate(field="longitude", value="108.0", evidence_text="108.0"),
            a_candidate(field="longitude", value="109.0", evidence_text="109.0"),
        ]) == {"longitude": ["108.0", "109.0"]}

    def test_the_same_value_twice_is_not_a_conflict(self):
        assert DialogueDecision.conflicts([
            a_candidate(value="大宁河"), a_candidate(value="大宁河")]) == {}

    def test_no_candidates_no_conflict(self):
        assert DialogueDecision.conflicts([]) == {}


class TestLocalValidation:
    """The workflow's own validators, not a second set written for the model."""

    def test_a_valid_coordinate_becomes_a_number(self):
        payload, problems = validated_payload([
            a_candidate(field="longitude", value="106.5", evidence_text="106.5"),
            a_candidate(field="latitude", value="29.6", evidence_text="29.6")])
        assert payload == {"longitude": 106.5, "latitude": 29.6}
        assert problems == []

    def test_an_impossible_coordinate_is_refused(self):
        payload, problems = validated_payload([
            a_candidate(field="longitude", value="999", evidence_text="999"),
            a_candidate(field="latitude", value="999", evidence_text="999")])
        assert payload == {}
        assert len(problems) == 2
        assert all("未通过校验" in item for item in problems)

    def test_an_impossible_date_is_refused(self):
        payload, problems = validated_payload([
            a_candidate(field="date", value="2024-02-31", evidence_text="2024-02-31")])
        assert payload == {}
        assert "不是有效日期" in problems[0]

    def test_a_real_leap_day_is_accepted(self):
        payload, _ = validated_payload([
            a_candidate(field="date", value="2024-02-29", evidence_text="2024-02-29")])
        assert payload == {"date": "2024-02-29"}

    def test_a_date_and_a_clock_time_are_normalised(self):
        payload, problems = validated_payload([
            a_candidate(field="start_datetime", value="2024-07-30 08:00",
                        evidence_text="08:00")])
        assert payload == {"start_datetime": "2024-07-30 08:00"}
        assert problems == []

    def test_a_nonsense_time_is_refused(self):
        payload, problems = validated_payload([
            a_candidate(field="start_datetime", value="下午三点", evidence_text="下午三点")])
        assert payload == {}
        assert problems

    def test_an_out_of_range_index_is_refused(self):
        _, problems = validated_payload([
            a_candidate(field="candidate_index", value="42", evidence_text="42")])
        assert problems

    def test_one_bad_value_does_not_discard_the_good_ones(self):
        payload, problems = validated_payload([
            a_candidate(field="longitude", value="106.5", evidence_text="106.5"),
            a_candidate(field="latitude", value="999", evidence_text="999")])
        assert payload == {"longitude": 106.5}
        assert len(problems) == 1

    def test_a_dangerous_river_name_is_refused_by_the_rules(self):
        """The name validators already reject shell and path syntax."""
        payload, problems = validated_payload([
            a_candidate(value="../../etc/passwd", evidence_text="../../etc/passwd")])
        assert payload == {}
        assert problems

    def test_every_refusal_is_reported(self):
        _, problems = validated_payload([
            a_candidate(field="date", value="2024-02-31", evidence_text="2024-02-31"),
            a_candidate(field="latitude", value="999", evidence_text="999")])
        assert len(problems) == 2
