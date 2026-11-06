"""Acceptance for the single-question clarification manager (stage C).

A clarification is the agent's answer to a fuzzy request. These tests pin the
two properties that make it usable — one question at a time, asked in a sensible
order — and the one that makes it safe: it is written by hand, so it works with
no model at all.
"""

from __future__ import annotations

import pytest

from slope_evaluation_agent.clarification import (
    TOPICS,
    TOPIC_ORDER,
    Clarification,
    ask,
    blocked,
    describe_recorded,
    next_topic,
)
from slope_evaluation_agent.intent import Intent
from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.state_machine import WorkflowState


pytestmark = [pytest.mark.unit]


def a_session(state: WorkflowState = WorkflowState.NEW, **facts) -> AgentSession:
    return AgentSession.model_validate({
        "session_id": "session-abcdef123456", "state": state.value, **facts})


class TestOneQuestionAtATime:
    def test_a_clarification_asks_exactly_one_thing(self):
        clarification = ask(a_session(), missing_fields=["river_name", "date", "lookback_years"])
        assert isinstance(clarification, Clarification)
        assert clarification.question.strip()
        # One question, not a form: the other missing fields are not asked about.
        assert "回看年数" not in clarification.question

    def test_at_most_three_examples_are_offered(self):
        for topic in TOPIC_ORDER + ("help",):
            session = a_session()
            clarification = ask(session)
            assert 0 < len(clarification.examples) <= 3, topic

    def test_examples_are_ordinary_phrases_not_commands(self):
        clarification = ask(a_session(), missing_fields=["river_name"])
        for example in clarification.examples:
            assert not example.startswith(("/", "-", ":")), example
            assert "=" not in example, example

    def test_the_render_includes_what_was_recorded(self):
        clarification = ask(a_session(), missing_fields=["date"],
                            recorded=["日期记为 2024-07-30"])
        rendered = clarification.render()
        assert "日期记为 2024-07-30" in rendered
        assert clarification.question in rendered

    def test_nothing_recorded_means_no_acknowledgement_line(self):
        rendered = ask(a_session(), missing_fields=["river_name"]).render()
        assert not rendered.startswith("好的，")

    def test_the_text_is_the_same_every_time(self):
        """Hand-written wording: no model, so no drift between runs."""
        first = ask(a_session(), missing_fields=["river_name"]).render()
        second = ask(a_session(), missing_fields=["river_name"]).render()
        assert first == second


class TestWhatToAskAbout:
    def test_a_missing_place_comes_first(self):
        assert next_topic(a_session(), missing_fields=["river_name"]) == "place"
        assert next_topic(a_session(), missing_fields=["longitude"]) == "place"

    def test_a_pending_boundary_is_asked_about_before_the_rainfall(self):
        session = a_session(WorkflowState.AWAITING_WATERSHED_CONFIRMATION,
                            review_id="basin-abcdef123456")
        assert next_topic(session, missing_fields=["date"]) == "watershed_confirmation"

    def test_the_rainfall_is_asked_about_once_the_boundary_is_confirmed(self):
        session = a_session(WorkflowState.WATERSHED_CONFIRMED, watershed_confirmed=True,
                            review_id="basin-abcdef123456")
        assert next_topic(session) == "rainfall"

    def test_a_known_rainfall_is_not_asked_about_again(self):
        session = a_session(WorkflowState.WATERSHED_CONFIRMED, watershed_confirmed=True,
                            review_id="basin-abcdef123456",
                            rainfall_request={"mode": "historical_max"})
        assert next_topic(session) != "rainfall"

    def test_a_planned_run_is_asked_about_starting(self):
        session = a_session(WorkflowState.HYDRO_PLANNED, watershed_confirmed=True,
                            review_id="basin-abcdef123456",
                            rainfall_request={"mode": "historical_max"},
                            hydro_planned=True)
        assert next_topic(session) == "start"

    def test_nothing_to_go_on_offers_the_way_in(self):
        assert next_topic(a_session()) == "help"

    def test_the_help_text_explains_what_this_agent_does_and_offers_a_way_in(self):
        clarification = ask(a_session(), ambiguity="没有从这句话里看出要做什么")
        assert "初步筛查" in clarification.question or "坡面稳定性" in clarification.question
        assert clarification.examples

    def test_a_clarification_never_claims_the_user_must_pick_an_example(self):
        from slope_evaluation_agent.clarification import TOPICS

        for topic, spec in TOPICS.items():
            text = Clarification(topic=topic, question=spec["question"],
                                 examples=list(spec["examples"])).render()
            for word in ("必须", "只能", "请输入命令", "选项："):
                assert word not in text, (topic, word)


class TestBlockedByTheStateMachine:
    def test_a_refusal_uses_the_gates_own_next_step(self):
        session = a_session(WorkflowState.AWAITING_WATERSHED_CONFIRMATION,
                            review_id="basin-abcdef123456")
        clarification = blocked(session, next_step="请明确确认或拒绝当前流域边界。")
        assert "请明确确认或拒绝当前流域边界。" in clarification.render()

    def test_a_refusal_does_not_leak_internal_state_or_operation_names(self):
        session = a_session(WorkflowState.AWAITING_WATERSHED_CONFIRMATION,
                            review_id="basin-abcdef123456")
        rendered = blocked(session, next_step="请明确确认或拒绝当前流域边界。").render()
        for internal in ("awaiting_watershed_confirmation", "build_rainfall",
                         "start_run", "WorkflowState"):
            assert internal not in rendered, internal

    def test_a_refusal_still_offers_examples(self):
        session = a_session(WorkflowState.AWAITING_WATERSHED_CONFIRMATION,
                            review_id="basin-abcdef123456")
        assert blocked(session, next_step="先确认边界。").examples


class TestRecordedEntitiesAreSpokenBack:
    def test_a_river_is_named(self):
        assert describe_recorded({"river_name": "大宁河"}) == ["河流记为 大宁河"]

    def test_a_rainfall_mode_is_put_into_words(self):
        assert describe_recorded({"rainfall_mode": "historical_max"}) == [
            "降雨改用资料期内样本最大日"]

    def test_empty_values_are_not_announced(self):
        assert describe_recorded({"date": None, "river_name": ""}) == []

    def test_an_unrecognised_field_is_not_announced(self):
        assert describe_recorded({"candidate_index": 2}) == []

    def test_a_date_is_named(self):
        assert describe_recorded({"date": "2024-07-30"}) == ["日期记为 2024-07-30"]


class TestTheLogView:
    def test_the_log_view_carries_no_question_text(self):
        clarification = ask(a_session(), missing_fields=["river_name"],
                            recorded=["河流记为 大宁河"])
        logged = clarification.log_safe()
        assert set(logged) == {"topic", "blocked_by", "examples", "recorded"}
        assert "大宁河" not in str(logged)


class TestEveryClarificationIsAuditSafe:
    """A clarification is a reply, and every reply goes through the numeric audit.

    The wording is hand-written and contains no quantities for exactly this
    reason: an example coordinate in the "where" question was enough to make
    thirteen different clarifications fail the audit and reach the user as
    "内部一致性检查未通过". The examples are illustrations, not evidence, so
    they must not look like evidence.
    """

    def test_no_rendering_carries_a_number_it_cannot_support(self):
        from slope_evaluation_agent.scientific_policy import numeric_audit

        for topic, spec in TOPICS.items():
            rendered = Clarification(topic=topic, question=spec["question"],
                                     examples=list(spec["examples"])).render()
            assert numeric_audit(rendered, {}) == [], topic

    def test_the_examples_are_not_copied_measurements(self):
        """A date inside an example is fine; a decimal quantity is not.

        "2024-07-30 全天" is a shape, and the audit strips dates. "108.03" would
        be a number the reply cannot account for — and a user copying it would
        run an analysis at a point they never chose.
        """
        import re

        decimal = re.compile(r"\d+\.\d+")
        for topic, spec in TOPICS.items():
            for example in spec["examples"]:
                assert not decimal.search(example), (topic, example)

    @pytest.mark.parametrize("topic", sorted(TOPICS))
    def test_a_clarification_survives_the_forbidden_phrase_audit(self, topic):
        from slope_evaluation_agent.scientific_policy import audit_reply

        spec = TOPICS[topic]
        rendered = Clarification(topic=topic, question=spec["question"],
                                 examples=list(spec["examples"])).render()
        assert audit_reply(rendered, {}) == []
