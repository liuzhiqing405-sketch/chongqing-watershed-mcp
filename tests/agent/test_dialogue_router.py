"""Acceptance for the conversational routing contract and the two-level router.

These pin down what a language model is *structurally* unable to say, not what a
prompt asks it not to say. A field that does not exist cannot be filled in, so
the strongest tests here assert the field sets exactly.

The model routes ordinary messages before keyword heuristics can act. Explicit
human decisions and validated workflow execution retain deterministic gates.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from slope_evaluation_agent.config import AgentSettings, DialogueSettings
from slope_evaluation_agent.dialogue_router import (
    ENTITY_FIELDS,
    MODEL_FORBIDDEN_INTENTS,
    STATE_CHANGING_ROUTES,
    DecisionRefused,
    DialogueDecision,
    DialogueRouter,
    EntityCandidate,
    RouteKind,
    missing_entities,
)
from slope_evaluation_agent.intent import Intent
from slope_evaluation_agent.llm import LLMRequestFailed
from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.state_machine import WorkflowState

from stubs import StubLLM


pytestmark = [pytest.mark.unit]


def a_candidate(**overrides) -> EntityCandidate:
    payload = {"field": "river_name", "value": "大宁河", "evidence_text": "大宁河"}
    payload.update(overrides)
    return EntityCandidate.model_validate(payload)


def a_decision(**overrides) -> DialogueDecision:
    payload = {"route": "workflow_action", "intent": "start_assessment",
               "confidence": 0.9}
    payload.update(overrides)
    return DialogueDecision.model_validate(payload)


def a_session(state: WorkflowState = WorkflowState.NEW, **facts) -> AgentSession:
    return AgentSession.model_validate({
        "session_id": "session-abcdef123456", "state": state.value, **facts})


class TestRouteKinds:
    def test_the_four_routes_are_exactly_the_documented_ones(self):
        assert {item.value for item in RouteKind} == {
            "workflow_action", "clarification", "knowledge_chat", "out_of_scope"}

    def test_only_a_workflow_action_may_change_state(self):
        assert STATE_CHANGING_ROUTES == {RouteKind.WORKFLOW_ACTION}
        assert a_decision().changes_workflow_state is True
        for kind in ("clarification", "knowledge_chat", "out_of_scope"):
            assert DialogueDecision(route=kind, confidence=0.5,
                                    ).changes_workflow_state is False


class TestNothingToCallAToolWith:
    """The decision is a description of intent, never an instruction to act."""

    def test_the_decision_has_exactly_the_declared_fields(self):
        assert set(DialogueDecision.model_fields) == {
            "route", "intent", "confidence", "candidates",
            "missing_fields", "topic", "ambiguity"}

    def test_a_candidate_has_exactly_the_declared_fields(self):
        assert set(EntityCandidate.model_fields) == {"field", "value", "evidence_text"}

    def test_no_field_name_looks_like_a_tool_command_or_path(self):
        forbidden = ("tool", "command", "script", "path", "method", "arguments",
                     "params", "payload", "url", "cwd", "mcp")
        names = [*DialogueDecision.model_fields, *EntityCandidate.model_fields] + \
            list(ENTITY_FIELDS)
        for name in names:
            assert not any(word in name.lower() for word in forbidden), name

    def test_a_model_returning_a_tool_name_is_rejected_outright(self):
        with pytest.raises(ValidationError):
            DialogueDecision.model_validate({"route": "knowledge_chat", "confidence": 0.6,
                                             "tool": "os.system"})

    def test_a_model_returning_an_argument_map_is_rejected_outright(self):
        with pytest.raises(ValidationError):
            DialogueDecision.model_validate({"route": "workflow_action",
                                             "intent": "start_assessment",
                                             "confidence": 0.9,
                                             "arguments": {"through": "qa"}})

    def test_a_candidate_carrying_a_path_field_is_rejected_outright(self):
        with pytest.raises(ValidationError):
            EntityCandidate.model_validate({"field": "river_name", "value": "x",
                                            "evidence_text": "x", "path": "D:\\runs"})

    def test_an_entity_field_outside_the_whitelist_is_refused(self):
        for name in ("tool", "command", "run_id", "review_id", "package_id"):
            with pytest.raises(ValidationError):
                EntityCandidate.model_validate({"field": name, "value": "x",
                                                "evidence_text": "x"})


class TestTheModelCannotDecide:
    """Refused at the door a model's answer comes through, not in the type.

    The router carries the *rules'* verdicts in this same type, and the rules are
    precisely the component allowed to record a confirmation — so the ban is on
    ``from_model``, which is where a model's reply is parsed.
    """

    def test_confirming_a_watershed_is_refused(self):
        with pytest.raises(DecisionRefused):
            DialogueDecision.from_model({"route": "workflow_action",
                                         "intent": "confirm_watershed", "confidence": 0.99})

    def test_rejecting_a_watershed_is_refused(self):
        with pytest.raises(DecisionRefused):
            DialogueDecision.from_model({"route": "workflow_action",
                                         "intent": "reject_watershed", "confidence": 0.99})

    def test_cancelling_a_run_is_refused(self):
        with pytest.raises(DecisionRefused):
            DialogueDecision.from_model({"route": "workflow_action",
                                         "intent": "cancel_run", "confidence": 0.99})

    def test_the_rules_may_still_record_a_confirmation(self):
        """The same type, built directly, carries the user's own decision."""
        decision = DialogueDecision(route=RouteKind.WORKFLOW_ACTION,
                                    intent=Intent.CONFIRM_WATERSHED, confidence=0.95)
        assert decision.intent is Intent.CONFIRM_WATERSHED

    def test_every_other_intent_is_still_allowed(self):
        for intent in Intent:
            if intent in MODEL_FORBIDDEN_INTENTS:
                continue
            route = "workflow_action" if intent is not Intent.UNKNOWN else "clarification"
            decision = DialogueDecision.from_model(
                {"route": route, "intent": intent.value, "confidence": 0.9})
            assert decision.intent is intent

    def test_a_workflow_action_must_name_an_intent(self):
        with pytest.raises(ValidationError):
            DialogueDecision(route="workflow_action", confidence=0.9)


class TestMissingFields:
    def test_only_entity_fields_may_be_reported_missing(self):
        with pytest.raises(ValidationError):
            a_decision(missing_fields=["hydro_run_id"])

    def test_duplicates_are_collapsed_keeping_the_first(self):
        decision = a_decision(missing_fields=["date", "river_name", "date"])
        assert decision.missing_fields == ["date", "river_name"]

    def test_a_start_request_needs_a_place_one_way_or_another(self):
        assert missing_entities(Intent.START_ASSESSMENT, {}) == [
            "administrative_area", "river_name", "longitude", "latitude"]
        assert missing_entities(Intent.START_ASSESSMENT, {"administrative_area":"永川区"}) == []
        assert missing_entities(Intent.START_ASSESSMENT, {"river_name": "大宁河"}) == []
        assert missing_entities(Intent.START_ASSESSMENT,
                                {"longitude": 108.0, "latitude": 30.0}) == []

    def test_a_read_only_intent_needs_nothing(self):
        for intent in (Intent.CHECK_STATUS, Intent.SHOW_RESULT, Intent.ASK_EXPLANATION):
            assert missing_entities(intent, {}) == []


class TestTheLogViewLeaksNothing:
    def test_no_evidence_value_or_topic_text_reaches_the_event_log(self):
        decision = a_decision(
            topic="大宁河的水文概念",
            ambiguity="用户说的可能是嘉陵江",
            candidates=[a_candidate(value="大宁河")])
        logged = json.dumps(decision.log_safe(), ensure_ascii=False)
        for secret in ("大宁河", "嘉陵江", "大宁河的水文概念"):
            assert secret not in logged, secret
        assert set(decision.log_safe()) == {
            "route", "intent", "confidence", "candidate_fields",
            "missing_fields", "has_topic", "has_ambiguity"}


class TestDialogueSettings:
    def test_the_defaults_match_the_plan(self):
        settings = DialogueSettings()
        assert settings.enabled is True
        assert settings.workflow_confidence_threshold == 0.85
        assert settings.clarification_confidence_threshold == 0.55
        assert settings.recent_turns == 16
        assert settings.allow_general_chat is True
        assert settings.allow_out_of_scope_reply is True

    def test_the_thresholds_must_be_ordered(self):
        with pytest.raises(ValidationError):
            DialogueSettings(workflow_confidence_threshold=0.4,
                             clarification_confidence_threshold=0.8)

    def test_equal_thresholds_are_allowed(self):
        settings = DialogueSettings(workflow_confidence_threshold=0.6,
                                    clarification_confidence_threshold=0.6)
        assert settings.workflow_confidence_threshold == 0.6

    def test_the_turn_window_is_bounded(self):
        with pytest.raises(ValidationError):
            DialogueSettings(recent_turns=500)

    def test_an_agent_with_no_configuration_still_gets_dialogue_defaults(self):
        assert AgentSettings.model_validate({}).dialogue.enabled is True

    def test_the_shipped_configs_carry_the_section(self, project_root):
        import yaml

        for name in ("agent.default.yaml", "agent.deepseek.yaml"):
            payload = yaml.safe_load(
                (project_root / "config" / name).read_text(encoding="utf-8"))
            assert "dialogue" in payload, name
            # It must survive validation, not merely be present.
            assert AgentSettings.model_validate(payload).dialogue.enabled is True


# ------------------------------------------------------------- stage B: router


def a_router(reply=None, **settings) -> DialogueRouter:
    return DialogueRouter(llm=StubLLM(reply) if reply is not None else None,
                          settings=DialogueSettings(**settings))


class TestModelRoutingAndHumanDecisions:
    """The model routes ordinary messages; explicit decisions remain human."""

    @pytest.mark.asyncio
    async def test_a_confirmation_never_reaches_the_model(self):
        model = StubLLM({"route": "out_of_scope", "confidence": 0.9})
        router = DialogueRouter(llm=model)
        decision = await router.decide(
            "确认该边界", a_session(WorkflowState.AWAITING_WATERSHED_CONFIRMATION,
                                    review_id="basin-abcdef123456"))
        assert decision.route is RouteKind.WORKFLOW_ACTION
        assert decision.intent is Intent.CONFIRM_WATERSHED
        assert model.calls == []

    @pytest.mark.asyncio
    async def test_a_rejection_never_reaches_the_model(self):
        model = StubLLM({"route": "out_of_scope", "confidence": 0.9})
        router = DialogueRouter(llm=model)
        decision = await router.decide(
            "不确认，换一个", a_session(WorkflowState.AWAITING_WATERSHED_CONFIRMATION,
                                        review_id="basin-abcdef123456"))
        assert decision.intent is Intent.REJECT_WATERSHED
        assert model.calls == []

    @pytest.mark.asyncio
    async def test_a_cancellation_never_reaches_the_model(self):
        model = StubLLM({"route": "knowledge_chat", "confidence": 0.9})
        router = DialogueRouter(llm=model)
        decision = await router.decide(
            "取消", a_session(WorkflowState.HYDRO_RUNNING,
                              hydro_run_id="hydro-run-abcdef123456"))
        assert decision.intent is Intent.CANCEL_RUN
        assert model.calls == []

    @pytest.mark.asyncio
    async def test_even_a_complete_rules_request_reaches_the_model(self):
        model = StubLLM({"route": "out_of_scope", "confidence": 0.9})
        router = DialogueRouter(llm=model)
        decision = await router.decide("分析大宁河", a_session())
        assert decision.route is RouteKind.OUT_OF_SCOPE
        assert len(model.calls) == 1

    @pytest.mark.asyncio
    async def test_the_rules_confidence_is_the_one_reported(self):
        router = a_router()
        decision = await router.decide("分析大宁河", a_session())
        assert decision.confidence == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_a_follow_up_the_rules_place_also_reaches_the_model(self):
        model = StubLLM({"route": "out_of_scope", "confidence": 0.9})
        router = DialogueRouter(llm=model)
        decision = await router.decide(
            "第 1 个", a_session(WorkflowState.OUTLET_CANDIDATES_READY,
                                 outlet_candidates=[{"longitude": 108.0, "latitude": 30.0}]))
        assert decision.route is RouteKind.OUT_OF_SCOPE
        assert len(model.calls) == 1


class TestNothingAmbiguousBecomesAConfirmation:
    """The confirmation gate is the one decision never taken by accident.

    "确认" is a substring of "不确认", of "没确认" and of "不确定", so a plain
    substring search answers yes to all three. The gate cannot afford that.
    """

    PENDING = {"state": WorkflowState.AWAITING_WATERSHED_CONFIRMATION,
               "review_id": "basin-abcdef123456"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("phrase", ["不确认", "先不确认", "不用这个", "换一个"])
    async def test_a_negated_confirmation_is_a_rejection(self, phrase):
        decision = await a_router().decide(phrase, a_session(**self.PENDING))
        assert decision.intent is Intent.REJECT_WATERSHED

    @pytest.mark.asyncio
    @pytest.mark.parametrize("phrase", ["没确认", "不确定", "继续", "可以帮我分析吗",
                                        "接着做吧", "就这样吧", "先看看"])
    async def test_an_ambiguous_phrase_is_never_a_confirmation(self, phrase):
        decision = await a_router().decide(phrase, a_session(**self.PENDING))
        assert decision.intent is not Intent.CONFIRM_WATERSHED, phrase

    @pytest.mark.asyncio
    @pytest.mark.parametrize("phrase", ["确认该边界", "确认", "确定", "同意", "就用这个"])
    async def test_an_explicit_confirmation_still_works(self, phrase):
        decision = await a_router().decide(phrase, a_session(**self.PENDING))
        assert decision.intent is Intent.CONFIRM_WATERSHED

    @pytest.mark.asyncio
    async def test_a_model_claiming_a_confirmation_cannot_get_through(self):
        router = DialogueRouter(llm=StubLLM({
            "route": "workflow_action", "intent": "confirm_watershed", "confidence": 1.0}))
        decision = await router.decide("继续", a_session(**self.PENDING))
        assert decision.intent is not Intent.CONFIRM_WATERSHED


class TestTheModelFillsBlanks:
    RAINY = "嗯，就那个七月三十号那天的雨吧"

    @pytest.mark.asyncio
    async def test_an_ambiguous_request_is_placed_by_the_model(self):
        model = StubLLM({"route": "workflow_action", "intent": "set_rainfall_period",
                         "confidence": 0.9,
                         "candidates": [{"field": "date", "value": "2024-07-30",
                                         "evidence_text": "七月三十号"}]})
        router = DialogueRouter(llm=model)
        decision = await router.decide(self.RAINY, a_session())
        assert model.calls, "模型没有被问到"
        assert decision.route is RouteKind.WORKFLOW_ACTION
        assert decision.intent is Intent.SET_RAINFALL_PERIOD

    @pytest.mark.asyncio
    async def test_the_placed_candidate_reaches_the_workflow_object(self):
        router = DialogueRouter(llm=StubLLM({
            "route": "workflow_action", "intent": "set_rainfall_period", "confidence": 0.9,
            "candidates": [{"field": "date", "value": "2024-07-30",
                            "evidence_text": "七月三十号"}]}))
        session = a_session()
        decision = await router.decide(self.RAINY, session)
        parsed, problems = router.to_parsed_intent(decision, self.RAINY, session)
        assert parsed.date == "2024-07-30"
        assert parsed.intent is Intent.SET_RAINFALL_PERIOD
        assert problems == []

    @pytest.mark.asyncio
    async def test_a_candidate_without_evidence_is_dropped(self):
        router = DialogueRouter(llm=StubLLM({
            "route": "workflow_action", "intent": "set_rainfall_period", "confidence": 0.9,
            "candidates": [{"field": "date", "value": "2024-08-15",
                            "evidence_text": "八月十五号"}]}))
        session = a_session()
        decision = await router.decide(self.RAINY, session)
        # Nothing usable survived, so the request is incomplete and must be asked about.
        assert decision.route is RouteKind.CLARIFICATION
        assert "date" in decision.missing_fields

    @pytest.mark.asyncio
    async def test_a_candidate_that_fails_validation_is_dropped(self):
        router = DialogueRouter(llm=StubLLM({
            "route": "workflow_action", "intent": "set_rainfall_period", "confidence": 0.9,
            "candidates": [{"field": "date", "value": "2024-02-31",
                            "evidence_text": "七月三十号"}]}))
        decision = await router.decide(self.RAINY, a_session())
        assert decision.route is RouteKind.CLARIFICATION
        assert "date" in decision.missing_fields

    @pytest.mark.asyncio
    async def test_conflicting_candidates_become_a_question(self):
        router = DialogueRouter(llm=StubLLM({
            "route": "workflow_action", "intent": "select_outlet", "confidence": 0.95,
            "candidates": [
                {"field": "longitude", "value": "106.5", "evidence_text": "106.5"},
                {"field": "longitude", "value": "109.5", "evidence_text": "109.5"}]}))
        decision = await router.decide("大概是 106.5 或者 109.5 那边", a_session())
        assert decision.route is RouteKind.CLARIFICATION
        assert decision.missing_fields == ["longitude"]

    @pytest.mark.asyncio
    async def test_a_low_confidence_action_becomes_a_question(self):
        router = DialogueRouter(llm=StubLLM({
            "route": "workflow_action", "intent": "set_rainfall_period", "confidence": 0.6,
            "candidates": [{"field": "date", "value": "2024-07-30",
                            "evidence_text": "七月三十号"}]}))
        decision = await router.decide(self.RAINY, a_session())
        assert decision.route is RouteKind.CLARIFICATION

    @pytest.mark.asyncio
    async def test_an_incomplete_action_asks_for_what_is_missing(self):
        router = DialogueRouter(llm=StubLLM({
            "route": "workflow_action", "intent": "start_assessment", "confidence": 0.95}))
        decision = await router.decide("嗯……那这个呢", a_session())
        assert decision.route is RouteKind.CLARIFICATION
        assert "river_name" in decision.missing_fields

    @pytest.mark.asyncio
    async def test_a_value_the_rules_found_beats_the_model_candidate(self):
        message = "大宁河，就那个七月三十号那天的雨吧"
        router = DialogueRouter(llm=StubLLM({
            "route": "workflow_action", "intent": "set_rainfall_period", "confidence": 0.9,
            "candidates": [{"field": "river_name", "value": "嘉陵江",
                            "evidence_text": "大宁河"}]}))
        session = a_session()
        decision = await router.decide(message, session)
        parsed, _ = router.to_parsed_intent(decision, message, session)
        assert parsed.river_name == "大宁河", "规则找到的取值被模型候选覆盖了"

    @pytest.mark.asyncio
    async def test_a_knowledge_question_is_routed_as_conversation(self):
        router = DialogueRouter(llm=StubLLM({
            "route": "knowledge_chat", "confidence": 0.9, "topic": "安全系数的含义"}))
        decision = await router.decide("安全系数越大越好吗？", a_session())
        assert decision.route is RouteKind.KNOWLEDGE_CHAT
        assert decision.changes_workflow_state is False

    @pytest.mark.asyncio
    async def test_the_model_is_shown_the_state_and_the_recent_turns(self):
        model = StubLLM({"route": "clarification", "confidence": 0.6})
        router = DialogueRouter(llm=model)
        session = a_session(WorkflowState.WATERSHED_CONFIRMED, river_name="大宁河")
        session.record_turn("user", "分析大宁河")
        session.record_turn("agent", "已记录确认")
        await router.decide("嗯……那这个呢", session)
        shown = json.loads(model.calls[0]["user"])
        assert shown["context"]["current_state"] == "watershed_confirmed"
        assert shown["context"]["known"]["river_name"] == "大宁河"
        assert [item["text"] for item in shown["context"]["recent_conversation"]] == [
            "分析大宁河", "已记录确认"]

    @pytest.mark.asyncio
    async def test_the_model_is_never_shown_a_path_or_a_key(self):
        model = StubLLM({"route": "clarification", "confidence": 0.6})
        router = DialogueRouter(llm=model)
        session = a_session()
        await router.decide("嗯……那这个呢", session)
        prompt = model.calls[0]["system"] + model.calls[0]["user"]
        for marker in ("D:\\", "C:\\", "sk-", "api_key", "Bearer"):
            assert marker not in prompt, marker

    @pytest.mark.asyncio
    async def test_the_recent_turn_window_is_bounded(self):
        model = StubLLM({"route": "clarification", "confidence": 0.6})
        router = DialogueRouter(llm=model, settings=DialogueSettings(recent_turns=2))
        session = a_session()
        for index in range(10):
            session.record_turn("user", f"第 {index} 句话")
        await router.decide("嗯……那这个呢", session)
        shown = json.loads(model.calls[0]["user"])
        assert len(shown["context"]["recent_conversation"]) == 2


class TestTheModelCannotDecideThroughTheRouter:
    @pytest.mark.asyncio
    async def test_a_confirm_intent_from_the_model_is_refused(self):
        router = DialogueRouter(llm=StubLLM({
            "route": "workflow_action", "intent": "confirm_watershed", "confidence": 1.0}))
        decision = await router.decide("嗯……那这个呢", a_session())
        assert decision.intent is not Intent.CONFIRM_WATERSHED
        assert decision.route is RouteKind.CLARIFICATION

    @pytest.mark.asyncio
    async def test_a_reject_intent_from_the_model_is_refused(self):
        router = DialogueRouter(llm=StubLLM({
            "route": "workflow_action", "intent": "reject_watershed", "confidence": 1.0}))
        decision = await router.decide("嗯……那这个呢", a_session())
        assert decision.intent is not Intent.REJECT_WATERSHED

    @pytest.mark.asyncio
    async def test_a_cancel_intent_from_the_model_is_refused(self):
        router = DialogueRouter(llm=StubLLM({
            "route": "workflow_action", "intent": "cancel_run", "confidence": 1.0}))
        decision = await router.decide("嗯……那这个呢", a_session())
        assert decision.intent is not Intent.CANCEL_RUN

    @pytest.mark.asyncio
    async def test_a_tool_field_from_the_model_is_refused(self):
        router = DialogueRouter(llm=StubLLM({
            "route": "knowledge_chat", "confidence": 0.9, "tool": "os.system"}))
        decision = await router.decide("嗯……那这个呢", a_session())
        assert decision.route is RouteKind.CLARIFICATION

    @pytest.mark.asyncio
    async def test_an_out_of_enum_route_is_refused(self):
        router = DialogueRouter(llm=StubLLM({
            "route": "run_swmm_now", "confidence": 0.9}))
        decision = await router.decide("嗯……那这个呢", a_session())
        assert decision.route is RouteKind.CLARIFICATION

    @pytest.mark.asyncio
    async def test_a_model_that_tries_to_place_a_new_river_name_cannot(self):
        """A name the user never typed cannot survive the grounding check."""
        router = DialogueRouter(llm=StubLLM({
            "route": "workflow_action", "intent": "start_assessment", "confidence": 0.99,
            "candidates": [{"field": "river_name", "value": "长江",
                            "evidence_text": "长江"}]}))
        decision = await router.decide("嗯……那个地方怎么样", a_session())
        assert decision.route is RouteKind.CLARIFICATION


class TestNoModelNoProblem:
    """DeepSeek being absent, slow or broken must not cost the workflow."""

    @pytest.mark.asyncio
    async def test_without_a_model_a_rules_request_still_executes(self):
        decision = await a_router().decide("分析大宁河", a_session())
        assert decision.route is RouteKind.WORKFLOW_ACTION
        assert decision.intent is Intent.START_ASSESSMENT

    @pytest.mark.asyncio
    async def test_without_a_model_a_follow_up_still_executes(self):
        decision = await a_router().decide(
            "第 1 个", a_session(WorkflowState.OUTLET_CANDIDATES_READY,
                                 outlet_candidates=[{"longitude": 108.0, "latitude": 30.0}]))
        assert decision.route is RouteKind.WORKFLOW_ACTION
        assert decision.intent is Intent.SELECT_OUTLET

    @pytest.mark.asyncio
    async def test_without_a_model_an_unknown_request_asks_instead_of_acting(self):
        decision = await a_router().decide("嗯……那这个呢", a_session())
        assert decision.route is RouteKind.CLARIFICATION
        assert decision.changes_workflow_state is False

    @pytest.mark.asyncio
    async def test_a_greeting_is_still_answered_without_a_model(self):
        """Ordinary conversation should not depend on the model being reachable."""
        decision = await a_router().decide("你好", a_session())
        assert decision.route is RouteKind.KNOWLEDGE_CHAT
        assert decision.topic == "greeting"

    @pytest.mark.asyncio
    async def test_a_concept_question_is_still_answered_without_a_model(self):
        decision = await a_router().decide("安全系数越大越好吗？", a_session())
        assert decision.route is RouteKind.KNOWLEDGE_CHAT
        assert decision.topic == "safety_factor"

    @pytest.mark.asyncio
    async def test_a_question_about_a_scene_is_not_answered_from_a_glossary(self):
        """"最近这场雨会不会不稳定" is a request for an analysis, not a concept."""
        decision = await a_router().decide("最近一次比较大的雨会不会不稳定？", a_session())
        assert decision.route is RouteKind.CLARIFICATION

    @pytest.mark.asyncio
    async def test_without_a_model_a_confirmation_is_still_a_confirmation(self):
        decision = await a_router().decide(
            "确认该边界", a_session(WorkflowState.AWAITING_WATERSHED_CONFIRMATION,
                                    review_id="basin-abcdef123456"))
        assert decision.intent is Intent.CONFIRM_WATERSHED

    @pytest.mark.asyncio
    async def test_a_timeout_falls_back(self):
        router = DialogueRouter(llm=StubLLM(TimeoutError("timed out")))
        decision = await router.decide("嗯……那这个呢", a_session())
        assert decision.route is RouteKind.CLARIFICATION

    @pytest.mark.asyncio
    async def test_an_http_failure_falls_back(self):
        router = DialogueRouter(llm=StubLLM(LLMRequestFailed("HTTP 500")))
        decision = await router.decide("嗯……那这个呢", a_session())
        assert decision.route is RouteKind.CLARIFICATION

    @pytest.mark.asyncio
    async def test_a_non_object_reply_falls_back(self):
        router = DialogueRouter(llm=StubLLM(["not", "an", "object"]))  # type: ignore[arg-type]
        decision = await router.decide("嗯……那这个呢", a_session())
        assert decision.route is RouteKind.CLARIFICATION

    @pytest.mark.asyncio
    async def test_a_broken_reply_does_not_stop_a_request_the_rules_placed(self):
        router = DialogueRouter(llm=StubLLM(TimeoutError("timed out")))
        decision = await router.decide("分析大宁河", a_session())
        assert decision.route is RouteKind.WORKFLOW_ACTION

    @pytest.mark.asyncio
    async def test_turning_the_dialogue_layer_off_returns_the_rules_verdict(self):
        router = DialogueRouter(llm=StubLLM({"route": "knowledge_chat", "confidence": 0.9}),
                                settings=DialogueSettings(enabled=False))
        assert router.llm_available is False
        decision = await router.decide("嗯……那这个呢", a_session())
        assert decision.route is RouteKind.CLARIFICATION

    @pytest.mark.asyncio
    async def test_general_chat_can_be_switched_off(self):
        router = DialogueRouter(llm=StubLLM({"route": "knowledge_chat", "confidence": 0.9}),
                                settings=DialogueSettings(allow_general_chat=False))
        decision = await router.decide("安全系数越大越好吗？", a_session())
        assert decision.route is not RouteKind.KNOWLEDGE_CHAT

    @pytest.mark.asyncio
    async def test_out_of_scope_replies_can_be_switched_off(self):
        router = DialogueRouter(
            llm=StubLLM({"route": "out_of_scope", "confidence": 0.9}),
            settings=DialogueSettings(allow_out_of_scope_reply=False))
        decision = await router.decide("帮我写首诗", a_session())
        assert decision.route is not RouteKind.OUT_OF_SCOPE
