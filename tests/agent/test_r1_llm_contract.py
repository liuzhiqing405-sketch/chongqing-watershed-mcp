"""R0 regression tests for R1: the language model is wired in, and fenced in.

The configuration carries a base URL, a model and a key, but nothing injects a
client into the running parser — ``build_llm_client`` exists and is never
called. These tests drive the *agent*, not the settings object, so they fail
until the model is genuinely reachable on the two paths it is allowed on, and
still refused everywhere else.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from slope_evaluation_agent.config import AgentSettings
from slope_evaluation_agent.intent import Intent, IntentParser
from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.response_rewriter import CanonicalReply, ResponseRewriter, SYSTEM_PROMPT
from slope_evaluation_agent.scientific_policy import EvidenceLedger
from slope_evaluation_agent.state_machine import WorkflowState


pytestmark = [pytest.mark.unit, pytest.mark.llm_contract]


class RecordingModel:
    """A language model that records what it was asked and answers on demand."""

    def __init__(self, reply: dict[str, Any] | None = None,
                 error: Exception | None = None) -> None:
        self.reply = reply or {"intent": "unknown", "confidence": 0.1}
        self.error = error
        self.calls: list[dict[str, str]] = []

    async def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        self.calls.append({"system": system, "user": user})
        if self.error is not None:
            raise self.error
        return self.reply


def session_at(state: WorkflowState = WorkflowState.NEW, **facts) -> AgentSession:
    return AgentSession.model_validate(
        {"session_id": "session-abcdef123456", "state": state.value, **facts})


def llm_settings(monkeypatch, tmp_path, **llm) -> AgentSettings:
    monkeypatch.setattr("slope_evaluation_agent.config.PROJECT_ROOT", tmp_path)
    return AgentSettings.model_validate({
        "sessions": {"root": str(tmp_path / "sessions")},
        "llm": {"base_url": "https://example.invalid/v1", "model": "test-model",
                "api_key": "sk-not-a-real-key", **llm},
    })


class TestTheAgentInjectsTheModel:
    def test_a_configured_model_reaches_the_parser(self, monkeypatch, tmp_path):
        """Plan R1: the running parser must report the model, not just the settings."""
        from slope_evaluation_agent.agent import SlopeEvaluationAgent

        settings = llm_settings(monkeypatch, tmp_path)
        monkeypatch.setattr("slope_evaluation_agent.agent.build_llm_client",
                            lambda *a, **k: RecordingModel(), raising=False)
        agent = SlopeEvaluationAgent(settings)
        assert agent.parser.llm_available is True, "配置了模型却没有注入解析器"

    def test_an_unconfigured_agent_has_no_model(self, monkeypatch, tmp_path):
        from slope_evaluation_agent.agent import SlopeEvaluationAgent

        monkeypatch.setattr("slope_evaluation_agent.config.PROJECT_ROOT", tmp_path)
        agent = SlopeEvaluationAgent(AgentSettings.model_validate({}))
        assert agent.parser.llm_available is False

    def test_an_explicitly_disabled_model_is_not_injected(self, monkeypatch, tmp_path):
        from slope_evaluation_agent.agent import SlopeEvaluationAgent

        settings = llm_settings(monkeypatch, tmp_path, enabled=False)
        agent = SlopeEvaluationAgent(settings)
        assert agent.parser.llm_available is False

    def test_the_client_is_built_from_the_settings(self, monkeypatch, tmp_path):
        """The agent must call the builder rather than construct its own."""
        from slope_evaluation_agent.agent import SlopeEvaluationAgent

        settings = llm_settings(monkeypatch, tmp_path)
        seen: list[Any] = []

        def fake_builder(llm_settings, logger=None):
            seen.append(llm_settings.model)
            return RecordingModel()

        monkeypatch.setattr("slope_evaluation_agent.agent.build_llm_client",
                            fake_builder, raising=False)
        SlopeEvaluationAgent(settings)
        assert seen == ["test-model"]


class TestNoNetworkWithoutConfiguration:
    @pytest.mark.asyncio
    async def test_an_unconfigured_parser_never_calls_out(self, monkeypatch, tmp_path):
        settings = llm_settings(monkeypatch, tmp_path, base_url=None, model=None)
        parser = IntentParser(llm=None)
        assert parser.llm_available is False
        parsed = await parser.parse("今天天气不错", session_at())
        assert parsed.intent is Intent.UNKNOWN


class TestTheModelMayOnlyFillInBlanks:
    @pytest.mark.asyncio
    async def test_an_ambiguous_help_question_may_be_placed_by_the_model(self):
        from slope_evaluation_agent.intent import IntentParser

        model = RecordingModel({"intent": "ask_explanation", "confidence": 0.9})
        parser = IntentParser(llm=model)
        parsed = await parser.parse("嗯……那这个呢", session_at())
        assert parsed.intent is Intent.ASK_EXPLANATION
        assert model.calls, "模型没有被调用"

    @pytest.mark.asyncio
    async def test_the_rules_decide_before_the_model_is_consulted(self):
        model = RecordingModel({"intent": "cancel_run", "confidence": 1.0})
        parser = IntentParser(llm=model)
        parsed = await parser.parse("分析大宁河", session_at())
        assert parsed.intent is Intent.START_ASSESSMENT
        assert model.calls == []


class TestTheModelCannotDecide:
    @pytest.mark.asyncio
    async def test_a_confirmation_never_comes_from_the_model(self):
        model = RecordingModel({"intent": "confirm_watershed", "confidence": 1.0})
        parser = IntentParser(llm=model)
        parsed = await parser.parse("嗯……那这个呢", session_at())
        assert parsed.intent is not Intent.CONFIRM_WATERSHED

    @pytest.mark.asyncio
    async def test_a_rejection_never_comes_from_the_model(self):
        model = RecordingModel({"intent": "reject_watershed", "confidence": 1.0})
        parser = IntentParser(llm=model)
        parsed = await parser.parse("嗯……那这个呢",
                                    session_at(WorkflowState.AWAITING_WATERSHED_CONFIRMATION,
                                               review_id="basin-abcdef123456"))
        assert parsed.intent is not Intent.REJECT_WATERSHED

    @pytest.mark.asyncio
    async def test_an_out_of_enum_intent_is_refused(self):
        model = RecordingModel({"intent": "run_swmm", "confidence": 1.0})
        parser = IntentParser(llm=model)
        parsed = await parser.parse("嗯……那这个呢", session_at())
        assert parsed.intent is Intent.UNKNOWN

    @pytest.mark.asyncio
    async def test_coordinates_from_the_model_are_refused(self):
        model = RecordingModel({"intent": "select_outlet", "longitude": 106.5,
                                "latitude": 29.6, "confidence": 1.0})
        parser = IntentParser(llm=model)
        parsed = await parser.parse("嗯……那这个呢", session_at())
        assert parsed.longitude is None and parsed.latitude is None

    @pytest.mark.asyncio
    async def test_a_run_id_from_the_model_is_refused(self, monkeypatch, tmp_path):
        from slope_evaluation_agent.agent import SlopeEvaluationAgent

        settings = llm_settings(monkeypatch, tmp_path)
        model = RecordingModel({"intent": "check_status", "confidence": 1.0,
                                "identifiers": {"hydro_run_id": "hydro-run-deadbeef0000"}})
        agent = SlopeEvaluationAgent(settings, parser=IntentParser(llm=model))
        session = agent.new_session()
        parsed = await agent.parser.parse("嗯……那这个呢", session)
        assert not parsed.identifiers.get("hydro_run_id")

    @pytest.mark.asyncio
    async def test_a_tool_name_from_the_model_is_refused(self):
        model = RecordingModel({"intent": "unknown", "confidence": 0.1,
                                "tool": "os.system", "command": "rm -rf /"})
        parser = IntentParser(llm=model)
        parsed = await parser.parse("嗯……那这个呢", session_at())
        payload = json.dumps(parsed.model_dump(), ensure_ascii=False)
        assert "os.system" not in payload and "rm -rf" not in payload


class TestTheModelFailsSafe:
    @pytest.mark.asyncio
    async def test_an_http_error_falls_back(self):
        from slope_evaluation_agent.llm import LLMRequestFailed

        parser = IntentParser(llm=RecordingModel(error=LLMRequestFailed("HTTP 500")))
        parsed = await parser.parse("分析大宁河", session_at())
        assert parsed.river_name == "大宁河"

    @pytest.mark.asyncio
    async def test_a_timeout_falls_back(self):
        parser = IntentParser(llm=RecordingModel(error=TimeoutError("timed out")))
        parsed = await parser.parse("分析大宁河", session_at())
        assert parsed.river_name == "大宁河"

    @pytest.mark.asyncio
    async def test_missing_fields_fall_back(self):
        parser = IntentParser(llm=RecordingModel({"confidence": 0.9}))
        parsed = await parser.parse("嗯……那这个呢", session_at())
        assert parsed.intent is Intent.UNKNOWN

    @pytest.mark.asyncio
    async def test_a_failure_does_not_break_the_turn(self, monkeypatch, tmp_path):
        from slope_evaluation_agent.agent import SlopeEvaluationAgent
        from helpers import converse

        settings = llm_settings(monkeypatch, tmp_path)
        agent = SlopeEvaluationAgent(
            settings, parser=IntentParser(llm=RecordingModel(error=TimeoutError("x"))))
        session = agent.new_session()
        replies = await converse(agent, session, ["分析大宁河"])
        assert replies[-1].text


class TestTheModelCannotRewriteNumbers:
    def test_rewrite_contract_requires_a_json_text_object(self):
        assert "JSON 对象" in SYSTEM_PROMPT
        assert "text" in SYSTEM_PROMPT
        assert "不要输出 JSON" not in SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_a_rewrite_must_preserve_all_protected_facts(self):
        model = RecordingModel({"text": "结果已经生成，请谨慎理解。"})
        rewriter = ResponseRewriter(model)
        canonical = CanonicalReply(
            intent="show_result", status="result",
            canonical_text="最小安全系数为 0.2752；累计降雨为 79.288 mm。")
        ledger = EvidenceLedger.from_evidence(EVIDENCE := {
            "minimum_safety_factor": 0.2752, "total_rain_mm": 79.288})
        outcome = await rewriter.rewrite(canonical, ledger)
        assert outcome.text == canonical.canonical_text
        assert outcome.used_model is False
        assert outcome.fallback_reason

    @pytest.mark.asyncio
    async def test_locked_literals_can_be_rephrased_but_not_changed(self):
        class EchoProtectedModel:
            async def complete_json(self, *, system: str, user: str):
                protected = user.split("待改写的正文：\n", 1)[1]
                return {"text": "复核结果：" + protected}

        rewriter = ResponseRewriter(EchoProtectedModel())
        canonical = CanonicalReply(
            intent="show_result", status="result",
            canonical_text="最小安全系数为 0.2752。")
        ledger = EvidenceLedger.from_evidence({"minimum_safety_factor": 0.2752})
        outcome = await rewriter.rewrite(canonical, ledger)
        assert "0.2752" in outcome.text
        assert outcome.used_model is True

    @pytest.mark.asyncio
    async def test_a_rewritten_reply_is_audited(self, build_agent, calls, monkeypatch):
        from helpers import PRELUDE, converse, prepared_agent

        agent, session = await prepared_agent(build_agent, calls, run_status="succeeded")

        class InventiveRewrite:
            async def complete_json(self, *, system: str, user: str):
                return {"text": "最小安全系数为 0.9。"}

        rewrites: list[str] = []

        async def rewrite(*, canonical, evidence, **kwargs):
            rewrites.append(canonical)
            return "最小安全系数为 0.9。"

        monkeypatch.setattr(agent, "_rewrite_reply", rewrite, raising=False)
        replies = await converse(agent, session, PRELUDE + ["开始运行", "结果怎么样"])
        assert rewrites, "改写路径没有被调用"
        assert "0.9" not in replies[-1].text

    @pytest.mark.asyncio
    async def test_the_model_receives_only_the_canonical_reply(self, build_agent, calls,
                                                               monkeypatch):
        """It must not be handed the session file or a raw MCP response."""
        from helpers import PRELUDE, converse, prepared_agent

        agent, session = await prepared_agent(build_agent, calls, run_status="succeeded")
        seen: list[str] = []

        async def rewrite(*, canonical, evidence, **kwargs):
            seen.append(str(canonical))
            return str(canonical)

        monkeypatch.setattr(agent, "_rewrite_reply", rewrite, raising=False)
        await converse(agent, session, PRELUDE + ["开始运行", "结果怎么样"])
        assert seen
        joined = " ".join(seen)
        for marker in ("session_id", "artifact_id", "internal_path", "sha256"):
            assert marker not in joined, marker


class TestTheAuditRecordIsSafe:
    @pytest.mark.asyncio
    async def test_the_event_log_records_no_key(self, build_agent, calls, monkeypatch):
        from helpers import converse

        settings = build_agent().settings
        agent = build_agent()
        session = agent.new_session()
        await converse(agent, session, ["分析大宁河"])
        raw = agent.store.events_path(session.session_id).read_text(encoding="utf-8")
        assert "sk-not-a-real-key" not in raw
        assert "Authorization" not in raw
