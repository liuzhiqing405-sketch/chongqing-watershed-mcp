"""R0 regression tests for R4 and R5.

R4: a follow-up question must be answered from the current session's evidence.
The handler currently calls ``explain`` without passing the assessment, so
"为什么最小安全系数这么低" can only return a generic definition.

R5: the session file must hold opaque ids and scientific facts, never a machine
path. It currently stores the exported package's absolute directory.
"""

from __future__ import annotations

import json
import re

import pytest

from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.state_machine import WorkflowState


pytestmark = [pytest.mark.unit]


#: Anything that names a place on this machine.
ABSOLUTE_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/])"          # drive letter
    r"|(?:\\\\[^\\/\s]+[\\/])"     # UNC
    r"|(?:/(?:home|usr|opt|var|etc|mnt|tmp)(?:/))"  # POSIX root
)

#: Credential-shaped content that must never be persisted.
CREDENTIAL = re.compile(r"(api[_-]?key|authorization|bearer\s+\S|sk-[A-Za-z0-9]{6,})",
                        re.IGNORECASE)


def a_session(**overrides) -> AgentSession:
    return AgentSession.model_validate(
        {"session_id": "session-abcdef123456", "state": "new", **overrides})


def assessment_with(*, minimum=0.2752, mean=1.4295, status="valid_preliminary",
                    **extra) -> dict:
    safety = {"minimum": minimum, "mean": mean, "p95": 2.0, "maximum": 2.0}
    safety.update(extra.pop("safety", {}))
    return {
        "assessment_status": status,
        "qa_passed": status == "valid_preliminary",
        "hydro_run_id": "hydro-run-abcdef123456",
        "factors": {
            "rainfall": {"total_rain_mm": 79.288, "peak_hourly_rain_mm": 6.9361,
                         "source": "ERA5 virtual stations"},
            "safety_factor": safety,
            "instability_extent_duration": {"max_unstable_fraction": 0.150457,
                                            "max_unstable_duration_h": 3.25},
            "hydraulic_pore_pressure": {"max_flow_depth_m": 0.0225,
                                        "max_velocity_ms": 0.345,
                                        "max_pore_water_pressure_kpa": 12.5},
        },
        "model_context": {"parameter_preset": "medium",
                          "mechanics_model": "reference_equation_9",
                          "swmm_dynamic_hydrograph_used": "True",
                          "subcatchment_count": 6},
        "interpretations": [],
        "warnings": [] if status == "valid_preliminary" else ["质量门槛未通过"],
        "limitations": ["本结果为区域尺度、降雨事件驱动的初步筛查与诊断。"],
        "evidence_sources": [],
    }


class TestExplanationsUseTheSessionEvidence:
    """Plan R4: the answer must quote the current result, not a definition."""

    def test_the_current_minimum_and_mean_appear(self):
        from slope_evaluation_agent.evaluation import explain
        from slope_evaluation_agent.intent import ParsedIntent

        text = explain(a_session(), ParsedIntent(topic="为什么最小安全系数这么低"),
                       "为什么最小安全系数这么低",
                       assessment_with())
        assert "0.275" in text, "回答没有引用当前会话的最小安全系数"
        assert "1.429" in text or "1.43" in text, "回答没有引用当前会话的均值"

    def test_a_missing_mean_does_not_crash(self):
        from slope_evaluation_agent.evaluation import explain
        from slope_evaluation_agent.intent import ParsedIntent

        text = explain(a_session(), ParsedIntent(topic="安全系数"),
                       "安全系数", assessment_with(mean=None))
        assert text
        assert "None" not in text

    def test_a_missing_minimum_does_not_crash(self):
        from slope_evaluation_agent.evaluation import explain
        from slope_evaluation_agent.intent import ParsedIntent

        text = explain(a_session(), ParsedIntent(topic="安全系数"),
                       "安全系数", assessment_with(minimum=None))
        assert text and "None" not in text

    def test_a_diagnostic_result_is_labelled_as_such(self):
        from slope_evaluation_agent.evaluation import explain
        from slope_evaluation_agent.intent import ParsedIntent

        text = explain(a_session(), ParsedIntent(topic="安全系数"), "安全系数",
                       assessment_with(status="diagnostic_only"))
        assert "诊断" in text
        assert "结论" in text or "不能" in text

    def test_without_a_result_only_terms_are_explained(self):
        from slope_evaluation_agent.evaluation import explain
        from slope_evaluation_agent.intent import ParsedIntent

        text = explain(a_session(), ParsedIntent(topic="安全系数"), "安全系数", None)
        assert "0.275" not in text and "1.429" not in text
        assert "计算" in text or "结果" in text

    def test_a_failed_run_explains_the_state_not_a_result(self):
        from slope_evaluation_agent.evaluation import explain
        from slope_evaluation_agent.intent import ParsedIntent

        text = explain(a_session(hydro_run_status="failed"),
                       ParsedIntent(topic="安全系数"), "安全系数", None)
        assert "失败" in text or "没有结果" in text or "计算" in text

    def test_no_causal_claim_about_the_why_question(self):
        from slope_evaluation_agent.evaluation import explain
        from slope_evaluation_agent.intent import ParsedIntent

        text = explain(a_session(), ParsedIntent(topic="为什么最小安全系数这么低"),
                       "为什么最小安全系数这么低", assessment_with())
        for phrase in ("导致", "因为降雨所以", "证明了", "必然"):
            assert phrase not in text, phrase

    def test_the_answer_states_the_scope(self):
        from slope_evaluation_agent.evaluation import explain
        from slope_evaluation_agent.intent import ParsedIntent

        text = explain(a_session(), ParsedIntent(topic="安全系数"), "安全系数",
                       assessment_with())
        assert "初步" in text or "筛查" in text or "诊断" in text


class TestTheHandlerLoadsTheEvidence:
    @pytest.mark.asyncio
    async def test_a_follow_up_quotes_the_finished_run(self, build_agent, calls,
                                                       monkeypatch):
        from helpers import PRELUDE, converse, prepared_agent

        agent, session = await prepared_agent(build_agent, calls, run_status="succeeded")
        replies = await converse(
            agent, session, PRELUDE + ["开始运行", "结果怎么样", "为什么最小安全系数这么低"])
        assert "0.275" in replies[-1].text, "追问没有引用当前会话的数值"

    @pytest.mark.asyncio
    async def test_two_sessions_do_not_share_evidence(self, build_agent, calls):
        """Evidence is looked up per session, so a second one starts empty."""
        from helpers import PRELUDE, converse, prepared_agent

        agent, first = await prepared_agent(build_agent, calls, run_status="succeeded")
        second = agent.new_session()
        async with agent:
            for text in PRELUDE + ["开始运行", "结果怎么样"]:
                await agent.handle(first, text)
            follow_up = await agent.handle(second, "为什么最小安全系数这么低")
        assert "0.275" not in follow_up.text, "第二个会话引用了别的会话的结果"
        assert "还没有" in follow_up.text or "先完成" in follow_up.text


class TestSessionHoldsNoMachinePath:
    def test_save_redacts_a_path_and_credential_in_user_text(self, tmp_path):
        from slope_evaluation_agent.config import AgentSettings
        from slope_evaluation_agent.session_store import SessionStore

        settings = AgentSettings.model_validate({"sessions": {"root": str(tmp_path)}})
        store = SessionStore(settings.sessions)
        session = store.create()
        session.record_turn(
            "user", "读取 D:\\secret\\input.tif，Authorization: Bearer sk-test-secret",
            maximum_turns=20, maximum_chars=2000)
        store.save(session)
        raw = store.session_path(session.session_id).read_text(encoding="utf-8")
        assert "D:\\secret" not in raw
        assert "sk-test-secret" not in raw
        assert "Authorization" not in raw

    @pytest.mark.asyncio
    async def test_a_real_flow_writes_no_absolute_path(self, build_agent, calls):
        """Plan R5: scan the session file after a real workflow."""
        from helpers import PRELUDE, converse

        agent = build_agent()
        session = agent.new_session()
        await converse(agent, session, PRELUDE)
        raw = agent.store.session_path(session.session_id).read_text(encoding="utf-8")
        found = ABSOLUTE_PATH.findall(raw)
        assert not found, f"会话文件包含绝对路径：{found[:3]}"

    @pytest.mark.asyncio
    async def test_a_real_flow_writes_no_credential(self, build_agent, calls):
        from helpers import PRELUDE, converse

        agent = build_agent()
        session = agent.new_session()
        await converse(agent, session, PRELUDE)
        raw = agent.store.session_path(session.session_id).read_text(encoding="utf-8")
        assert not CREDENTIAL.search(raw)

    @pytest.mark.asyncio
    async def test_a_real_flow_writes_no_traceback(self, build_agent, calls):
        from helpers import PRELUDE, converse

        agent = build_agent()
        session = agent.new_session()
        await converse(agent, session, PRELUDE)
        raw = agent.store.session_path(session.session_id).read_text(encoding="utf-8")
        assert "Traceback" not in raw
        assert 'File "' not in raw

    def test_the_schema_has_no_path_shaped_field(self):
        """A field that holds a directory cannot be persisted by accident."""
        fields = AgentSession.model_fields
        assert "model_input_package_ref" not in fields, (
            "会话模型仍保留可能存放本机路径的字段")

    def test_ids_remain_usable_after_minimisation(self):
        """Minimisation must not remove the ids later steps need."""
        fields = AgentSession.model_fields
        for needed in ("review_id", "rainfall_package_id", "model_input_package_id",
                       "hydro_input_id", "hydro_run_id"):
            assert needed in fields, needed

    def test_a_run_plan_keeps_no_machine_path(self):
        """The service's plan names its engine directory; the session must not.

        The stub carries ``engine.directory`` exactly because the real service
        does. A stub that omitted it let a session persist an absolute path
        while every stub-driven scan stayed green — so the projection, not the
        scan's absence, has to be what removes it.
        """
        from stubs import plan_payload

        from slope_evaluation_agent.models import minimise_plan

        raw = plan_payload()
        assert ABSOLUTE_PATH.search(json.dumps(raw, ensure_ascii=False)), (
            "桩的计划载荷里没有路径，这条断言就失去意义")

        minimised = minimise_plan(raw)
        assert not ABSOLUTE_PATH.search(json.dumps(minimised, ensure_ascii=False))
        # What the session still needs survives the projection.
        assert minimised["engine"]["ok"] is True
        assert minimised["input"]["input_id"] == "hydro-input-abcdef123456"
        assert len(minimised["stages"]) == 11
        assert minimised["parameter_preset"] == "medium"


class TestOldSessionsMigrate:
    def test_an_old_session_with_a_path_migrates(self, tmp_path):
        from slope_evaluation_agent.config import AgentSettings
        from slope_evaluation_agent.session_store import SessionStore

        settings = AgentSettings.model_validate({"sessions": {"root": str(tmp_path)}})
        store = SessionStore(settings.sessions)
        session = store.create()
        path = store.session_path(session.session_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = 0
        payload["model_input_package_ref"] = "D:" + chr(92) + "runs" + chr(92) + "pkg"
        payload["unknown_field"] = {"nested": ["D:" + chr(92) + "somewhere"]}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        loaded = store.load(session.session_id)
        assert loaded.session_id == session.session_id
        rewritten = path.read_text(encoding="utf-8")
        assert not ABSOLUTE_PATH.search(rewritten), "迁移后仍保留绝对路径"

    def test_a_corrupt_old_session_does_not_crash(self, tmp_path):
        from slope_evaluation_agent.config import AgentSettings
        from slope_evaluation_agent.session_store import SessionError, SessionStore

        settings = AgentSettings.model_validate({"sessions": {"root": str(tmp_path)}})
        store = SessionStore(settings.sessions)
        session = store.create()
        store.session_path(session.session_id).write_text("{ broken", encoding="utf-8")
        with pytest.raises(SessionError):
            store.load(session.session_id)


class TestDisplayDoesNotLeakPaths:
    @pytest.mark.asyncio
    async def test_status_output_has_no_absolute_path(self, build_agent, calls):
        from helpers import PRELUDE, converse
        from slope_evaluation_agent.cli import describe_session

        agent = build_agent()
        session = agent.new_session()
        await converse(agent, session, PRELUDE)
        assert not ABSOLUTE_PATH.search(describe_session(agent.store, session.session_id))

    @pytest.mark.asyncio
    async def test_inspect_output_has_no_absolute_path(self, build_agent, calls):
        from helpers import PRELUDE, converse
        from slope_evaluation_agent.cli import dump_session

        agent = build_agent()
        session = agent.new_session()
        await converse(agent, session, PRELUDE)
        payload = dump_session(agent.store, session.session_id, event_limit=50)
        assert not ABSOLUTE_PATH.search(json.dumps(payload, ensure_ascii=False))
