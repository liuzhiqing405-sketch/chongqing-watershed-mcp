"""Phase 5 and 6 acceptance: the two workflows end to end.

The agent is driven through real turns against in-process servers that present
the real tool signatures, so the gates, the state transitions, the id plumbing
and the failure branches are all exercised as a conversation would exercise
them.
"""

from __future__ import annotations

import json

import pytest

from slope_evaluation_agent.agent import SlopeEvaluationAgent
from slope_evaluation_agent.config import AgentSettings, SessionSettings
from slope_evaluation_agent.intent import IntentParser
from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.session_store import SessionStore
from slope_evaluation_agent.state_machine import WorkflowState

from stubs import (
    PNG_BYTES,
    StubHydroClient,
    StubSpatialClient,
    candidates_payload,
    envelope,
    hydro_server,
    import_payload,
    model_input_payload,
    plan_payload,
    rainfall_payload,
    review_payload,
    run_result_payload,
    run_status_payload,
    spatial_server,
    validation_payload,
    visualization_payload,
)


REVIEW_ID = "basin-abcdef123456"
INPUT_ID = "hydro-input-abcdef123456"
RUN_ID = "hydro-run-abcdef123456"


from helpers import (
    INPUT_ID,
    PRELUDE,
    DRAW_AREA,
    step,
    REVIEW_ID,
    RUN_ID,
    Calls,
    converse,
    hydro_handlers,
    hydro_server,
    prepared_agent,
    run,
    spatial_handlers,
    spatial_server,
)


class TestSpatialWorkflow:
    @pytest.mark.asyncio
    async def test_naming_a_river_loads_a_map_and_asks_for_a_click(self, build_agent, calls):
        agent = build_agent()
        session = agent.new_session()
        reply = await run(agent, session, "分析大宁河")
        assert session.state == WorkflowState.OUTLET_CANDIDATES_READY.value
        assert session.river_name == "大宁河"
        assert "绘制闭合评价边界" in reply.text
        assert "候选" not in reply.text
        assert session.outlet_candidates == []
        assert reply.outlet_map and reply.images
        assert calls.spatial == ["render_outlet_candidates"]

    @pytest.mark.asyncio
    async def test_coordinates_locate_a_map_and_wait_for_drawing(self, build_agent, calls):
        agent = build_agent()
        session = agent.new_session()
        reply = await run(agent, session, "东经 109.5、北纬 31.4")
        assert session.state == WorkflowState.OUTLET_CANDIDATES_READY.value
        assert session.review_id is None
        assert session.watershed_confirmed is False
        assert "绘制闭合评价边界" in reply.text
        assert calls.spatial == ["render_outlet_candidates"]

    @pytest.mark.asyncio
    async def test_the_watershed_figure_is_returned_as_an_image(self, build_agent):
        import base64

        agent = build_agent()
        session = agent.new_session()
        reply = await run(agent, session, "东经 109.5、北纬 31.4")
        assert reply.images and reply.images[0]["mime_type"] == "image/png"
        assert base64.b64decode(reply.images[0]["data"]) == PNG_BYTES

    @pytest.mark.asyncio
    async def test_choosing_a_point_by_coordinates(self, build_agent, calls):
        agent = build_agent()
        session = agent.new_session()
        await converse(agent, session, ["分析大宁河", "经度 109.5、纬度 31.4"])
        assert session.state == WorkflowState.OUTLET_CANDIDATES_READY.value
        # The boundary figure is rendered as part of the review step.
        assert calls.spatial[:2] == ["render_outlet_candidates", "render_outlet_candidates"]
        assert "delineate_from_outlet" not in calls.spatial

    @pytest.mark.asyncio
    async def test_an_out_of_range_candidate_index_is_refused(self, build_agent, calls):
        agent = build_agent()
        session = agent.new_session()
        replies = await converse(agent, session, ["分析大宁河", "第 9 个"])
        assert "绘制闭合评价边界" in replies[-1].text
        assert calls.spatial == ["render_outlet_candidates"]

    @pytest.mark.asyncio
    async def test_rainfall_cannot_be_built_before_confirmation(self, build_agent, calls):
        """Plan §6: an unconfirmed boundary blocks the rainfall package."""
        agent = build_agent()
        session = agent.new_session()
        replies = await converse(agent, session, ["东经 109.5、北纬 31.4", DRAW_AREA, "2024-07-30 全天"])
        reply = replies[-1]
        assert reply.ok is False
        assert "确认" in reply.text
        assert "build_virtual_rainfall" not in calls.spatial

    @pytest.mark.asyncio
    async def test_confirming_runs_the_whole_intake_when_the_period_is_known(
            self, build_agent, calls):
        agent = build_agent()
        session = agent.new_session()
        await converse(agent, session, ["东经 109.5、北纬 31.4", DRAW_AREA, "2024-07-30 全天", "确认该边界"])
        assert session.watershed_confirmed is True
        assert session.rainfall_package_id == "rain-abcdef123456"
        assert session.model_input_package_id == "package-abcdef123456"
        assert session.hydro_input_id == INPUT_ID
        assert session.hydro_input_validated is True
        assert session.hydro_planned is True
        assert session.state == WorkflowState.HYDRO_PLANNED.value
        assert calls.spatial[-3:] == ["build_virtual_rainfall", "prepare_model_inputs",
                                      "confirm_watershed"] or "prepare_model_inputs" in calls.spatial
        assert "import_hydro_input_package" in calls.hydro

    @pytest.mark.asyncio
    async def test_confirming_without_a_period_stops_and_asks(self, build_agent, calls):
        agent = build_agent()
        session = agent.new_session()
        replies = await converse(agent, session, ["东经 109.5、北纬 31.4", DRAW_AREA, "确认该边界"])
        reply = replies[-1]
        assert session.watershed_confirmed is True
        assert session.state == WorkflowState.WATERSHED_CONFIRMED.value
        assert "降雨" in reply.text
        assert "build_virtual_rainfall" not in calls.spatial

    @pytest.mark.asyncio
    async def test_rejecting_keeps_the_record_and_allows_reselection(self, build_agent, calls):
        agent = build_agent()
        session = agent.new_session()
        replies = await converse(agent, session,
                                 ["东经 109.5、北纬 31.4", DRAW_AREA, "拒绝，重新选", "分析大宁河"])
        # The replies share the session object, so assert on the text and on the
        # facts recorded by each step rather than on a mutated state field.
        assert '拒绝' in replies[2].text and '修改' in replies[2].text
        assert session.watershed_confirmed is False
        assert session.state == WorkflowState.OUTLET_CANDIDATES_READY.value
        assert calls.spatial.count("confirm_watershed") == 1
        assert calls.spatial[-1] == "render_outlet_candidates"

    @pytest.mark.asyncio
    async def test_a_failed_spatial_gate_blocks_confirmation(self, build_agent, calls):
        server = spatial_server(spatial_handlers(
            calls, create_drawn_study_area=lambda: dict(review_payload(qa_passed=False),selection_mode="drawn_study_area",outlet={})))
        agent = build_agent(spatial=server)
        session = agent.new_session()
        replies = await converse(agent, session, ["东经 109.5、北纬 31.4", DRAW_AREA, "确认"])
        assert "不能确认或计算" in replies[1].text
        assert replies[2].ok is False
        assert session.watershed_confirmed is False

    @pytest.mark.asyncio
    async def test_historical_maximum_wording_is_accurate(self, build_agent):
        agent = build_agent()
        session = agent.new_session()
        replies = await converse(agent, session,
                                 ["东经 109.5、北纬 31.4", DRAW_AREA, "确认该边界", "用历史最大日"])
        reply = replies[-1]
        assert "样本最大" in reply.text
        assert reply.facts['rainfall_package']['selection_mode'] == 'historical_max'
        assert 'maximum complete calendar-day' in reply.facts['rainfall_package']['selection_rule']
        assert session.rainfall_request["mode"] == "historical_max"

    @pytest.mark.asyncio
    async def test_the_rainfall_package_mentions_virtual_stations(self, build_agent):
        agent = build_agent()
        session = agent.new_session()
        replies = await converse(agent, session,
                                 ["东经 109.5、北纬 31.4", DRAW_AREA, "确认该边界", "2024-07-30 全天"])
        assert replies[-1].facts['rainfall_package']['stations']
        assert '布站依据' not in replies[-1].text
        assert len(replies[-1].text) < 250

    @pytest.mark.asyncio
    async def test_ids_come_from_the_server_not_from_the_agent(self, build_agent, calls):
        agent = build_agent()
        session = agent.new_session()
        await converse(agent, session,
                       ["东经 109.5、北纬 31.4", DRAW_AREA, "确认该边界", "2024-07-30 全天"])
        # Every id is exactly what the stub returned, never a constructed one.
        assert session.review_id == REVIEW_ID
        assert session.rainfall_package_id == "rain-abcdef123456"
        assert session.model_input_package_id == "package-abcdef123456"


# -------------------------------------------------------------------- hydro flow


PRELUDE = ["东经 109.5、北纬 31.4", DRAW_AREA, "确认该边界", "2024-07-30 全天"]


async def prepared_agent(build_agent, calls, *, run_status: str | None = None):
    """An agent and a session whose build agent reports the given run status."""
    overrides = {}
    if run_status is not None:
        overrides = {
            "get_hydro_run_status": lambda: envelope(
                data=run_status_payload(status=run_status)),
            "get_hydro_run_result": lambda: envelope(
                ok=run_status == "succeeded", status=run_status,
                data=run_result_payload(status=run_status,
                                        qa_passed=run_status == "succeeded")),
        }
    agent = build_agent(hydro=hydro_server(hydro_handlers(calls, **overrides)))
    session = agent.new_session()
    return agent, session


class TestHydroWorkflow:
    @pytest.mark.asyncio
    async def test_the_run_cannot_start_before_a_plan_is_shown(self, build_agent, calls):
        """Plan section 6: start_hydro_run is unreachable without a plan."""
        agent, session = await prepared_agent(build_agent, calls)
        async with agent:
            for text in PRELUDE:
                await step(agent, session, text)
            # Simulate a session where the plan was never shown.
            session.hydro_planned = False
            agent.store.save(session)
            reply = await agent.handle(session, "开始运行")
        assert reply.ok is False
        assert "计划" in reply.text
        assert "start_hydro_run" not in calls.hydro

    @pytest.mark.asyncio
    async def test_plan_is_shown_before_anything_runs(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls)
        replies = await converse(agent, session, PRELUDE)
        assert session.hydro_planned is True
        assert session.last_plan is not None
        assert "开始运行" in replies[-1].text
        assert session.parameter_preset == 'medium'
        assert '默认参数' in replies[-1].text

    @pytest.mark.asyncio
    async def test_starting_returns_immediately(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls)
        replies = await converse(agent, session, PRELUDE + ["开始运行"])
        assert session.hydro_run_id == RUN_ID
        assert session.state == WorkflowState.HYDRO_RUNNING.value
        assert RUN_ID not in replies[-1].text  # internal identifiers stay in session state
        assert "后台" in replies[-1].text

    @pytest.mark.asyncio
    async def test_status_is_queried_not_remembered(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls)
        await converse(agent, session, PRELUDE + ["开始运行", "状态", "进度到哪了"])
        assert calls.hydro.count("get_hydro_run_status") >= 2
        assert session.hydro_run_id == RUN_ID

    @pytest.mark.asyncio
    async def test_polling_does_not_start_a_second_run(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls)
        await converse(agent, session, PRELUDE + ["开始运行", "状态", "状态", "状态"])
        assert calls.hydro.count("start_hydro_run") == 1

    @pytest.mark.asyncio
    async def test_a_succeeded_run_reaches_assessment_ready(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls, run_status="succeeded")
        replies = await converse(agent, session, PRELUDE + ["开始运行", "结果怎么样"])
        reply = replies[-1]
        assert session.state == WorkflowState.ASSESSMENT_READY.value
        assert session.qa_passed is True
        assert reply.ok is True
        assert "初步评价" in reply.text
        assert "0.275" in reply.text  # the minimum, verbatim from the result

    @pytest.mark.asyncio
    async def test_result_while_still_running_reports_progress(self, build_agent, calls):
        """Plan scenario A: result requests query first, then decide."""
        agent, session = await prepared_agent(build_agent, calls)
        replies = await converse(agent, session, PRELUDE + ["开始运行", "结果怎么样"])
        assert replies[-1].ok is True
        assert session.state == WorkflowState.HYDRO_RUNNING.value

    @pytest.mark.asyncio
    async def test_a_qa_failed_run_is_diagnostic_only(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls, run_status="qa_failed")
        replies = await converse(agent, session, PRELUDE + ["开始运行", "结果怎么样"])
        reply = replies[-1]
        assert session.state == WorkflowState.HYDRO_QA_FAILED.value
        assert reply.ok is False
        assert "诊断" in reply.text
        assert "stability" in reply.text  # the failing gate is named

    @pytest.mark.asyncio
    async def test_assessment_is_never_reached_from_a_failed_run(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls, run_status="qa_failed")
        await converse(agent, session,
                       PRELUDE + ["开始运行", "结果怎么样", "结果怎么样", "结果怎么样"])
        assert session.state != WorkflowState.ASSESSMENT_READY.value
        assert agent.store.load_assessment(session.session_id) is None

    @pytest.mark.asyncio
    async def test_a_failed_run_reports_no_stability_verdict(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls, run_status="failed")
        replies = await converse(agent, session, PRELUDE + ["开始运行", "结果怎么样"])
        assert replies[-1].ok is False
        assert session.state == WorkflowState.HYDRO_FAILED.value

    @pytest.mark.asyncio
    async def test_cancel_only_affects_the_session_task(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls)
        replies = await converse(agent, session, PRELUDE + ["开始运行", "取消"])
        assert session.state == WorkflowState.HYDRO_CANCELLED.value
        assert RUN_ID in replies[-1].text
        assert calls.hydro.count("cancel_hydro_run") == 1

    @pytest.mark.asyncio
    async def test_cancel_is_refused_before_a_run_exists(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls)
        replies = await converse(agent, session, PRELUDE + ["取消"])
        assert replies[-1].ok is False
        assert "cancel_hydro_run" not in calls.hydro

    @pytest.mark.asyncio
    async def test_a_failed_validation_stops_the_chain(self, build_agent, calls):
        server = hydro_server(hydro_handlers(
            calls, validate_hydro_input=lambda: envelope(
                ok=False, status="invalid", summary="输入校验未通过",
                data=validation_payload(ok=False))))
        agent = build_agent(hydro=server)
        session = agent.new_session()
        replies = await converse(agent, session, PRELUDE)
        assert session.hydro_input_validated is False
        assert session.state != WorkflowState.HYDRO_PLANNED.value
        assert "plan_hydro_run" not in calls.hydro
        assert "未通过" in replies[-1].text

    @pytest.mark.asyncio
    async def test_visualization_is_created_and_returned(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls, run_status="succeeded")
        replies = await converse(agent, session,
                                 PRELUDE + ["开始运行", "看最小安全系数的图"])
        reply = replies[-1]
        assert reply.images and reply.images[0]["mime_type"] == "image/png"
        assert session.visualizations
        assert "create_hydro_visualization" in calls.hydro

    @pytest.mark.asyncio
    async def test_asking_for_the_catalog_lists_options(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls, run_status="succeeded")
        replies = await converse(agent, session, PRELUDE + ["开始运行", "看有哪些图"])
        assert "图件类型" in replies[-1].text or "safety" in replies[-1].text


class TestSessionContinuity:
    @pytest.mark.asyncio
    async def test_a_stale_status_is_requeried_not_trusted(self, build_agent, calls):
        """A resumed session must ask the service, not believe what it stored."""
        agent, session = await prepared_agent(build_agent, calls, run_status="succeeded")
        async with agent:
            for text in PRELUDE + ["开始运行"]:
                await step(agent, session, text)
            # Simulate a session file written while the job was still going.
            session.hydro_run_status = "running"
            session.state = WorkflowState.HYDRO_RUNNING.value
            agent.store.save(session)
            reply = await agent.reconcile(session)
        assert reply is not None
        assert session.state == WorkflowState.HYDRO_SUCCEEDED.value
        assert session.hydro_run_id == RUN_ID
        assert calls.hydro.count("get_hydro_run_status") >= 1

    @pytest.mark.asyncio
    async def test_reconcile_does_not_create_a_duplicate_task(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls, run_status="succeeded")
        async with agent:
            for text in PRELUDE + ["开始运行"]:
                await step(agent, session, text)
            started = calls.hydro.count("start_hydro_run")
            session.hydro_run_status = "running"
            session.state = WorkflowState.HYDRO_RUNNING.value
            await agent.reconcile(session)
        assert calls.hydro.count("start_hydro_run") == started == 1

    @pytest.mark.asyncio
    async def test_reconcile_reports_nothing_while_still_running(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls)
        async with agent:
            for text in PRELUDE + ["开始运行"]:
                await step(agent, session, text)
            assert await agent.reconcile(session) is None
        assert session.state == WorkflowState.HYDRO_RUNNING.value

    @pytest.mark.asyncio
    async def test_identifiers_survive_a_reload_from_disk(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls)
        await converse(agent, session, PRELUDE)
        reloaded = agent.store.load(session.session_id)
        assert reloaded.review_id == REVIEW_ID
        assert reloaded.rainfall_package_id == "rain-abcdef123456"
        assert reloaded.hydro_input_id == INPUT_ID
        assert reloaded.hydro_input_validated is True
        assert reloaded.hydro_planned is True
        assert reloaded.watershed_confirmed is True
        assert reloaded.state == WorkflowState.HYDRO_PLANNED.value

    @pytest.mark.asyncio
    async def test_confirm_does_not_misfire_without_a_pending_watershed(self, build_agent):
        agent = build_agent()
        session = agent.new_session()
        reply = await run(agent, session, "确认")
        assert reply.ok is False
        assert session.watershed_confirmed is False

    @pytest.mark.asyncio
    async def test_confirm_does_not_misfire_on_an_already_decided_watershed(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls)
        replies = await converse(agent, session, PRELUDE + ["确认"])
        assert session.watershed_confirmed is True
        assert replies[-1].ok is False
        assert calls.spatial.count("confirm_watershed") == 1


class TestEventLogIntegrity:
    @pytest.mark.asyncio
    async def test_every_tool_call_is_logged_around_the_call(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls)
        await converse(agent, session, PRELUDE)
        events = agent.store.read_events(session.session_id)
        kinds = [item["event"] for item in events]
        assert kinds.count("tool_call") == kinds.count("tool_result")
        assert "state_transition" in kinds
        assert "intent" in kinds

    @pytest.mark.asyncio
    async def test_the_log_never_contains_a_credential(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls)
        await converse(agent, session, PRELUDE)
        raw = agent.store.events_path(session.session_id).read_text(encoding="utf-8")
        for marker in ("api_key", "sk-", "Bearer "):
            assert marker not in raw

    @pytest.mark.asyncio
    async def test_intents_are_recorded_with_confidence(self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls)
        await converse(agent, session, PRELUDE)
        intents = [item for item in agent.store.read_events(session.session_id)
                   if item["event"] == "intent"]
        assert intents and all("confidence" in item for item in intents)


class TestTheRunLifetimeIsStatedUpFront:
    """The transport decides a job's lifetime, so the wording must follow it.

    A user who starts a long job and then closes the CLI should have been told
    which of the two things will happen *before* choosing to leave, not only in
    the exit notice they have already stopped reading.
    """

    @pytest.mark.asyncio
    async def test_over_stdio_starting_a_run_says_it_ends_with_the_process(
            self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls)
        replies = await converse(agent, session, PRELUDE + ["开始运行"])
        text = replies[-1].text
        assert '关闭服务会终止' in text
        assert '保持服务运行' in text
        assert len(text) < 150

    @pytest.mark.asyncio
    async def test_over_a_persistent_service_no_such_warning_is_added(
            self, build_agent, calls):
        agent, session = await prepared_agent(build_agent, calls)
        hydro = agent.settings.mcp.hydro.model_copy(
            update={"transport": "streamable-http",
                    "url": "http://127.0.0.1:8766/mcp"})
        agent.settings = agent.settings.model_copy(
            update={"mcp": agent.settings.mcp.model_copy(update={"hydro": hydro})})
        replies = await converse(agent, session, PRELUDE + ["开始运行"])
        text = replies[-1].text
        assert "终止这个任务" not in text
        assert "stdio" not in text
        assert "后台" in text, "提交任务本身的话术不变"
