"""Regression cases from the eight-scenario diagnosis, isolated from user history."""
from copy import deepcopy

import pytest

from helpers import PRELUDE, RUN_ID, step
from slope_evaluation_agent.dialogue_router import DialogueDecision, RouteKind
from slope_evaluation_agent.intent import Intent, ParsedIntent
from slope_evaluation_agent.input_binding import input_signature
from slope_evaluation_agent.state_machine import Operation, WorkflowError, require_operation
from slope_evaluation_agent.task_workspace import revision_rows
from test_artifact_downloads import an_agent, CSV_ID
from stubs import StubLLM, envelope, hydro_server, run_status_payload, run_result_payload
from helpers import hydro_handlers
from slope_evaluation_agent.task_workspace import task_snapshot
from slope_evaluation_agent.evaluation import build_assessment


@pytest.mark.asyncio
@pytest.mark.parametrize("state,skipped", [
    ("watershed_confirmed", []),
    ("rainfall_ready", ["build_virtual_rainfall"]),
    ("model_inputs_ready", ["build_virtual_rainfall", "prepare_model_inputs"]),
    ("hydro_input_imported", ["build_virtual_rainfall", "prepare_model_inputs", "import_hydro_input_package"]),
    ("hydro_input_validated", ["build_virtual_rainfall", "prepare_model_inputs", "import_hydro_input_package", "validate_hydro_input"]),
])
async def test_goal_continuation_reuses_completed_stages_and_stops_at_plan(build_agent, calls, state, skipped):
    agent = build_agent()
    session = agent.new_session()
    session.state = state
    session.review_id = "basin-abcdef123456"
    session.watershed_confirmed = True
    session.watershed_qa_passed = True
    session.rainfall_request = {"mode": "explicit_period", "date": "2009-08-08"}
    if "build_virtual_rainfall" in skipped:
        session.rainfall_package_id = "rain-abcdef123456"
    if "prepare_model_inputs" in skipped:
        session.model_input_package_id = "package-abcdef123456"
    if "import_hydro_input_package" in skipped:
        session.hydro_input_id = "hydro-input-abcdef123456"
    if "validate_hydro_input" in skipped:
        session.hydro_input_validated = True
    agent.router.llm = StubLLM({"route": "workflow_action", "intent": "start_hydro_run", "confidence": .99})
    async with agent:
        reply = await agent.handle(session, "沿用已确认范围，完成后续建模并生成计划，按我批准的计划完成运行")
    assert reply.ok, reply.text
    assert session.hydro_planned and session.state == "hydro_planned"
    assert "start_hydro_run" not in calls.hydro
    assert not set(skipped) & set(calls.hydro + calls.spatial)


@pytest.mark.asyncio
async def test_bare_continue_does_not_approve_an_existing_plan(build_agent, calls):
    agent = build_agent()
    session = agent.new_session()
    async with agent:
        for text in PRELUDE:
            await step(agent, session, text)
        reply = await agent.handle(session, "继续")
    assert session.hydro_planned and not session.hydro_run_id
    assert "start_hydro_run" not in calls.hydro


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["请先取消当前任务，然后我再修改降雨", "停止当前运行，之后怎么恢复？"])
async def test_explicit_compound_cancellation_reaches_the_real_cancel_handler(build_agent, calls, message):
    agent = build_agent()
    session = agent.new_session()
    session.state, session.hydro_run_id, session.hydro_run_status = "hydro_running", RUN_ID, "running"
    async with agent:
        reply = await agent.handle(session, message)
    assert reply.ok and session.state == "hydro_cancelled"
    assert calls.hydro == ["cancel_hydro_run"]


@pytest.mark.asyncio
async def test_future_panel_approval_is_not_treated_as_cancel_permission(build_agent, calls):
    agent = build_agent()
    session = agent.new_session()
    session.state, session.hydro_run_id, session.hydro_run_status = "hydro_running", RUN_ID, "running"
    async with agent:
        reply = await agent.handle(session, "取消当前运行，我会在面板批准")
    assert session.state == "hydro_running" and session.pending_question
    assert reply.facts["pending_user_control"] == "cancel_run" and not calls.hydro


@pytest.mark.asyncio
async def test_cancel_then_edit_rainfall_prepares_a_new_plan_without_submitting(build_agent, calls):
    agent = build_agent()
    session = agent.new_session()
    async with agent:
        for text in PRELUDE:
            await step(agent, session, text)
        await agent.handle(session, "开始运行")
        reply = await agent.handle(session, "先取消当前运行，确认停止后把降雨改为2009-08-09，展示计划，暂不启动")
    assert reply.ok, reply.text
    assert session.rainfall_request["date"] == "2009-08-09"
    assert session.state == "hydro_planned" and not session.hydro_run_id
    assert calls.hydro.count("start_hydro_run") == calls.hydro.count("cancel_hydro_run") == 1


@pytest.mark.asyncio
async def test_start_is_deferred_until_announced_panel_approval(build_agent, calls):
    agent = build_agent()
    session = agent.new_session()
    async with agent:
        for text in PRELUDE:
            await step(agent, session, text)
        agent.router.llm = StubLLM({"route": "workflow_action", "intent": "start_hydro_run", "confidence": .99})
        await agent.handle(session, "启动当前运行，我会在面板批准")
    assert session.hydro_planned and not session.hydro_run_id
    assert "start_hydro_run" not in calls.hydro


@pytest.mark.asyncio
async def test_merely_withholding_start_does_not_prepare_or_modify_inputs(build_agent, calls):
    agent = build_agent()
    session = agent.new_session()
    session.state, session.review_id, session.watershed_confirmed = "watershed_confirmed", "basin-abcdef123456", True
    session.rainfall_request = {"date": "2009-08-08", "mode": "explicit_period"}
    before = deepcopy(task_snapshot(session))
    agent.router.llm = StubLLM({"route": "workflow_action", "intent": "start_hydro_run", "confidence": .99})
    async with agent:
        await agent.handle(session, "暂不启动")
    assert not calls.spatial and not calls.hydro
    after = task_snapshot(session)
    # Routing may record the last understood intent; physical inputs and the
    # approval/plan state must remain untouched.
    before.pop("intent")
    after.pop("intent")
    assert after == before


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["不要取消当前运行", "如果失败就取消当前运行", "‘取消当前运行’是什么意思", "我会在批准后取消当前运行", "取消当前运行，算了先不要取消"])
async def test_negated_hypothetical_and_quoted_cancellation_never_grant_control(build_agent, message):
    agent = build_agent()
    session = agent.new_session()
    agent.router.llm = StubLLM({"route": "knowledge_chat", "intent": "ask_explanation", "confidence": .99})
    decision = await agent.router.decide(message, session)
    assert decision.intent is not Intent.CANCEL_RUN


@pytest.mark.asyncio
async def test_cancel_acknowledgement_does_not_claim_a_still_running_job_has_stopped(build_agent, calls):
    agent = build_agent(hydro=hydro_server(hydro_handlers(calls,
        cancel_hydro_run=envelope(data=run_status_payload(status="running")))))
    session = agent.new_session()
    session.state, session.hydro_run_id, session.hydro_run_status = "hydro_running", RUN_ID, "running"
    async with agent:
        reply = await agent.handle(session, "取消当前运行")
    assert session.state == "hydro_running" and "仍报告" in reply.text


@pytest.mark.asyncio
async def test_saved_scheme_lookup_is_read_only_and_never_substitutes_current_values(build_agent, calls):
    agent = build_agent()
    session = agent.new_session()
    session.state, session.hydro_run_id, session.hydro_run_status = "assessment_ready", RUN_ID, "succeeded"
    session.administrative_area = "区域A"
    session.rainfall_request = {"date": "2009-08-08", "mode": "explicit_period"}
    session.parameter_overrides = {"saturated_cohesion_kpa": 12.0}
    saved = build_assessment(session, run_result_payload())
    agent.store.save_assessment(session.session_id, saved)
    revision = agent.checkpoint_task(session, "方案A")
    session.hydro_run_id = "hydro-run-222222222222"
    session.rainfall_request["date"] = "2009-08-09"
    session.parameter_overrides = {"saturated_cohesion_kpa": 20.0}
    current_evidence = {"current": "must survive"}
    agent.store.save_assessment(session.session_id, current_evidence)
    before = deepcopy(task_snapshot(session))
    agent.router.llm = StubLLM({"route": "knowledge_chat", "intent": "ask_explanation", "confidence": .99})
    async with agent:
        reply = await agent.handle(session, "查询区域A中标记A的已保存方案，报告日期、参数和最小Fs，保持当前任务不变")
    assert reply.ok, reply.text
    assert "2009-08-08" in reply.text and "0.2752" in reply.text and "12.0" in reply.text
    assert "2009-08-09" not in reply.text and revision in reply.text
    assert before == task_snapshot(session)
    assert agent.store.load_assessment(session.session_id) == current_evidence
    assert not calls.hydro and not calls.spatial


@pytest.mark.asyncio
async def test_explanation_and_download_are_both_delivered_without_a_new_run(build_agent, calls):
    agent, session = an_agent(build_agent, calls)
    agent.router.llm = StubLLM({"route": "workflow_action", "intent": "download_artifact", "confidence": .99})
    async with agent:
        reply = await agent.handle(session, "解释当前结果并下载对应数据表")
    assert reply.ok, reply.text
    assert reply.facts["artifact"]["artifact_id"] == CSV_ID
    assert "0.2752" in reply.text
    assert "get_hydro_run_result" in calls.hydro and "start_hydro_run" not in calls.hydro


@pytest.mark.asyncio
async def test_historical_maximum_keeps_the_explicit_archive_end_year(build_agent):
    agent = build_agent()
    session = agent.new_session()
    session.state, session.review_id, session.watershed_confirmed = "watershed_confirmed", "basin-abcdef123456", True
    agent.router.llm = StubLLM({"route": "workflow_action", "intent": "set_rainfall_period", "confidence": .99})
    async with agent:
        reply = await agent.handle(session, "用截至2010年的过去11年资料期样本最大日准备模型")
    assert reply.ok, reply.text
    assert session.rainfall_request["mode"] == "historical_max"
    assert session.rainfall_request["end_year"] == 2010
    assert session.rainfall_request["lookback_years"] == 11


@pytest.mark.asyncio
async def test_unresolved_new_rainfall_does_not_silently_reuse_old_event(build_agent, calls):
    agent = build_agent()
    session = agent.new_session()
    session.rainfall_request = {"date": "2009-08-08", "mode": "explicit_period"}
    agent.router.llm = StubLLM({"route": "clarification", "intent": "set_rainfall_period", "confidence": .62,
        "candidates": [{"field": "date", "value": "2009-08-09", "evidence_text": "2009-08-09"}],
        "ambiguity": "确认要更换降雨日期吗？"})
    async with agent:
        await agent.handle(session, "改为2009-08-09")
    assert session.rainfall_request["date"] == "2009-08-08"
    assert not agent._rainfall_is_specified(session)
    with pytest.raises(WorkflowError, match="尚未澄清"):
        agent._rainfall_request(session, ParsedIntent(intent=Intent.SET_RAINFALL_PERIOD))
    assert not calls.spatial and not calls.hydro


@pytest.mark.asyncio
@pytest.mark.parametrize("route", [RouteKind.WORKFLOW_ACTION, RouteKind.CLARIFICATION])
async def test_rainfall_change_invalidates_legacy_results_without_rainfall_package(build_agent, route):
    agent = build_agent()
    session = agent.new_session()
    session.state = "assessment_ready"
    session.watershed_confirmed = True
    session.hydro_run_id = RUN_ID
    session.hydro_run_status = "succeeded"
    session.rainfall_request = {"mode": "explicit_period", "date": "2009-08-08"}
    agent.store.save_assessment(session.session_id, {"old": True})
    before = session.model_dump(mode="json")
    parsed = ParsedIntent(intent=Intent.SET_RAINFALL_PERIOD, date="2009-08-09")
    recorded = agent._remember_entities(session, parsed)
    # Clarification must also invalidate a committed input, without running it.
    decision = DialogueDecision(route=route, intent=Intent.SET_RAINFALL_PERIOD,
                                confidence=.9, missing_fields=["station_count"] if route is RouteKind.CLARIFICATION else [])
    async with agent:
        await agent._dispatch_decision(session, decision, parsed, recorded, before, "改为2009-08-09")
    assert session.hydro_run_id is None
    assert not agent.store.load_assessment(session.session_id)
    old = next(row for row in revision_rows(agent.store, session.session_id)
               if row["snapshot"]["hydro_run_id"] == RUN_ID)
    assert old["snapshot"]["rainfall_request"]["date"] == "2009-08-08"


@pytest.mark.asyncio
async def test_new_run_remembers_input_binding_and_rejects_silent_parameter_drift(build_agent):
    agent = build_agent()
    session = agent.new_session()
    async with agent:
        for text in PRELUDE:
            await step(agent, session, text)
        assert session.planned_input_signature == input_signature(session)
        await agent.handle(session, "开始运行")
        assert session.run_input_signature == input_signature(session)
    session.state = "hydro_succeeded"
    session.parameter_overrides = {"saturated_cohesion_kpa": 15.0}
    for operation in (Operation.SHOW_RESULT, Operation.DOWNLOAD_ARTIFACT, Operation.SHOW_VISUALIZATION):
        with pytest.raises(WorkflowError, match="输入已改变"):
            require_operation(session, operation)


@pytest.mark.asyncio
async def test_registered_csv_can_be_downloaded_without_first_rendering_a_figure(build_agent, calls):
    agent, session = an_agent(build_agent, calls)
    async with agent:
        reply = await agent._handle_download_artifact(session,
            ParsedIntent(intent=Intent.DOWNLOAD_ARTIFACT), "下载对应的数据表")
        assert reply.facts["artifact"]["artifact_id"] == CSV_ID
        download = await agent.download_artifact(session, CSV_ID)
        assert b"".join([chunk async for chunk in download.chunks])
    assert not session.visualizations
    assert "create_hydro_visualization" not in calls.hydro


@pytest.mark.asyncio
async def test_generic_download_lists_existing_artifacts_instead_of_claiming_absence(build_agent, calls):
    agent, session = an_agent(build_agent, calls)
    async with agent:
        reply = await agent._handle_download_artifact(session,
            ParsedIntent(intent=Intent.DOWNLOAD_ARTIFACT), "下载现有成果")
    assert reply.facts["artifacts"] and reply.kind == "question"
    assert "没有可下载" not in reply.text


@pytest.mark.asyncio
async def test_real_service_public_descriptor_is_accepted_without_weakening_permissions(build_agent, calls):
    from watershed_hydro_mcp.export_service import public_descriptor
    from test_artifact_downloads import artifacts_payload
    raw = deepcopy(artifacts_payload()["artifacts"][1])
    raw.pop("formats")
    raw["internal_path"] = "C:/private/summary.csv"
    public = public_descriptor(raw)
    agent, session = an_agent(build_agent, calls,
        list_hydro_artifacts=envelope(data={"artifacts": [public]}))
    async with agent:
        reply = await agent._handle_download_artifact(session,
            ParsedIntent(intent=Intent.DOWNLOAD_ARTIFACT), "下载现有数据表")
    assert reply.facts["artifact"]["artifact_id"] == CSV_ID
    assert "private" not in str(reply.facts) and "internal_path" not in str(session.process_artifacts)


@pytest.mark.asyncio
async def test_incomplete_simulator_descriptor_still_does_not_grant_download_rights(build_agent, calls):
    from test_artifact_downloads import artifacts_payload
    rows = deepcopy(artifacts_payload()["artifacts"])
    for row in rows:
        row.pop("formats")
    agent, session = an_agent(build_agent, calls,
        list_hydro_artifacts=envelope(data={"artifacts": rows}))
    async with agent:
        reply = await agent._handle_download_artifact(session,
            ParsedIntent(intent=Intent.DOWNLOAD_ARTIFACT), "下载现有数据表")
    assert not session.process_artifacts and not reply.facts.get("artifact")


@pytest.mark.asyncio
async def test_download_never_substitutes_a_previously_shown_figure_of_another_kind(build_agent, calls):
    agent, session = an_agent(build_agent, calls)
    async with agent:
        await agent.handle(session, "看最小安全系数的图")
        reply = await agent._handle_download_artifact(session, ParsedIntent(
            intent=Intent.DOWNLOAD_ARTIFACT, visualization_kind="rainfall_cumulative"), "下载累积降雨图")
    assert "artifact" not in reply.facts


@pytest.mark.asyncio
async def test_direct_download_refuses_a_file_after_silent_input_drift(build_agent, calls):
    from slope_evaluation_agent.artifacts import DownloadError
    agent, session = an_agent(build_agent, calls)
    session.run_input_signature = input_signature(session)
    async with agent:
        await agent.refresh_process_artifacts(session)
        session.parameter_overrides = {"deposit_thickness_m": 1.5}
        with pytest.raises(DownloadError, match="输入已改变"):
            await agent.download_artifact(session, CSV_ID)


@pytest.mark.parametrize("result", [
    {"run_id": RUN_ID, "status": {"run_id": "hydro-run-other"}},
    {"run_id": RUN_ID, "status": {"input_id": "hydro-input-other"}},
])
def test_contradictory_run_or_input_ids_are_rejected(build_agent, result):
    agent = build_agent()
    session = agent.new_session()
    session.hydro_run_id, session.hydro_input_id = RUN_ID, "hydro-input-abcdef123456"
    with pytest.raises(Exception, match="输入版本"):
        agent._validate_result_binding(session, result)


@pytest.mark.asyncio
async def test_a_plan_cannot_be_submitted_after_its_input_changes(build_agent, calls):
    agent = build_agent()
    session = agent.new_session()
    async with agent:
        for text in PRELUDE:
            await step(agent, session, text)
        session.parameter_overrides = {"deposit_thickness_m": 1.5}
        reply = await agent.handle(session, "开始运行")
    assert not reply.ok and "start_hydro_run" not in calls.hydro


def test_archive_year_interval_is_inclusive():
    from slope_evaluation_agent.temporal_inputs import historical_period
    assert historical_period("使用2000—2010年资料期样本最大日") == {"end_year": 2010, "lookback_years": 11}
