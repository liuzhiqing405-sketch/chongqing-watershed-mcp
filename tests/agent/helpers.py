"""Helpers shared by the workflow and injection test suites.

The ``build_agent`` fixture lives in ``conftest.py`` so both modules can use it;
everything else is a plain function here.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import Image

from slope_evaluation_agent.agent import SlopeEvaluationAgent
from slope_evaluation_agent.models import AgentSession

from stubs import (
    PNG_BYTES,
    StubHydroClient,
    StubSpatialClient,
    candidates_payload,
    envelope,
    hydro_server,
    import_payload,
    model_input_payload,
    outlet_map_payload,
    plan_payload,
    rainfall_payload,
    review_payload,
    run_result_payload,
    run_status_payload,
    spatial_server,
    visualization_artifacts_payload,
    validation_payload,
    visualization_payload,
)


REVIEW_ID = "basin-abcdef123456"
INPUT_ID = "hydro-input-abcdef123456"
RUN_ID = "hydro-run-abcdef123456"

#: The turns that carry a session from nothing to a shown run plan.
DRAW_AREA = "@submit_drawn_area"
PRELUDE = ["东经 109.5、北纬 31.4", DRAW_AREA, "确认该边界", "2024-07-30 全天"]


class Calls:
    """Records what the stub servers were asked to do."""

    def __init__(self) -> None:
        self.spatial: list[str] = []
        self.hydro: list[str] = []


def _recorder(store: list[str]):
    """A handler wrapper that records the call and resolves callable payloads."""

    def record(name: str, payload: Any):
        def handler():
            store.append(name)
            return payload() if callable(payload) else payload
        return handler

    return record


def spatial_handlers(calls: Calls, **overrides) -> dict:
    record = _recorder(calls.spatial)
    drawn=dict(review_payload(),selection_mode="drawn_study_area",outlet={})
    drawn["qa"]["hydrologic_check"]={"checked":True,"passed":True,"major_outlet_count":1,
                                    "external_upstream_area_km2":0,"warnings":[]}
    handlers = {
        "recommend_outlets": record("recommend_outlets", candidates_payload()),
        "render_outlet_candidates": record("render_outlet_candidates", outlet_map_payload()),
        "delineate_from_outlet": record("delineate_from_outlet", review_payload()),
        "create_drawn_study_area": record("create_drawn_study_area",drawn),
        "render_process_artifact": record(
            "render_process_artifact", Image(data=PNG_BYTES, format="png")),
        "confirm_watershed": record("confirm_watershed", envelope(data={
            "review_id": REVIEW_ID, "status": "confirmed", "confirmed": True,
            "reviewer_note": "用户确认"})),
        "build_virtual_rainfall": record("build_virtual_rainfall", rainfall_payload()),
        "prepare_model_inputs": record("prepare_model_inputs", model_input_payload()),
    }
    # Default successful rainfall must reflect the requested calendar window.
    # Explicit test overrides retain their exact payload, including mismatches.
    handlers["build_virtual_rainfall"].use_requested_period = True
    for name, payload in overrides.items():
        handlers[name] = record(name, payload)
    return handlers


def hydro_handlers(calls: Calls, **overrides) -> dict:
    record = _recorder(calls.hydro)
    handlers = {
        "get_hydro_capabilities": record("get_hydro_capabilities", envelope(data={
            "service": "watershed-hydro-stability", "engine": {"ok": True},
            "modules": {"ok": True}, "stages": [{"stage_id": "sync"}],
            "parameter_presets": ["low", "medium", "high"],
            "parameter_bounds": {"saturated_cohesion_kpa": [0.0, 60.0],
                                 "deposit_thickness_m": [0.2, 5.0]},
            "registered_inputs": 0, "limitations": []})),
        "import_hydro_input_package": record("import_hydro_input_package",
                                             envelope(data=import_payload())),
        "validate_hydro_input": record("validate_hydro_input",
                                       envelope(data=validation_payload())),
        "plan_hydro_run": record("plan_hydro_run", envelope(data=plan_payload())),
        "start_hydro_run": record("start_hydro_run", envelope(data=run_status_payload())),
        "get_hydro_run_status": record("get_hydro_run_status",
                                       envelope(data=run_status_payload())),
        "get_hydro_run_result": record("get_hydro_run_result",
                                       envelope(data=run_result_payload())),
        "cancel_hydro_run": record("cancel_hydro_run",
                                   envelope(data=run_status_payload(status="cancelled"))),
        "create_hydro_visualization": record(
            "create_hydro_visualization",
            envelope(data=visualization_payload(status="queued"))),
        "get_hydro_visualization_status": record(
            "get_hydro_visualization_status",
            envelope(data=visualization_payload(status="succeeded"))),
        "list_hydro_visualization_options": record(
            "list_hydro_visualization_options",
            envelope(data={"options": [{"kind": "safety_minimum_factor",
                                        "stage_id": "safety",
                                        "title": "最小安全系数"}], "existing": []})),
        "get_hydro_visualization": record(
            "get_hydro_visualization", Image(data=PNG_BYTES, format="png")),
        "list_visualization_artifacts": record(
            "list_visualization_artifacts",
            envelope(data=visualization_artifacts_payload())),
    }
    for name, payload in overrides.items():
        handlers[name] = record(name, payload)
    return handlers


def agent_with(spatial=None, hydro=None, *, store=None, parser=None, root=None):
    """An agent over the given clients.

    ``root`` defaults to a private temporary directory, so a test can never
    write into the repository's own session store.
    """
    import tempfile
    from pathlib import Path

    from slope_evaluation_agent.config import AgentSettings
    from slope_evaluation_agent.intent import IntentParser
    from slope_evaluation_agent.session_store import SessionStore

    if store is not None:
        settings = store.settings.model_copy() if hasattr(store, "settings") else None
        settings = AgentSettings.model_validate(
            {"sessions": {"root": str(store.root)}})
    else:
        base = Path(root) if root is not None else Path(tempfile.mkdtemp(prefix="slope-agent-"))
        settings = AgentSettings.model_validate(
            {"sessions": {"root": str(base / "sessions")}})
        store = SessionStore(settings.sessions)
    return SlopeEvaluationAgent(
        settings,
        spatial=spatial or StubSpatialClient(spatial_server({})),
        hydro=hydro or StubHydroClient(hydro_server({})),
        store=store,
        parser=parser or IntentParser(),
    )


async def run(agent: SlopeEvaluationAgent, session: AgentSession, text: str):
    async with agent:
        return await step(agent,session,text)


async def step(agent,session,text):
    if text==DRAW_AREA:
        return await agent.submit_study_area(session,session.administrative_area,
            [[109.48,31.38],[109.52,31.38],[109.52,31.42],[109.48,31.42]],
            map_id=session.selection_context.get("map_id"))
    return await agent.handle(session,text)


async def converse(agent: SlopeEvaluationAgent, session: AgentSession,
                   turns: list[str]):
    """One connection for a whole scenario, as the CLI holds it for a session."""
    replies = []
    async with agent:
        for text in turns:
            replies.append(await step(agent,session,text))
    return replies


async def prepared_agent(build_agent, calls: Calls, *, run_status: str | None = None):
    """An agent and a session ready to be driven; the status stub is chosen here."""
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
