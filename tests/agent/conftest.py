"""Tests shared by the agent suites.

These isolate the agent from the developer's environment: a stray
``SLOPE_AGENT_LLM_API_KEY`` in the shell must never change a test outcome, and
no test may read or write the real session directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


#: Every variable the agent reads. Cleared before each test.
AGENT_ENVIRONMENT = (
    "SLOPE_AGENT_CONFIG",
    "SLOPE_AGENT_LLM_BASE_URL",
    "SLOPE_AGENT_LLM_MODEL",
    "SLOPE_AGENT_LLM_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Run each test with no agent environment variables set."""
    for name in AGENT_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture
def write_config(tmp_path):
    """Write a YAML config file and return its path."""
    import yaml

    def _write(payload: dict, name: str = "agent.yaml") -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def calls():
    """A recorder for the stub servers' tool calls."""
    from helpers import Calls

    return Calls()


@pytest.fixture
def build_agent(tmp_path, calls):
    """A factory for agents backed by in-process stub servers."""
    from helpers import hydro_handlers, hydro_server, spatial_handlers, spatial_server
    from slope_evaluation_agent.agent import SlopeEvaluationAgent
    from slope_evaluation_agent.config import AgentSettings
    from slope_evaluation_agent.intent import IntentParser
    from slope_evaluation_agent.session_store import SessionStore
    from stubs import StubHydroClient, StubSpatialClient

    def _build(*, spatial=None, hydro=None, recorder=None) -> SlopeEvaluationAgent:
        settings = AgentSettings.model_validate({
            "sessions": {"root": str(tmp_path / "sessions"),
                         "max_conversation_turns": 20},
        })
        store = SessionStore(settings.sessions)
        return SlopeEvaluationAgent(
            settings,
            spatial=StubSpatialClient(spatial or spatial_server(
                spatial_handlers(recorder or calls))),
            hydro=StubHydroClient(hydro or hydro_server(
                hydro_handlers(recorder or calls))),
            store=store,
            parser=IntentParser(),
        )

    return _build
