"""Ensure the live benchmark cannot claim success from fallbacks or tool bypasses."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("understanding_acceptance", ROOT / "scripts/accept_model_understanding.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def row(**changes):
    result = dict(model_source="model", route="knowledge_chat", entities={},
                  tool_calls=[], state="new", review_id=None,
                  watershed_confirmed=False, knowledge_fallback=False)
    result.update(changes)
    return result


@pytest.mark.parametrize("changes", [
    {"model_source":"model_fallback"},
    {"route":"workflow_action"},
    {"tool_calls":[{"tool":"render_outlet_candidates"}]},
    {"knowledge_fallback":True},
])
def test_question_cannot_pass_with_fallback_or_execution(changes):
    expected = bench.turn("重庆有哪些河", "knowledge_chat", no_tools=True)
    assert bench.score(expected, row(**changes))
    assert not bench.score(expected, row())


def test_replacing_river_requires_dropping_old_coordinate():
    expected = bench.turn("改成大宁河", "workflow_action",
                          entities={"river_name":"大宁河"}, no_coordinate=True)
    assert bench.score(expected, row(route="workflow_action",
        entities={"river_name":"大宁河", "longitude":105.9}))


def test_knowledge_must_preserve_pending_boundary():
    expected = bench.turn("讲讲孔隙水压力", "knowledge_chat", preserve_review=True)
    assert not bench.score(expected, row(state="awaiting_watershed_confirmation",
                                       review_id="basin-acceptance"))
    assert bench.score(expected, row(state="awaiting_watershed_confirmation",
                                    review_id="basin-acceptance", watershed_confirmed=True))


@pytest.mark.asyncio
async def test_fixture_refuses_every_computational_operation():
    from slope_evaluation_agent.mcp_clients import AgentMCPError
    for label in ("spatial", "hydro"):
        fixture = bench.FixtureMCP(label)
        for operation in ("delineate_from_outlet", "confirm_watershed", "start_hydro_run"):
            with pytest.raises(AgentMCPError):
                await getattr(fixture, operation)()


@pytest.mark.asyncio
async def test_production_agent_can_use_isolated_map_fixture(tmp_path):
    from slope_evaluation_agent.agent import SlopeEvaluationAgent
    from slope_evaluation_agent.config import AgentSettings
    from slope_evaluation_agent.dialogue_router import DialogueRouter
    from slope_evaluation_agent.response_rewriter import ResponseRewriter
    from stubs import StubLLM
    settings = AgentSettings()
    settings.sessions.root = str(tmp_path)
    model = StubLLM({"route":"workflow_action", "intent":"start_assessment", "confidence":.98,
        "candidates":[{"field":"administrative_area", "value":"永川区", "evidence_text":"永川区"}]})
    spatial, hydro = bench.FixtureMCP("spatial"), bench.FixtureMCP("hydro")
    agent = SlopeEvaluationAgent(settings, spatial=spatial, hydro=hydro,
        router=DialogueRouter(llm=model), rewriter=ResponseRewriter(None))
    session = agent.new_session()
    reply = await agent.handle(session, "评价永川区的小流域")
    assert reply.outlet_map and session.administrative_area == "永川区"
    assert not session.review_id and not session.hydro_run_id and not hydro.calls


def test_corpus_contains_original_failure_patterns_and_28_turns():
    cases = bench.scenarios()
    assert len({s["id"] for s in cases}) == len(cases)
    assert sum(len(s["turns"]) for s in cases) == 28
    texts = [t["text"] for s in cases for t in s["turns"]]
    assert "重庆有哪些河" in texts and "就看前面说的那条河" in texts
