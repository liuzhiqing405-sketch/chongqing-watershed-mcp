"""Continuation keeps current inputs; a saved-version rollback remains separate.

These are execution-contract tests with model decisions supplied explicitly.
They do not measure the real model's intent accuracy; that needs a live replay.
"""

import copy
import json
from pathlib import Path

import pytest

from slope_evaluation_agent.conversation_responder import ConversationResponder
from slope_evaluation_agent.task_workspace import task_snapshot
from stubs import StubLLM
from test_web_conversation import a_client, a_session, say, web_for
from test_web_study_area import area_tools


pytestmark = pytest.mark.unit
CASES = json.loads((Path(__file__).parent / "fixtures" / "resume_dialogue_cases.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
async def test_recorded_resume_requests_keep_inputs_and_only_open_map(tmp_path, calls, case):
    app, agent, _ = web_for(tmp_path, calls, spatial=area_tools())
    model = StubLLM({})
    agent.router.llm = model
    agent.responder = ConversationResponder(llm=StubLLM({"answer": "这是原理说明，当前评价设置保持不变。"}))
    async with agent, a_client(app) as client:
        sid = await a_session(client)
        session = agent.store.load(sid)
        session.state = "outlet_candidates_ready"
        session.administrative_area = "永川区"
        session.selection_mode = "drawn_study_area"
        session.selection_context = {"map_id": "existing-map", "location_mode": "district"}
        session.intent = "start_assessment"
        session.rainfall_request = {"mode": "explicit_period", "date": case["seed_date"]}
        session.pending_workflow = {
            "intent": "start_assessment",
            "entities": {"administrative_area": "永川区", "date": case["seed_date"]},
            "unresolved_fields": [],
        }
        agent.store.save(session)
        for turn in case["turns"]:
            before = copy.deepcopy(task_snapshot(agent.store.load(sid)))
            question = turn["mode"] == "question"
            model.reply = {"route": "knowledge_chat" if question else "workflow_action",
                           "intent": "ask_explanation" if question else "start_assessment",
                           "confidence": .96}
            response = await say(client, sid, turn["text"])
            current = agent.store.load(sid)
            if question:
                assert task_snapshot(current) == before
                assert not calls.spatial and not calls.hydro
            else:
                assert response["reply"]["outlet_map"]["administrative_area"] == "永川区"
                assert calls.spatial == ["list_administrative_areas", "render_administrative_dem"]
                assert current.selection_context["map_id"] != "existing-map"
            assert current.rainfall_request == {"mode": "explicit_period", "date": case["seed_date"]}
            assert not current.watershed_confirmed and current.hydro_run_id is None
            assert not calls.hydro
            assert response["reply"]["ok"]
        events = agent.store.read_events(sid)
        assert not any(row["event"] == "task_restored" for row in events)


@pytest.mark.asyncio
async def test_explicit_saved_version_restore_still_restores_previous_inputs(tmp_path, calls):
    app, agent, _ = web_for(tmp_path, calls, spatial=area_tools(), model_reply={
        "route": "workflow_action", "intent": "restore_task", "confidence": .98})
    async with agent, a_client(app) as client:
        sid = await a_session(client)
        session = agent.store.load(sid)
        session.administrative_area = "永川区"
        session.rainfall_request = {"mode": "explicit_period", "date": "2024-02-29"}
        agent.checkpoint_task(session, "修改日期前")
        session.rainfall_request["date"] = "2024-03-01"
        agent.store.save(session)
        response = await say(client, sid, "撤销刚才的日期修改，恢复到保存的上一版本。")
        assert response["reply"]["ok"]
        assert agent.store.load(sid).rainfall_request["date"] == "2024-02-29"
        assert not calls.spatial and not calls.hydro
        assert any(row["event"] == "task_restored" for row in agent.store.read_events(sid))
