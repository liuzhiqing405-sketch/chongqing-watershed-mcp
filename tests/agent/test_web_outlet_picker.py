"""The interactive outlet picker (Web API).

What these tests pin down: the picker walks the same gate as naming a river in
chat, it hands the browser everything it needs to turn a click into a
coordinate, and choosing a point does not delineate anything by itself — the
chosen coordinates still have to be sent as a message, which is what runs
delineation and its manual-confirmation gate.
"""

from __future__ import annotations

import pytest

from stubs import outlet_map_payload

from test_web_conversation import a_client, a_session, say, web_for


@pytest.mark.unit
class TestTheSitingMap:
    @pytest.mark.asyncio
    async def test_naming_a_river_in_chat_opens_a_manual_map(self, tmp_path, calls):
        app, agent, _ = web_for(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            body = await say(client, session_id, "分析大宁河")
            reply = body["reply"]
            assert reply["ok"] is True
            assert reply["outlet_map"]["url"] == reply["images"][0]["url"]
            assert reply["outlet_map"]["bbox_wgs84"] == [109.30, 31.20, 109.90, 31.70]
            assert "image_base64" not in reply["outlet_map"]
            assert "候选" not in reply["text"]
            assert agent.store.load(session_id).outlet_candidates == []
            assert calls.spatial == ["render_outlet_candidates"]
            events = agent.store.read_events(session_id)
            event = next(item for item in events if item.get("tool") == "render_outlet_candidates")
            assert event["arguments"]["max_candidates"] == 0
            restored = (await client.get(f"/api/v1/sessions/{session_id}")).json()
            assert "绘制闭合评价边界" in restored["pending_action"]
            assert "第 1 个" not in restored["pending_action"]

    @pytest.mark.asyncio
    async def test_it_returns_no_candidates_and_a_served_map(self, tmp_path, calls):
        app, agent, _ = web_for(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            response = await client.post(
                f"/api/v1/sessions/{session_id}/outlet-candidates",
                json={"river_name": "大宁河", "max_candidates": 5})
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["matched_names"] == ["大宁河"]
            assert body["candidates"] == []
            assert body["session"]["state"] == "outlet_candidates_ready"
            assert "recommend_outlets" not in calls.spatial
            assert body["map"]["bbox_wgs84"] == [109.30, 31.20, 109.90, 31.70]
            assert body["map"]["axes_rect"][0] > 0
            assert "render_outlet_candidates" in calls.spatial
            # The base64 payload must not be echoed to the browser: only the URL.
            assert "image_base64" not in body["map"]
            image = await client.get(body["map"]["url"])
            assert image.status_code == 200
            assert image.headers["content-type"] == "image/png"

    @pytest.mark.asyncio
    async def test_the_gate_and_state_come_from_the_agent(self, tmp_path, calls):
        """Generating the map advances the workflow exactly as naming a river does."""
        app, agent, _ = web_for(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            before = (await client.get(f"/api/v1/sessions/{session_id}")).json()
            response = await client.post(
                f"/api/v1/sessions/{session_id}/outlet-candidates",
                json={"river_name": "大宁河"})
            after = response.json()["session"]
        assert response.status_code == 200
        assert before["state"] != after["state"]
        assert after["state"] == "outlet_candidates_ready"

    @pytest.mark.asyncio
    async def test_it_never_delineates_on_its_own(self, tmp_path, calls):
        app, agent, _ = web_for(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            await client.post(f"/api/v1/sessions/{session_id}/outlet-candidates",
                              json={"river_name": "大宁河"})
        assert "delineate_from_outlet" not in calls.spatial

    @pytest.mark.asyncio
    async def test_a_blank_name_is_rejected_before_any_tool_call(self, tmp_path, calls):
        app, agent, _ = web_for(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            response = await client.post(
                f"/api/v1/sessions/{session_id}/outlet-candidates",
                json={"river_name": "   "})
        assert response.status_code == 422
        assert "render_outlet_candidates" not in calls.spatial

    @pytest.mark.asyncio
    async def test_an_unknown_river_answers_without_a_map(self, tmp_path, calls):
        """A river the layer does not have is a normal answer, not a failure."""
        app, agent, _ = web_for(tmp_path, calls, spatial={
            "render_outlet_candidates": outlet_map_payload(matched=False)})
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            response = await client.post(
                f"/api/v1/sessions/{session_id}/outlet-candidates",
                json={"river_name": "不存在河"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["candidates"] == []
        assert body["map"]["url"] is None
        assert body["message"]

    @pytest.mark.asyncio
    async def test_a_coordinate_only_relocates_the_drawing_map(self, tmp_path, calls):
        app, agent, _ = web_for(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            response = await client.post(
                f"/api/v1/sessions/{session_id}/outlet-candidates",
                json={"river_name": "大宁河"})
            assert response.json()["candidates"] == []
            candidate = {"longitude": 109.5, "latitude": 31.4}
            assert "delineate_from_outlet" not in calls.spatial
            # What the browser sends when the user confirms the picked point.
            await say(client, session_id,
                      f"经度 {candidate['longitude']}、纬度 {candidate['latitude']}")
        assert "delineate_from_outlet" not in calls.spatial
        assert calls.spatial.count("render_outlet_candidates")==2
