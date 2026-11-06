import pytest
from unittest.mock import patch
from test_web_conversation import web_for, a_client, a_session, say
from test_web_study_area import area_tools

POINTS=[[109.48,31.38],[109.52,31.38],[109.52,31.42],[109.48,31.42]]


@pytest.mark.asyncio
@pytest.mark.parametrize("location",[{"river_name":"大宁河"},{"longitude":109.5,"latitude":31.4}])
async def test_both_location_modes_require_an_explicit_polygon_before_confirmation(tmp_path,calls,location):
    app,agent,_=web_for(tmp_path,calls,spatial=area_tools())
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        loaded=await client.post(f"/api/v1/sessions/{sid}/study-area-map",json=location)
        assert loaded.status_code==200,loaded.text
        result=loaded.json()
        assert result["selection_mode"]=="drawn_study_area" and result["map_id"]
        assert not agent.store.load(sid).review_id
        assert result["map"]["url"] and "image_base64" not in result["map"]
        assert not calls.hydro and "delineate_from_outlet" not in calls.spatial
        await say(client,sid,"确认")
        assert "confirm_watershed" not in calls.spatial
        with patch.object(agent.spatial,"create_drawn_study_area",wraps=agent.spatial.create_drawn_study_area) as create:
            submitted=await client.post(f"/api/v1/sessions/{sid}/study-area",json={
                "map_id":result["map_id"],"coordinates":POINTS})
            assert create.call_args.kwargs["coordinates"]==POINTS
        assert submitted.status_code==200,submitted.text
        assert submitted.json()["session"]["state"]=="awaiting_watershed_confirmation"
        assert "create_drawn_study_area" in calls.spatial
        assert "delineate_from_outlet" not in calls.spatial and not calls.hydro
        event=next(e for e in agent.store.read_events(sid) if e.get("tool")=="create_drawn_study_area")
        assert event["arguments"]["administrative_area"] is None
        assert (await say(client,sid,"确认"))["session"]["watershed_confirmed"]


@pytest.mark.asyncio
async def test_relocating_invalidates_old_polygons_even_without_a_district(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,spatial=area_tools())
    async with agent,a_client(app) as client:
        sid=await a_session(client);endpoint=f"/api/v1/sessions/{sid}/study-area-map"
        old=(await client.post(endpoint,json={"river_name":"大宁河"})).json()
        new=(await client.post(endpoint,json={"longitude":109.5,"latitude":31.4})).json()
        assert old["map_id"]!=new["map_id"]
        for body in [{"coordinates":POINTS},{"map_id":old["map_id"],"coordinates":POINTS},
                     {"map_id":new["map_id"],"coordinates":[[105,29],[106,29],[106,30]]}]:
            refused=await client.post(f"/api/v1/sessions/{sid}/study-area",json=body)
            assert refused.status_code==409
        assert "create_drawn_study_area" not in calls.spatial and not agent.store.load(sid).review_id


@pytest.mark.asyncio
async def test_coordinate_chat_opens_a_drawing_map_instead_of_auto_delineating(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls)
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        response=await say(client,sid,"东经109.5、北纬31.4")
        assert response["reply"]["outlet_map"]["selection_mode"]=="drawn_study_area"
        assert response["reply"]["outlet_map"]["location_mode"]=="coordinates"
        assert not agent.store.load(sid).review_id
        assert not agent.store.load(sid).requested_outlet
        assert calls.spatial==["render_outlet_candidates"] and not calls.hydro


@pytest.mark.asyncio
async def test_model_can_refer_to_the_previous_coordinate_location(tmp_path,calls):
    from stubs import StubLLM
    app,agent,_=web_for(tmp_path,calls)
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        await say(client,sid,"东经109.5、北纬31.4")
        model=StubLLM({"route":"workflow_action","intent":"start_assessment","confidence":.95})
        agent.router.llm=model
        response=await say(client,sid,"就看刚才那个位置")
        assert response["reply"]["outlet_map"]["location_mode"]=="coordinates"
        assert not agent.store.load(sid).review_id
        assert calls.spatial==["render_outlet_candidates"]*2 and not calls.hydro


@pytest.mark.asyncio
@pytest.mark.parametrize("location",[{"longitude":109.5},{"river_name":" "},
    {"administrative_area":"永川区","river_name":"大宁河"},{"longitude":200,"latitude":31}])
async def test_bad_locations_are_rejected_before_tool_calls(tmp_path,calls,location):
    app,agent,_=web_for(tmp_path,calls)
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        response=await client.post(f"/api/v1/sessions/{sid}/study-area-map",json=location)
        assert response.status_code==422 and not calls.spatial
