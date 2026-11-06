import base64
import pytest
from test_web_conversation import web_for,a_client,a_session,say
from stubs import outlet_map_payload,review_payload,envelope


def area_tools():
    record=review_payload()
    record.update(selection_mode="drawn_study_area",administrative_area="永川区",outlet={})
    record["qa"]["hydrologic_check"]={"checked":True,"passed":True,"major_outlet_count":1,
                                      "external_upstream_area_km2":0,"warnings":[]}
    district_map=outlet_map_payload()["map"]
    district_map["bbox_wgs84"]=[105.6,28.9,106.2,29.6]
    return {
        "list_administrative_areas":lambda:envelope(data={"areas":[{"name":"永川区","code":"500118"}]}),
        "render_administrative_dem":lambda:envelope(data={"administrative_area":"永川区",
            "selection_mode":"drawn_study_area","message":"请在 DEM 上绘制闭合边界。",
            "map":district_map}),
        "create_drawn_study_area":lambda:envelope(data=record)}


@pytest.mark.asyncio
async def test_changed_map_content_gets_new_url_without_duplicate_same_content(tmp_path,calls):
    tools=area_tools()
    first=base64.b64encode(b'old map layer').decode()
    second=base64.b64encode(b'new HydroRIVERS map layer').decode()
    current=[first]
    original=tools['render_administrative_dem']
    def render():
        result=original()
        result['data']['map']['image_base64']=current[0]
        return result
    tools['render_administrative_dem']=render
    app,agent,_=web_for(tmp_path,calls,spatial=tools)
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        path=f'/api/v1/sessions/{sid}/study-area-map'
        old=(await client.post(path,json={'administrative_area':'永川区'})).json()['map']['url']
        current[0]=second
        new=(await client.post(path,json={'administrative_area':'永川区'})).json()['map']['url']
        again=(await client.post(path,json={'administrative_area':'永川区'})).json()['map']['url']
        assert old!=new and new==again
        assert (await client.get(old)).content==b'old map layer'
        assert (await client.get(new)).content==b'new HydroRIVERS map layer'


@pytest.mark.asyncio
async def test_drawing_path_requires_review_confirmation_before_rainfall(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,spatial=area_tools())
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        loaded=await client.post(f"/api/v1/sessions/{sid}/study-area-map",json={"administrative_area":"永川区"})
        assert loaded.status_code==200,loaded.text
        assert "image_base64" not in loaded.json()["map"]
        submitted=await client.post(f"/api/v1/sessions/{sid}/study-area",json={"administrative_area":"永川区",
             "coordinates":[[105.9,29.4],[105.93,29.4],[105.93,29.43],[105.9,29.4]]})
        assert submitted.status_code==200,submitted.text
        assert submitted.json()["reply"]["ok"]
        assert submitted.json()["session"]["state"]=="awaiting_watershed_confirmation"
        assert submitted.json()['session']['selection_mode'] == 'drawn_study_area'
        assert "确认" in submitted.json()["reply"]["text"]
        assert not agent.store.load(sid).watershed_confirmed
        assert not calls.hydro and "delineate_from_outlet" not in calls.spatial
        assert (await say(client,sid,"确认"))["session"]["watershed_confirmed"]


@pytest.mark.asyncio
async def test_changed_district_does_not_submit_stale_map_coordinates(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls,spatial=area_tools())
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        await client.post(f"/api/v1/sessions/{sid}/study-area-map",json={"administrative_area":"永川区"})
        result=await client.post(f"/api/v1/sessions/{sid}/study-area",json={"administrative_area":"武隆区",
            "coordinates":[[107,29],[107.1,29],[107.1,29.1]]})
        assert result.status_code==409
        assert not agent.store.load(sid).review_id


@pytest.mark.asyncio
async def test_model_distinguishes_district_task_from_river_name(tmp_path,calls):
    model={"route":"workflow_action","intent":"start_assessment","confidence":.95,
           "candidates":[{"field":"administrative_area","value":"永川区","evidence_text":"永川区"}]}
    app,agent,_=web_for(tmp_path,calls,spatial=area_tools(),model_reply=model)
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        result=await say(client,sid,"我想在永川区自己画线进行坡面稳定性分析")
        assert result["reply"]["outlet_map"]["selection_mode"]=="drawn_study_area"
        assert agent.store.load(sid).administrative_area=="永川区"
        assert not agent.store.load(sid).river_name
        assert not agent.store.load(sid).review_id and not calls.hydro


@pytest.mark.asyncio
@pytest.mark.parametrize("diagnostic",[None,{"checked":True,"passed":False,"major_outlet_count":2,
    "external_upstream_area_km2":3.4,"errors":["边界切断上游，未建外部入流。"],"warnings":[]}])
async def test_hydrology_failure_or_old_service_blocks_confirmation_even_if_top_level_qa_passes(tmp_path,calls,diagnostic):
    tools=area_tools()
    payload=tools["create_drawn_study_area"]()
    if diagnostic is None:
        payload["data"]["qa"].pop("hydrologic_check")
    else:
        payload["data"]["qa"]["hydrologic_check"]=diagnostic
        payload["data"]["qa"]["errors"]=diagnostic["errors"]
    tools["create_drawn_study_area"]=lambda:payload
    app,agent,_=web_for(tmp_path,calls,spatial=tools)
    async with agent,a_client(app) as client:
        sid=await a_session(client)
        await client.post(f"/api/v1/sessions/{sid}/study-area-map",json={"administrative_area":"永川区"})
        submitted=await client.post(f"/api/v1/sessions/{sid}/study-area",json={"administrative_area":"永川区",
            "coordinates":[[105.9,29.4],[105.93,29.4],[105.93,29.43]]})
        assert submitted.status_code==200,submitted.text
        data=submitted.json()
        assert data["session"]["state"]=="watershed_qa_failed"
        assert data["reply"]["spatial_qa_passed"] is False
        assert data["reply"]["hydrologic_qa_passed"] is (None if diagnostic is None else False)
        assert "不能确认" in data["reply"]["text"]
        await say(client,sid,"确认该边界")
        assert not agent.store.load(sid).watershed_confirmed
        assert "confirm_watershed" not in calls.spatial and not calls.hydro
