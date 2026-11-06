"""Cross-provider regressions from the frozen Gemini/DeepSeek engineering tasks."""
import json

import pytest

from slope_evaluation_agent.dialogue_router import DialogueRouter, RouteKind
from slope_evaluation_agent.intent import Intent, ParsedIntent, RuleParser
from slope_evaluation_agent.llm import LLMRequestFailed, OpenAICompatibleClient
from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.config import LLMSettings
from slope_evaluation_agent.state_machine import WorkflowError
from helpers import spatial_handlers, spatial_server
from stubs import StubLLM, rainfall_payload
from test_relay_model_families import HTTP, response, gemini_reply
from test_grounded_result_dialogue import assessment, Model
from slope_evaluation_agent.result_responder import answer_result, _rainfall_window_problems


MESSAGE = ('请在重庆市东经107.761828°、北纬30.134155°附近开展完整评价。'
           '先按这个坐标加载地图，我来提交并确认固定的闭合评价边界，范围可以跨区县。'
           '降雨采用北京时间2024-06-03 00:00至2024-06-03 19:00，共20个小时记录。')


@pytest.mark.parametrize('text,start,end', [
    ('2024-06-03 00:00至2024-06-03 19:00', '2024-06-03 00:00', '2024-06-03 19:00'),
    ('2012-07-12 12:00至2012-07-13 07:00', '2012-07-12 12:00', '2012-07-13 07:00'),
    ('2024年6月3日00时至2024年6月3日19时', '2024-06-03 00:00', '2024-06-03 19:00'),
    ('2024-06-03T00:00:00—2024-06-03T19:00:00', '2024-06-03 00:00', '2024-06-03 19:00'),
])
def test_full_timestamp_range_retains_both_dates(text, start, end):
    parsed = RuleParser().parse('降雨采用'+text, AgentSession(session_id='session-regression123', state='new'))
    assert (parsed.start_datetime, parsed.end_datetime, parsed.date) == (start, end, None)


@pytest.mark.asyncio
@pytest.mark.parametrize('style', ['deepseek', 'gemini', 'coarse_date'])
async def test_model_route_kept_and_complete_user_window_survives_omitted_candidates(build_agent, style):
    candidates = [{'field':'longitude','value':107.761828,'evidence_text':'107.761828'},
                  {'field':'latitude','value':30.134155,'evidence_text':'30.134155'}]
    if style == 'deepseek':
        candidates += [{'field':'start_datetime','value':'2024-06-03 00:00','evidence_text':'2024-06-03 00:00'},
                       {'field':'end_datetime','value':'2024-06-03 19:00','evidence_text':'2024-06-03 19:00'}]
    if style == 'coarse_date':
        candidates += [{'field':'date','value':'2024-06-03','evidence_text':'2024-06-03'}]
    model = StubLLM({'route':'workflow_action','intent':'start_assessment','confidence':.95,
                     'candidates':candidates})
    agent = build_agent(); agent.router.llm = model
    current = agent.new_session()
    async with agent:
        reply = await agent.handle(current, MESSAGE)
    assert reply.outlet_map and 'list_administrative_areas' not in reply.facts
    assert current.rainfall_request['start_datetime'] == '2024-06-03 00:00'
    assert current.rainfall_request['end_datetime'] == '2024-06-03 19:00'
    assert 'date' not in current.rainfall_request
    assert not current.watershed_confirmed and not current.hydro_run_id
    assert agent.router.last_source == 'model' and len(model.calls) == 1


@pytest.mark.asyncio
async def test_explicit_coordinate_map_request_wins_over_city_context(build_agent, calls):
    agent=build_agent(); current=agent.new_session()
    agent.router.llm=StubLLM({'route':'workflow_action','intent':'start_assessment','confidence':.95,
        'candidates':[{'field':'administrative_area','value':'重庆市','evidence_text':'重庆市'},
                      {'field':'longitude','value':107.761828,'evidence_text':'107.761828'},
                      {'field':'latitude','value':30.134155,'evidence_text':'30.134155'}]})
    async with agent:
        reply=await agent.handle(current,MESSAGE)
    assert reply.outlet_map and 'render_outlet_candidates' in calls.spatial
    assert 'list_administrative_areas' not in calls.spatial and not calls.hydro


@pytest.mark.asyncio
async def test_knowledge_with_identical_numbers_does_not_record_or_execute_inputs(build_agent, calls):
    agent=build_agent();current=agent.new_session()
    agent.router.llm=StubLLM({'route':'knowledge_chat','intent':'ask_explanation','confidence':.99})
    async with agent:
        await agent.handle(current,'如果降雨采用2024-06-03 00:00至2024-06-03 19:00，时间口径如何理解？')
    assert not current.rainfall_request and not calls.spatial and not calls.hydro


@pytest.mark.asyncio
@pytest.mark.parametrize('window',['2024-06-03 19:00至2024-06-03 00:00',
                                  '2024-02-30 00:00至2024-02-30 19:00'])
async def test_invalid_full_window_clarifies_without_reverting_to_whole_day(window):
    router=DialogueRouter(llm=StubLLM({'route':'workflow_action','intent':'set_rainfall_period','confidence':.99,
        'candidates':[{'field':'date','value':'2024-06-03','evidence_text':window}]}))
    current=AgentSession(session_id='session-regression123', state='new')
    decision=await router.decide('降雨改为'+window,current)
    parsed,_=router.to_parsed_intent(decision,window,current)
    assert decision.route is RouteKind.CLARIFICATION
    assert parsed.date is parsed.start_datetime is parsed.end_datetime is None


@pytest.mark.asyncio
async def test_mismatched_rainfall_response_cannot_prepare_inputs(build_agent, calls):
    payload=rainfall_payload(mode='explicit_period')
    payload.update(event_start='2024-06-03 00:00:00', event_end='2024-06-03 23:00:00', timestamp_count=24)
    agent=build_agent(spatial=spatial_server(spatial_handlers(calls,build_virtual_rainfall=payload)))
    current=agent.new_session();current.review_id='basin-abcdef123456'
    current.watershed_confirmed=True;current.watershed_qa_passed=True;current.state='watershed_confirmed'
    current.rainfall_request={'mode':'explicit_period','start_datetime':'2024-06-03 00:00','end_datetime':'2024-06-03 19:00'}
    async with agent:
        reply=await agent._run_intake(current,ParsedIntent(intent=Intent.CONFIRM_WATERSHED),'确认边界',include_rainfall=True)
    assert not reply.ok and not current.rainfall_package_id
    assert 'prepare_model_inputs' not in calls.spatial and not calls.hydro
    assert any(e['event']=='rainfall_package_rejected' for e in agent.store.read_events(current.session_id))


@pytest.mark.asyncio
@pytest.mark.parametrize('valid_second', [True,False])
async def test_native_gemini_json_retry_is_bounded_and_keeps_thinking(monkeypatch, valid_second):
    settings=LLMSettings(provider='gemini',model='gemini-3.8-flash',base_url='https://4router.net/v1',
        api_key='fake-not-real',api_format='gemini_generate_content',temperature=None,max_retries=1)
    malformed=gemini_reply();malformed['candidates'][0]['content']['parts']=[{'text':'{'}]
    http=HTTP(response(malformed),response(gemini_reply() if valid_second else malformed))
    client=OpenAICompatibleClient(settings);monkeypatch.setattr(client,'_client',lambda:http)
    if valid_second:
        assert await client.complete_json(system='JSON',user='task')=={'ok':True}
    else:
        with pytest.raises(LLMRequestFailed,match='JSON'):
            await client.complete_json(system='JSON',user='task')
    assert len(http.requests)==2
    assert all('thinkingConfig' not in body['generationConfig'] for _,_,body in http.requests)
    assert http.requests[0][2] == http.requests[1][2]


@pytest.mark.asyncio
async def test_result_answer_must_disambiguate_event_minimum_area():
    model=Model({'answer':'Fs<1面积为2.0 km²，占有效计算面积10.0 km²的20.00%，QA通过。'},
                {'answer':'按事件内每个像元的最小FS统计，曾FS<1的面积为2.0 km²，占有效计算面积10.0 km²的20.00%，QA通过。'})
    result=await answer_result(llm=model, question='事件内曾FS<1的面积和分母是多少，注明时间口径',
        assessment=assessment(),background={},catalog=[],read_artifact=None,fallback='已核验的摘要')
    assert result.used_model and len(model.calls)==2
    assert '逐像元' in json.loads(model.calls[1]['user'])['revision_request']


@pytest.mark.parametrize('text,bad', [
    ('降雨事件全时段1995-11-12 15:00至1995-11-13 06:00内逐像元最小值。', True),
    ('降雨输入窗口1995-11-12 15:00至1995-11-13 04:00。模拟窗口1995-11-12 15:00至1995-11-13 06:00。', False),
    ('按事件内逐像元最小FS；模拟窗口1995-11-12 15:00至1995-11-13 06:00。', False),
])
def test_rainfall_input_and_simulation_windows_have_distinct_meanings(text, bad):
    background={'prepared_rainfall':{'event_start':'1995-11-12 15:00:00','event_end':'1995-11-13 04:00:00'}}
    assert bool(_rainfall_window_problems(text,background)) is bad
