"""Read-only evidence repairs without changing the routing/action contract."""
import copy
import json

import pytest

from slope_evaluation_agent.dialogue_router import DialogueRouter
from slope_evaluation_agent.evaluation import build_assessment
from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.result_responder import answer_result
from slope_evaluation_agent.scientific_policy import EvidenceLedger, audit_reply
from slope_evaluation_agent.temporal_inputs import resolve_relative_rainfall
from stubs import StubLLM


def source():
    return {'run_id': 'hydro-run-evidence123', 'status': {'status': 'succeeded'},
            'quality': {'passed': True},
            'safety': {'available': True, 'minimum_safety_factor': .58316,
                       'max_unstable_fraction': .12, 'unstable_area_km2': 18.07342},
            'overlay_evidence': {'intersected_polygon_count': 17, 'total_polygon_count': 25,
                                 'overlap_landslide_area_km2': .8, 'total_landslide_area_km2': 2.0}}


def assessment():
    session = AgentSession(session_id='session-evidence123', hydro_run_id='hydro-run-evidence123',
                           state='hydro_succeeded')
    return build_assessment(session, source())


def test_raw_evidence_is_preserved_and_not_aliased():
    raw = source()
    built = build_assessment(AgentSession(session_id='session-evidence123', state='hydro_succeeded'), raw)
    raw['overlay_evidence']['total_polygon_count'] = 99
    assert built['result_details']['overlay_evidence']['total_polygon_count'] == 25
    assert built['result_details']['safety']['minimum_safety_factor'] == .58316


@pytest.mark.parametrize('text', [
    'Fs<1面积为18.07 km²，对应面积为18,073,420 m²。',
    '最小安全系数为0.58316。',
    '图斑数量比例为68.00%，面积覆盖率为40.00%。',
])
def test_traceable_values_conversions_and_distinct_denominators_pass(text):
    assert not audit_reply(text, EvidenceLedger.from_assessment(assessment()))


@pytest.mark.parametrize('text', [
    '最小安全系数为0.19。', '面积覆盖率为68.00%。',
    '最小安全系数为0.58316 m²。', 'Fs<1面积为18.07 m²。',
])
def test_wrong_values_units_and_count_area_substitution_still_fail(text):
    assert audit_reply(text, EvidenceLedger.from_assessment(assessment()))


def test_no_ratio_invented_from_zero_missing_or_inconsistent_denominator():
    raw = source()
    raw['overlay_evidence']['total_polygon_count'] = 0
    raw['overlay_evidence']['total_landslide_area_km2'] = .3
    built = build_assessment(AgentSession(session_id='session-evidence123', state='hydro_succeeded'), raw)
    assert not any(x['operation'] == 'divide' for x in built['derived_evidence'])
    assert all('sources' in item for item in built['derived_evidence'])


@pytest.mark.parametrize('text', [
    '请确认2008年2月29日的年份。', '8月5日是指哪一年？',
    '当前任务run-plan-20090804尚未结束。',
])
def test_calendar_and_opaque_ids_do_not_block_clarification(text):
    assert not audit_reply(text, {})


def test_invalid_calendar_is_not_exempt_and_physical_numbers_remain_audited():
    assert audit_reply('2009年2月30日', {})
    assert audit_reply('最小安全系数为0.42。', {})


@pytest.mark.parametrize('message,expected', [
    ('降雨改为原日期四天之后', '2025-07-02'),
    ('改用两天以前', '2025-06-26'),
])
def test_postfix_calendar_offsets(message, expected):
    request = {'mode': 'explicit_period', 'date': '2025-06-28'}
    before = copy.deepcopy(request)
    assert resolve_relative_rainfall(message, request).values['date'] == expected
    assert request == before


def test_negated_postfix_offset_does_not_become_an_edit():
    assert not resolve_relative_rainfall('不要改用四天之后', {'date': '2025-06-28'}).values


@pytest.mark.asyncio
async def test_model_place_candidate_loses_action_prefix_but_keeps_actual_place():
    router = DialogueRouter(llm=StubLLM({'route': 'workflow_action', 'intent': 'start_assessment',
        'confidence': .96, 'candidates': [{'field': 'administrative_area', 'value': '显示永川区',
                                         'evidence_text': '显示永川区'}]}))
    session = AgentSession(session_id='session-evidence123', state='new')
    decision = await router.decide('显示永川区底图', session)
    parsed, _ = router.to_parsed_intent(decision, '显示永川区底图', session)
    assert parsed.administrative_area == '永川区'


@pytest.mark.asyncio
async def test_result_model_receives_raw_and_derived_evidence_without_new_actions():
    class Model:
        async def complete_json(self, **kwargs):
            evidence = json.loads(kwargs['user'])['assessment']
            assert evidence['result_details']['overlay_evidence']['total_polygon_count'] == 25
            assert evidence['derived_evidence']
            return {'reads': [], 'answer': '图斑数量比例为68.00%，面积覆盖率为40.00%。'}
    async def read(request):
        raise AssertionError('No extra tool should be needed')
    result = await answer_result(llm=Model(), question='解释数量和面积的差别', assessment=assessment(),
        background={}, catalog=[], read_artifact=read, fallback='摘要。')
    assert result.used_model and not result.reads


@pytest.mark.asyncio
async def test_unrelated_explicit_run_is_not_silently_replaced(build_agent, calls):
    agent = build_agent()
    session = agent.new_session()
    session.hydro_run_id = 'hydro-run-evidence123'
    agent.router.llm = StubLLM({'route': 'knowledge_chat', 'intent': 'ask_explanation',
                               'confidence': .99, 'topic': 'current_result'})
    reply = await agent.handle(session, '查询run-other-123的结果')
    assert '未与当前会话关联' in reply.text
    assert not calls.hydro and not calls.spatial


def test_response_from_a_different_run_is_rejected(build_agent):
    agent = build_agent()
    session = agent.new_session()
    session.hydro_run_id = 'run-current-123'
    with pytest.raises(Exception, match='运行编号'):
        agent._validate_result_binding(session, {'run_id': 'run-other-123'})
