"""Actual M7 requests, independent transport evidence, and negative controls."""
from copy import deepcopy
import hashlib

import pytest

from slope_evaluation_agent.intent import Intent, ParsedIntent
from slope_evaluation_agent.result_delivery import requested_deliverables
from slope_evaluation_agent.input_binding import input_signature
from stubs import StubLLM, envelope, visualization_payload, run_result_payload
from test_artifact_downloads import an_agent, artifacts_payload, TIFF_BYTES, TIFF_ID


RESULT_REQUEST = ('请读取刚完成的本次结果，给出最小FS、事件内曾出现FS<1的面积和面积比例，'
                  '注明面积分母、时间口径和QA状态，显示对应的最小FS图。')
DOWNLOAD_REQUEST = ('请提供本次稳定性结果的一个可下载文件，并列出本次运行编号、'
                    '实际降雨起止时间及参数来源，供我复查。')


def saved_sources(session):
    session.rainfall_context = {'event_start':'2024-07-30 00:00:00',
                                'event_end':'2024-07-30 19:00:00'}
    session.rainfall_request = {'source_timezone':'Asia/Shanghai'}
    session.last_plan = {'parameter_preset':'medium', 'parameter_overrides':{}}


@pytest.mark.parametrize('route,intent', [('workflow_action','show_result'),
    ('workflow_action','show_visualization'), ('knowledge_chat','ask_explanation')])
@pytest.mark.asyncio
async def test_actual_result_request_delivers_image_with_one_render(build_agent, calls, route, intent):
    agent, session = an_agent(build_agent, calls)
    agent.router.llm = StubLLM({'route':route,'intent':intent,'confidence':.99,
                              'topic':'current_result'})
    async with agent:
        reply = await agent.handle(session, RESULT_REQUEST)
    assert reply.ok and reply.images and reply.visualizations, reply.text
    assert calls.hydro.count('create_hydro_visualization') == 1
    assert 'get_hydro_run_result' in calls.hydro and 'get_hydro_visualization' in calls.hydro
    assert '0.2752' in reply.text
    assert reply.facts['result_delivery']['completed'] == {'figure':True, 'result':True}
    assert 'start_hydro_run' not in calls.hydro


@pytest.mark.asyncio
async def test_actual_download_request_delivers_one_relevant_file_and_sources(build_agent, calls):
    items = deepcopy(artifacts_payload()['artifacts'])
    other = deepcopy(items[0]);other.update(artifact_id='artifact-unrelated01',title='dem.tif',stage_id='delineate')
    duplicate = deepcopy(items[0]);duplicate.update(artifact_id='artifact-export0001',stage_id='export')
    agent, session = an_agent(build_agent, calls,
        list_hydro_artifacts=envelope(data={'artifacts':[other, duplicate, *items]}))
    saved_sources(session)
    agent.router.llm = StubLLM({'route':'workflow_action','intent':'download_artifact','confidence':.99})
    async with agent:
        reply = await agent.handle(session, DOWNLOAD_REQUEST)
        assert reply.ok and not reply.audit, reply.text
        descriptor = reply.facts['artifact']
        assert descriptor['artifact_id'] == TIFF_ID
        stream = await agent.download_artifact(session, descriptor['artifact_id'])
        try:
            raw = b''.join([chunk async for chunk in stream.chunks])
            assert raw == TIFF_BYTES and hashlib.sha256(raw).hexdigest() == stream.sha256
        finally:
            await stream.aclose()
    assert session.hydro_run_id in reply.text
    assert '2024-07-30 00:00:00' in reply.text and '2024-07-30 19:00:00' in reply.text
    assert 'medium' in reply.text and '未覆盖' in reply.text
    assert reply.facts['result_delivery']['completed'] == {'download':True,'provenance':True}
    assert 'start_hydro_run' not in calls.hydro and 'create_hydro_visualization' not in calls.hydro


@pytest.mark.parametrize('text', [
    '不要出图或下载，只解释最小FS数值。',
    '如果我要最小FS图和下载文件，应该怎么操作？',
    '“显示最小FS图并下载文件”是之前的话，现在只问这个数值的含义。',
])
def test_negation_hypothetical_and_quotation_do_not_add_delivery(text):
    wanted = requested_deliverables(text, ParsedIntent(intent=Intent.ASK_EXPLANATION))
    assert not wanted.figure and not wanted.download


@pytest.mark.asyncio
async def test_unavailable_image_keeps_values_and_marks_incomplete(build_agent, calls):
    agent, session = an_agent(build_agent, calls,
        list_hydro_visualization_options=envelope(data={'options':[{
            'kind':'safety_minimum_factor','stage_id':'safety','available':False}]}))
    agent.router.llm = StubLLM({'route':'workflow_action','intent':'show_result','confidence':.99})
    async with agent:
        reply = await agent.handle(session, RESULT_REQUEST)
    assert not reply.ok and not reply.images and '0.2752' in reply.text
    assert reply.facts['result_delivery']['completed'] == {'figure':False,'result':True}
    assert 'create_hydro_visualization' not in calls.hydro


@pytest.mark.asyncio
async def test_dirty_input_cannot_render_saved_run_via_compound_request(build_agent, calls):
    agent, session = an_agent(build_agent, calls)
    session.run_input_signature = input_signature(session)
    session.parameter_overrides = {'saturated_cohesion_kpa': 8}
    agent.router.llm = StubLLM({'route':'workflow_action','intent':'show_result','confidence':.99})
    async with agent:
        reply = await agent.handle(session, RESULT_REQUEST)
    assert not reply.ok
    assert 'create_hydro_visualization' not in calls.hydro


@pytest.mark.asyncio
async def test_missing_rainfall_evidence_is_explicit_not_invented(build_agent, calls):
    agent, session = an_agent(build_agent, calls)
    agent.router.llm = StubLLM({'route':'workflow_action','intent':'download_artifact','confidence':.99})
    async with agent:
        reply = await agent.handle(session, DOWNLOAD_REQUEST)
    assert not reply.ok and '未保存完整证据' in reply.text
    assert reply.facts['result_delivery']['completed']['provenance'] is False


@pytest.mark.asyncio
async def test_rejected_model_answer_keeps_area_and_denominator_in_audited_fallback(build_agent, calls):
    payload = run_result_payload()
    payload['safety']['spatial_distribution'] = {
        'available':True, 'basis':'event_minimum_per_pixel',
        'calculation_area_km2':392.71842281545327,
        'below_one_area_km2':59.98845728497427,
        'below_one_fraction':.15275182879099133,
        'at_or_above_one_fraction':.8472481712090086,
    }
    agent, session = an_agent(build_agent, calls, get_hydro_run_result=envelope(data=payload))
    agent.llm = StubLLM({'reads':[], 'answer':'本次最小安全系数为9。'})
    agent.router.llm = StubLLM({'route':'workflow_action','intent':'show_result','confidence':.99})
    async def forbidden_rewrite(*args, **kwargs):
        raise AssertionError('A rejected answer must use the audited fallback without another rewrite')
    agent._rewrite_reply = forbidden_rewrite
    async with agent:
        reply = await agent.handle(session, '给出事件内曾出现FS<1的面积和比例，注明分母、时间口径与QA。')
    assert reply.ok and not reply.audit, reply.text
    import re
    areas = [float(value) for value in re.findall(r'([\d.]+) km²', reply.text)]
    assert any(abs(value-59.98845728497427) < 5e-5 for value in areas)
    assert any(abs(value-392.71842281545327) < 5e-5 for value in areas)
    assert '15.28%' in reply.text and '事件最小' in reply.text and '质量检查通过' in reply.text
    assert reply.answer_source == 'mcp_evidence'
