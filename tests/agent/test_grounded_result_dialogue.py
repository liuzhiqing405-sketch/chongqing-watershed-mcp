import json
from pathlib import Path

import pytest

from slope_evaluation_agent.evaluation import build_assessment,describe_result_summary
from slope_evaluation_agent.result_responder import answer_result
from slope_evaluation_agent.scientific_policy import EvidenceLedger,audit_reply
from slope_evaluation_agent.web import STATIC_ROOT
from slope_evaluation_agent.models import AgentSession
from stubs import run_result_payload, visualization_artifacts_payload, envelope, build_server
from helpers import hydro_handlers
from slope_evaluation_agent.mcp_clients import HydroMCPClient


class Model:
    def __init__(self,*replies): self.replies=list(replies); self.calls=[]
    async def complete_json(self,**kwargs):
        self.calls.append(kwargs)
        return self.replies[min(len(self.calls)-1,len(self.replies)-1)]


def assessment():
    session=AgentSession(session_id='session-abcdef123456', state='hydro_succeeded')
    result=run_result_payload()
    result['safety']['spatial_distribution']={
        'available':True,'basis':'event_minimum_per_pixel','below_one_fraction':.2,
        'at_or_above_one_fraction':.8,'below_one_area_km2':2.,'calculation_area_km2':10.,
        'largest_below_one_sector':'西北部',
        'sectors':{'西北部':{'below_one_fraction':.5,'share_of_all_below_one':1.},
                   '东南部':{'below_one_fraction':0.,'share_of_all_below_one':0.}}}
    return build_assessment(session,result)


async def run(model,read=None,catalog=None,assessment_value=None):
    return await answer_result(llm=model,question='评测区域稳定性',assessment=assessment_value or assessment(),
        background={},catalog=catalog or [],read_artifact=read,fallback='计算结果摘要。')


@pytest.mark.asyncio
async def test_model_interprets_actual_distribution_instead_of_rewriting_labels():
    model=Model({'answer':'事件最小Fs栅格中，Fs<1区域占20.00%，其余区域占80.00%。低值面积在西北部最多，应优先核查。','reads':[]})
    result=await run(model)
    assert result.used_model and '西北部' in result.text
    prompt=json.loads(model.calls[0]['user'])
    assert prompt['assessment']['factors']['spatial_distribution']['basis']=='event_minimum_per_pixel'
    assert '色标' in model.calls[0]['system']


@pytest.mark.asyncio
async def test_model_selects_and_reads_actual_parameters_with_no_workflow_actions():
    model=Model({'reads':[{'artifact_id':'artifact-params','search':'forest'}]},
                {'answer':'林地的黏聚力参数为7.0 kPa，取值来自本次参数表。'})
    called=[]
    async def read(request):
        called.append(request.model_dump())
        return {'rows':[{'landuse':'forest','cohesion_kpa':7.0}]}
    result=await run(model,read,[{'artifact_id':'artifact-params','title':'actual_parameters.csv'}])
    assert result.used_model and len(called)==1
    assert called[0]['search']=='forest'
    assert json.loads(model.calls[1]['user'])['artifact_reads'][0]['rows'][0]['cohesion_kpa']==7


@pytest.mark.asyncio
@pytest.mark.parametrize('passed', [True, False])
@pytest.mark.parametrize('complete_catalog', [True, False])
async def test_agent_routes_file_question_and_reads_over_mcp_without_changing_task(build_agent,calls,passed,complete_catalog):
    descriptor=visualization_artifacts_payload()['artifacts'][1]
    descriptor.update(title='actual_parameters.csv',stage_id='build')
    artifact_id=descriptor['artifact_id']
    result=run_result_payload()
    result['quality']['passed']=passed
    handlers=hydro_handlers(calls,**{
        'get_hydro_run_result':envelope(data=result),
        'list_hydro_artifacts':envelope(data={'artifacts':[descriptor]}),
        'list_hydro_readable_artifacts':envelope(data={'artifacts':[descriptor]}),
        'inspect_hydro_artifact':envelope(data={
            'artifact_id':artifact_id,'kind':'table','read_only':True,
            'rows':[{'landuse':'forest','cohesion_kpa':7.0}]})})
    tools=set(HydroMCPClient.required_tools)|{'inspect_hydro_artifact'}
    if complete_catalog:
        tools.add('list_hydro_readable_artifacts')
    agent=build_agent(hydro=build_server('watershed-hydro-stability',handlers,tools=tools))
    session=agent.new_session()
    session.hydro_run_id='hydro-run-abcdef123456'
    session.state='hydro_succeeded' if passed else 'hydro_qa_failed'
    before=(session.state,session.hydro_run_id,session.watershed_confirmed)
    agent.router.llm=Model({'route':'knowledge_chat','intent':'ask_explanation',
                            'confidence':.98,'topic':'task_artifacts'})
    agent.llm=Model({'reads':[{'artifact_id':artifact_id,'search':'forest'}]},
                    {'answer':'本次林地黏聚力参数为7.0 kPa，来自当前任务的参数表。'})
    async with agent:
        reply=await agent.handle(session,'这次模型实际采用的林地黏聚力是多少？请查看参数文件。')
    assert reply.answer_source=='grounded_model_result'
    assert '7.0 kPa' in reply.text and not reply.audit
    expected=['get_hydro_run_result','list_hydro_artifacts']
    if complete_catalog:
        expected.append('list_hydro_readable_artifacts')
    assert calls.hydro==expected+['inspect_hydro_artifact']
    assert calls.spatial==[]
    assert before==(session.state,session.hydro_run_id,session.watershed_confirmed)


@pytest.mark.asyncio
async def test_unregistered_artifact_or_attempted_tool_execution_cannot_be_read():
    called=[]
    async def read(request): called.append(request)
    model=Model({'reads':[{'artifact_id':'artifact-other-task'}]})
    result=await run(model,read,[{'artifact_id':'artifact-this-task'}])
    assert not result.used_model and not called
    action=await run(Model({'answer':'已经运行。','tool':'start_hydro_run'}))
    assert not action.used_model


@pytest.mark.asyncio
async def test_invented_metrics_and_confused_time_basis_are_repaired_or_rejected():
    model=Model({'answer':'最小安全系数为0.9，最大失稳比例为99.99%。'},
                {'answer':'最小安全系数为0.275，同一时刻最大失稳比例为15.05%。'})
    result=await run(model)
    assert result.used_model and len(model.calls)==2
    assert '99.99' not in result.text
    bad=await run(Model({'answer':'同一时刻最大失稳比例为20.00%。'}))
    assert not bad.used_model
    ledger=EvidenceLedger.from_assessment(assessment())
    assert audit_reply('最小安全系数为0.5。',ledger)


@pytest.mark.asyncio
async def test_wrong_geographic_conclusion_is_not_accepted():
    result=await run(Model({'answer':'低值面积在东南部最多。'}))
    assert not result.used_model


@pytest.mark.asyncio
async def test_failed_quality_or_missing_model_never_yields_a_stability_verdict():
    failed=assessment(); failed['assessment_status']='diagnostic_only'; failed['qa_passed']=False
    result=await run(Model({'answer':'计算已证明整体稳定。'}),assessment_value=failed)
    assert not result.used_model
    fallback=await run(None)
    assert not fallback.used_model and fallback.fallback_reason=='model_unavailable'
    assert audit_reply(describe_result_summary(assessment()),EvidenceLedger.from_assessment(assessment()))==[]


def test_sidebar_and_figure_details_default_to_concise_surfaces():
    html=(STATIC_ROOT/'index.html').read_text(encoding='utf-8')
    workspace=(STATIC_ROOT/'task_workspace.js').read_text(encoding='utf-8')
    js=(STATIC_ROOT/'app.js').read_text(encoding='utf-8')
    assert '边界与水文检查' not in html and 'task-quality' not in workspace
    assert "detailsTitle.textContent = '图件说明与下载'" in js
    assert 'DeepSeek 受控改写' not in js and '确定性回答' not in js
