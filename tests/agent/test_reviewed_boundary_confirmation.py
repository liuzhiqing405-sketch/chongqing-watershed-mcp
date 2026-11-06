"""A reviewed boundary confirmation must retain its real pending review."""
import pytest

from slope_evaluation_agent.dialogue_router import DialogueRouter,RouteKind
from slope_evaluation_agent.intent import Intent
from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.state_machine import WorkflowState
from helpers import DRAW_AREA,step,REVIEW_ID,INPUT_ID
from stubs import StubLLM

CONFIRM='我已核对这个评价范围和边界QA，确认边界。'

def pending():
    return AgentSession(session_id='session-abcdef123456',state='awaiting_watershed_confirmation',
                        review_id=REVIEW_ID,watershed_qa_passed=True)

@pytest.mark.asyncio
@pytest.mark.parametrize('text',[
 CONFIRM,'我已经核对了当前评价范围与边界QA，确认当前边界。',
 '我已检查这个选区和质量检查结果；我确认这个边界。',
 '我已复核该流域边界无误。确认该流域边界。',
 '我已经查看过边界QA，确认本次边界。'])
async def test_personal_review_and_explicit_confirmation_bypass_model(text):
    model=StubLLM({'route':'workflow_action','intent':'start_assessment','confidence':1})
    router=DialogueRouter(llm=model)
    decision=await router.decide(text,pending())
    assert decision.intent is Intent.CONFIRM_WATERSHED
    assert decision.route is RouteKind.WORKFLOW_ACTION and router.last_source=='human'
    assert model.calls==[]

@pytest.mark.asyncio
@pytest.mark.parametrize('text',[
 '如果我已核对这个评价范围和边界QA，确认边界。',
 '我已核对这个评价范围和边界QA，如果通过就确认边界。',
 '我已核对这个评价范围和边界QA，确认边界吗？',
 '我已核对这个评价范围和边界QA，不确认边界。',
 '我已核对这个评价范围和边界QA，你确认边界。',
 '我已核对这个评价范围和边界QA，帮我确认边界。',
 '我已核对这个评价范围和边界QA，“确认边界”。',
 '他说我已核对这个评价范围和边界QA，确认边界。',
 '我已核对这个评价范围和边界QA，确认边界，但先不要确认。',
 '我已核对这个评价范围和边界QA，确认边界；撤回确认。',
 '我还没核对这个评价范围和边界QA，确认边界。',
 '我已核对这个评价范围和边界QA，随后可以确认边界。'])
async def test_review_wording_does_not_turn_conditions_or_mentions_into_confirmation(text):
    model=StubLLM({'route':'knowledge_chat','intent':'ask_explanation','confidence':.9,
                   'candidates':[],'missing_fields':[],'topic':'engineering_general'})
    router=DialogueRouter(llm=model)
    decision=await router.decide(text,pending())
    assert decision.intent is not Intent.CONFIRM_WATERSHED
    assert router.last_source!='human' or decision.intent is Intent.REJECT_WATERSHED

@pytest.mark.asyncio
async def test_reviewed_confirmation_needs_a_pending_review():
    session=AgentSession(session_id='session-abcdef123456',state='new')
    router=DialogueRouter(llm=StubLLM({'route':'clarification','intent':'unknown','confidence':.5,'ambiguity':'请先提交边界'}))
    assert (await router.decide(CONFIRM,session)).intent is not Intent.CONFIRM_WATERSHED

@pytest.mark.asyncio
@pytest.mark.parametrize('known_rainfall',[False,True])
async def test_actual_m1_confirm_retains_review_and_continues_the_existing_intake(build_agent,calls,known_rainfall):
    agent=build_agent();session=agent.new_session()
    async with agent:
        await step(agent,session,'东经 109.5、北纬 31.4');await step(agent,session,DRAW_AREA)
        if known_rainfall:
            session.rainfall_request={'mode':'explicit_period','start_datetime':'2024-06-03 00:00',
                                     'end_datetime':'2024-06-03 19:00','source_timezone':'Asia/Shanghai'}
        model=StubLLM({'route':'workflow_action','intent':'start_assessment','confidence':1})
        agent.router=DialogueRouter(llm=model,settings=agent.settings.dialogue,rules=agent.parser.rules)
        maps_before=calls.spatial.count('render_outlet_candidates')
        await agent.handle(session,CONFIRM)
    assert session.watershed_confirmed and session.review_id==REVIEW_ID and session.watershed_qa_passed
    assert calls.spatial.count('confirm_watershed')==1
    assert calls.spatial.count('render_outlet_candidates')==maps_before
    assert model.calls==[]
    if known_rainfall:
        assert session.hydro_input_id==INPUT_ID and session.hydro_input_validated and session.hydro_planned
        assert session.state==WorkflowState.HYDRO_PLANNED.value
    else:assert session.state==WorkflowState.WATERSHED_CONFIRMED.value

@pytest.mark.asyncio
async def test_review_phrase_cannot_override_a_failed_boundary_check(build_agent,calls):
    agent=build_agent();session=agent.new_session()
    async with agent:
        await step(agent,session,'东经 109.5、北纬 31.4');await step(agent,session,DRAW_AREA)
        session.watershed_qa_passed=False
        agent.router=DialogueRouter(llm=StubLLM({'route':'workflow_action','intent':'start_assessment','confidence':1}),
                                   settings=agent.settings.dialogue,rules=agent.parser.rules)
        await agent.handle(session,CONFIRM)
    assert not session.watershed_confirmed and 'confirm_watershed' not in calls.spatial
    assert session.review_id==REVIEW_ID
