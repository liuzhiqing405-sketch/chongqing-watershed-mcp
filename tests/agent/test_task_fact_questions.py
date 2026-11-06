import json

import pytest

from slope_evaluation_agent.conversation_responder import ConversationResponder
from slope_evaluation_agent.engineering_knowledge import knowledge_answer_problems
from slope_evaluation_agent.task_context import task_background
from stubs import StubLLM
from helpers import run
from test_web_conversation import web_for, a_client, say


def saved_task(agent):
    session = agent.new_session()
    session.state = 'hydro_qa_failed'
    session.administrative_area = '南川区'
    session.hydro_run_id = 'hydro-run-123456abcdef'
    session.hydro_run_status = 'qa_failed'
    session.watershed_confirmed = True
    session.qa_passed = False
    session.rainfall_request = {'date':'2024-07-23','mode':'explicit_period'}
    session.run_status_detail = {
        'status':'qa_failed','stages':['swmm','qa'],'completed_stages':['swmm','qa'],
        'qa_errors':['填洼最大抬高超过门槛，需要复核洼地与DEM',
                     '全部站点降雨总量为0，事件窗口可能选错']}
    return session


@pytest.mark.asyncio
@pytest.mark.parametrize('question,answer', [
    ('为何QA不通过', '任务已经完成计算，但质量门槛未通过：全部站点降雨总量为0，事件窗口可能选错。'),
    ('这个流域是哪个区', '当前任务选择的是南川区，边界已经确认。'),
])
async def test_task_questions_accept_saved_history_and_keep_state(build_agent,calls,question,answer):
    agent = build_agent(); session = saved_task(agent)
    agent.router.llm = StubLLM({'route':'knowledge_chat','confidence':.95,'topic':'用户自定义主题'})
    model = StubLLM({'answer':answer})
    agent.responder = ConversationResponder(llm=model)
    before = (session.state,session.administrative_area,session.hydro_run_id,session.rainfall_request.copy())
    reply = await run(agent,session,question)
    assert reply.text == answer
    assert reply.answer_source == 'general_model_knowledge'
    assert reply.facts['knowledge_fallback'] is None
    sent = json.loads(model.calls[0]['user'])['task_background']
    assert sent['location']['administrative_area'] == '南川区'
    assert sent['run']['detail']['qa_errors'] == session.run_status_detail['qa_errors']
    assert before == (session.state,session.administrative_area,session.hydro_run_id,session.rainfall_request)
    assert calls.spatial == calls.hydro == []


@pytest.mark.asyncio
@pytest.mark.parametrize('question,needle', [
    ('为何QA不通过','全部站点降雨总量为0'),
    ('为什么质量门槛没过','填洼最大抬高超过门槛'),
    ('这个流域是哪个区','南川区'),
    ('我选的区域在哪里','南川区'),
])
@pytest.mark.parametrize('failure', [None, TimeoutError('offline'), {'answer':'我已经调用MCP完成计算。'}])
async def test_task_fact_fallback_answers_question_instead_of_generic_error(build_agent,calls,question,needle,failure):
    agent = build_agent(); session = saved_task(agent)
    agent.router.llm = StubLLM({'route':'knowledge_chat','confidence':.95,'topic':'qa'})
    agent.responder = ConversationResponder(llm=None if failure is None else StubLLM(failure))
    reply = await run(agent,session,question)
    assert needle in reply.text
    assert '请检查模型配置' not in reply.text
    assert reply.answer_source == 'saved_task_context'
    assert reply.audit_scope == 'task_context'
    assert session.state == 'hydro_qa_failed'
    assert calls.spatial == calls.hydro == []
    if isinstance(failure,dict):
        events = agent.store.events(session.session_id).read()
        record = next(row for row in reversed(events) if row['event'] == 'knowledge_reply')
        assert record['fallback_reason'] == 'knowledge_policy_rejected'
        assert record['rejection_reasons']


def test_background_only_authorizes_saved_facts_not_new_actions_or_metrics(build_agent):
    context = task_background(saved_task(build_agent()))
    assert not knowledge_answer_problems('任务已经完成计算。边界已确认。',task_background=context)
    assert not knowledge_answer_problems('该流域计算结果未通过质量验证。',task_background=context)
    assert not knowledge_answer_problems('任务编号 hydro-run-123456abcdef。',task_background=context)
    for text in ['我已经完成计算。','本轮已运行SWMM。','本次流域安全系数为1.5。',
                 '任务已经取消。','任务编号 hydro-run-ffffffffffff。']:
        assert knowledge_answer_problems(text,task_background=context), text
    context['run']['detail']['completed_stages'] = ['swmm']
    assert knowledge_answer_problems('任务已经完成计算。',task_background=context)


@pytest.mark.asyncio
async def test_general_qa_concept_does_not_fall_back_to_unrelated_task_errors(build_agent):
    agent=build_agent();session=saved_task(agent)
    reply=await ConversationResponder().respond_general(question='QA的一般原理是什么？',session=session,topic='qa')
    assert reply.answer_source == 'builtin_knowledge'
    assert '全部站点降雨总量为0' not in reply.text


@pytest.mark.asyncio
async def test_web_restores_task_fact_provenance_without_claiming_qa_passed(tmp_path,calls):
    app,agent,_=web_for(tmp_path,calls)
    session=saved_task(agent);agent.store.save(session)
    agent.router.llm=StubLLM({'route':'knowledge_chat','confidence':.95,'topic':'流域行政区归属'})
    agent.responder=ConversationResponder()
    async with agent,a_client(app) as client:
        result=await say(client,session.session_id,'这个流域是哪个区')
        reply=result['reply']
        assert reply['answer_source']=='saved_task_context'
        assert reply['audit_scope']=='task_context'
        assert reply['audit_passed'] is None
        assert '南川区' in reply['text']
        restored=(await client.get('/api/v1/sessions/'+session.session_id)).json()
        assert restored['conversation'][-1]['answer_source']=='saved_task_context'
        assert restored['state']=='hydro_qa_failed'
    assert calls.spatial == calls.hydro == []
