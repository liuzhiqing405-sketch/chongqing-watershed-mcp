import pytest
from slope_evaluation_agent.models import AgentSession
from stubs import envelope
from helpers import hydro_server, hydro_handlers
from slope_evaluation_agent.task_workspace import revision_rows


@pytest.mark.asyncio
@pytest.mark.parametrize('prior',['hydro_succeeded','assessment_ready'])
async def test_quality_recheck_revokes_verdict_without_losing_task(build_agent,calls,prior):
    agent=build_agent(hydro=hydro_server(hydro_handlers(calls,get_hydro_run_status=envelope(data={
        'status':'qa_failed','qa_passed':False,'run_id':'hydro-run-abcdef123456','completed_stages':['delineate']}))))
    session=AgentSession(session_id='session-abcdef123456',state=prior,
        hydro_run_id='hydro-run-abcdef123456',hydro_run_status='succeeded',
        review_id='basin-abcdef123456',watershed_confirmed=True,qa_passed=True)
    agent.store.save(session)
    agent.store.save_assessment(session.session_id,{'assessment_status':'valid_preliminary'})
    await agent.refresh_run_status(session)
    assert session.state=='hydro_qa_failed' and session.qa_passed is False
    assert session.hydro_run_id=='hydro-run-abcdef123456' and session.watershed_confirmed
    assert not (agent.store.directory(session.session_id)/'final_assessment.json').exists()
    assert revision_rows(agent.store,session.session_id)
