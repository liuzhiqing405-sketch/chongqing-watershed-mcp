"""Regressions grounded in the first live DeepSeek acceptance failures."""
import json

import pytest
from pydantic import ValidationError

from slope_evaluation_agent.dialogue_router import DialogueDecision, DialogueRouter, RouteKind
from slope_evaluation_agent.intent import Intent, RuleParser
from slope_evaluation_agent.llm import LLMRequestFailed
from slope_evaluation_agent.models import AgentSession
from stubs import StubLLM
from test_web_conversation import a_client, a_session, say, web_for


def session(**kwargs):
    return AgentSession(session_id="session-aabbccddee12",state="new",**kwargs)


def test_output_schema_is_distinct_from_workflow_reference():
    schema = DialogueRouter.schema()
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    expected = {"route","intent","confidence","candidates","missing_fields","topic","ambiguity"}
    assert set(schema["properties"]) == set(schema["required"]) == expected
    assert "workflow_inputs" not in schema["properties"]
    assert "intent_meanings" not in schema["properties"]
    assert not {"confirm_watershed","reject_watershed","cancel_run"} & set(schema["$defs"]["Intent"]["enum"])
    DialogueDecision.from_model(DialogueRouter.output_example())


@pytest.mark.asyncio
async def test_actual_request_separates_output_and_reference_data():
    model = StubLLM({"route":"knowledge_chat","intent":"ask_explanation","confidence":.9})
    await DialogueRouter(llm=model).decide("水流如何影响边坡稳定性",session())
    request = json.loads(model.calls[0]["user"])
    assert request["workflow_reference"]["required_inputs"]["start_assessment"] == [
        ["administrative_area"],["river_name"],["longitude","latitude"]]
    assert set(request["output_example"]) == set(request["schema"]["properties"])
    assert "绝不能作为返回字段" in model.calls[0]["system"]


@pytest.mark.asyncio
async def test_final_invalid_schema_keeps_field_level_diagnostic():
    bad = {"route":"workflow_action","intent":"start_assessment","confidence":.95,
           "workflow_inputs":{"administrative_area":"永川区"}}
    router = DialogueRouter(llm=StubLLM(bad))
    await router.decide("对，就是永川区",session())
    assert router.last_source == "model_fallback"
    assert router.last_diagnostic == {"reason":"invalid_route_schema","recovered":False,
        "issues":[{"field":"workflow_inputs","type":"extra_forbidden"}]}
    with pytest.raises(ValidationError):
        DialogueDecision.from_model(bad)


@pytest.mark.parametrize("text,name",[
    ("这次换成大宁河","大宁河"),
    ("本次改为嘉陵江","嘉陵江"),
    ("现在切换到芙蓉江","芙蓉江"),
    ("接下来重新分析临江河","临江河"),
    ("换成那个大宁河","大宁河"),
    ("帮我看看长江","长江"),
    ("长江","长江"),
    ("乌江","乌江"),
])
def test_fallback_name_extraction_does_not_include_action_prefix(text,name):
    assert RuleParser().parse(text,session()).river_name == name


@pytest.mark.asyncio
async def test_transport_failure_uses_clean_river_and_drops_old_coordinate():
    class Offline:
        async def complete_json(self,**kwargs):
            raise LLMRequestFailed("offline",reason="transport_error")
    router = DialogueRouter(llm=Offline())
    current = session(selection_context={"location_mode":"coordinates",
        "map_center":{"longitude":106.1,"latitude":29.4}})
    current.state = "outlet_candidates_ready"
    decision = await router.decide("这次换成大宁河",current)
    parsed,_ = router.to_parsed_intent(decision,"这次换成大宁河",current)
    assert parsed.river_name == "大宁河" and parsed.longitude is None and parsed.latitude is None
    assert router.last_source == "model_fallback"


@pytest.mark.asyncio
async def test_failed_model_preserves_task_and_asks_contextual_question(tmp_path,calls):
    app,agent,_ = web_for(tmp_path,calls,model_reply={"route":"workflow_action",
        "intent":"start_assessment","confidence":.9,"workflow_inputs":{}})
    async with agent,a_client(app) as client:
        sid = await a_session(client)
        current = agent.store.load(sid)
        pending = {"intent":"start_assessment","entities":{"administrative_area":"永川区"},
                   "unresolved_fields":[],"user_request":"我想评价永川区的小流域"}
        current.pending_workflow = pending
        agent.store.save(current)
        result = await say(client,sid,"我明白了，按之前说的地方开始评价")
        assert "永川区" in result["reply"]["text"] and "加载 DEM 地图" in result["reply"]["text"]
        assert "这个智能体负责" not in result["reply"]["text"]
        assert result["reply"]["route"] == "clarification"
        assert agent.store.load(sid).pending_workflow == pending
        assert not calls.spatial and not calls.hydro


@pytest.mark.asyncio
async def test_unknown_fallback_cannot_resume_disputed_location():
    router = DialogueRouter()
    current = session(pending_workflow={"intent":"start_assessment",
        "entities":{"administrative_area":"永川区"},"unresolved_fields":["administrative_area"]})
    decision = await router.decide("嗯，还是前面那个",current)
    assert decision.route is RouteKind.CLARIFICATION
    assert "永川区" not in (decision.ambiguity or "")


@pytest.mark.asyncio
async def test_recovered_contract_allows_task_reference_from_pending_context():
    class Repair:
        def __init__(self): self.requests=[]
        async def complete_json(self,*,system,user):
            self.requests.append(json.loads(user))
            good = {"route":"workflow_action","intent":"start_assessment","confidence":.9,
                    "candidates":[],"missing_fields":[],"topic":None,"ambiguity":None}
            return {**good,"workflow_inputs":{}} if len(self.requests)==1 else good
    model=Repair()
    router=DialogueRouter(llm=model)
    current=session(pending_workflow={"intent":"start_assessment",
        "entities":{"administrative_area":"永川区"},"unresolved_fields":[]})
    text="我明白了，按之前说的地方开始评价"
    decision=await router.decide(text,current)
    parsed,_=router.to_parsed_intent(decision,text,current)
    assert decision.route is RouteKind.WORKFLOW_ACTION and parsed.administrative_area == "永川区"
    assert router.last_source == "model" and router.last_diagnostic["recovered"]
    assert model.requests[1]["schema_repair"]["issues"][0]["field"] == "workflow_inputs"
