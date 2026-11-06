import pytest

from slope_evaluation_agent.dialogue_router import DialogueRouter, RouteKind
from slope_evaluation_agent.intent import Intent, RuleParser
from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.scientific_policy import audit_reply
from slope_evaluation_agent.response_writer import describe_rainfall_package
from slope_evaluation_agent.task_workspace import revision_rows
from stubs import StubLLM, rainfall_payload, envelope


def session():
    return AgentSession(session_id="session-abcdef123456", state="watershed_confirmed",
                        review_id="basin-abcdef123456", watershed_confirmed=True, watershed_qa_passed=True,
                        rainfall_request={"mode": "explicit_period", "date": "2024-07-30", "station_count": 4})


@pytest.mark.parametrize("text,count,mode", [
    ("设立6个虚拟雨站", 6, "manual"), ("设置六个虚拟雨量站", 6, "manual"),
    ("虚拟雨站数量改成十二个", 12, "manual"), ("布设一百个雨站", 100, "manual"),
    ("将4个雨站改成8个雨站", 8, "manual"),
    ("恢复按面积自动布站", None, "auto"), ("根据面积自动确定雨站数量", None, "auto"),
    ("根据面积自行判断需要多少个雨站", None, "auto"),
])
def test_rules_extract_numeric_chinese_manual_and_auto_preferences(text, count, mode):
    parsed = RuleParser().parse(text, session())
    assert parsed.intent is Intent.SET_RAINFALL_PERIOD
    assert parsed.station_count == count and parsed.station_count_mode == mode


def actual_package(count=3, *, auto=True):
    value = rainfall_payload()
    value["stations"] = [{"station_key": f"VS{i+1:02d}"} for i in range(count)]
    value["station_count"] = count
    value["station_selection"] = {"mode": "auto" if auto else "manual", "station_count": count,
        "area_km2": 123.091, "area_per_station_km2": 50., "stations_per_unit": 1, "minimum": 2, "maximum": 20}
    return value


def test_automatic_explanation_passes_scientific_numeric_audit():
    package = actual_package()
    text = describe_rainfall_package(package)
    assert "按选区面积自动" in text and "123.09" in text
    assert audit_reply(text, {"rainfall_package": package}) == []


@pytest.mark.asyncio
async def test_manual_then_auto_rebuilds_inputs_and_keeps_prior_version(build_agent):
    agent = build_agent()
    current = agent.new_session()
    source = session()
    for key in ("state", "review_id", "watershed_confirmed", "watershed_qa_passed", "rainfall_request"):
        setattr(current, key, getattr(source, key))
    sent = []
    async def build(request):
        assert current.hydro_input_id is None and current.model_input_package_id is None
        sent.append(request)
        return envelope(data=actual_package(request.station_count or 3, auto=request.station_count is None))
    agent.spatial.build_virtual_rainfall = build
    async with agent:
        reply = await agent.handle(current, "设立6个虚拟雨站")
        assert reply.ok and not reply.audit, reply.text
        assert sent[-1].station_count == 6 and sent[-1].date == "2024-07-30"
        assert current.rainfall_request["station_count"] == 6
        reply = await agent.handle(current, "恢复按面积自动布站")
        assert reply.ok and not reply.audit, reply.text
        assert sent[-1].station_count is None and sent[-1].date == "2024-07-30"
    assert current.rainfall_request.get("station_count") is None
    assert current.rainfall_context["station_count"] == 3
    assert current.rainfall_context["station_selection"]["mode"] == "auto"
    assert any(row["snapshot"]["rainfall_request"].get("station_count") == 6
               for row in revision_rows(agent.store, current.session_id))


@pytest.mark.asyncio
async def test_count_only_without_event_records_preference_and_does_not_compute(build_agent, calls):
    agent = build_agent()
    current = agent.new_session()
    async with agent:
        reply = await agent.handle(current, "设立六个虚拟雨站")
    assert current.rainfall_request["station_count"] == 6
    assert not calls.spatial and not calls.hydro
    assert not reply.audit and "降雨" in reply.text


@pytest.mark.asyncio
async def test_reset_without_event_does_not_repeat_old_manual_count(build_agent,calls):
    agent=build_agent();current=agent.new_session()
    current.rainfall_request={'station_count':6}
    async with agent:
        reply=await agent.handle(current,'恢复按面积自动布站')
    assert current.rainfall_request.get('station_count') is None
    assert not reply.audit and '自动确定' in reply.text and '设为6' not in reply.text
    assert not calls.spatial and not calls.hydro


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["设立0个雨站", "设立101个雨站", "布设6.5个雨量站", "设置负三个雨站", "设立十二点五个雨站"])
async def test_invalid_counts_clarify_without_mutating_existing_inputs(build_agent, text):
    agent = build_agent()
    current = agent.new_session()
    current.rainfall_request = {"mode": "explicit_period", "date": "2024-07-30", "station_count": 4}
    async with agent:
        reply = await agent.handle(current, text)
    assert current.rainfall_request["station_count"] == 4
    assert not reply.audit and "整数" in reply.text
    assert current.last_route == "clarification"


@pytest.mark.asyncio
async def test_model_question_route_does_not_change_station_preferences():
    current = session()
    router = DialogueRouter(llm=StubLLM({"route": "knowledge_chat", "confidence": .95,
                                        "topic": "rainfall", "candidates": []}))
    decision = await router.decide("6个虚拟雨站够吗？", current)
    parsed, errors = router.to_parsed_intent(decision, "6个虚拟雨站够吗？", current)
    assert decision.route is RouteKind.KNOWLEDGE_CHAT and not errors
    assert parsed.station_count is None and parsed.station_count_mode is None
    assert current.rainfall_request["station_count"] == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("message,fields,expected", [
    ("只修改为六个，其他沿用", [("station_count",6),("station_count_mode","manual")],6),
    ("恢复按面积自动布站", [("station_count_mode","auto")],None),
])
async def test_model_can_extract_count_from_context_or_reset(message,fields,expected):
    router=DialogueRouter(llm=StubLLM({"route":"workflow_action","intent":"set_rainfall_period",
        "confidence":.98,"candidates":[{"field":key,"value":value,"evidence_text":message}
                                           for key,value in fields]}))
    current=session()
    decision=await router.decide(message,current)
    parsed,errors=router.to_parsed_intent(decision,message,current)
    assert not errors and decision.route is RouteKind.WORKFLOW_ACTION
    assert parsed.station_count==expected
    assert parsed.station_count_mode==("auto" if expected is None else "manual")
    assert parsed.date=="2024-07-30"
