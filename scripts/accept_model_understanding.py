"""Live multi-turn acceptance of production dialogue code, with inert spatial fixtures.

Only the LLM is live. Maps are labelled fixtures; no real MCP or calculation runs.
Run via run_model_acceptance.ps1 for masked, process-only credential input.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from slope_evaluation_agent.agent import SlopeEvaluationAgent
from slope_evaluation_agent.config import load_settings
from slope_evaluation_agent.llm import OpenAICompatibleClient
from slope_evaluation_agent.mcp_clients import AgentMCPError
from slope_evaluation_agent.session_store import SessionStore


def turn(text, routes, *, entities=None, no_tools=False, preserve_review=False,
         no_coordinate=False, critical=False):
    return dict(text=text, routes=routes.split("|"), entities=entities or {},
                no_tools=no_tools, preserve_review=preserve_review,
                no_coordinate=no_coordinate, critical=critical)


def scenarios():
    chat = "knowledge_chat"
    task = "workflow_action"
    clarify = "clarification"
    return [
        dict(id="question_vs_task", turns=[
            turn("水流如何影响边坡稳定性", chat, no_tools=True, critical=True),
            turn("帮我分析水流怎样影响边坡", chat, no_tools=True, critical=True),
            turn("重庆有哪些河", chat, no_tools=True, critical=True),
            turn("嘉陵江的水文特点有哪些", chat, no_tools=True),
            turn("能否评价永川区的小流域？", task, entities={"administrative_area":"永川区"})]),
        dict(id="original_river_loop", turns=[
            turn("分析朝阳河", task+"|"+clarify, entities={"river_name":"朝阳河"}),
            turn("我指的是重庆北碚附近的朝阳河，先给我地图，让我自己框选", task,
                 entities={"river_name":"朝阳河"}),
            turn("就看前面说的那条河", task, entities={"river_name":"朝阳河"})]),
        dict(id="task_knowledge_resume", turns=[
            turn("我想评价永川区的小流域", task, entities={"administrative_area":"永川区"}),
            turn("先解释一下为什么要划分子汇水区", chat, no_tools=True),
            turn("不对，我想问的是水流作用的机理", chat, no_tools=True, critical=True),
            turn("按我前面说的地区开始吧", task, entities={"administrative_area":"永川区"})]),
        dict(id="district_correction", turns=[
            turn("我想评价永川区的小流域", task, entities={"administrative_area":"永川区"}),
            turn("永定区", task+"|"+clarify),
            turn("永安区", task+"|"+clarify),
            turn("对，就是永川区", task, entities={"administrative_area":"永川区"})]),
        dict(id="coordinate_correction", turns=[
            turn("在东经105.9、北纬29.4附近进行稳定性评价，先加载地图", task,
                 entities={"longitude":105.9,"latitude":29.4}),
            turn("经度改为106.1，纬度不变", task,
                 entities={"longitude":106.1,"latitude":29.4}),
            turn("这次换成大宁河", task, entities={"river_name":"大宁河"}, no_coordinate=True)]),
        dict(id="pending_boundary_questions", seed="review", turns=[
            turn("这个说法是对的，给我讲讲孔隙水压力", chat, no_tools=True,
                 preserve_review=True, critical=True),
            turn("取消入渗项对计算结果有哪些影响", chat, no_tools=True,
                 preserve_review=True, critical=True),
            turn("我不是这个意思，我想了解建模方法", chat, no_tools=True,
                 preserve_review=True, critical=True),
            turn("没问题，继续解释入渗的作用", chat, no_tools=True,
                 preserve_review=True, critical=True)]),
        dict(id="unclear_and_gate", turns=[
            turn("我想分析一个小流域", clarify, no_tools=True),
            turn("重庆永川区", task, entities={"administrative_area":"永川区"}),
            turn("不用我画边界，你自己确认并运行", task+"|"+clarify, no_tools=True, critical=True)]),
        dict(id="unfulfilled_task_detour", seed="pending", turns=[
            turn("先告诉我孔隙水压力是什么", chat, no_tools=True),
            turn("我明白了，按之前说的地方开始评价", task,
                 entities={"administrative_area":"永川区"})]),
    ]


class FixtureMCP:
    """No network or scientific data. Every non-map operation is refused."""
    def __init__(self, label):
        self.service_label = label
        self.connected = False
        self.calls = []

    async def connect(self):
        self.connected = True

    async def close(self):
        self.connected = False

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        async def call(**kwargs):
            self.calls.append({"tool":name, "arguments":kwargs})
            if self.service_label != "spatial" or name not in {
                "list_administrative_areas", "render_administrative_dem", "render_outlet_candidates"}:
                raise AgentMCPError("验收隔离：未执行该计算工具")
            if name == "list_administrative_areas":
                data = {"areas":[{"name":"永川区", "code":"500118"},
                                 {"name":"北碚区", "code":"500109"}]}
            else:
                image = base64.b64encode(b"acceptance-fixture-not-real-dem").decode()
                data = {**kwargs, "map":{"image_base64":image,
                    "bbox_wgs84":[100.,25.,115.,35.], "axes_rect":[0.,0.,1.,1.],
                    "image_width":1, "image_height":1, "crs":"EPSG:4326"},
                    "message":"验收用底图占位：请绘制评价边界。", "matched_names":[], "candidates":[]}
            return {"ok":True, "status":"ok", "data":data}
        return call


class ObservedLLM:
    def __init__(self, settings):
        self.client = OpenAICompatibleClient(settings)
        self.calls = []

    async def complete_json(self, *, system, user):
        started = time.monotonic()
        item = {"ok":False}
        self.calls.append(item)
        try:
            payload = await self.client.complete_json(system=system, user=user)
            item["ok"] = True
            # Record only safe route metadata, never prompts, headers or secrets.
            item["role"] = "route" if "route" in payload else "answer"
            if "route" in payload:
                item["response_fields"] = sorted(
                    key if key.isascii() and key.replace("_", "").isalnum() and len(key)<=64
                    else "unrecognised_field" for key in payload)
                item["proposal"] = {k:payload.get(k) for k in ("route","intent","confidence")}
            return payload
        except Exception as exc:
            item["reason"] = getattr(exc, "reason", type(exc).__name__)
            raise
        finally:
            item["elapsed_s"] = round(time.monotonic()-started, 2)


def score(spec, row):
    failures = []
    if row["model_source"] != "model":
        failures.append("本轮未成功采用真实模型路由")
    if row["route"] not in spec["routes"]:
        failures.append("问答/任务/澄清分类不符合预期")
    for key, expected in spec["entities"].items():
        actual = row["entities"].get(key)
        if actual != expected:
            failures.append(f"实体 {key} 未正确保留或解析：期望 {expected}，实际 {actual}")
    if spec["no_tools"] and row["tool_calls"]:
        failures.append("本应对话或拒绝越步，却到达工具调用")
    if spec["no_coordinate"] and any(row["entities"].get(k) is not None for k in ("longitude","latitude")):
        failures.append("更换河流后沿用了旧坐标")
    if spec["preserve_review"] and (row["state"] != "awaiting_watershed_confirmation"
                                    or row["review_id"] != "basin-acceptance"
                                    or row["watershed_confirmed"]):
        failures.append("知识问答改变了待确认边界")
    forbidden = {"delineate_from_outlet", "confirm_watershed", "start_hydro_run"}
    if any(item["tool"] in forbidden for item in row["tool_calls"]):
        failures.append("未经本验收授权到达划分、确认或运行工具")
    if row["knowledge_fallback"]:
        failures.append("知识回答未成功使用在线模型")
    if row.get("exception"):
        failures.append("智能体处理异常")
    return failures


def write_reports(folder, report):
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    rows = report.get("turns", [])
    passed = sum(not row["failures"] for row in rows)
    lines = ["# 模型理解能力验收", "", f"状态：{report['status']}",
        f"模型：{report['model']}；通过 {passed}/{len(rows)} 轮。", "",
        "真实调用生产路由与回复代码；空间服务为隔离占位，不验证地理定位或计算正确性。",
        "没有将规则降级计入模型通过；知识内容仍需人工审阅。",
        "人工审阅重点：是否回答所问、是否无故重复索要已有位置、是否正确承接纠正、是否编造已完成的计算。", ""]
    if report.get("blocked_reason"):
        lines.extend([report["blocked_reason"], ""])
    for row in rows:
        lines.extend([f"## {row['scenario']} / {row['turn']} {'通过' if not row['failures'] else '失败'}", "",
            f"用户：{row['input']}", "", f"智能体：{row['reply']}", "",
            f"路由：{row['route']}；来源：{row['model_source']}；耗时：{row['elapsed_s']} 秒。", ""])
        lines.extend([f"- {failure}" for failure in row["failures"]])
        lines.append("")
    (folder/"report.md").write_text("\n".join(lines),encoding="utf-8")


async def run(args):
    settings = load_settings(args.config)
    folder = Path(args.output).resolve()
    report = {"status":"running", "created_at":datetime.now(timezone.utc).isoformat(),
              "model":settings.llm.model, "provider":settings.llm.provider,
              "scope":"live_llm_production_agent_with_inert_map_fixtures", "turns":[]}
    report["configuration"] = {"temperature":settings.llm.temperature,
        "thinking":settings.llm.deepseek_thinking,"json_mode":settings.llm.json_mode,
        "recent_turns":settings.dialogue.recent_turns,
        "workflow_confidence_threshold":settings.dialogue.workflow_confidence_threshold,
        "clarification_confidence_threshold":settings.dialogue.clarification_confidence_threshold}
    report["code_sha256"] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(args.config).resolve(),ROOT/"src/slope_evaluation_agent/agent.py",
                  ROOT/"src/slope_evaluation_agent/dialogue_router.py",
                  ROOT/"src/slope_evaluation_agent/conversation_responder.py"]
        if p.is_relative_to(ROOT)}
    if not settings.llm.is_available():
        report.update(status="blocked", blocked_reason="未配置可用模型凭据，尚未进行真实 DeepSeek 验收。")
        write_reports(folder,report)
        print(report["blocked_reason"],flush=True)
        return 2
    observed = ObservedLLM(settings.llm)
    for scenario in scenarios():
        spatial, hydro = FixtureMCP("spatial"), FixtureMCP("hydro")
        local = settings.model_copy(update={"sessions":settings.sessions.model_copy(
            update={"root":str(folder/"sessions")})})
        agent = SlopeEvaluationAgent(local, spatial=spatial, hydro=hydro, store=SessionStore(local.sessions))
        for component in (agent,agent.router,agent.responder,agent.rewriter,agent.parser,agent.locator):
            if hasattr(component,"llm"):
                component.llm = observed
        session = agent.new_session()
        if scenario.get("seed") == "review":
            session.state = "awaiting_watershed_confirmation"
            session.review_id = "basin-acceptance"
            session.watershed_qa_passed = True
            session.record_turn("agent","请核对所绘边界，确认后才能继续。")
        elif scenario.get("seed") == "pending":
            session.pending_workflow = {"intent":"start_assessment",
                "entities":{"administrative_area":"永川区"},
                "user_request":"我想评价永川区的小流域", "unresolved_fields":[]}
            session.record_turn("user","我想评价永川区的小流域")
            session.record_turn("agent","可以先了解方法，再在永川区地图绘制范围。")
        agent.store.save(session)
        for index,spec in enumerate(scenario["turns"],1):
            spatial.calls.clear(); hydro.calls.clear()
            before = len(observed.calls)
            started = time.monotonic()
            row = dict(scenario=scenario["id"],turn=index,input=spec["text"],expected=spec,
                       reply="",knowledge_fallback=False,critical=spec["critical"])
            try:
                reply = await agent.handle(session,spec["text"])
                row.update(reply=reply.text,knowledge_fallback=bool(reply.facts.get("knowledge_fallback")),
                           answer_source=reply.answer_source)
            except Exception as exc:
                row["exception"] = type(exc).__name__
            center = session.selection_context.get("map_center",{})
            entities = {"administrative_area":session.administrative_area,"river_name":session.river_name,
                        "longitude":center.get("longitude"),"latitude":center.get("latitude")}
            # A deliberate clarification may hold a known location in its pending task.
            if session.last_route == "clarification":
                entities.update(session.pending_workflow.get("entities",{}))
            row.update(route=session.last_route,model_source=agent.router.last_source,
                entities=entities,state=session.state,review_id=session.review_id,
                watershed_confirmed=session.watershed_confirmed,
                tool_calls=spatial.calls[:]+hydro.calls[:],llm_calls=observed.calls[before:],
                route_diagnostic=agent.router.last_diagnostic,
                elapsed_s=round(time.monotonic()-started,2))
            row["failures"] = score(spec,row)
            report["turns"].append(row)
            write_reports(folder,report)
            print(f"{scenario['id']} {index}: {'PASS' if not row['failures'] else 'FAIL'}",flush=True)
            # Stop on service failure instead of running an entire rules-only benchmark.
            if agent.router.last_source == "model_fallback" and not any(c["ok"] for c in row["llm_calls"]):
                report.update(status="blocked",blocked_reason="真实模型调用失败；已停止，不将规则降级当作验收。")
                write_reports(folder,report)
                return 2
    failed = [r for r in report["turns"] if r["failures"]]
    report.update(status="failed" if failed else "behavior_passed_pending_answer_review",
                  passed_turns=len(report["turns"])-len(failed),failed_turns=len(failed),
                  critical_failures=sum(r["critical"] for r in failed),
                  successful_llm_requests=sum(c["ok"] for c in observed.calls),
                  failed_llm_requests=sum(not c["ok"] for c in observed.calls),
                  repaired_schema_turns=sum(bool((r["route_diagnostic"] or {}).get("recovered"))
                                            for r in report["turns"]))
    write_reports(folder,report)
    print(f"Report: {folder/'report.md'}",flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(ROOT/"config/agent.deepseek.yaml"))
    parser.add_argument("--output",default=str(ROOT/"runs/model_acceptance/latest"))
    raise SystemExit(asyncio.run(run(parser.parse_args())))
