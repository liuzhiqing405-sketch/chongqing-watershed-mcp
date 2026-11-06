"""Command-line entry point for the slope evaluation agent.

Commands:

    slope-evaluation-agent config                 resolved configuration, no secrets
    slope-evaluation-agent new-session            create a session and print its id
    slope-evaluation-agent chat                   a conversation in a new session
    slope-evaluation-agent chat --session <id>    continue an existing conversation
    slope-evaluation-agent resume --session <id>  alias of chat --session
    slope-evaluation-agent status --session <id>  state, what is pending, last error
    slope-evaluation-agent cancel --session <id>  cancel the session's run
    slope-evaluation-agent inspect --session <id> dump session, events, evidence

The CLI holds one connection to each MCP server for the duration of a
conversation, so a long run does not block it: the agent submits the job and
returns, and the CLI can be closed at any point. Reopening with ``resume``
re-queries the service rather than trusting what was stored.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from typing import Any, Sequence

from . import DISCLAIMER_ZH, SCOPE_STATEMENT, __version__
from .agent import AgentReply, SlopeEvaluationAgent
from .config import AgentSettings, configure_logging, load_settings
from .session_store import SessionError, SessionStore
from .state_machine import pending_action


PROGRAM = "slope-evaluation-agent"

HELP_EPILOGUE = f"{DISCLAIMER_ZH}\n\n对话中输入 :help 查看可用指令，:quit 退出。"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            f"{SCOPE_STATEMENT}。\n\n"
            "通过 MCP 编排 chongqing-spatial-mcp 与 watershed-hydro-stability-mcp："
            "自然语言请求 -> 出口候选 -> 流域划分 -> 人工确认 -> 降雨与输入包 -> "
            "水文任务 -> 结果解释。所有数值均来自 MCP 返回，本程序不做科学计算。"
        ),
        epilog=HELP_EPILOGUE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"{PROGRAM} {__version__}")
    parser.add_argument("--config", metavar="PATH",
                        help="agent 配置文件；默认 config/agent.default.yaml，缺失则用内置默认值")
    parser.add_argument("--verbose", action="store_true", help="输出调试日志")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    show = sub.add_parser("config", help="显示解析后的配置（不含密钥）")
    show.add_argument("--json", action="store_true", help="以 JSON 输出")

    sub.add_parser("new-session", help="新建会话并输出编号")

    chat = sub.add_parser("chat", help="开始或继续对话")
    chat.add_argument("--session", metavar="ID", help="继续指定会话")

    resume = sub.add_parser("resume", help="继续一个已有会话")
    resume.add_argument("--session", metavar="ID", required=True)

    status = sub.add_parser("status", help="显示会话状态与待办")
    status.add_argument("--session", metavar="ID")
    status.add_argument("--json", action="store_true")

    cancel = sub.add_parser("cancel", help="取消该会话绑定的运行任务")
    cancel.add_argument("--session", metavar="ID")

    inspect = sub.add_parser("inspect", help="导出会话、事件、证据与评价")
    inspect.add_argument("--session", metavar="ID")
    inspect.add_argument("--json", action="store_true")
    inspect.add_argument("--events", type=int, default=20, help="显示最近多少条事件")
    return parser


# ------------------------------------------------------------------ rendering


def render_reply(reply: AgentReply, settings: AgentSettings) -> int:
    """Print a reply, saving any figures beside the session. Returns the count."""
    if reply.images:
        directory = settings.sessions.resolved_root().parent / "images"
        directory.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(reply.images, start=1):
            suffix = ".png" if image.get("mime_type", "").endswith("png") else ".bin"
            path = directory / f"{reply.session.session_id}-{index}{suffix}"
            path.write_bytes(base64.b64decode(image.get("data") or ""))
            print(f"[图件已保存：{path}]")
    print(reply.text)
    return len(reply.images)


async def save_download(agent, session, reply: AgentReply,
                        settings: AgentSettings) -> str | None:
    """Save the file a download reply points at, beside the session.

    The command line has no button to click, so the deterministic route has to
    deliver the file itself. The bytes are streamed straight to disk in the same
    chunks the browser would receive; nothing is held in memory whole. A failure
    is reported and never aborts the conversation.
    """
    from .models import ArtifactDescriptor

    payload = (reply.facts or {}).get("artifact")
    if not isinstance(payload, dict) or not payload.get("artifact_id"):
        return None
    try:
        descriptor = ArtifactDescriptor.model_validate(payload)
        download = await agent.download_artifact(session, descriptor.artifact_id)
    except Exception as exc:  # noqa: BLE001 - a failed download is not a failed turn
        print(f"[成果未能下载：{exc}]")
        return None
    directory = settings.sessions.resolved_root().parent / "downloads"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session.session_id}-{download.filename}"
    try:
        with path.open("wb") as stream:
            async for chunk in download.chunks:
                stream.write(chunk)
    except OSError as exc:
        print(f"[成果写入失败：{exc}]")
        return None
    qualifier = "（QA 未通过，仅供诊断）" if download.diagnostic_only else ""
    print(f"[成果已保存{qualifier}：{path}]")
    return str(path)


def describe_session(store: SessionStore, session_id: str) -> str:
    session = store.load(session_id)
    lines = [
        f"会话编号：{session.session_id}",
        f"状态：{session.state}",
        f"待办：{session.pending_action()}",
        f"更新时间：{session.updated_at}",
    ]
    for label, value in (("审核编号", session.review_id),
                         ("降雨包", session.rainfall_package_id),
                         ("输入包", session.model_input_package_id),
                         ("水文输入", session.hydro_input_id),
                         ("水文任务", session.hydro_run_id),
                         ("任务状态", session.hydro_run_status),
                         ("质量门槛", session.qa_passed)):
        if value is not None:
            lines.append(f"{label}：{value}")
    if session.last_error:
        lines.append(f"最近错误：{session.last_error.get('message')}"
                     f"（{session.last_error.get('code')}）")
    return "\n".join(lines)


def dump_session(store: SessionStore, session_id: str, *, event_limit: int) -> dict[str, Any]:
    return {
        "session": store.load(session_id).model_dump(mode="json"),
        "events": store.read_events(session_id, limit=event_limit),
        "evidence": store.load_evidence(session_id),
        "assessment": store.load_assessment(session_id),
    }


# -------------------------------------------------------------------- actions


async def _new_session(settings: AgentSettings) -> int:
    store = SessionStore(settings.sessions)
    session = store.create(language=settings.language)
    print(session.session_id)
    return 0


async def _status(settings: AgentSettings, session_id: str | None, as_json: bool) -> int:
    store = SessionStore(settings.sessions)
    identifier = session_id or store.latest_session_id()
    if not identifier:
        print("没有找到任何会话。用 new-session 新建一个。", file=sys.stderr)
        return 2
    if as_json:
        print(json.dumps(store.load(identifier).model_dump(mode="json"),
                         ensure_ascii=False, indent=2))
        return 0
    print(describe_session(store, identifier))
    return 0


async def _inspect(settings: AgentSettings, session_id: str | None, as_json: bool,
                   event_limit: int) -> int:
    store = SessionStore(settings.sessions)
    identifier = session_id or store.latest_session_id()
    if not identifier:
        print("没有找到任何会话。", file=sys.stderr)
        return 2
    payload = dump_session(store, identifier, event_limit=event_limit)
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(describe_session(store, identifier))
    print(f"\n最近 {len(payload['events'])} 条事件：")
    for item in payload["events"]:
        detail = item.get("tool") or item.get("intent") or item.get("to") or ""
        print(f"  {item.get('at', '')[:19]}  {item.get('event')}  {detail}")
    if payload["assessment"]:
        print("\n评价状态：", payload["assessment"].get("assessment_status"))
    return 0


async def _cancel(settings: AgentSettings, session_id: str | None) -> int:
    store = SessionStore(settings.sessions)
    identifier = session_id or store.latest_session_id()
    if not identifier:
        print("没有找到任何会话。", file=sys.stderr)
        return 2
    session = store.load(identifier)
    if not session.hydro_run_id:
        print("该会话没有已提交的运行任务。")
        return 0
    agent = SlopeEvaluationAgent(settings, store=store)
    async with agent:
        reply = await agent.handle(session, "取消")
    render_reply(reply, settings)
    return 0 if reply.ok else 1


# ------------------------------------------------------------------ chat loop


def _exit_notice(session, settings: AgentSettings | None = None) -> str:
    """What actually happens to the session's run when the CLI exits.

    With a persistent service the job keeps running and the message says so.
    With an stdio server the service is this process's child, so the job ends
    with it — and the message must not claim otherwise.
    """
    if session.state != "hydro_running" or not session.hydro_run_id:
        return f"已退出。可用 resume --session {session.session_id} 继续会话。"
    persistent = bool(settings and settings.mcp.hydro.transport == "streamable-http")
    if persistent:
        return (
            f"已退出。任务 {session.hydro_run_id} 由常驻服务继续执行。\n"
            f"稍后可用 resume --session {session.session_id} 查询同一任务。"
        )
    return (
        f"已退出。任务 {session.hydro_run_id} 仍在运行，"
        "但本进程退出后它会被终止。\n"
        "请保持本进程运行直到任务结束，或改用常驻服务"
        "（服务端 --transport streamable-http，客户端 mcp.hydro.transport）。"
    )


CHAT_HELP = """可用输入：
  :help          显示本帮助
  :status        显示当前会话状态与待办
  :quit / :exit  退出（常驻服务下任务继续；stdio 下随本进程结束）
  其它任何文字   作为对话内容交给 Agent

示例：
  分析大宁河 2024-07-30 全天
  东经 108.03、北纬 30.44
  第 1 个
  确认该边界
  用历史最大日
  开始运行
  状态
  结果怎么样
  看最小安全系数的图
"""


async def _chat(settings: AgentSettings, session_id: str | None,
                logger) -> int:
    store = SessionStore(settings.sessions)
    if session_id:
        if not store.exists(session_id):
            print(f"找不到会话：{session_id}", file=sys.stderr)
            return 2
        session = store.load(session_id)
        print(f"已恢复会话 {session.session_id}（状态：{session.state}）")
    else:
        session = store.create(language=settings.language)
        print(f"新会话：{session.session_id}")

    agent = SlopeEvaluationAgent(settings, store=store)
    async with agent:
        # A resumed session may have a job that finished while the CLI was away.
        if session_id:
            try:
                notice = await agent.reconcile(session)
            except Exception as exc:  # noqa: BLE001 - report, never traceback
                notice = None
                logger.debug("恢复检查失败：%s", exc)
            if notice is not None:
                render_reply(notice, settings)
        print(f"当前待办：{pending_action(session.state)}")
        print("输入 :help 查看指令，:quit 退出。\n")

        while True:
            try:
                text = input("你> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n" + _exit_notice(session, settings))
                return 0
            if not text:
                continue
            if text in {":quit", ":exit"}:
                print(_exit_notice(session, settings))
                return 0
            if text == ":help":
                print(CHAT_HELP)
                continue
            if text == ":status":
                print(describe_session(store, session.session_id))
                continue
            try:
                reply = await agent.handle(session, text)
            except Exception as exc:  # noqa: BLE001 - never show a traceback
                logger.debug("处理输入失败", exc_info=True)
                print(f"处理这条消息时出现问题：{exc}\n"
                      f"当前待办：{pending_action(session.state)}")
                continue
            render_reply(reply, settings)
            await save_download(agent, session, reply, settings)
            print(f"\n[状态：{session.state}；待办：{pending_action(session.state)}]\n")


# ----------------------------------------------------------------------- main


def _ensure_utf8_streams() -> None:
    """Write user-facing text as UTF-8 regardless of the console code page.

    Without this, Chinese messages are mangled on a Windows console whose page
    is not UTF-8 — which is every default console here.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # a replaced or closed stream
                pass


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_utf8_streams()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2

    try:
        settings = load_settings(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2
    logger = configure_logging(settings, verbose=args.verbose)

    try:
        if args.command == "config":
            return _cmd_config(settings, args)
        if args.command == "new-session":
            return asyncio.run(_new_session(settings))
        if args.command == "status":
            return asyncio.run(_status(settings, getattr(args, "session", None),
                                       getattr(args, "json", False)))
        if args.command == "inspect":
            return asyncio.run(_inspect(settings, getattr(args, "session", None),
                                        getattr(args, "json", False),
                                        getattr(args, "events", 20)))
        if args.command == "cancel":
            return asyncio.run(_cancel(settings, getattr(args, "session", None)))
        if args.command in {"chat", "resume"}:
            return asyncio.run(_chat(settings, getattr(args, "session", None), logger))
    except SessionError as exc:
        print(f"会话错误：{exc.user_message()}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - a user-facing message, not a traceback
        logger.debug("命令失败", exc_info=True)
        print(f"执行失败：{exc}", file=sys.stderr)
        print("如需详细堆栈，请加 --verbose 重试。", file=sys.stderr)
        return 1

    parser.print_help()
    return 2


def _cmd_config(settings: AgentSettings, args: argparse.Namespace) -> int:
    payload = settings.redacted()
    if getattr(args, "json", False):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"配置文件        : {settings.source}")
    print(f"语言            : {settings.language}")
    print(f"空间 MCP        : {' '.join([settings.mcp.spatial.resolved_command()] + settings.mcp.spatial.args)}")
    print(f"  工作目录      : {settings.mcp.spatial.resolved_cwd()}")
    print(f"水文 MCP        : {' '.join([settings.mcp.hydro.resolved_command()] + settings.mcp.hydro.args)}")
    print(f"  工作目录      : {settings.mcp.hydro.resolved_cwd()}")
    print(f"会话目录        : {settings.sessions.resolved_root()}")
    print(f"轮询            : 起始 {settings.polling.initial_interval_s}s，"
          f"上限 {settings.polling.maximum_interval_s}s，"
          f"退避 {settings.polling.backoff_factor}x")
    print(f"语言模型        : {'可用' if settings.llm.is_available() else '不可用'}"
          f"（{settings.llm.unavailable_reason() or '已配置'}）")
    print(f"API Key         : {'已设置' if payload['llm']['api_key'] == 'set' else '未设置'}")
    print()
    print("说明：" + DISCLAIMER_ZH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
