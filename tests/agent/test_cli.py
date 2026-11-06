"""Phase 9 acceptance: the command-line product.

The interactive loop is driven by a scripted ``input``, so a whole conversation
can be exercised without a terminal. What matters here is that the CLI can be
closed mid-run, that a resumed session re-queries the run, and that nothing a
user sees is a traceback.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from slope_evaluation_agent import __version__
from slope_evaluation_agent.cli import describe_session, main, render_reply
from slope_evaluation_agent.config import AgentSettings, PROJECT_ROOT, SessionSettings
from slope_evaluation_agent.session_store import SessionStore
from slope_evaluation_agent.state_machine import WorkflowState


@pytest.fixture
def agent_settings(tmp_path) -> AgentSettings:
    return AgentSettings.model_validate(
        {"sessions": {"root": str(tmp_path / "sessions")}})


@pytest.fixture
def cli(agent_settings, monkeypatch):
    """Run the CLI with a fixed session root and scripted input."""
    def _run(argv: list[str], *, lines: list[str] | None = None):
        monkeypatch.setattr("slope_evaluation_agent.cli.load_settings",
                            lambda explicit=None: agent_settings)
        if lines is not None:
            feed = iter(lines)

            def scripted_input(prompt: str = "") -> str:
                # Exhausting the script ends the conversation, the way a closed
                # stdin would; returning "" forever would spin the loop.
                try:
                    return next(feed)
                except StopIteration:
                    raise EOFError from None

            monkeypatch.setattr("builtins.input", scripted_input)
        return main(argv)

    return _run


class TestSessionCommands:
    def test_new_session_prints_an_id(self, cli, capsys):
        assert cli(["new-session"]) == 0
        session_id = capsys.readouterr().out.strip()
        assert session_id.startswith("session-")

    def test_status_reports_state_and_pending_action(self, cli, agent_settings, capsys):
        store = SessionStore(agent_settings.sessions)
        session = store.create()
        assert cli(["status", "--session", session.session_id]) == 0
        out = capsys.readouterr().out
        assert session.session_id in out
        assert "状态" in out
        assert "待办" in out

    def test_status_json_parses(self, cli, agent_settings, capsys):
        store = SessionStore(agent_settings.sessions)
        session = store.create()
        assert cli(["status", "--session", session.session_id, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["session_id"] == session.session_id
        assert payload["state"] == "new"

    def test_status_without_a_session_is_a_clean_message(self, cli, capsys):
        assert cli(["status"]) == 2
        assert "没有找到" in capsys.readouterr().err

    def test_status_of_a_missing_session_reports_without_a_traceback(self, cli, capsys):
        assert cli(["status", "--session", "session-ffffffffffff"]) == 2
        captured = capsys.readouterr()
        assert "Traceback" not in captured.err

    def test_inspect_dumps_events_and_evidence(self, cli, agent_settings, capsys):
        store = SessionStore(agent_settings.sessions)
        session = store.create()
        store.save_evidence(session.session_id, {"factors": {}})
        assert cli(["inspect", "--session", session.session_id, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["session"]["session_id"] == session.session_id
        assert payload["events"]
        assert payload["evidence"] == {"factors": {}}

    def test_cancel_refuses_without_a_run(self, cli, agent_settings, capsys):
        store = SessionStore(agent_settings.sessions)
        session = store.create()
        assert cli(["cancel", "--session", session.session_id]) == 0
        assert "没有已提交的运行任务" in capsys.readouterr().out

    def test_a_broken_config_is_reported_without_a_traceback(self, tmp_path, capsys):
        broken = tmp_path / "broken.yaml"
        broken.write_text("language: [not, scalar]\n", encoding="utf-8")
        assert main(["--config", str(broken), "status"]) == 2
        assert "Traceback" not in capsys.readouterr().err

    def test_no_command_prints_help(self, cli, capsys):
        assert cli([]) == 2
        assert "usage" in capsys.readouterr().out


class TestChatLoop:
    def test_help_and_quit(self, cli, capsys):
        assert cli(["chat"], lines=[":help", ":quit"]) == 0
        out = capsys.readouterr().out
        assert "可用输入" in out
        assert "已退出" in out

    def test_end_of_input_exits_cleanly(self, cli, capsys):
        assert cli(["chat"], lines=[]) == 0
        assert "已退出" in capsys.readouterr().out

    def test_status_command_inside_the_loop(self, cli, capsys):
        assert cli(["chat"], lines=[":status", ":quit"]) == 0
        assert "状态" in capsys.readouterr().out

    def test_blank_lines_are_ignored(self, cli, capsys):
        assert cli(["chat"], lines=["", "  ", ":quit"]) == 0

    def test_resuming_a_missing_session_is_a_clean_error(self, cli, capsys):
        assert cli(["resume", "--session", "session-ffffffffffff"]) == 2
        assert "找不到会话" in capsys.readouterr().err


class TestLongRunDoesNotBlock:
    def test_the_loop_exits_while_a_job_is_still_running(self, cli, agent_settings,
                                                         capsys, monkeypatch):
        """Plan section 12 phase 9: the CLI may be closed mid-run."""
        from helpers import Calls, hydro_handlers, hydro_server, spatial_handlers, spatial_server
        from slope_evaluation_agent.agent import SlopeEvaluationAgent
        from slope_evaluation_agent.intent import IntentParser
        from stubs import StubHydroClient, StubSpatialClient

        calls = Calls()
        store = SessionStore(agent_settings.sessions)
        session = store.create()

        agent = SlopeEvaluationAgent(
            agent_settings,
            spatial=StubSpatialClient(spatial_server(spatial_handlers(calls))),
            hydro=StubHydroClient(hydro_server(hydro_handlers(calls))),
            store=store, parser=IntentParser(),
        )
        monkeypatch.setattr("slope_evaluation_agent.cli.SlopeEvaluationAgent",
                            lambda *a, **k: agent)

        # This test exercises closing the CLI after starting an already planned
        # run. Map selection now requires a browser polygon submission.
        session.state=WorkflowState.HYDRO_PLANNED.value
        session.review_id="basin-abcdef123456"
        session.watershed_confirmed=True
        session.watershed_qa_passed=True
        session.hydro_input_id="hydro-input-abcdef123456"
        session.hydro_input_validated=True
        session.hydro_planned=True
        store.save(session)
        lines = ["开始运行", ":quit"]
        assert cli(["chat", "--session", session.session_id], lines=lines) == 0

        reloaded = store.load(session.session_id)
        assert reloaded.hydro_run_id == "hydro-run-abcdef123456"
        assert reloaded.state == WorkflowState.HYDRO_RUNNING.value

    def test_resuming_requeries_the_run(self, cli, agent_settings, capsys, monkeypatch):
        """A resumed session asks the service; it does not trust the file."""
        from helpers import Calls, hydro_handlers, hydro_server
        from slope_evaluation_agent.agent import SlopeEvaluationAgent
        from slope_evaluation_agent.intent import IntentParser
        from stubs import StubHydroClient
        from helpers import spatial_handlers, spatial_server
        from stubs import StubSpatialClient
        from helpers import run_status_payload, envelope

        calls = Calls()
        store = SessionStore(agent_settings.sessions)
        session = store.create()
        session.review_id = "basin-abcdef123456"
        session.hydro_run_id = "hydro-run-abcdef123456"
        session.hydro_run_status = "running"
        session.state = WorkflowState.HYDRO_RUNNING.value
        store.save(session)

        agent = SlopeEvaluationAgent(
            agent_settings,
            spatial=StubSpatialClient(spatial_server(spatial_handlers(calls))),
            hydro=StubHydroClient(hydro_server(hydro_handlers(
                calls,
                get_hydro_run_status=lambda: envelope(
                    data=run_status_payload(status="succeeded"))))),
            store=store, parser=IntentParser(),
        )
        monkeypatch.setattr("slope_evaluation_agent.cli.SlopeEvaluationAgent",
                            lambda *a, **k: agent)

        assert cli(["resume", "--session", session.session_id], lines=[":quit"]) == 0
        assert store.load(session.session_id).state == WorkflowState.HYDRO_SUCCEEDED.value


class TestRendering:
    def test_an_image_is_saved_and_its_path_printed(self, tmp_path, capsys):
        import base64

        from helpers import PNG_BYTES
        from slope_evaluation_agent.agent import AgentReply
        from slope_evaluation_agent.models import AgentSession

        settings = AgentSettings.model_validate(
            {"sessions": {"root": str(tmp_path / "sessions")}})
        session = AgentSession.model_validate(
            {"session_id": "session-abcdef123456", "state": "new"})
        reply = AgentReply(text="图件已生成。", session=session,
                           images=[{"mime_type": "image/png",
                                    "data": base64.b64encode(PNG_BYTES).decode()}])
        assert render_reply(reply, settings) == 1
        out = capsys.readouterr().out
        assert "图件已保存" in out
        saved = list((tmp_path / "images").glob("*.png"))
        assert saved and saved[0].read_bytes() == PNG_BYTES

    def test_a_reply_without_images_prints_only_text(self, tmp_path, capsys):
        from slope_evaluation_agent.agent import AgentReply
        from slope_evaluation_agent.models import AgentSession

        settings = AgentSettings.model_validate(
            {"sessions": {"root": str(tmp_path / "sessions")}})
        session = AgentSession.model_validate(
            {"session_id": "session-abcdef123456", "state": "new"})
        render_reply(AgentReply(text="只有文字。", session=session), settings)
        assert capsys.readouterr().out.strip() == "只有文字。"

    def test_the_description_names_the_pending_action(self, tmp_path):
        from slope_evaluation_agent.session_store import SessionStore

        settings = SessionSettings(root=str(tmp_path / "sessions"))
        store = SessionStore(settings)
        session = store.create()
        description = describe_session(store, session.session_id)
        assert "绘制评价范围" in description
        assert "河流名称" in description


class TestEntryPoint:
    def test_module_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "slope_evaluation_agent.cli", "--help"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT))
        assert result.returncode == 0
        for command in ("chat", "new-session", "resume", "status", "cancel", "inspect"):
            assert command in result.stdout

    def test_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "slope_evaluation_agent.cli", "--version"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT))
        assert result.returncode == 0
        assert __version__ in result.stdout

    def test_a_failing_command_does_not_print_a_traceback(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "slope_evaluation_agent.cli",
             "--config", str(tmp_path / "missing.yaml"), "status"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT))
        assert result.returncode == 2
        assert "Traceback" not in result.stderr
        assert "配置错误" in result.stderr
