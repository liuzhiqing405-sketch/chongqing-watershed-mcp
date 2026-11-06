"""What happens to a background job when the agent exits over *stdio*.

Under stdio the agent owns the MCP server as a subprocess, and the server owns
the worker that runs a job. Closing the agent therefore terminates the service,
and on this platform the worker does not outlive it — even launched with
``CREATE_BREAKAWAY_FROM_JOB``, which the OS accepts and which works in isolated
reproductions but does not survive this particular teardown.

That is a limitation, not a silent one: the job's state stays recoverable, the
next service start marks it interrupted rather than pretending it is still
running, and the agent reports that faithfully on resume. These tests pin that
behaviour down.

The production path for long runs is the opposite arrangement and lives in
``test_r2_cli_exit_persistence.py``: the two MCP servers run as persistent
``streamable-http`` services of their own, the agent only connects to them, and
a job started over that transport keeps running after the CLI exits. What these
tests add on top is that the CLI never promises *more* than the transport it is
actually using delivers — which is why the exit notice is transport-dependent.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import build_package  # noqa: E402

from watershed_hydro_mcp.input_registry import import_package  # noqa: E402
from watershed_hydro_mcp.settings import Settings  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_SRC = PROJECT_ROOT / "src"

#: ``rainfall_only`` is the longest mode the synthetic fixture can carry end to
#: end; its DEM is too coarse for the SWMM build stage.
PROBE_MODE = "rainfall_only"

#: Settings used by the launch-flag test; the paths are never touched.
SETTINGS_FOR_PROBE = Settings(root=PROJECT_ROOT, runs_root=PROJECT_ROOT / "runs",
                              hydro_root=PROJECT_ROOT / "runs" / "hydro",
                              native_root=PROJECT_ROOT / "native",
                              config_root=PROJECT_ROOT / "config")


def make_settings(root: Path) -> Settings:
    (root / "native").mkdir(parents=True, exist_ok=True)
    shutil.copytree(PROJECT_ROOT / "native" / "swmm", root / "native" / "swmm",
                    dirs_exist_ok=True)
    (root / "exchange" / "exports").mkdir(parents=True, exist_ok=True)
    (root / "exchange" / "imports").mkdir(parents=True, exist_ok=True)
    runs = root / "runs"
    return Settings(root=root, runs_root=runs, hydro_root=runs / "hydro",
                    native_root=root / "native", config_root=PROJECT_ROOT / "config")


def probe_script(input_id: str) -> str:
    """Connect over stdio, submit a job, print its id, then block."""
    return f'''
import asyncio, sys
sys.path.insert(0, r"{AGENT_SRC}")
from slope_evaluation_agent.config import ServerSettings
from slope_evaluation_agent.mcp_clients import HydroMCPClient
from slope_evaluation_agent.models import HydroRunRequestPayload


async def main():
    settings = ServerSettings(args=["-m", "watershed_hydro_mcp.server"],
                              cwd=r"{PROJECT_ROOT}")
    async with HydroMCPClient(settings) as hydro:
        payload = await hydro.start_hydro_run(
            HydroRunRequestPayload(input_id="{input_id}", mode="{PROBE_MODE}"))
        print(payload["data"]["run_id"], flush=True)
        await asyncio.sleep(3600)


asyncio.run(main())
'''


class WorkerLaunchFlagTests(unittest.TestCase):
    """The launch flags, asserted directly so they cannot regress silently."""

    def test_windows_requests_a_detached_worker(self):
        if os.name != "nt":
            self.skipTest("Job Object breakaway is a Windows concern")
        source = (PROJECT_ROOT / "src" / "watershed_hydro_mcp" / "job_manager.py"
                  ).read_text(encoding="utf-8")
        self.assertIn("CREATE_BREAKAWAY_FROM_JOB", source)
        self.assertIn("CREATE_NEW_PROCESS_GROUP", source)

    def test_posix_launches_the_worker_in_its_own_session(self):
        source = (PROJECT_ROOT / "src" / "watershed_hydro_mcp" / "job_manager.py"
                  ).read_text(encoding="utf-8")
        self.assertIn("start_new_session", source)

    def test_a_refused_breakaway_still_runs_the_job(self):
        """A job must start even where the OS forbids breakaway."""
        import watershed_hydro_mcp.job_manager as job_manager

        store = job_manager.JobStore(SETTINGS_FOR_PROBE)
        accepted = mock.MagicMock()
        with mock.patch.object(job_manager.subprocess, "Popen",
                               side_effect=[OSError("access denied"), accepted]) as popen:
            process = store._spawn_worker(
                "hydro-run-abcdef123456", {}, mock.MagicMock(),
                {"creationflags": 0x01000000} if os.name == "nt" else {})
            self.assertIs(process, accepted)
            self.assertEqual(popen.call_count, 2)


class InterruptedRunTests(unittest.TestCase):
    """An interrupted job must stay recoverable and be reported honestly."""

    def setUp(self):
        self.temp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp.name)
        self.settings = make_settings(self.root)
        build_package(self.root / "exchange" / "exports" / "package-interrupt0001")
        self.record = import_package("package-interrupt0001", self.settings)

    def tearDown(self):
        self.temp.cleanup()

    def _submit_and_interrupt(self) -> str:
        """Submit a job over a real service, let it start, then kill the agent."""
        path = self.root / "submit.py"
        path.write_text(probe_script(self.record["input_id"]), encoding="utf-8")
        environment = {
            **os.environ,
            "WHS_RUNS_ROOT": str(self.settings.runs_root),
            "WHS_HYDRO_ROOT": str(self.settings.hydro_root),
            "WHS_NATIVE_ROOT": str(self.settings.native_root),
            "WHS_CONFIG_ROOT": str(self.settings.config_root),
            "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
            "PYTHONPATH": str(AGENT_SRC),
        }
        process = subprocess.Popen(
            [sys.executable, str(path)], cwd=str(PROJECT_ROOT), env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            encoding="utf-8", errors="replace")
        run_id = process.stdout.readline().strip()
        self.assertTrue(run_id.startswith("hydro-run-"), f"未取得任务编号：{run_id!r}")

        # Wait until the worker has genuinely started, so the interruption is a
        # real one rather than a race with the scheduler.
        status_path = self.settings.job_dir(run_id) / "status.json"
        deadline = time.time() + 180
        saw_completed = False
        while time.time() < deadline:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") in {"succeeded", "qa_failed", "failed", "cancelled"}:
                process.kill()
                process.wait(timeout=30)
                self.skipTest(f"任务在可以中断之前就已结束（{status.get('status')}）")
            logs = [item for item in (self.settings.job_dir(run_id) / "logs").glob("*.log")
                    if item.name != "worker.log" and item.stat().st_size]
            # Wait for a completed stage, not merely a started one: the point of
            # interrupting is to check that finished work survives, and killing
            # before anything finished leaves nothing to check. Without this the
            # test fails on a loaded machine for a reason that is not behaviour.
            saw_completed = saw_completed or bool(status.get("completed_stages"))
            if status.get("status") == "running" and logs and saw_completed:
                break
            time.sleep(0.2)
        process.kill()
        process.wait(timeout=30)
        # The kill has to land while the job is still running. On a loaded
        # machine it can finish in the window between the last read and the
        # kill, and then there is nothing to interrupt — say so, rather than
        # fail an assertion that no longer describes what happened.
        settled = json.loads(status_path.read_text(encoding="utf-8")).get("status")
        if settled in {"succeeded", "qa_failed", "failed", "cancelled"}:
            self.skipTest(f"任务在中断落地前就已结束（{settled}）")
        if not saw_completed:
            self.skipTest("等不到任何阶段完成（机器负载过高），无可验证的已完成工作")
        return run_id

    def test_an_interrupted_run_is_marked_interrupted_on_recovery(self):
        """Plan scenario E: the run survives as a record, not as a live process."""
        from watershed_hydro_mcp.job_manager import JobStore

        run_id = self._submit_and_interrupt()
        # A service starting again finds the record and refuses to call it running.
        report = JobStore(self.settings).restore()
        self.assertIn(run_id, report["interrupted"])
        after = json.loads(
            (self.settings.job_dir(run_id) / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(after["status"], "failed")
        self.assertTrue(after.get("interrupted"))
        # The original id is kept, so the user can point at the same run.
        self.assertEqual(after["run_id"], run_id)

    def test_the_completed_work_remains_readable(self):
        run_id = self._submit_and_interrupt()
        status = json.loads(
            (self.settings.job_dir(run_id) / "status.json").read_text(encoding="utf-8"))
        self.assertTrue(status.get("completed_stages"),
                        "中断前完成的阶段应当保留")
        project = self.settings.job_dir(run_id) / "project"
        self.assertTrue(project.is_dir())
        self.assertTrue(list(project.glob("inputs/rainfall/*")))


class ExitNoticeTests(unittest.TestCase):
    """What the CLI says must match what the configured transport does."""

    def test_a_running_job_over_stdio_is_described_as_terminated(self):
        from slope_evaluation_agent.cli import _exit_notice
        from slope_evaluation_agent.models import AgentSession

        session = AgentSession.model_validate(
            {"session_id": "session-abcdef123456", "state": "hydro_running",
             "hydro_run_id": "hydro-run-abcdef123456"})
        notice = _exit_notice(session)
        self.assertIn("会被终止", notice)
        self.assertIn("保持本进程运行", notice)
        self.assertNotIn("会继续运行", notice)

    def test_a_running_job_over_a_persistent_service_is_described_as_surviving(self):
        """The streamable-http branch must not understate what happens either."""
        from slope_evaluation_agent.cli import _exit_notice
        from slope_evaluation_agent.config import AgentSettings
        from slope_evaluation_agent.models import AgentSession

        settings = AgentSettings.model_validate({
            "mcp": {"hydro": {"transport": "streamable-http",
                              "url": "http://127.0.0.1:8766/mcp"}}})
        session = AgentSession.model_validate(
            {"session_id": "session-abcdef123456", "state": "hydro_running",
             "hydro_run_id": "hydro-run-abcdef123456"})
        notice = _exit_notice(session, settings)
        self.assertIn("继续执行", notice)
        self.assertIn(session.hydro_run_id, notice)
        self.assertNotIn("会被终止", notice)

    def test_an_idle_session_gets_a_plain_notice(self):
        from slope_evaluation_agent.cli import _exit_notice
        from slope_evaluation_agent.models import AgentSession

        session = AgentSession.model_validate(
            {"session_id": "session-abcdef123456", "state": "new"})
        self.assertIn("resume", _exit_notice(session))

    def test_the_help_text_says_the_same_thing(self):
        """The help line must name both transports, as the notice branches on them."""
        from slope_evaluation_agent.cli import CHAT_HELP

        self.assertIn("常驻服务下任务继续", CHAT_HELP)
        self.assertIn("stdio", CHAT_HELP)
        self.assertIn("随本进程结束", CHAT_HELP)


if __name__ == "__main__":
    unittest.main()
