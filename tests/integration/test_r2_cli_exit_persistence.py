"""R0 regression test for R2: a hydro job must outlive the CLI.

Process-level, not stubbed. A real hydro MCP service is started over
``streamable-http``, a real agent process submits a real job through it and then
exits, and the test proves — from process liveness and from the job's own status
file — that the run kept going. A second agent process then recovers the session
and queries the same ``hydro_run_id`` without submitting anything.

This is the acceptance the plan calls blocking, so it is marked ``windows_process``
and must not be skipped on a Windows acceptance machine.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import build_package  # noqa: E402

from watershed_hydro_mcp.input_registry import import_package  # noqa: E402
from watershed_hydro_mcp.settings import Settings  # noqa: E402


pytestmark = [pytest.mark.real_mcp, pytest.mark.windows_process]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_SRC = PROJECT_ROOT / "src"

#: The mode used for the probe. The synthetic fixture cannot carry the full
#: pipeline (its DEM is too coarse for the SWMM build stage), so the probe runs
#: the longest mode it can take to completion.
PROBE_MODE = "rainfall_only"

#: How long a job is allowed to take once the CLI has gone.
JOB_TIMEOUT_S = 900
#: How long to wait for the service to answer a health check.
SERVICE_BOOT_TIMEOUT_S = 60


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_settings(root: Path) -> Settings:
    (root / "native").mkdir(parents=True, exist_ok=True)
    shutil.copytree(PROJECT_ROOT / "native" / "swmm", root / "native" / "swmm",
                    dirs_exist_ok=True)
    (root / "exchange" / "exports").mkdir(parents=True, exist_ok=True)
    (root / "exchange" / "imports").mkdir(parents=True, exist_ok=True)
    runs = root / "runs"
    return Settings(root=root, runs_root=runs, hydro_root=runs / "hydro",
                    native_root=root / "native", config_root=PROJECT_ROOT / "config")


def service_environment(settings: Settings) -> dict[str, str]:
    return {
        **os.environ,
        "WHS_RUNS_ROOT": str(settings.runs_root),
        "WHS_HYDRO_ROOT": str(settings.hydro_root),
        "WHS_NATIVE_ROOT": str(settings.native_root),
        "WHS_CONFIG_ROOT": str(settings.config_root),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "PYTHONPATH": str(AGENT_SRC),
        # The fixture is intentionally small. Hold the worker until the test has
        # observed it and closed the submitting Agent, then release it.
        "WHS_TEST_HOLD_WORKER": "1",
    }


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(
                    handle, ctypes.byref(exit_code)):
                return False
            return int(exit_code.value) == 259
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def worker_pid_from_status(status_path: Path) -> int | None:
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    structured = payload.get("worker_pid")
    if structured:
        return int(structured)
    match = re.search(r"pid (\d+)", str(payload.get("summary") or ""))
    return int(match.group(1)) if match else None


class PersistentService:
    """A hydro MCP service running as its own process, over HTTP."""

    def __init__(self, settings: Settings, port: int) -> None:
        self.settings = settings
        self.port = port
        self.url = f"http://127.0.0.1:{port}/mcp"
        self.log_path = settings.root / "hydro_service.log"
        self.process: subprocess.Popen | None = None

    def start(self) -> "PersistentService":
        handle = self.log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "watershed_hydro_mcp.server",
             "--transport", "streamable-http", "--host", "127.0.0.1",
             "--port", str(self.port)],
            cwd=str(PROJECT_ROOT), env=service_environment(self.settings),
            stdout=handle, stderr=subprocess.STDOUT)
        self._wait_until_listening()
        return self

    def _wait_until_listening(self) -> None:
        deadline = time.time() + SERVICE_BOOT_TIMEOUT_S
        while time.time() < deadline:
            with socket.socket() as sock:
                sock.settimeout(0.5)
                if sock.connect_ex(("127.0.0.1", self.port)) == 0:
                    return
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"服务启动即退出：{self.log_path.read_text(encoding='utf-8', errors='replace')[-500:]}")
            time.sleep(0.4)
        raise RuntimeError(f"服务未在 {SERVICE_BOOT_TIMEOUT_S}s 内监听 {self.port}")

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=30)

    @property
    def pid(self) -> int:
        return self.process.pid if self.process is not None else 0


def agent_environment(settings: Settings, service: PersistentService,
                      config_path: Path) -> dict[str, str]:
    return {
        **service_environment(settings),
        "SLOPE_AGENT_CONFIG": str(config_path),
    }


def write_agent_config(path: Path, settings: Settings, service: PersistentService) -> Path:
    """An agent config pointing at the persistent service over HTTP."""
    path.write_text(
        "schema_version: 1\n"
        "language: zh\n"
        "mcp:\n"
        "  hydro:\n"
        "    transport: streamable-http\n"
        f"    url: {service.url}\n"
        "    startup_timeout_s: 15\n"
        "    call_timeout_s: 180\n"
        "  spatial:\n"
        f"    transport: streamable-http\n"
        f"    url: {service.url}\n"
        "    startup_timeout_s: 15\n"
        "    call_timeout_s: 180\n"
        f"sessions:\n  root: {json.dumps(str(settings.hydro_root.parent / 'agent' / 'sessions'))}\n"
        "polling:\n  initial_interval_s: 2\n  maximum_interval_s: 5\n",
        encoding="utf-8")
    return path


class CliExitPersistenceTests(unittest.TestCase):
    """Plan scenario C.4 and the Definition of Done's blocking item."""

    def setUp(self):
        # The plan asks for test scratch space inside the repository rather than
        # in whatever the machine's temp directory happens to permit.
        scratch = PROJECT_ROOT / ".scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = TemporaryDirectory(ignore_cleanup_errors=True, dir=str(scratch))
        self.root = Path(self.temp.name)
        self.settings = make_settings(self.root)
        build_package(self.root / "exchange" / "exports" / "package-persist00001")
        self.record = import_package("package-persist00001", self.settings)
        self.port = free_port()
        self.service = PersistentService(self.settings, self.port).start()
        self.config = write_agent_config(self.root / "agent.yaml", self.settings, self.service)

    def tearDown(self):
        self.service.stop()
        self.temp.cleanup()

    # ------------------------------------------------------------------ helpers
    def _run_agent_program(self, program: str, timeout: int = 300) -> str:
        path = self.root / f"program-{abs(hash(program)) % 10**8}.py"
        path.write_text(program, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(path)], cwd=str(PROJECT_ROOT),
            env=agent_environment(self.settings, self.service, self.config),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout)
        self.assertEqual(result.returncode, 0,
                         f"程序失败：{result.stderr[-800:]}")
        return result.stdout

    def _submit_and_exit_while_worker_is_live(self) -> tuple[dict[str, str], int]:
        """Observe the worker first, then let the submitting Agent exit.

        The old test used ``subprocess.run`` and only looked for the worker after
        the Agent had already exited.  On a fast machine rainfall_only could
        finish in that gap, turning a lifecycle test into a timing lottery.  A
        one-line stdin handshake makes the ordering deterministic without
        slowing or modifying the production worker.
        """
        program = self._submit_program(wait_for_release=True)
        path = self.root / f"program-{abs(hash(program)) % 10**8}.py"
        path.write_text(program, encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, str(path)], cwd=str(PROJECT_ROOT),
            env=agent_environment(self.settings, self.service, self.config),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace")
        assert process.stdout is not None
        first_line = process.stdout.readline()
        try:
            submission = json.loads(first_line)
        except json.JSONDecodeError as exc:
            stderr = process.stderr.read() if process.stderr is not None else ""
            process.kill()
            self.fail(f"提交进程没有输出运行编号：{first_line!r}\n{stderr[-800:]}")
            raise exc  # unreachable; helps type checkers
        worker_pid = self._wait_for_worker(submission["run_id"])
        self.assertTrue(process_alive(worker_pid), "释放 Agent 前 worker 已退出")
        assert process.stdin is not None
        process.stdin.write("exit\n")
        process.stdin.flush()
        stdout, stderr = process.communicate(timeout=300)
        self.assertEqual(process.returncode, 0, f"提交进程失败：{stderr[-800:]}")
        self.assertFalse(process_alive(process.pid), "提交 Agent 没有按握手退出")
        self.assertTrue(process_alive(worker_pid), "Agent 退出时 worker 没有存活")
        release = self.settings.job_dir(submission["run_id"]) / "test-worker-release"
        release.write_text("continue", encoding="utf-8")
        return submission, worker_pid

    def _submit_program(self, *, wait_for_release: bool = False) -> str:
        """Submit a job through the persistent service, print the session and run."""
        return f'''
import asyncio, json, sys
sys.path.insert(0, r"{AGENT_SRC}")
from slope_evaluation_agent.agent import SlopeEvaluationAgent
from slope_evaluation_agent.config import load_settings
from slope_evaluation_agent.models import HydroRunRequestPayload


async def main():
    settings = load_settings(r"{self.config}")
    agent = SlopeEvaluationAgent(settings)
    session = agent.new_session()
    async with agent:
        payload = await agent.hydro.start_hydro_run(
            HydroRunRequestPayload(input_id="{self.record['input_id']}", mode="{PROBE_MODE}"))
        run_id = payload["data"]["run_id"]
        session.hydro_input_id = "{self.record['input_id']}"
        session.hydro_run_id = run_id
        session.hydro_run_status = payload["data"]["status"]
        session.state = "hydro_running"
        agent.store.save(session)
        print(json.dumps({{"session_id": session.session_id, "run_id": run_id}}), flush=True)
        {"await asyncio.to_thread(sys.stdin.readline)" if wait_for_release else ""}


asyncio.run(main())
'''

    def _wait_for_worker(self, run_id: str, timeout_s: float = 120) -> int:
        """Wait until the job is running with a live worker; return its pid."""
        status_path = self.settings.job_dir(run_id) / "status.json"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            pid = worker_pid_from_status(status_path)
            if pid and process_alive(pid):
                return pid
            time.sleep(0.3)
        self.fail(f"worker 未在 {timeout_s}s 内启动："
                  f"{status_path.read_text(encoding='utf-8', errors='replace')[:400]}")

    def _wait_for_terminal(self, run_id: str, timeout_s: float = JOB_TIMEOUT_S) -> dict:
        status_path = self.settings.job_dir(run_id) / "status.json"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(1.0)
                continue
            if payload.get("status") in {"succeeded", "qa_failed", "failed", "cancelled"}:
                return payload
            time.sleep(1.0)
        self.fail(f"任务未在 {timeout_s}s 内到达终态")

    # -------------------------------------------------------------------- tests
    def test_a_job_continues_after_the_agent_process_exits(self):
        """The blocking acceptance: CLI exit must not stop the run."""
        submission, worker_pid = self._submit_and_exit_while_worker_is_live()
        run_id = submission["run_id"]

        # The program that submitted the job has already exited. The service and
        # the worker must both still be there.
        self.assertTrue(process_alive(self.service.pid), "持久服务随 CLI 一起退出了")
        self.assertTrue(process_alive(worker_pid), "worker 随 CLI 一起退出了")

        # From here on no agent process exists: the submitting program has
        # returned and this test owns no part of the job. Whatever happens to
        # the run next happens because the service and its worker kept going.
        status_path = self.settings.job_dir(run_id) / "status.json"
        before = json.loads(status_path.read_text(encoding="utf-8")).get("completed_stages") or []
        final = self._wait_for_terminal(run_id)
        self.assertIn(final["status"], {"succeeded", "qa_failed"}, final)
        self.assertFalse(final.get("interrupted"))
        after = final.get("completed_stages") or []
        self.assertGreater(len(after), len(before),
                           f"CLI 退出后任务没有继续推进（{before} -> {after}）")
        # A job that had died with the CLI would sit at `running` until some
        # restart noticed; reaching a terminal state, un-interrupted, with no
        # agent alive is the acceptance the plan calls blocking.
        self.assertTrue(process_alive(self.service.pid), "持久服务在任务结束前退出了")

    def test_a_new_process_recovers_the_same_run(self):
        """A fresh agent must query the original run, not submit another."""
        submission, _ = self._submit_and_exit_while_worker_is_live()
        run_id, session_id = submission["run_id"], submission["session_id"]
        final = self._wait_for_terminal(run_id)
        self.assertIn(final["status"], {"succeeded", "qa_failed"})

        recover = f'''
import asyncio, json, sys
sys.path.insert(0, r"{AGENT_SRC}")
from slope_evaluation_agent.agent import SlopeEvaluationAgent
from slope_evaluation_agent.config import load_settings


async def main():
    settings = load_settings(r"{self.config}")
    agent = SlopeEvaluationAgent(settings)
    session = agent.store.load("{session_id}")
    async with agent:
        status = await agent.refresh_run_status(session)
        # The plan's Definition of Done also asks for assessment or diagnostic
        # output from a real run, so the recovered session is taken one step
        # further than a status query.
        reply = await agent.show_result(session)
    print(json.dumps({{"run_id": session.hydro_run_id, "status": status.get("status"),
                      "completed": len(status.get("completed_stages") or []),
                      "assessment_status": session.qa_passed,
                      "reply": reply.text, "ok": reply.ok}}))


asyncio.run(main())
'''
        recovered = json.loads(self._run_agent_program(recover))
        self.assertEqual(recovered["run_id"], run_id, "恢复后使用了不同的任务编号")
        self.assertEqual(recovered["status"], final["status"])
        self.assertGreater(recovered["completed"], 0)
        # A real run reaches a real verdict: a preliminary assessment when the
        # gates passed, a diagnostic one when they did not. Either way the user
        # gets prose from the assessment path, and it must not carry a path.
        self.assertTrue(recovered["reply"].strip(), "恢复后没有产出评价或诊断文本")
        self.assertNotIn("Traceback", recovered["reply"])
        self.assertNotRegex(recovered["reply"], r"[A-Za-z]:[\\/]")
        self.assertTrue(
            any(marker in recovered["reply"]
                for marker in ("初步评价", "诊断", "没有可用于评价的结果")),
            f"恢复后的文本不是评价/诊断路径产出的：{recovered['reply'][:200]}")

        jobs = [item for item in self.settings.job_dir(run_id).parent.iterdir()
                if item.is_dir()]
        self.assertEqual(len(jobs), 1, f"重进后产生了额外的任务目录：{[j.name for j in jobs]}")

        # R5 over *real* MCP responses, not stubs: a real flow leaves a session
        # that names this machine nowhere and carries no credential.
        self._assert_session_is_minimal(session_id)

    def _assert_session_is_minimal(self, session_id: str) -> None:
        directory = self.settings.runs_root / "agent" / "sessions" / session_id
        files = [path for path in directory.rglob("*") if path.is_file()]
        self.assertTrue(files, f"会话目录里没有文件：{directory}")
        for path in files:
            raw = path.read_text(encoding="utf-8", errors="replace")
            name = path.name
            self.assertNotRegex(raw, r"[A-Za-z]:[\\/]", f"{name} 含绝对路径")
            self.assertNotRegex(raw, r"\\\\[^\\/\s]+[\\/]", f"{name} 含 UNC 路径")
            self.assertNotRegex(raw, r"Traceback", f"{name} 含 traceback")
            self.assertNotRegex(raw, r"(?i)(api[_-]?key|authorization|bearer\s+\S)",
                                f"{name} 含凭据形状的内容")

    def test_stopping_the_service_is_reported_not_hidden(self):
        """A dead service must produce a connection error, not a traceback."""
        self.service.stop()
        check = f'''
import asyncio, sys
sys.path.insert(0, r"{AGENT_SRC}")
from slope_evaluation_agent.agent import SlopeEvaluationAgent
from slope_evaluation_agent.config import load_settings
from slope_evaluation_agent.mcp_clients import AgentMCPError


async def main():
    settings = load_settings(r"{self.config}")
    agent = SlopeEvaluationAgent(settings)
    session = agent.new_session()
    session.hydro_input_id = "{self.record['input_id']}"
    session.hydro_run_id = "hydro-run-abcdef123456"
    session.state = "hydro_running"
    async with agent:
        reply = await agent.handle(session, "状态")
    print(reply.text)


asyncio.run(main())
'''
        text = self._run_agent_program(check)
        self.assertNotIn("Traceback", text)
        self.assertTrue("连接" in text or "无法" in text or "启动" in text, text[:200])
