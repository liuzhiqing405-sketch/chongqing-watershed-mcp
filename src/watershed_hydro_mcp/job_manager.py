"""Background job management for hydro runs.

``start_hydro_run`` returns a run id immediately; a fixed worker module then
executes the planned stages in a separate process, updating an atomically
written ``status.json`` after each one. The service can be restarted at any
time and recover what it finds on disk.
"""

from __future__ import annotations

import logging
import os
import json
import hashlib
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from . import planner
from .input_registry import load_record, validate_input
from .settings import SETTINGS, Settings
from .state import new_id, read_json, utc_now, write_json


logger = logging.getLogger("watershed_hydro_mcp.job_manager")


RUN_ID_PREFIX = "hydro-run-"

#: Bound on concurrent work, so one caller cannot saturate the machine.
MAXIMUM_CONCURRENT_HYDRO_RUNS = 1

TERMINAL_STATUSES = {"succeeded", "qa_failed", "failed", "cancelled"}

#: A run whose worker process vanished is reported as interrupted, not running.
ACTIVE_STATUSES = {"queued", "running"}


class JobError(ValueError):
    """Raised when a job cannot be created or manipulated."""


class JobStore:
    """Owns the running workers and the queue behind them."""

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self.settings = settings
        self.logger = logger
        #: Identifies this service process, so a status file can say which
        #: instance launched its worker.
        from uuid import uuid4

        self.instance_id = f"hydro-service-{uuid4().hex[:12]}"
        self._lock = threading.RLock()
        self._processes: dict[str, subprocess.Popen] = {}
        self._scheduler: threading.Thread | None = None

    # ------------------------------------------------------------------ paths
    def job_dir(self, run_id: str) -> Path:
        return self.settings.job_dir(run_id)

    def status_path(self, run_id: str) -> Path:
        return self.job_dir(run_id) / "status.json"

    def job_path(self, run_id: str) -> Path:
        return self.job_dir(run_id) / "job.json"

    # ---------------------------------------------------------------- reading
    def status(self, run_id: str) -> dict[str, Any]:
        path = self.status_path(run_id)
        if not path.is_file():
            raise JobError(f"任务不存在：{run_id}")
        record = read_json(path)
        if record.get("status") in ACTIVE_STATUSES:
            with self._lock:
                alive = run_id in self._processes
            if not alive and record.get("status") == "running":
                record["status"] = "failed"
                record["summary"] = "worker process is no longer running"
                record["finished_at"] = record.get("finished_at") or utc_now()
                write_json(path, record)
        return record

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        root = self.settings.jobs_root
        if not root.is_dir():
            return []
        rows: list[dict[str, Any]] = []
        for directory in root.iterdir():
            status_path = directory / "status.json"
            if not directory.is_dir() or not status_path.is_file():
                continue
            try:
                record = read_json(status_path)
            except ValueError:
                continue
            rows.append({
                "run_id": record.get("run_id", directory.name),
                "status": record.get("status"),
                "mode": record.get("mode"),
                "input_id": record.get("input_id"),
                "created_at": record.get("created_at"),
                "updated_at": record.get("updated_at"),
                "summary": record.get("summary"),
            })
        rows.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return rows[:limit]

    # --------------------------------------------------------------- writing
    def record_quality_report(self, run_id: str, report: dict[str, Any]) -> None:
        """Keep terminal status consistent with a rechecked QA report."""
        from .qa import write_gates
        with self._lock:
            status = self.status(run_id)
            if status.get("status") in ACTIVE_STATUSES:
                raise JobError("任务仍在执行，无法重检质量门槛")
            write_gates(self.job_dir(run_id), report)
            status["qa_passed"] = bool(report["passed"]) if report.get("gates_applied") else None
            status["qa_gates_applied"] = bool(report.get("gates_applied"))
            status["qa_rechecked_at"] = report["checked_at"]
            status["qa_failed_gates"] = list(report.get("failed") or [])
            status["qa_errors"] = [error for gate in report.get("gates") or [] for error in gate.get("errors") or []]
            status["updated_at"] = utc_now()
            if status.get("status") in {"succeeded", "qa_failed"} and report.get("gates_applied"):
                status["status"] = "succeeded" if report["passed"] else "qa_failed"
                status["summary"] = "计算完成，重新检查的质量门槛通过" if report["passed"] else "计算完成，但重新检查的质量门槛未通过"
            write_json(self.status_path(run_id), status)

    def start(self, request, settings: Settings | None = None) -> dict[str, Any]:
        settings = settings or self.settings
        record = load_record(request.input_id, settings)
        validation = validate_input(request.input_id, settings)
        if not validation["ok"]:
            details = "；".join(validation["errors"][:3])
            raise JobError(f"输入校验未通过，任务未启动：{details}")
        if not settings.engine_present:
            raise JobError(
                "SWMM 引擎缺失，无法运行建模阶段；请检查 native/swmm 下的 runswmm.exe、swmm5.dll 和 vcomp140.dll"
            )
        run_id = new_id(RUN_ID_PREFIX)
        job_dir = self.job_dir(run_id)
        job_dir.mkdir(parents=True, exist_ok=False)
        (job_dir / "logs").mkdir()
        plan = planner.build_project(run_id, record, request, settings)
        stages = [stage.stage_id for stage in planner.stages_for(request.mode, request.through)]
        job = {
            "run_id": run_id,
            "input_id": request.input_id,
            "mode": request.mode,
            "through": request.through,
            "stages": stages,
            "parameter_preset": request.parameter_preset,
            "parameter_overrides": request.parameter_overrides.present(),
            "notes": request.notes,
            "domain_policy": (record.get("provenance") or {}).get("domain_policy") or {},
            "input_limitations": list(record.get("limitations") or []),
            "created_at": utc_now(),
            "input_validation": {
                "ok": True,
                "checked_at": utc_now(),
                "errors": validation["errors"],
                "warnings": validation["warnings"],
                "sha256": hashlib.sha256(
                    json.dumps(validation, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest(),
            },
            "plan": plan,
        }
        write_json(self.job_path(run_id), job)
        status = {
            "run_id": run_id,
            "status": "queued",
            "mode": request.mode,
            "input_id": request.input_id,
            "stages": stages,
            "completed_stages": [],
            "current_stage": None,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "summary": "已排队，等待执行槽位",
            "project": plan["project"],
        }
        write_json(self.status_path(run_id), status)
        self._ensure_scheduler()
        self._pump()
        return status

    def cancel(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            process = self._processes.get(run_id)
        status = self.status(run_id)
        if status.get("status") in TERMINAL_STATUSES:
            return status
        if process is not None:
            self._terminate_process_tree(process)
        status.update({
            "status": "cancelled",
            "summary": "用户取消了该任务",
            "finished_at": utc_now(),
            "updated_at": utc_now(),
        })
        write_json(self.status_path(run_id), status)
        with self._lock:
            self._processes.pop(run_id, None)
        self._pump()
        return status

    # ------------------------------------------------------------ scheduling
    def _ensure_scheduler(self) -> None:
        with self._lock:
            if self._scheduler is not None and self._scheduler.is_alive():
                return
            self._scheduler = threading.Thread(target=self._loop, name="hydro-scheduler", daemon=True)
            self._scheduler.start()

    def _loop(self) -> None:
        import time

        while True:
            self._pump()
            time.sleep(1.0)

    def _pump(self) -> None:
        # A completed worker must release its slot even when clients only poll
        # the status endpoint. Otherwise every subsequent run stays queued.
        self.reap()
        with self._lock:
            running = len(self._processes)
            free = MAXIMUM_CONCURRENT_HYDRO_RUNS - running
            if free <= 0:
                return
            queued = [
                row for row in self.list_jobs(limit=200)
                if row.get("status") == "queued"
            ]
            for row in reversed(queued):  # oldest first
                if free <= 0:
                    break
                self._launch(row["run_id"])
                free -= 1

    def _launch(self, run_id: str) -> None:
        job_dir = self.job_dir(run_id)
        status_path = self.status_path(run_id)
        # Claim the slot before starting the process: a fast run can reach a
        # terminal status before Popen returns, and writing "running" afterwards
        # would overwrite the worker's own result.
        try:
            status = read_json(status_path)
        except ValueError:
            return
        if status.get("status") != "queued":
            return
        status.update({"status": "running", "started_at": utc_now(), "updated_at": utc_now(),
                       "summary": "worker 已启动"})
        write_json(status_path, status)

        log_path = job_dir / "logs" / "worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        # The worker builds its own Settings from the environment, so the
        # service's roots have to travel with the command.
        environment["WHS_RUNS_ROOT"] = str(self.settings.runs_root)
        environment["WHS_HYDRO_ROOT"] = str(self.settings.hydro_root)
        environment["WHS_NATIVE_ROOT"] = str(self.settings.native_root)
        environment["WHS_CONFIG_ROOT"] = str(self.settings.config_root)
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["PYTHONUTF8"] = "1"
        # Unbuffered so worker.log shows progress while a run is in flight,
        # rather than only after the process exits.
        environment["PYTHONUNBUFFERED"] = "1"
        handle = log_path.open("a", encoding="utf-8")
        popen_options: dict[str, Any] = {}
        if os.name == "nt":
            # CREATE_NEW_PROCESS_GROUP so cancelling can stop the worker's own
            # children (runswmm.exe) as a group. CREATE_BREAKAWAY_FROM_JOB so a
            # worker outlives the service that launched it: MCP clients put the
            # service in a Job Object that is killed when the client exits, and
            # the worker would otherwise be killed with it — which would make
            # "start a run, close the agent, come back later" impossible.
            popen_options["creationflags"] = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
            )
        else:
            popen_options["start_new_session"] = True
        try:
            process = self._spawn_worker(run_id, environment, handle, popen_options)
        finally:
            handle.close()
        self._processes[run_id] = process
        # Record the pid as a field, not only inside a sentence: recovery has to
        # compare it against the worker's own heartbeat after a restart.
        current = read_json(status_path)
        if current.get("status") == "running":
            current.update({"summary": f"worker 已启动（pid {process.pid}）",
                            "worker_pid": process.pid,
                            "worker_started_at": current.get("started_at") or utc_now(),
                            "service_instance": self.instance_id,
                            "service_pid": os.getpid(),
                            "updated_at": utc_now()})
            write_json(status_path, current)

    def _spawn_worker(self, run_id: str, environment: dict[str, str], handle,
                      popen_options: dict[str, Any]) -> "subprocess.Popen":
        """Start one worker, falling back if the OS refuses a detached spawn.

        A job object may forbid breakaway; if it does, the run still has to
        start, so the flag is dropped and the worker lives no longer than the
        service — the degraded behaviour, not a failure.
        """
        command = [sys.executable, "-m", "watershed_hydro_mcp.worker",
                   "--run-id", run_id]
        base: dict[str, Any] = {
            "cwd": str(self.settings.root),
            "stdout": handle,
            "stderr": subprocess.STDOUT,
            "env": environment,
        }
        try:
            return subprocess.Popen(command, **base, **popen_options)
        except OSError as exc:
            if "creationflags" not in popen_options:
                raise
            self.logger.warning("worker 无法脱离服务进程独立运行（%s）；"
                                "任务仍会执行，但服务退出时会被终止", exc)
            degraded = {**popen_options,
                        "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
            return subprocess.Popen(command, **base, **degraded)

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen) -> None:
        """Stop a worker and every native process it launched.

        On Windows, ``terminate`` only stops the Python worker and may leave
        ``runswmm.exe`` running. Workers are therefore launched in their own
        process group and cancelled as a tree. POSIX uses an isolated session
        for the equivalent guarantee.
        """
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=20)
            return
        except (OSError, subprocess.SubprocessError):
            pass
        try:
            process.terminate()
            process.wait(timeout=5)
        except (OSError, subprocess.SubprocessError):
            try:
                process.kill()
            except OSError:
                pass

    def reap(self) -> None:
        """Reconcile in-memory processes with what finished on disk."""
        with self._lock:
            finished = [run_id for run_id, process in self._processes.items()
                        if process.poll() is not None]
            for run_id in finished:
                self._processes.pop(run_id, None)
        for run_id in finished:
            try:
                status = read_json(self.status_path(run_id))
            except ValueError:
                continue
            if status.get("status") in ACTIVE_STATUSES:
                status.update({
                    "status": "failed",
                    "summary": "worker 进程退出但未写入最终状态",
                    "finished_at": utc_now(),
                    "updated_at": utc_now(),
                })
                write_json(self.status_path(run_id), status)

    #: How stale a heartbeat may be before its worker counts as gone.
    HEARTBEAT_STALE_SECONDS = 30.0

    def worker_liveness(self, run_id: str) -> dict[str, Any]:
        """Evidence about a run's worker, from the heartbeat it writes.

        Returning ``alive: False`` with a reason is as important as returning
        ``alive: True``: the status file alone says ``running`` whether the
        worker is working or dead, and the two must not be confused.
        """
        import time

        path = self.job_dir(run_id) / "heartbeat.json"
        if not path.is_file():
            return {"alive": False, "reason": "没有心跳记录"}
        try:
            beat = read_json(path)
        except ValueError:
            return {"alive": False, "reason": "心跳记录无法读取"}
        try:
            age = time.time() - float(beat.get("at") or 0.0)
        except (TypeError, ValueError):
            return {"alive": False, "reason": "心跳时间戳无效"}
        pid = int(beat.get("pid") or 0)
        if age > self.HEARTBEAT_STALE_SECONDS:
            return {"alive": False, "pid": pid, "age_seconds": round(age, 1),
                    "reason": f"心跳已停止 {age:.0f} 秒"}
        if pid and not self._pid_alive(pid):
            return {"alive": False, "pid": pid, "age_seconds": round(age, 1),
                    "reason": "心跳对应的进程不存在"}
        return {"alive": True, "pid": pid, "age_seconds": round(age, 1)}

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """Whether a process with this id exists, on this platform."""
        if pid <= 0:
            return False
        if os.name == "nt":
            # ``tasklist`` may return an empty result in restricted desktop or
            # service sessions even when the process exists. Query the process
            # handle directly so recovery does not mislabel a live worker as
            # interrupted merely because a shell utility was unavailable.
            import ctypes

            process_query_limited_information = 0x1000
            still_active = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(process_query_limited_information,
                                          False, int(pid))
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return int(exit_code.value) == still_active
            finally:
                kernel32.CloseHandle(handle)
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return False
        return True

    def restore(self) -> dict[str, Any]:
        """Recover after a service restart, from evidence rather than assumption.

        A run recorded as active is checked against its worker's heartbeat:

        * a live worker means the job is genuinely still running — it outlived
          this service — and its status is left alone;
        * anything else means the worker is gone, and the run is marked
          ``interrupted`` so nobody reads ``running`` as progress. Its completed
          stages and artifacts stay on disk, and the user can resubmit from the
          same input.
        """
        recovered = {"interrupted": [], "finished": [], "still_running": []}
        for row in self.list_jobs(limit=500):
            run_id = row["run_id"]
            if row.get("status") not in ACTIVE_STATUSES:
                recovered["finished"].append(run_id)
                continue
            status = read_json(self.status_path(run_id))
            liveness = self.worker_liveness(run_id)
            status["worker_liveness"] = liveness
            if liveness["alive"]:
                status.update({
                    "summary": f"服务重启后检测到 worker 仍在运行（pid {liveness['pid']}）",
                    "updated_at": utc_now(),
                })
                write_json(self.status_path(run_id), status)
                recovered["still_running"].append(run_id)
                continue
            status.update({
                "status": "failed",
                "summary": f"worker 已不在运行（{liveness['reason']}）；"
                           "已完成阶段与成果仍可读取，可基于同一输入重新提交",
                "interrupted": True,
                "finished_at": status.get("finished_at") or utc_now(),
                "updated_at": utc_now(),
            })
            write_json(self.status_path(run_id), status)
            recovered["interrupted"].append(run_id)
        return recovered


STORE = JobStore(SETTINGS)
