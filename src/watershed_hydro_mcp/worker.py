"""Fixed worker entry point for one hydro run.

The server constructs this command; a client can never add an argument:

    python -m watershed_hydro_mcp.worker --run-id <id>

Every stage is imported from the packaged computation core and executed in
order. Output goes to ``logs/<stage>.log`` and ``status.json`` is rewritten
atomically after each stage so an observer always sees a consistent state.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import os
import sys
import threading
import time
import traceback
from pathlib import Path


def quality_failure_details(report):
    """Small, actionable QA details for polling clients, without large rasters/geometry."""
    report = report or {}
    return {"qa_failed_gates": list(report.get("failed") or []),
            "qa_errors": [error for gate in report.get("gates") or []
                          for error in gate.get("errors") or []]}


def _start_heartbeat(job_dir: Path, run_id: str) -> threading.Event:
    """Keep a liveness record the service can check after a restart.

    A worker that outlives its service is otherwise indistinguishable from one
    that died with it: the status file would say ``running`` either way. The
    heartbeat carries the pid and a timestamp, which is evidence.
    """
    import json
    import os
    import threading
    import time

    from .state import utc_now

    stop = threading.Event()
    path = job_dir / "heartbeat.json"

    def beat() -> None:
        while not stop.is_set():
            try:
                payload = {"run_id": run_id, "pid": os.getpid(),
                           "at": time.time(), "at_iso": utc_now()}
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(payload), encoding="utf-8")
                os.replace(temporary, path)
            except OSError:
                pass
            stop.wait(5.0)

    threading.Thread(target=beat, name=f"heartbeat-{run_id}", daemon=True).start()
    return stop


def _wait_at_test_barrier(job_dir: Path, run_id: str) -> None:
    """Deterministic process-lifecycle barrier used only by integration tests.

    A tiny rainfall-only fixture can finish before a parent test has time to
    observe its worker.  When the explicit test environment switch is set, the
    worker announces that it is alive and waits for a per-run release file.
    Production environments never set the switch and take no extra branch.
    """
    if os.environ.get("WHS_TEST_HOLD_WORKER") != "1":
        return
    ready = job_dir / "test-worker-ready"
    release = job_dir / "test-worker-release"
    ready.write_text(run_id, encoding="utf-8")
    deadline = time.monotonic() + 180.0
    while not release.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("integration-test worker barrier timed out")
        time.sleep(0.05)


def _configure_environment(run_id: str, settings) -> tuple[Path, Path]:
    """Point the computation core at this run before anything imports it."""
    job_dir = settings.job_dir(run_id)
    project = job_dir / "project"
    if not project.is_dir():
        raise SystemExit(f"job {run_id} has no project directory")
    config = project / "config.yaml"
    if not config.is_file():
        raise SystemExit(f"job {run_id} has no config.yaml")
    os.environ["WHS_RUN_ROOT"] = str(project.resolve())
    os.environ["WHS_CONFIG"] = str(config.resolve())
    os.environ["WHS_NATIVE_DIR"] = str(settings.native_root.resolve())
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    return job_dir, project


def _register_stage_artifacts(registry, project: Path, stage_id: str) -> None:
    """Record the files a stage is responsible for producing."""
    from .artifact_registry import STAGE_KINDS

    if stage_id not in STAGE_KINDS:
        return
    patterns = {
        "delineate": ("inputs/dem_derived/**/*",),
        "rainfall": ("inputs/rainfall/*",),
        "rainfall_plot": ("inputs/rainfall/figures/*",),
        "build": ("swmm/model/*",),
        "swmm": ("swmm/results/*",),
        "safety": ("rapid_safety_factor_results/*",),
        "analysis": ("analysis_results/*",),
        # The export stage lays out a folder tree; every file in it is a result.
        "export": ("final_results/**/*",),
    }.get(stage_id, ())
    for pattern in patterns:
        for path in sorted(project.glob(pattern)):
            if path.is_file():
                registry.register(path, stage_id)


def _run_qa(job_dir: Path, project: Path, completed: list[str], settings,
            registry=None) -> dict:
    from . import qa

    report = qa.run_gates(project, completed, settings)
    qa.write_gates(job_dir, report)
    if registry is not None:
        # The gates are a property of the run, not of one file, so the verdict
        # is stamped across the registry once — here, after the last stage that
        # registers anything. A download must be able to say whether what it is
        # handing over came from a run that passed.
        registry.stamp_qa("passed" if report.get("passed") else "failed")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one planned hydro/stability job.")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    from .job_manager import RUN_ID_PREFIX
    from .settings import SETTINGS
    from .state import read_json, utc_now, write_json

    run_id = args.run_id
    if not run_id.startswith(RUN_ID_PREFIX):
        raise SystemExit("invalid run id")

    settings = SETTINGS
    job_dir, project = _configure_environment(run_id, settings)

    from . import planner
    from .artifact_registry import ArtifactRegistry

    heartbeat = _start_heartbeat(job_dir, run_id)
    _wait_at_test_barrier(job_dir, run_id)
    job = read_json(job_dir / "job.json")
    status_path = job_dir / "status.json"
    status = read_json(status_path)
    logs = job_dir / "logs"
    logs.mkdir(exist_ok=True)
    registry = ArtifactRegistry(job_dir, run_id, project)
    stages = {stage.stage_id: stage for stage in planner.STAGES}
    planned = list(job.get("stages") or [])
    completed: list[str] = []

    for stage_id in planned:
        stage = stages.get(stage_id)
        if stage is None:
            status.update({"status": "failed", "summary": f"未知阶段：{stage_id}",
                           "updated_at": utc_now(), "finished_at": utc_now()})
            write_json(status_path, status)
            return 1
        status.update({"status": "running", "current_stage": stage_id,
                       "summary": f"正在执行：{stage.title}", "updated_at": utc_now()})
        write_json(status_path, status)
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                if stage_id == "qa":
                    report = _run_qa(job_dir, project, completed, settings, registry)
                else:
                    module = importlib.import_module(stage.module)
                    module.main()
        except BaseException:  # noqa: BLE001 - a stage failure must be recorded, not lost
            text = buffer.getvalue() + "\n--- TRACEBACK ---\n" + traceback.format_exc()
            (logs / f"{stage_id}.log").write_text(text, encoding="utf-8")
            status.update({
                "status": "failed",
                "current_stage": stage_id,
                "summary": f"阶段失败：{stage.title}",
                "error": traceback.format_exc().strip().splitlines()[-1][:500],
                "completed_stages": completed,
                "updated_at": utc_now(),
                "finished_at": utc_now(),
                "log": f"logs/{stage_id}.log",
            })
            write_json(status_path, status)
            return 1
        text = buffer.getvalue()
        (logs / f"{stage_id}.log").write_text(text, encoding="utf-8")
        completed.append(stage_id)
        if stage_id != "qa":
            _register_stage_artifacts(registry, project, stage_id)
        status.update({"completed_stages": completed, "updated_at": utc_now()})
        write_json(status_path, status)

    qa_report = None
    qa_path = job_dir / "qa.json"
    if qa_path.is_file():
        from .state import read_json as _read

        qa_report = _read(qa_path)

    if qa_report is not None and not qa_report.get("passed", True):
        final_status = "qa_failed"
        summary = "计算完成，但质量门槛未通过，结果不得作为已验证结论使用"
    else:
        final_status = "succeeded"
        if qa_report is not None and not qa_report.get("gates_applied", True):
            summary = "全部计划阶段完成（本次运行未包含需要质量检查的阶段）"
        elif qa_report is not None:
            summary = "全部计划阶段完成，质量检查通过"
        else:
            summary = "全部计划阶段完成"

    gates_applied = None if qa_report is None else bool(qa_report.get("gates_applied", True))
    status.update({
        "status": final_status,
        "current_stage": None,
        "summary": summary,
        "completed_stages": completed,
        "updated_at": utc_now(),
        "finished_at": utc_now(),
        "qa_passed": None if qa_report is None or not gates_applied else bool(qa_report.get("passed")),
        "qa_gates_applied": gates_applied,
        **quality_failure_details(qa_report),
    })
    write_json(status_path, status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
