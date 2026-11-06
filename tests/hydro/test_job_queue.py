"""Queue progression must not depend on a client inspecting a finished run."""
from unittest.mock import Mock

from watershed_hydro_mcp.job_manager import JobStore
from watershed_hydro_mcp.settings import Settings
from watershed_hydro_mcp.state import read_json, write_json


def store_with_queue(tmp_path, finished_state="succeeded", exit_code=0):
    settings = Settings(tmp_path, tmp_path / "runs", tmp_path / "hydro",
                        tmp_path / "native", tmp_path / "config")
    store = JobStore(settings)
    for run_id, state, created in (("hydro-run-old", finished_state, "2026-01-01"),
                                   ("hydro-run-next", "queued", "2026-01-02")):
        store.job_dir(run_id).mkdir(parents=True)
        write_json(store.status_path(run_id), {
            "run_id": run_id, "status": state, "created_at": created,
            "completed_stages": ["sync"],
        })
    process = Mock()
    process.poll.return_value = exit_code
    store._processes["hydro-run-old"] = process
    store._launch = Mock()
    return store


def test_completed_worker_releases_slot_without_status_poll(tmp_path):
    store = store_with_queue(tmp_path)
    store._pump()
    store._launch.assert_called_once_with("hydro-run-next")
    assert "hydro-run-old" not in store._processes
    assert read_json(store.status_path("hydro-run-old"))["status"] == "succeeded"


def test_live_worker_keeps_slot(tmp_path):
    store = store_with_queue(tmp_path, "running", None)
    store._pump()
    store._launch.assert_not_called()
    assert "hydro-run-old" in store._processes


def test_crashed_worker_fails_old_run_and_starts_next(tmp_path):
    store = store_with_queue(tmp_path, "running", 1)
    store._pump()
    store._launch.assert_called_once_with("hydro-run-next")
    old = read_json(store.status_path("hydro-run-old"))
    assert old["status"] == "failed"
    assert old["completed_stages"] == ["sync"]
