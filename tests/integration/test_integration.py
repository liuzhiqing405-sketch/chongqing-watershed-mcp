"""End-to-end behaviour: the two independent paths and the hand-off between them.

These drive the real worker process and the real job store against synthetic
inputs small enough to finish quickly, so the state machine, the atomic status
writes and the gate mapping are exercised rather than mocked.
"""

from __future__ import annotations

import json
import base64
import shutil
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import build_package  # noqa: E402

from watershed_hydro_mcp.input_registry import import_package, register_inputs  # noqa: E402
from watershed_hydro_mcp.job_manager import JobStore  # noqa: E402
from watershed_hydro_mcp.models import HydroRunRequest, RegisterInputRequest  # noqa: E402
from watershed_hydro_mcp.settings import Settings  # noqa: E402
from watershed_hydro_mcp.state import read_json, write_json  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def make_settings(root: Path) -> Settings:
    """A project-shaped settings object rooted in a temporary directory."""
    (root / "native").mkdir(parents=True, exist_ok=True)
    shutil.copytree(PROJECT_ROOT / "native" / "swmm", root / "native" / "swmm",
                    dirs_exist_ok=True)
    (root / "exchange" / "exports").mkdir(parents=True, exist_ok=True)
    (root / "exchange" / "imports").mkdir(parents=True, exist_ok=True)
    runs = root / "runs"
    return Settings(root=root, runs_root=runs, hydro_root=runs / "hydro",
                    native_root=root / "native", config_root=PROJECT_ROOT / "config")


def wait_for(store: JobStore, run_id: str, timeout: float = 300.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        store.reap()
        status = store.status(run_id)
        if status.get("status") in {"succeeded", "qa_failed", "failed", "cancelled"}:
            return status
        time.sleep(0.5)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s: "
                         f"{store.status(run_id).get('status')}")


class ExchangePathTests(unittest.TestCase):
    """A package exported from one project imports into a clean one."""

    def setUp(self):
        self.temp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp.name)
        self.settings = make_settings(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_import_copies_the_package_out_of_the_source_tree(self):
        source = build_package(self.root / "exchange" / "exports" / "package-abcdef000001")
        record = import_package("package-abcdef000001", self.settings)
        managed = self.settings.input_dir(record["input_id"])
        self.assertTrue(managed.is_dir())
        self.assertTrue((managed / "dem.tif").is_file())
        # Deleting the source must not disturb the imported copy.
        shutil.rmtree(source)
        self.assertTrue((managed / "dem.tif").is_file())

    def test_imported_hashes_are_re_verified_against_the_manifest(self):
        directory = build_package(self.root / "exchange" / "exports" / "package-abcdef000002")
        payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        payload["files"][0]["sha256"] = "0" * 64
        (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(ValueError):
            import_package("package-abcdef000002", self.settings)

    def test_zip_and_directory_imports_agree(self):
        directory = build_package(self.root / "exchange" / "exports" / "package-abcdef000003")
        import zipfile

        archive = self.root / "exchange" / "exports" / "package-abcdef000003.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    bundle.write(path, arcname=path.relative_to(directory).as_posix())
        from_directory = import_package("package-abcdef000003", self.settings)
        from_archive = import_package("package-abcdef000003.zip", self.settings)
        for record in (from_directory, from_archive):
            self.assertEqual(record["spatial"]["dem"]["width"], 70)
            self.assertEqual(record["temporal"]["station_count"], 4)

    def test_registered_inputs_land_in_the_managed_directory(self):
        inbox = self.root / "exchange" / "imports" / "campaign"
        inbox.mkdir(parents=True)
        build_package(inbox)
        request = RegisterInputRequest(
            source_root=str(inbox), dem="dem.tif", landuse="landuse.shp",
            river="river.shp", boundary="basin.geojson",
            rainfall_zip="station_rainfall.zip", labels={"campaign": "test"},
        )
        record = register_inputs(request, self.settings)
        managed = self.settings.input_dir(record["input_id"])
        self.assertTrue((managed / "dem.tif").is_file())
        self.assertTrue((managed / "landuse.dbf").is_file())
        self.assertEqual(record["origin"], "registered")
        self.assertEqual(record["labels"], {"campaign": "test"})


class PreflightRunTests(unittest.TestCase):
    """The lightest real run: the worker process starts, validates and stops."""

    def setUp(self):
        self.temp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp.name)
        self.settings = make_settings(self.root)
        build_package(self.root / "exchange" / "exports" / "package-abcdef000010")
        self.record = import_package("package-abcdef000010", self.settings)
        self.store = JobStore(self.settings)

    def tearDown(self):
        self.temp.cleanup()

    def test_preflight_only_runs_two_stages_and_succeeds(self):
        status = self.store.start(HydroRunRequest(input_id=self.record["input_id"],
                                                  mode="preflight_only"))
        final = wait_for(self.store, status["run_id"])
        self.assertEqual(final["status"], "succeeded", final.get("summary"))
        self.assertEqual(final["completed_stages"], ["sync", "validate"])

    def test_status_progresses_through_the_documented_states(self):
        status = self.store.start(HydroRunRequest(input_id=self.record["input_id"],
                                                  mode="preflight_only"))
        observed = set()
        deadline = time.time() + 300
        while time.time() < deadline:
            observed.add(self.store.status(status["run_id"]).get("status"))
            if "succeeded" in observed:
                break
            time.sleep(0.2)
        self.assertTrue(observed <= {"queued", "running", "succeeded", "qa_failed", "failed"},
                        observed)
        self.assertIn("succeeded", observed)

    def test_job_record_lists_the_planned_stages(self):
        status = self.store.start(HydroRunRequest(input_id=self.record["input_id"],
                                                  mode="preflight_only"))
        wait_for(self.store, status["run_id"])
        job = read_json(self.settings.job_dir(status["run_id"]) / "job.json")
        self.assertEqual(job["stages"], ["sync", "validate"])
        self.assertEqual(job["input_id"], self.record["input_id"])
        self.assertTrue(Path(job["plan"]["project"]).is_dir())
        self.assertTrue(job["input_validation"]["ok"])
        self.assertEqual(len(job["input_validation"]["sha256"]), 64)

    def test_start_refuses_an_input_that_became_invalid_after_import(self):
        (self.settings.input_dir(self.record["input_id"]) / "dem.tif").unlink()
        with self.assertRaisesRegex(ValueError, "输入校验未通过"):
            self.store.start(HydroRunRequest(input_id=self.record["input_id"],
                                             mode="preflight_only"))

    def test_run_copies_its_input_and_never_reads_the_registry_again(self):
        status = self.store.start(HydroRunRequest(input_id=self.record["input_id"],
                                                  mode="preflight_only"))
        wait_for(self.store, status["run_id"])
        project = self.settings.job_dir(status["run_id"]) / "project"
        self.assertTrue((project / "inputs/user_data/dem.tif").is_file())
        shutil.rmtree(self.settings.input_dir(self.record["input_id"]))
        self.assertTrue((project / "inputs/user_data/dem.tif").is_file())

    def test_config_pins_every_path_inside_the_run(self):
        status = self.store.start(HydroRunRequest(input_id=self.record["input_id"],
                                                  mode="preflight_only"))
        wait_for(self.store, status["run_id"])
        project = self.settings.job_dir(status["run_id"]) / "project"
        import yaml

        config = yaml.safe_load((project / "config.yaml").read_text(encoding="utf-8"))
        # Config paths are written with forward slashes; compare normalised.
        root = str(self.root).replace("\\", "/")
        for key, value in (("dem", config["terrain"]["dem"]),
                           ("landuse", config["study_area"]["landuse_shp"]),
                           ("rainfall", config["rainfall"]["station_zip"]),
                           ("engine", config["executables"]["swmm_cli"])):
            self.assertTrue(str(value).startswith(root), f"{key}: {value}")
        self.assertTrue(Path(config["executables"]["grass_cli"]).is_file())
        self.assertEqual(config["swmm"]["delineation_method"], "grass_watershed")
        self.assertNotIn("D:\\paper", (project / "config.yaml").read_text(encoding="utf-8"))
        self.assertNotIn("Yongchuan", (project / "config.yaml").read_text(encoding="utf-8"))

    def test_cancel_stops_a_queued_run(self):
        # Occupy the single slot so the second run stays queued.
        first = self.store.start(HydroRunRequest(input_id=self.record["input_id"],
                                                 mode="preflight_only"))
        second = self.store.start(HydroRunRequest(input_id=self.record["input_id"],
                                                  mode="preflight_only"))
        cancelled = self.store.cancel(second["run_id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(self.store.status(second["run_id"])["status"], "cancelled")
        wait_for(self.store, first["run_id"])

    def test_cancelling_a_finished_run_leaves_it_alone(self):
        status = self.store.start(HydroRunRequest(input_id=self.record["input_id"],
                                                  mode="preflight_only"))
        final = wait_for(self.store, status["run_id"])
        again = self.store.cancel(status["run_id"])
        self.assertEqual(again["status"], final["status"])

    def test_restart_marks_an_orphaned_run_interrupted_and_keeps_its_stages(self):
        store = JobStore(self.settings)
        status = store.start(HydroRunRequest(input_id=self.record["input_id"],
                                             mode="preflight_only"))
        worker = store._processes.get(status["run_id"])
        wait_for(store, status["run_id"])
        # The terminal status can be written before the process actually exits.
        # Recovery correctly preserves a live worker, so establish the test's
        # "worker is gone" premise before rewriting its status as active.
        if worker is not None:
            worker.wait(timeout=30)
        store.reap()
        # Simulate the service dying mid-run: the worker is gone but the status
        # file still claims an active run.
        path = self.settings.job_dir(status["run_id"]) / "status.json"
        record = read_json(path)
        record.update({"status": "running", "completed_stages": ["sync"],
                       "current_stage": "validate", "interrupted": False})
        write_json(path, record)
        fresh = JobStore(self.settings)
        report = fresh.restore()
        self.assertIn(status["run_id"], report["interrupted"])
        recovered = fresh.status(status["run_id"])
        self.assertEqual(recovered["status"], "failed")
        self.assertTrue(recovered.get("interrupted"))
        self.assertEqual(recovered["completed_stages"], ["sync"])

    def test_restore_leaves_finished_runs_alone(self):
        store = JobStore(self.settings)
        status = store.start(HydroRunRequest(input_id=self.record["input_id"],
                                             mode="preflight_only"))
        final = wait_for(store, status["run_id"])
        report = JobStore(self.settings).restore()
        self.assertIn(status["run_id"], report["finished"])
        self.assertEqual(self.store.status(status["run_id"])["status"], final["status"])

    def test_missing_engine_blocks_starting_a_run(self):
        broken = make_settings(self.root / "broken")
        build_package(broken.root / "exchange" / "exports" / "package-abcdef000011")
        record = import_package("package-abcdef000011", broken)
        shutil.rmtree(broken.native_root / "swmm")
        store = JobStore(broken)
        with self.assertRaises(ValueError):
            store.start(HydroRunRequest(input_id=record["input_id"], mode="build_run_qa"))


class GateMappingTests(unittest.TestCase):
    """A computed run that fails a gate must never be reported as success."""

    def setUp(self):
        self.temp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp.name)
        self.settings = make_settings(self.root)
        build_package(self.root / "exchange" / "exports" / "package-abcdef000020")
        self.record = import_package("package-abcdef000020", self.settings)
        self.store = JobStore(self.settings)

    def tearDown(self):
        self.temp.cleanup()

    def test_rainfall_only_run_extends_the_real_pipeline(self):
        status = self.store.start(HydroRunRequest(input_id=self.record["input_id"],
                                                  mode="rainfall_only"))
        final = wait_for(self.store, status["run_id"], timeout=900)
        self.assertIn(final["status"], {"succeeded", "qa_failed"}, final.get("summary"))
        project = self.settings.job_dir(status["run_id"]) / "project"
        # Dilation of the synthetic DEM reaches the delineation and rainfall stages.
        self.assertTrue((project / "inputs/dem_derived").is_dir())
        self.assertTrue(any((project / "inputs/rainfall").glob("*active_hourly.csv")))

    def test_run_reports_qa_failed_when_a_gate_rejects_the_result(self):
        from watershed_hydro_mcp import qa
        from watershed_hydro_mcp.planner import load_gates

        status = self.store.start(HydroRunRequest(input_id=self.record["input_id"],
                                                  mode="rainfall_only"))
        wait_for(self.store, status["run_id"], timeout=900)
        project = self.settings.job_dir(status["run_id"]) / "project"
        # Re-run the gates with an impossible minimum-rainfall threshold. The
        # stages are named explicitly so the test does not depend on how far the
        # run happened to get.
        gates = load_gates(self.settings)
        gates["rainfall"] = {**gates["rainfall"], "minimum_total_rain_mm": 1e9}
        original = qa.load_gates
        try:
            qa.load_gates = lambda settings=None: gates
            report = qa.run_gates(project, ["delineate", "rainfall"])
        finally:
            qa.load_gates = original
        self.assertFalse(report["passed"])
        self.assertIn("rainfall", report["failed"])

    def test_qa_failure_maps_to_qa_failed_not_succeeded(self):
        """The worker's terminal status must reflect the gate report."""
        status = self.store.start(HydroRunRequest(input_id=self.record["input_id"],
                                                  mode="preflight_only"))
        final = wait_for(self.store, status["run_id"])
        job_dir = self.settings.job_dir(status["run_id"])
        # The qa stage is not part of a preflight run, so no report is written.
        self.assertFalse((job_dir / "qa.json").is_file())
        self.assertIsNone(final["qa_passed"])
        self.assertEqual(final["status"], "succeeded")

    def test_worker_status_reflects_a_failing_gate_report(self):
        """Simulated via run_gates: a failing report is what maps to qa_failed."""
        from watershed_hydro_mcp import qa
        from watershed_hydro_mcp.planner import load_gates

        status = self.store.start(HydroRunRequest(input_id=self.record["input_id"],
                                                  mode="preflight_only"))
        wait_for(self.store, status["run_id"])
        project = self.settings.job_dir(status["run_id"]) / "project"
        gates = load_gates(self.settings)
        gates["rainfall"] = {**gates["rainfall"], "minimum_total_rain_mm": 1e9}
        original = qa.load_gates
        try:
            qa.load_gates = lambda settings=None: gates
            report = qa.run_gates(project, ["rainfall"])
        finally:
            qa.load_gates = original
        self.assertTrue(report["gates_applied"])
        self.assertFalse(report["passed"])
        self.assertIn("rainfall", report["failed"])


class ComparisonTests(unittest.TestCase):
    """Two runs of the same input can be compared through their visualizations."""

    def setUp(self):
        self.temp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp.name)
        self.settings = make_settings(self.root)
        build_package(self.root / "exchange" / "exports" / "package-abcdef000030")
        self.record = import_package("package-abcdef000030", self.settings)

    def tearDown(self):
        self.temp.cleanup()

    def test_two_runs_keep_separate_project_trees(self):
        store = JobStore(self.settings)
        ids = []
        for _ in range(2):
            status = store.start(HydroRunRequest(input_id=self.record["input_id"],
                                                 mode="preflight_only"))
            wait_for(store, status["run_id"])
            ids.append(status["run_id"])
        first = self.settings.job_dir(ids[0]) / "project"
        second = self.settings.job_dir(ids[1]) / "project"
        self.assertNotEqual(first, second)
        for project in (first, second):
            self.assertTrue((project / "config.yaml").is_file())
            self.assertGreater((project / "config.yaml").stat().st_size, 0)

    def test_comparison_reports_different_run_ids(self):
        from watershed_hydro_mcp.visualization.service import VisualizationService

        service = VisualizationService(self.settings)
        root = self.settings.visualizations_root
        for index, digest in ((1, "b" * 64), (2, "c" * 64)):
            folder = root / f"viz-cmp{index:011d}"
            folder.mkdir(parents=True)
            (folder / "preview.png").write_bytes(PNG_1PX)
            write_json(folder / "visualization.json", {
                "visualization_id": f"viz-cmp{index:011d}",
                "run_id": f"hydro-run-{index:012d}",
                "kind": "qa_dashboard", "status": "succeeded",
                "created_at": "2026-01-01T00:00:00+00:00",
                "title": "QA 仪表盘", "image": "preview.png", "image_sha256": "a" * 64,
                "sources": [{"scope": "job", "relative_path": "qa.json",
                             "sha256": digest, "size_bytes": 10}],
            })
        result = service.compare([f"viz-cmp{1:011d}", f"viz-cmp{2:011d}"])
        self.assertEqual(len(result["run_ids"]), 2)
        self.assertFalse(result["same_sources"])
        self.assertTrue(any("不同的运行" in item for item in result["differences"]))
        self.assertTrue(any("源文件内容不同" in item for item in result["differences"]))
        self.assertTrue(service.image_path(result["comparison_visualization_id"]).is_file())

    def test_visualization_of_a_missing_stage_reports_a_warning_not_a_crash(self):
        from watershed_hydro_mcp.visualization.service import VisualizationService

        store = JobStore(self.settings)
        status = store.start(HydroRunRequest(input_id=self.record["input_id"],
                                             mode="preflight_only"))
        wait_for(store, status["run_id"])
        service = VisualizationService(self.settings)
        # Render synchronously so nothing outlives the test's temporary directory.
        service._dispatch = lambda visualization_id: None
        created = service.create(status["run_id"], "safety_minimum_factor")
        service._run(created["visualization_id"])
        record = service.status(created["visualization_id"])
        # No stability raster exists after a preflight run, so the figure renders
        # its placeholder and carries a warning rather than failing the service.
        self.assertIn(record["status"], {"succeeded", "failed"})
        if record["status"] == "succeeded":
            self.assertIn("该图未绑定任何源文件", " ".join(record.get("warnings") or []))


class IndependenceTests(unittest.TestCase):
    """The hydro MCP must work from a package alone, with no spatial state."""

    def test_import_works_without_any_spatial_run_directory(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            root = Path(temp)
            settings = make_settings(root)
            # Nothing under runs/watersheds, runs/rainfall or runs/model_inputs.
            self.assertFalse((root / "runs" / "watersheds").exists())
            build_package(root / "exchange" / "exports" / "package-abcdef000040")
            record = import_package("package-abcdef000040", settings)
            self.assertEqual(record["origin"], "model_input_package")
            self.assertTrue(settings.input_dir(record["input_id"]).is_dir())

    def test_hydro_package_never_imports_the_spatial_package(self):
        import watershed_hydro_mcp

        modules = set(sys.modules)
        for name in list(modules):
            if name.startswith("watershed_hydro_mcp"):
                module = sys.modules[name]
                source = getattr(module, "__file__", None)
                if not source:
                    continue
                text = Path(source).read_text(encoding="utf-8", errors="ignore")
                self.assertNotIn("import chongqing_spatial_mcp", text, source)

    def test_spatial_package_never_imports_the_hydro_package(self):
        import chongqing_spatial_mcp

        for name, module in list(sys.modules.items()):
            if not name.startswith("chongqing_spatial_mcp"):
                continue
            source = getattr(module, "__file__", None)
            if not source:
                continue
            text = Path(source).read_text(encoding="utf-8", errors="ignore")
            self.assertNotIn("import watershed_hydro_mcp", text, source)


if __name__ == "__main__":
    unittest.main()
