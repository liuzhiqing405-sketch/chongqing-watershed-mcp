"""Unit tests for planning, quality gates, artifacts and job state."""

from __future__ import annotations

import json
import base64
import sys
import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import write_dem, write_polygon_shapefile  # noqa: E402

from watershed_hydro_mcp import planner, qa  # noqa: E402
from watershed_hydro_mcp import job_manager, server as hydro_server_module  # noqa: E402
from watershed_hydro_mcp.artifact_registry import ArtifactRegistry, describe  # noqa: E402
from watershed_hydro_mcp.input_registry import INPUT_ID_PREFIX  # noqa: E402
from watershed_hydro_mcp.models import HydroRunRequest, ParameterOverrides  # noqa: E402
from watershed_hydro_mcp.planner import (  # noqa: E402
    MODE_STAGES, STAGE_IDS, PlanningError, STAGES, check_engine, stages_for,
)
from watershed_hydro_mcp.settings import Settings  # noqa: E402
from watershed_hydro_mcp.state import envelope, failure, read_json, write_json  # noqa: E402
from watershed_hydro_mcp.visualization import catalog as viz_catalog  # noqa: E402
from watershed_hydro_mcp.visualization.renderer import RENDERERS  # noqa: E402
from watershed_hydro_mcp.visualization.service import VisualizationService  # noqa: E402


PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def temporary_settings(root: Path) -> Settings:
    runs = root / "runs"
    return Settings(root=root, runs_root=runs, hydro_root=runs / "hydro",
                    native_root=root / "native", config_root=Path(
                        __file__).resolve().parents[2] / "config")


class StageWhitelistTests(unittest.TestCase):
    def test_whitelist_matches_the_planned_pipeline(self):
        self.assertEqual(STAGE_IDS, (
            "sync", "validate", "delineate", "rainfall", "build",
            "rainfall_plot", "swmm", "safety", "analysis", "export", "qa",
        ))

    def test_every_mode_is_a_whitelist_subset(self):
        for mode, selected in MODE_STAGES.items():
            for stage_id in selected:
                self.assertIn(stage_id, STAGE_IDS, f"{mode} names {stage_id}")

    def test_modes_are_ordered_subsequences(self):
        for mode, selected in MODE_STAGES.items():
            ordered = [stage.stage_id for stage in STAGES if stage.stage_id in selected]
            self.assertEqual(list(selected), ordered, mode)

    def test_preflight_stops_before_modelling(self):
        ids = [stage.stage_id for stage in stages_for("preflight_only", None)]
        self.assertEqual(ids, ["sync", "validate"])

    def test_rainfall_mode_excludes_modelling(self):
        ids = [stage.stage_id for stage in stages_for("rainfall_only", None)]
        for excluded in ("build", "swmm", "safety", "analysis"):
            self.assertNotIn(excluded, ids)

    def test_no_stage_module_can_be_supplied_by_a_client(self):
        for stage in STAGES:
            self.assertTrue(stage.module.startswith("watershed_hydro_mcp.core.")
                            or stage.module == "watershed_hydro_mcp.qa")


class RequestValidationTests(unittest.TestCase):
    def test_input_id_prefix_is_enforced(self):
        with self.assertRaises(ValueError):
            HydroRunRequest(input_id="basin-abcdef", mode="build_run_qa")

    def test_mode_is_constrained(self):
        with self.assertRaises(ValueError):
            HydroRunRequest(input_id="hydro-input-abcdef123456", mode="everything")

    def test_overrides_drop_absent_fields(self):
        overrides = ParameterOverrides(saturated_cohesion_kpa=15.0).present()
        self.assertEqual(overrides, {"saturated_cohesion_kpa": 15.0})


class ParameterBoundsTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.settings = temporary_settings(Path(self.temp.name))
        self.defaults = {"stability": {"saturated_cohesion_kpa": 12.0}}

    def tearDown(self):
        self.temp.cleanup()

    def test_preset_is_applied(self):
        result = planner._override_stability(self.defaults, "medium", {}, self.settings)
        self.assertEqual(result["parameter_preset"], "medium")
        self.assertEqual(result["saturated_cohesion_kpa"], 6.0)

    def test_override_within_bounds_is_applied(self):
        result = planner._override_stability(
            self.defaults, "medium", {"saturated_cohesion_kpa": 20.0}, self.settings)
        self.assertEqual(result["saturated_cohesion_kpa"], 20.0)

    def test_override_outside_bounds_is_refused(self):
        with self.assertRaises(PlanningError):
            planner._override_stability(
                self.defaults, "medium", {"saturated_cohesion_kpa": 500.0}, self.settings)

    def test_unknown_preset_is_refused(self):
        with self.assertRaises(PlanningError):
            planner._override_stability(self.defaults, "nonexistent", {}, self.settings)


class EventWindowTests(unittest.TestCase):
    def test_missing_event_falls_back_to_one_day(self):
        start, end, sim_end = planner._event_times({})
        self.assertEqual((end - start).total_seconds() / 3600, 23.0)
        self.assertGreater(sim_end, end)

    def test_reversed_window_is_refused(self):
        with self.assertRaises(PlanningError):
            planner._event_times({"event": {"start_datetime": "2024-06-02 00:00",
                                            "end_datetime": "2024-06-01 00:00"}})

    def test_window_longer_than_the_station_format_is_refused(self):
        with self.assertRaises(PlanningError):
            planner._event_times({"event": {"start_datetime": "2024-01-01 00:00",
                                            "end_datetime": "2024-02-15 00:00"}})

    def test_explicit_simulation_end_is_honoured(self):
        _, _, sim_end = planner._event_times({"event": {
            "start_datetime": "2024-06-01 00:00", "end_datetime": "2024-06-01 05:00",
            "simulation_end_datetime": "2024-06-01 12:00:00"}})
        self.assertEqual(sim_end.hour, 12)


class EngineCheckTests(unittest.TestCase):
    def test_engine_present_in_this_project(self):
        report = check_engine()
        self.assertTrue(report["ok"], report)
        self.assertTrue(all(report["files"].values()))

    def test_missing_engine_is_reported_not_raised(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            settings = temporary_settings(Path(temp))
            report = check_engine(settings)
            self.assertFalse(report["ok"])


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.project = Path(self.temp.name)
        self.gates = planner.load_gates()

    def tearDown(self):
        self.temp.cleanup()

    def test_empty_project_fails_every_gate(self):
        report = qa.run_gates(self.project, ["delineate", "rainfall", "swmm", "safety", "export"])
        self.assertFalse(report["passed"])
        self.assertIn("spatial", report["failed"])

    def test_gates_are_skipped_for_stages_that_did_not_run(self):
        report = qa.run_gates(self.project, ["delineate"])
        self.assertEqual([item["name"] for item in report["gates"]], ["spatial"])

    def test_single_dominant_subcatchment_fails_the_spatial_gate(self):
        directory = self.project / "inputs/dem_derived"
        directory.mkdir(parents=True)
        (directory / "subcatchments_summary.csv").write_text(
            "name,area_ha,source\nG001,900.0,test\nG002,10.0,test\n", encoding="utf-8")
        result = qa.spatial_gate(self.project, self.gates["spatial"])
        self.assertFalse(result["passed"])
        self.assertTrue(any("最大子汇水区" in item for item in result["errors"]))

    def test_balanced_subcatchments_pass_the_spatial_gate(self):
        directory = self.project / "inputs/dem_derived"
        directory.mkdir(parents=True)
        rows = "\n".join(f"G{index:03d},50.0,test" for index in range(1, 21))
        (directory / "subcatchments_summary.csv").write_text(
            f"name,area_ha,source\n{rows}\n", encoding="utf-8")
        result = qa.spatial_gate(self.project, self.gates["spatial"])
        self.assertTrue(result["passed"], result["errors"])

    def test_rainfall_gate_reads_station_tables(self):
        directory = self.project / "inputs/rainfall"
        directory.mkdir(parents=True)
        (directory / "event_station_rainfall_summary.csv").write_text(
            "station_key,station_id,station_name,lon,lat,total_mm,peak_mm_h\n"
            "ST01,VS01,雨站01,108.0,30.5,12.5,3.2\n"
            "ST02,VS02,雨站02,108.25,30.25,10.0,2.8\n", encoding="utf-8")
        (directory / "event_station_observations_long.csv").write_text(
            "station_key,station_id,station_name,lon,lat,datetime_beijing,rain_mm\n"
            "ST01,VS01,雨站01,108.0,30.5,2024-06-01 00:00:00,1.0\n"
            "ST01,VS01,雨站01,108.0,30.5,2024-06-01 01:00:00,2.2\n"
            "ST02,VS02,雨站02,108.25,30.25,2024-06-01 00:00:00,0.5\n"
            "ST02,VS02,雨站02,108.25,30.25,2024-06-01 01:00:00,2.3\n", encoding="utf-8")
        (directory / "event_active_hourly.csv").write_text(
            "datetime_beijing,rain_mm\n2024-06-01 00:00,1.5\n2024-06-01 01:00,1.0\n",
            encoding="utf-8")
        result = qa.rainfall_gate(self.project, self.gates["rainfall"])
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["facts"]["stations"], 2)
        self.assertEqual(result["facts"]["timestamps"], 2)

    def test_rainfall_gate_labels_single_station_and_respects_stricter_config(self):
        directory = self.project / "inputs/rainfall"
        directory.mkdir(parents=True)
        (directory / "event_station_rainfall_summary.csv").write_text(
            "station_key,station_id,station_name,lon,lat,total_mm,peak_mm_h\n"
            "ST01,VS01,雨站01,108.0,30.5,12.5,3.2\n", encoding="utf-8")
        (directory / "event_station_observations_long.csv").write_text(
            "station_key,datetime_beijing,rain_mm\n"
            "ST01,2024-06-01 00:00:00,1.0\nST01,2024-06-01 01:00:00,2.2\n", encoding="utf-8")
        (directory / "event_active_hourly.csv").write_text(
            "datetime_beijing,rain_mm\n2024-06-01 00:00,1.5\n", encoding="utf-8")
        result = qa.rainfall_gate(self.project, self.gates["rainfall"])
        self.assertTrue(result["passed"],result)
        self.assertEqual(result["facts"]["spatial_rainfall_assumption"],"single_station_uniform")
        result = qa.rainfall_gate(self.project, {**self.gates["rainfall"],"minimum_stations":2})
        self.assertFalse(result["passed"])
        self.assertTrue(any("雨站" in item for item in result["errors"]))

    def test_absurd_rainfall_is_flagged(self):
        directory = self.project / "inputs/rainfall"
        directory.mkdir(parents=True)
        (directory / "event_station_rainfall_summary.csv").write_text(
            "station_key,station_id,station_name,lon,lat,total_mm,peak_mm_h\n"
            "ST01,VS01,雨站01,108.0,30.5,9000.0,9000.0\n"
            "ST02,VS02,雨站02,108.25,30.25,10.0,2.8\n", encoding="utf-8")
        (directory / "event_station_observations_long.csv").write_text(
            "station_key,datetime_beijing,rain_mm\n"
            "ST01,2024-06-01 00:00:00,9000.0\nST01,2024-06-01 01:00:00,2.2\n"
            "ST02,2024-06-01 00:00:00,0.5\nST02,2024-06-01 01:00:00,2.3\n", encoding="utf-8")
        (directory / "event_active_hourly.csv").write_text(
            "datetime_beijing,rain_mm\n2024-06-01 00:00,1.5\n2024-06-01 01:00,1.0\n",
            encoding="utf-8")
        result = qa.rainfall_gate(self.project, self.gates["rainfall"])
        self.assertFalse(result["passed"])

    def _write_report(self, body: str) -> None:
        directory = self.project / "swmm/results"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.rpt").write_text(body, encoding="utf-8")

    def _write_inp(self) -> None:
        directory = self.project / "swmm/model"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.inp").write_text(
            "[OUTFALLS]\n;;Name Elevation Type\nO001 291.1 FREE\nO002 274.8 FREE\n",
            encoding="utf-8")

    def _report_body(self, error_lines: str = "", subcatchments: int = 3,
                     continuity: float = -0.12) -> str:
        rows = "\n".join(
            f"  S{index:03d}" + "  79.30       0.00       0.00      72.96       6.26       0.00       6.26       2.74       0.07   0.079"
            for index in range(1, subcatchments + 1)
        )
        return (
            "  Runoff Quantity Continuity\n  **************************\n"
            f"  Continuity Error (%) ..... {continuity}\n"
            "  Subcatchment Runoff Summary\n  ***************************\n\n"
            "  ------------------------------------------------------------------\n"
            "                            Total      Total      Total      Total\n"
            "                           Precip      Runon       Evap      Infil\n"
            "  Subcatchment                 mm         mm         mm         mm\n"
            "  ------------------------------------------------------------------\n"
            f"{rows}\n\n\n"
            f"{error_lines}"
        )

    def test_swmm_gate_accepts_a_healthy_report(self):
        self._write_report(self._report_body())
        self._write_inp()
        result = qa.swmm_gate(self.project, self.gates["swmm"])
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["facts"]["subcatchment_rows"], 3)
        self.assertEqual(result["facts"]["outfall_count"], 2)

    def test_swmm_gate_flags_error_lines(self):
        self._write_report(self._report_body(error_lines="ERROR 04: some fatal problem\n"))
        self._write_inp()
        result = qa.swmm_gate(self.project, self.gates["swmm"])
        self.assertFalse(result["passed"])
        self.assertTrue(any("错误行" in item for item in result["errors"]))

    def test_swmm_gate_flags_large_continuity_error(self):
        self._write_report(self._report_body(continuity=42.5))
        self._write_inp()
        result = qa.swmm_gate(self.project, self.gates["swmm"])
        self.assertFalse(result["passed"])
        self.assertTrue(any("连续性误差" in item for item in result["errors"]))

    def test_swmm_gate_flags_a_model_without_outfalls(self):
        directory = self.project / "swmm/model"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.inp").write_text("[OUTFALLS]\n;;Name Elevation Type\n", encoding="utf-8")
        self._write_report(self._report_body())
        result = qa.swmm_gate(self.project, self.gates["swmm"])
        self.assertFalse(result["passed"])
        self.assertTrue(any("出水口" in item for item in result["errors"]))

    def test_stability_gate_reads_the_summary(self):
        directory = self.project / "rapid_safety_factor_results"
        directory.mkdir(parents=True)
        (directory / "event_enhanced_safety_factor_summary.csv").write_text(
            "metric,value\nminimum_safety_factor_min,0.42\nminimum_safety_factor_mean,1.43\n"
            "minimum_safety_factor_p95,2.10\nminimum_safety_factor_max,24.0\n"
            "max_unstable_fraction,0.05\nmechanics_model,reference_equation_9\n"
            "parameter_preset,medium\nswmm_dynamic_source,subcatchment_hydrograph\n"
            "swmm_dynamic_hydrograph_used,True\n", encoding="utf-8")
        result = qa.stability_gate(self.project, self.gates["stability"])
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual(result["facts"]["minimum_safety_factor"], 0.42)
        self.assertEqual(result["facts"]["mean_safety_factor"], 1.43)

    def test_stability_gate_flags_a_nonpositive_factor(self):
        directory = self.project / "rapid_safety_factor_results"
        directory.mkdir(parents=True)
        (directory / "event_enhanced_safety_factor_summary.csv").write_text(
            "metric,value\nminimum_safety_factor_min,0.0\nminimum_safety_factor_max,10.0\n",
            encoding="utf-8")
        result = qa.stability_gate(self.project, self.gates["stability"])
        self.assertFalse(result["passed"])

    def test_stability_gate_requires_the_global_minimum(self):
        directory = self.project / "rapid_safety_factor_results"
        directory.mkdir(parents=True)
        (directory / "event_enhanced_safety_factor_summary.csv").write_text(
            "metric,value\nminimum_safety_factor_mean,1.4\nminimum_safety_factor_max,10.0\n",
            encoding="utf-8")
        result = qa.stability_gate(self.project, self.gates["stability"])
        self.assertFalse(result["passed"])
        self.assertTrue(any("全局最小值" in item for item in result["errors"]))

    def test_export_gate_requires_present_files(self):
        directory = self.project / "final_results"
        directory.mkdir(parents=True)
        (directory / "present.csv").write_text("a\n1\n", encoding="utf-8")
        rows = "\n".join(
            f"role{index},{self.project}/src{index}.csv,{directory}/present.csv,0.001"
            for index in range(6)
        )
        (directory / "final_results_manifest.csv").write_text(
            f"role,source,final_file,size_mb\n{rows}\n", encoding="utf-8")
        result = qa.export_gate(self.project, self.gates["export"])
        self.assertTrue(result["passed"], result["errors"])

    def test_export_gate_detects_a_missing_exported_file(self):
        directory = self.project / "final_results"
        directory.mkdir(parents=True)
        rows = "\n".join(
            f"role{index},{self.project}/src{index}.csv,{directory}/gone{index}.csv,0.001"
            for index in range(6)
        )
        (directory / "final_results_manifest.csv").write_text(
            f"role,source,final_file,size_mb\n{rows}\n", encoding="utf-8")
        result = qa.export_gate(self.project, self.gates["export"])
        self.assertFalse(result["passed"])
        self.assertTrue(any("不存在" in item for item in result["errors"]))

    def test_gate_report_round_trips(self):
        report = qa.run_gates(self.project, ["delineate"])
        qa.write_gates(self.project, report)
        self.assertEqual(qa.read_gates(self.project)["passed"], report["passed"])

    def test_a_crashing_gate_is_reported_not_raised(self):
        original = qa.GATES
        try:
            qa.GATES = (("spatial", lambda project, gates: (_ for _ in ()).throw(RuntimeError("boom"))),)
            report = qa.run_gates(self.project, ["delineate"])
        finally:
            qa.GATES = original
        self.assertFalse(report["passed"])
        self.assertTrue(any("执行失败" in item for item in report["gates"][0]["errors"]))


class ArtifactRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.job = Path(self.temp.name) / "job"
        self.project = self.job / "project"
        self.project.mkdir(parents=True)
        self.registry = ArtifactRegistry(self.job, "hydro-run-test0001", self.project)

    def tearDown(self):
        self.temp.cleanup()

    def test_registered_artifact_records_a_hash(self):
        path = self.project / "out.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        record = self.registry.register(path, "export")
        self.assertEqual(record["internal_path"], "out.csv")
        self.assertEqual(len(record["sha256"]), 64)
        self.assertGreater(record["size_bytes"], 0)
        self.assertTrue(record["ready"])

    def test_registering_twice_does_not_duplicate(self):
        path = self.project / "out.csv"
        path.write_text("a\n1\n", encoding="utf-8")
        self.registry.register(path, "export")
        self.registry.register(path, "export")
        self.assertEqual(len(self.registry.all()), 1)

    def test_missing_file_is_not_registered(self):
        self.assertIsNone(self.registry.register(self.project / "absent.csv", "export"))

    def test_index_survives_a_reload(self):
        path = self.project / "out.csv"
        path.write_text("x\n", encoding="utf-8")
        record = self.registry.register(path, "export")
        reloaded = ArtifactRegistry(self.job, "hydro-run-test0001", self.project)
        self.assertEqual(reloaded.by_id(record["artifact_id"])["internal_path"], "out.csv")

    def test_renderers_are_reported(self):
        raster = self.project / "r.tif"
        write_dem(raster)
        record = self.registry.register(raster, "delineate")
        self.assertIn("raster_png", record["available_renderers"])
        self.assertIsNotNone(record["crs"])
        self.assertEqual(len(record["bbox"]), 4)

    def test_text_artifact_is_readable(self):
        path = self.project / "log.txt"
        path.write_text("hello", encoding="utf-8")
        record = describe(path, run_id="hydro-run-test0001", stage_id="export",
                          project_root=self.project)
        self.assertEqual(record["available_renderers"], ["text"])
        self.assertEqual(record["media_type"], "text/plain")


class VisualizationCatalogTests(unittest.TestCase):
    def test_every_entry_has_an_implemented_renderer(self):
        for item in viz_catalog.CATALOG:
            self.assertIn(item.renderer, RENDERERS, item.kind)

    def test_kinds_are_unique(self):
        kinds = [item.kind for item in viz_catalog.CATALOG]
        self.assertEqual(len(kinds), len(set(kinds)))

    def test_every_stage_has_at_least_one_visualization(self):
        covered = {item.stage_id for item in viz_catalog.CATALOG}
        for stage_id in ("sync", "validate", "delineate", "rainfall", "rainfall_plot",
                         "build", "swmm", "safety", "analysis", "qa", "export"):
            self.assertIn(stage_id, covered, stage_id)

    def test_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            viz_catalog.by_kind("no_such_plot")

    def test_catalog_exposes_no_code(self):
        for item in viz_catalog.catalog():
            self.assertNotIn("renderer", item)
            self.assertNotIn("callable", item)


class VisualizationServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.temp.name)
        self.settings = temporary_settings(self.root)
        self.run_id = "hydro-run-viztest0001"
        job = self.settings.job_dir(self.run_id)
        (job / "project").mkdir(parents=True)
        write_json(job / "status.json", {"run_id": self.run_id, "status": "succeeded",
                                         "completed_stages": []})
        self.service = VisualizationService(self.settings)
        self.service.pump = lambda: None  # keep dispatch synchronous in the test

    def tearDown(self):
        self.temp.cleanup()

    def test_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            self.service.create(self.run_id, "not_a_kind")

    def test_unknown_run_is_refused(self):
        with self.assertRaises(ValueError):
            self.service.create("hydro-run-missing0001", "export_index")

    def test_image_requires_a_finished_render(self):
        record = {"visualization_id": "viz-test00000001", "status": "queued"}
        directory = self.settings.visualizations_root / "viz-test00000001"
        directory.mkdir(parents=True)
        write_json(directory / "visualization.json", record)
        with self.assertRaises(ValueError):
            self.service.image_path("viz-test00000001")

    def test_compare_needs_two_ids(self):
        with self.assertRaises(ValueError):
            self.service.compare(["viz-only-one"])

    def test_compare_reports_shared_sources(self):
        directory = self.settings.visualizations_root
        for index in (1, 2):
            folder = directory / f"viz-compare{index:06d}"
            folder.mkdir(parents=True)
            (folder / "preview.png").write_bytes(PNG_1PX)
            write_json(folder / "visualization.json", {
                "visualization_id": f"viz-compare{index:06d}",
                "run_id": self.run_id, "kind": "export_index", "status": "succeeded",
                "created_at": f"2026-01-0{index}T00:00:00+00:00",
                "title": "成果索引",
                "image": "preview.png",
                "image_sha256": "a" * 64,
                "sources": [{"relative_path": "final_results/x.csv", "sha256": "b" * 64,
                             "size_bytes": 1}],
            })
        result = self.service.compare(["viz-compare000001", "viz-compare000002"])
        self.assertTrue(result["same_sources"])
        self.assertEqual(result["differences"], [])
        self.assertTrue(self.service.image_path(result["comparison_visualization_id"]).is_file())

    def test_compare_detects_changed_sources(self):
        directory = self.settings.visualizations_root
        for index, digest in ((1, "b" * 64), (2, "c" * 64)):
            folder = directory / f"viz-changed{index:06d}"
            folder.mkdir(parents=True)
            (folder / "preview.png").write_bytes(PNG_1PX)
            write_json(folder / "visualization.json", {
                "visualization_id": f"viz-changed{index:06d}",
                "run_id": f"hydro-run-{index}", "kind": "export_index", "status": "succeeded",
                "created_at": "2026-01-01T00:00:00+00:00",
                "title": "成果索引",
                "image": "preview.png",
                "image_sha256": "a" * 64,
                "sources": [{"relative_path": "final_results/x.csv", "sha256": digest,
                             "size_bytes": 1}],
            })
        result = self.service.compare(["viz-changed000001", "viz-changed000002"])
        self.assertFalse(result["same_sources"])
        self.assertTrue(result["differences"])


class RegressionTests(unittest.TestCase):
    def test_swmm_inspection_tool_calls_the_result_parser_not_itself(self):
        available = {
            "available": True, "subcatchment_count": 3, "outfall_count": 1,
            "routing_network_present": False, "limitation": "独立出水口",
        }
        with mock.patch.object(hydro_server_module.STORE, "status", return_value={"status": "succeeded"}), \
                mock.patch.object(hydro_server_module, "inspect_swmm_model_result",
                                  return_value=available) as parser:
            result = hydro_server_module.inspect_swmm_model("hydro-run-regression01")
        self.assertTrue(result["ok"])
        self.assertEqual(result["facts"]["subcatchments"], 3)
        parser.assert_called_once()

    def test_windows_cancellation_terminates_the_complete_process_tree(self):
        process = mock.Mock()
        process.pid = 4321
        process.poll.return_value = None
        process.wait.return_value = 0
        with mock.patch.object(job_manager.os, "name", "nt"), \
                mock.patch.object(job_manager.subprocess, "run") as taskkill:
            job_manager.JobStore._terminate_process_tree(process)
        taskkill.assert_called_once()
        self.assertEqual(taskkill.call_args.args[0],
                         ["taskkill", "/PID", "4321", "/T", "/F"])


class EnvelopeTests(unittest.TestCase):
    def test_envelope_has_every_documented_key(self):
        payload = envelope(ok=True, status="succeeded", summary="ok")
        self.assertEqual(set(payload), {"ok", "status", "summary", "facts", "warnings",
                                        "limitations", "next_actions", "data"})

    def test_failure_sets_ok_false(self):
        payload = failure("bad input")
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status"], "failed")


if __name__ == "__main__":
    unittest.main()


class AtomicWriteTests(unittest.TestCase):
    """A reader must never be able to kill a writer.

    On Windows ``os.replace`` fails with a sharing violation while another
    process holds the destination open for reading, and polling a job's
    ``status.json`` does exactly that. Without a retry, a worker died on its own
    progress write and a finished computation was lost — observed in the real
    process test as ``PermissionError: [WinError 5] ... .tmp-*.json -> status.json``
    followed by a job stuck at ``running``. This pins the retry down.
    """

    def test_a_write_survives_a_concurrent_reader(self):
        import threading
        import time

        with TemporaryDirectory() as temp:
            path = Path(temp) / "status.json"
            write_json(path, {"status": "queued"})

            holding = threading.Event()

            def hold_open() -> None:
                with path.open("r", encoding="utf-8") as stream:
                    stream.read()
                    holding.set()
                    # Keep the handle open across the writer's first attempt.
                    time.sleep(0.4)

            reader = threading.Thread(target=hold_open, daemon=True)
            reader.start()
            self.assertTrue(holding.wait(5.0), "读者线程没能打开文件")
            try:
                write_json(path, {"status": "running"})
            finally:
                reader.join(5.0)

            self.assertEqual(read_json(path)["status"], "running")

    def test_a_half_written_file_is_never_left_behind(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "status.json"
            write_json(path, {"status": "queued"})
            write_json(path, {"status": "running", "completed_stages": ["sync"]})
            # The reader sees one complete document, never a merge of the two.
            self.assertEqual(read_json(path),
                             {"status": "running", "completed_stages": ["sync"]})
            self.assertEqual(list(Path(temp).glob(".tmp-*")), [])
