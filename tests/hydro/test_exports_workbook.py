"""Derived downloads: a workbook from a table, and data from a comparison.

Two things are being protected here. A workbook is a real spreadsheet that says
where its numbers came from and refuses rather than truncating. A numeric
comparison is *data*, written out and registered with the rule that produced it —
a picture of a difference is not the difference.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import write_dem  # noqa: E402

from watershed_hydro_mcp import workbook as workbook_module  # noqa: E402
from watershed_hydro_mcp.artifact_registry import ArtifactRegistry  # noqa: E402
from watershed_hydro_mcp.export_service import ExportError, ExportService  # noqa: E402
from watershed_hydro_mcp.settings import Settings  # noqa: E402
from watershed_hydro_mcp.state import write_json  # noqa: E402
from watershed_hydro_mcp.visualization.service import VisualizationService  # noqa: E402


RUN_ID = "hydro-run-abcdef123456"


def temporary_settings(root: Path) -> Settings:
    runs = root / "runs"
    return Settings(root=root, runs_root=runs, hydro_root=runs / "hydro",
                    native_root=root / "native", config_root=Path(
                        __file__).resolve().parents[2] / "config")


def a_csv(path: Path, rows: int = 3, *, timezone_column: str = "datetime_beijing") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{timezone_column},representative_fs,note"]
    lines += [f"2024-07-30 {hour:02d}:00:00,{0.27 + hour / 100:.4f},-"
              for hour in range(rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def a_geotiff(path: Path, *, factor: float = 1.0) -> Path:
    import rasterio

    path.parent.mkdir(parents=True, exist_ok=True)
    write_dem(path)
    with rasterio.open(path) as source:
        values = source.read(1)
        profile = source.profile.copy()
    with rasterio.open(path, "w", **profile) as sink:
        sink.write(values * factor, 1)
    return path


class WorkbookTests(unittest.TestCase):
    """A workbook is a real spreadsheet that says where its numbers came from."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _build(self, text: str, **metadata):
        source = self.root / "summary.csv"
        source.write_text(text, encoding="utf-8")
        target = self.root / "summary.xlsx"
        info = workbook_module.build_workbook(source, target, metadata={
            "run_id": RUN_ID, "artifact_id": "artifact-abcdef123456",
            "source_title": "summary.csv", "source_sha256": "a" * 64,
            "qa_status": "passed", **metadata})
        return target, info

    def test_it_is_a_real_workbook_not_a_renamed_csv(self):
        from openpyxl import load_workbook

        target, _ = self._build("a,b\n1,x\n")
        self.assertEqual(target.read_bytes()[:2], b"PK")
        self.assertEqual(load_workbook(target).sheetnames[0], "元数据")

    def test_the_provenance_sheet_names_the_source_and_its_digest(self):
        from openpyxl import load_workbook

        target, _ = self._build("a,b\n1,x\n")
        rows = dict(row for row in load_workbook(target)["元数据"].iter_rows(values_only=True))
        self.assertEqual(rows["运行编号"], RUN_ID)
        self.assertEqual(rows["来源成果编号"], "artifact-abcdef123456")
        self.assertEqual(rows["来源 SHA-256"], "a" * 64)
        self.assertEqual(rows["QA 状态"], "passed")
        self.assertIn("Asia/Shanghai", str(rows["时间列时区"]))
        self.assertIn("派生文件", str(rows["文件性质"]))

    def test_the_data_sheet_is_frozen_filtered_and_uniquely_named(self):
        from openpyxl import load_workbook

        target, _ = self._build("a,a,b\n1,2,3\n4,5,6\n")
        sheet = load_workbook(target)["summary"]
        header = [cell.value for cell in next(sheet.iter_rows(max_row=1))]
        self.assertEqual(header, ["a", "a_1", "b"])
        self.assertEqual(sheet.freeze_panes, "A2")
        self.assertEqual(sheet.auto_filter.ref, "A1:C1")

    def test_values_keep_their_kind(self):
        from openpyxl import load_workbook

        target, _ = self._build(
            "datetime_beijing,value,note\n"
            "2024-07-30 08:00:00,0.2752,ok\n"
            "2024-07-30 09:00:00,NaN,\n")
        rows = list(load_workbook(target)["summary"].iter_rows(min_row=2, values_only=True))
        self.assertIsInstance(rows[0][0], datetime)
        self.assertEqual(rows[0][1], 0.2752)
        self.assertIsNone(rows[1][1])
        self.assertIsNone(rows[1][2])

    def test_a_table_too_large_is_refused_and_leaves_nothing_behind(self):
        body = "a,b\n" + "".join(f"{index},x\n" for index in range(60))
        with mock.patch.object(workbook_module, "MAXIMUM_ROWS", 10):
            with self.assertRaises(workbook_module.WorkbookTooLarge) as caught:
                self._build(body)
        self.assertIn("CSV", str(caught.exception))
        self.assertFalse((self.root / "summary.xlsx").exists())
        self.assertEqual(list(self.root.glob("*.partial")), [])

    def test_an_empty_table_is_refused(self):
        with self.assertRaises(workbook_module.WorkbookTooLarge):
            self._build("")

    def test_a_worksheet_name_is_made_acceptable(self):
        self.assertEqual(workbook_module.sheet_title("a/b:c*d?e[f]g"), "abcdefg")
        self.assertLessEqual(len(workbook_module.sheet_title("x" * 80)), 31)


class PrepareExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = temporary_settings(self.root)
        self.exports = ExportService(self.settings)
        self.job_dir = self.settings.job_dir(RUN_ID)
        self.project = self.job_dir / "project"
        self.project.mkdir(parents=True)
        self.registry = ArtifactRegistry(self.job_dir, RUN_ID, self.project)

    def tearDown(self):
        self.temp.cleanup()

    def _a_registered_table(self) -> dict:
        path = a_csv(self.project / "analysis_results/summary.csv")
        return self.registry.register(path, "analysis")

    def test_a_csv_becomes_a_registered_workbook(self):
        source = self._a_registered_table()
        result = self.exports.prepare_export(RUN_ID, source["artifact_id"], "xlsx")
        derived = result["artifact"]
        self.assertFalse(result["reused"])
        self.assertEqual(derived["format"], "xlsx")
        self.assertEqual(derived["exportable"], [])
        stored = ArtifactRegistry(self.job_dir, RUN_ID, self.project).by_id(
            derived["artifact_id"])
        self.assertEqual(stored["derived_from_artifact_ids"], [source["artifact_id"]])
        self.assertEqual(stored["derivation"]["from"], "csv")
        self.assertIn("workbook", stored["producer"])
        self.assertTrue(stored["producer_version"])
        self.assertEqual(stored["derivation"]["source_sha256"], source["sha256"])

    def test_asking_twice_reuses_the_same_artifact(self):
        source = self._a_registered_table()
        first = self.exports.prepare_export(RUN_ID, source["artifact_id"], "xlsx")
        second = self.exports.prepare_export(RUN_ID, source["artifact_id"], "xlsx")
        self.assertTrue(second["reused"])
        self.assertEqual(first["artifact"]["artifact_id"], second["artifact"]["artifact_id"])

    def test_a_changed_source_is_not_reused(self):
        """A workbook made from a file that has since changed is a different
        workbook, and must not be handed over under the old id."""
        source = self._a_registered_table()
        first = self.exports.prepare_export(RUN_ID, source["artifact_id"], "xlsx")
        a_csv(self.project / "analysis_results/summary.csv", rows=9)
        replaced = self.registry.register(
            self.project / "analysis_results/summary.csv", "analysis")
        second = self.exports.prepare_export(RUN_ID, replaced["artifact_id"], "xlsx")
        self.assertFalse(second["reused"])
        self.assertNotEqual(first["artifact"]["artifact_id"],
                            second["artifact"]["artifact_id"])

    def test_a_raster_has_no_workbook_form(self):
        raster = self.registry.register(
            a_geotiff(self.project / "rapid_safety_factor_results/fs.tif"), "safety")
        with self.assertRaises(ExportError):
            self.exports.prepare_export(RUN_ID, raster["artifact_id"], "xlsx")

    def test_an_unknown_format_is_refused_by_name(self):
        source = self._a_registered_table()
        with self.assertRaises(ExportError) as caught:
            self.exports.prepare_export(RUN_ID, source["artifact_id"], "docx")
        self.assertIn("docx", str(caught.exception))

    def test_an_unknown_artifact_is_refused(self):
        with self.assertRaises(ExportError):
            self.exports.prepare_export(RUN_ID, "artifact-000000000000", "xlsx")

    def test_no_partial_workbook_is_left_behind(self):
        self._a_registered_table()
        self.exports.prepare_export(RUN_ID, self.registry.all()[0]["artifact_id"], "xlsx")
        self.assertEqual(list(self.project.glob("exports/*.partial")), [])


class ComparisonRasterTests(unittest.TestCase):
    """A difference is data. The figure is only a picture of it.

    A numeric comparison is between the *same* source file from two different
    runs — the service requires the relative names to match, because comparing
    two unrelated rasters would produce a difference that means nothing.
    """

    RUN_A = "hydro-run-aaaaaa111111"
    RUN_B = "hydro-run-bbbbbb222222"
    RELATIVE = "rapid_safety_factor_results/pkg_enhanced_minimum_safety_factor.tif"

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = temporary_settings(self.root)
        self.service = VisualizationService(self.settings)
        for run_id in (self.RUN_A, self.RUN_B):
            project = self.settings.job_dir(run_id) / "project"
            project.mkdir(parents=True)
            write_json(self.settings.job_dir(run_id) / "status.json",
                       {"run_id": run_id, "status": "succeeded"})

    def tearDown(self):
        self.temp.cleanup()

    def _a_figure_in(self, run_id: str, *, factor: float) -> str:
        project = self.settings.job_dir(run_id) / "project"
        path = a_geotiff(project / self.RELATIVE, factor=factor)
        record = ArtifactRegistry(self.settings.job_dir(run_id), run_id,
                                  project).register(path, "safety")
        visualization_id = f"viz-{run_id[-6:]}{abs(hash(run_id)) % 10**4:04d}"
        directory = self.service.directory(visualization_id)
        directory.mkdir(parents=True, exist_ok=True)
        # A side-by-side comparison embeds the finished figure, so a real one
        # has to exist: a stub header would only prove that the reader crashes.
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        figure = plt.figure(figsize=(1, 1))
        figure.savefig(directory / f"{visualization_id}.png", dpi=40)
        plt.close(figure)
        write_json(directory / "visualization.json", {
            "visualization_id": visualization_id, "run_id": run_id,
            "kind": "safety_minimum_factor", "stage_id": "safety",
            "title": "最小安全系数", "status": "succeeded",
            "sources": [{"scope": "project", "relative_path": self.RELATIVE,
                         "sha256": record["sha256"], "size_bytes": record["size_bytes"]}],
            "image": f"{visualization_id}.png", "image_sha256": "a" * 64,
            "qa_failed": False, "limitations": [],
        })
        return visualization_id

    def test_a_difference_is_written_out_and_registered(self):
        import rasterio

        before = self._a_figure_in(self.RUN_A, factor=1.0)
        after = self._a_figure_in(self.RUN_B, factor=0.5)
        result = self.service.compare([before, after], "difference")

        stored = self.service.status(result["comparison_visualization_id"])
        self.assertEqual(len(stored["source_artifact_ids"]), 1)
        registry = ArtifactRegistry(self.settings.job_dir(stored["run_id"]),
                                    stored["run_id"],
                                    self.settings.job_dir(stored["run_id"]) / "project")
        derived = registry.by_id(stored["source_artifact_ids"][0])
        self.assertEqual(derived["kind"], "comparison")
        self.assertEqual(derived["units"], "dimensionless")
        self.assertEqual(len(derived["derived_from_artifact_ids"]), 2)
        self.assertEqual(derived["derivation"]["formula"], "after - before")
        self.assertIn("mask_rule", derived["derivation"])

        path = registry.resolve(derived)
        self.assertTrue(path.is_file())
        with rasterio.open(path) as sink:
            values = sink.read(1, masked=True)
            self.assertEqual(sink.crs.to_string(), "EPSG:32649")
            self.assertEqual(sink.nodata, -9999.0)
            # after = before * 0.5, so every difference is negative.
            self.assertLess(float(values.mean()), 0)

    def test_a_percent_change_records_its_zero_rule(self):
        before = self._a_figure_in(self.RUN_A, factor=1.0)
        after = self._a_figure_in(self.RUN_B, factor=1.2)
        result = self.service.compare([before, after], "percent_change")
        stored = self.service.status(result["comparison_visualization_id"])
        registry = ArtifactRegistry(self.settings.job_dir(stored["run_id"]),
                                    stored["run_id"],
                                    self.settings.job_dir(stored["run_id"]) / "project")
        derived = registry.by_id(stored["source_artifact_ids"][0])
        self.assertIn("zero_rule", derived["derivation"])
        self.assertIn("abs(before)", derived["derivation"]["formula"])

    def test_a_side_by_side_comparison_derives_no_new_raster(self):
        """A layout is not a computation: nothing numeric is created for it.

        What it *does* offer is the two figures' own source files, because
        "compare these two" is a request to look at both of them.
        """
        before = self._a_figure_in(self.RUN_A, factor=1.0)
        after = self._a_figure_in(self.RUN_B, factor=0.5)
        result = self.service.compare([before, after], "side_by_side")
        stored = self.service.status(result["comparison_visualization_id"])
        self.assertNotIn("derivation", stored)
        self.assertEqual(len(stored["source_artifact_ids"]), 2)
        for run_id in (self.RUN_A, self.RUN_B):
            project = self.settings.job_dir(run_id) / "project"
            self.assertEqual(list(project.glob("comparison_results/*")), [])

    def test_no_partial_raster_is_left_behind(self):
        before = self._a_figure_in(self.RUN_A, factor=1.0)
        after = self._a_figure_in(self.RUN_B, factor=0.5)
        self.service.compare([before, after], "difference")
        for run_id in (self.RUN_A, self.RUN_B):
            project = self.settings.job_dir(run_id) / "project"
            self.assertEqual(list(project.glob("comparison_results/*.partial")), [])
