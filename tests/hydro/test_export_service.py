"""What a caller may download, and what must never reach it.

The rule this file exists to hold: a download is a **registered artifact of this
run**, reached by an opaque id. Nothing is inferred from a file name, nothing is
promised that is not on disk, and no server path travels with the descriptor —
because a caller that cannot see a path cannot ask for one.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import write_dem, write_polygon_shapefile  # noqa: E402

from watershed_hydro_mcp.artifact_registry import (  # noqa: E402
    ArtifactRegistry,
    _available_renderers,
    describe,
    media_type,
    units_for,
)
from watershed_hydro_mcp.export_service import (  # noqa: E402
    ExportService,
    public_descriptor,
    role_for,
)
from watershed_hydro_mcp.settings import Settings  # noqa: E402
from watershed_hydro_mcp.state import write_json  # noqa: E402
from watershed_hydro_mcp.visualization.service import VisualizationService  # noqa: E402


RUN_ID = "hydro-run-abcdef123456"


def temporary_settings(root: Path) -> Settings:
    runs = root / "runs"
    return Settings(root=root, runs_root=runs, hydro_root=runs / "hydro",
                    native_root=root / "native", config_root=Path(
                        __file__).resolve().parents[2] / "config")


def a_geotiff(path: Path) -> Path:
    """A real GeoTIFF, in a directory that exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_dem(path)


def a_csv(path: Path, rows: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["datetime_beijing,representative_fs,units_note"]
    lines += [f"2024-07-30 {hour:02d}:00:00,{0.27 + hour / 100:.4f},-"
              for hour in range(rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class RegistryMetadataTests(unittest.TestCase):
    """The registered metadata has to describe the file, not the machine."""

    def test_a_csv_is_a_csv(self):
        """``mimetypes`` reads the Windows registry, where .csv belongs to Excel.

        The same run therefore used to register ``application/vnd.ms-excel`` on
        this machine and ``text/csv`` on another — and a browser handed the
        first value saves a comma-separated file under an ``.xls`` name.
        """
        self.assertEqual(media_type(Path("x.csv")), "text/csv")
        self.assertEqual(media_type(Path("x.tsv")), "text/tab-separated-values")

    def test_the_swmm_formats_are_stated_explicitly(self):
        self.assertEqual(media_type(Path("model.inp")), "text/plain")
        self.assertEqual(media_type(Path("model.rpt")), "text/plain")
        self.assertEqual(media_type(Path("model.out")), "application/octet-stream")
        self.assertEqual(media_type(Path("bundle.zip")), "application/zip")

    def test_a_csv_is_readable_as_text_and_as_a_table(self):
        """The table branch used to be unreachable: `.csv` is in TEXT_SUFFIXES,
        and the text test came first."""
        self.assertEqual(_available_renderers(Path("x.csv")), ["text", "table"])
        self.assertEqual(_available_renderers(Path("x.rpt")), ["text"])
        self.assertEqual(_available_renderers(Path("x.out")), [])

    def test_a_whole_file_unit_is_only_claimed_when_the_file_is_one_quantity(self):
        self.assertEqual(units_for(Path("pkg_enhanced_minimum_safety_factor.tif")),
                         "dimensionless")
        self.assertEqual(units_for(Path("pkg_enhanced_maximum_flow_depth_m.tif")), "m")
        self.assertEqual(units_for(Path("pkg_enhanced_maximum_velocity_ms.tif")), "m/s")
        self.assertEqual(units_for(Path("pkg_enhanced_maximum_pore_water_pressure_kpa.tif")),
                         "kPa")
        self.assertEqual(units_for(Path("pkg_enhanced_unstable_duration_h.tif")), "h")
        # A multi-column table has no single unit, and saying it does would be
        # an invention.
        self.assertIsNone(units_for(Path("pkg_active_hourly.csv")))


class DescribeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = temporary_settings(self.root)
        self.project = self.settings.job_dir(RUN_ID) / "project"
        self.project.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def test_raster_metadata_matches_what_rasterio_reads(self):
        import rasterio

        path = a_geotiff(self.project / "dem.tif")
        record = describe(path, run_id=RUN_ID, stage_id="validate",
                          project_root=self.project)
        with rasterio.open(path) as source:
            self.assertEqual(record["width"], source.width)
            self.assertEqual(record["height"], source.height)
            self.assertEqual(record["count"], source.count)
            self.assertEqual(record["dtype"], source.dtypes[0])
            self.assertEqual(record["nodata"], source.nodata)
            self.assertEqual(record["crs"], source.crs.to_string())
            for written, actual in zip(record["transform"], tuple(source.transform)[:6]):
                self.assertAlmostEqual(written, actual, places=6)
        self.assertEqual(record["units"], "m")

    def test_table_metadata_describes_the_columns(self):
        path = a_csv(self.project / "summary.csv")
        record = describe(path, run_id=RUN_ID, stage_id="safety",
                          project_root=self.project)
        self.assertEqual(record["columns"],
                         ["datetime_beijing", "representative_fs", "units_note"])
        self.assertEqual(record["rows"], 3)
        self.assertEqual(record["time_column"], "datetime_beijing")
        self.assertEqual(record["media_type"], "text/csv")

    def test_the_recorded_path_is_relative(self):
        path = a_csv(self.project / "summary.csv")
        record = describe(path, run_id=RUN_ID, stage_id="safety",
                          project_root=self.project)
        self.assertEqual(record["internal_path"], "summary.csv")
        self.assertFalse(Path(record["internal_path"]).is_absolute())

    def test_an_unjudged_artifact_says_so_rather_than_claiming_it_passed(self):
        path = a_csv(self.project / "summary.csv")
        record = describe(path, run_id=RUN_ID, stage_id="safety",
                          project_root=self.project)
        self.assertIsNone(record["qa_status"])


class RegistryLookupTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = temporary_settings(self.root)
        self.project = self.settings.job_dir(RUN_ID) / "project"
        self.project.mkdir(parents=True)
        self.registry = ArtifactRegistry(self.settings.job_dir(RUN_ID), RUN_ID, self.project)

    def tearDown(self):
        self.temp.cleanup()

    def test_a_reregistered_path_replaces_its_record(self):
        """A re-run must not leave two records for one file."""
        path = a_csv(self.project / "summary.csv")
        first = self.registry.register(path, "safety")
        second = self.registry.register(path, "safety")
        self.assertEqual(len(self.registry.all()), 1)
        self.assertNotEqual(first["artifact_id"], second["artifact_id"])

    def test_lookup_needs_the_hash_to_match(self):
        """A file that has changed is not the file a figure was drawn from.

        The figure recorded the digest of what it read; a lookup that carried
        only the path would happily hand back whatever now sits at that path.
        """
        path = a_csv(self.project / "summary.csv")
        record = self.registry.register(path, "safety")
        self.assertEqual(
            self.registry.by_path_and_hash("summary.csv", record["sha256"])["artifact_id"],
            record["artifact_id"])
        self.assertEqual(
            self.registry.by_path_and_hash("summary.csv")["artifact_id"],
            record["artifact_id"])
        # The file changed since, so the digest no longer matches any record.
        a_csv(path, rows=5)
        self.assertIsNone(self.registry.by_path_and_hash("summary.csv", "0" * 64))

    def test_the_qa_verdict_is_stamped_across_the_registry(self):
        a_csv(self.project / "a.csv")
        a_csv(self.project / "b.csv")
        self.registry.register(self.project / "a.csv", "safety")
        self.registry.register(self.project / "b.csv", "analysis")
        self.assertEqual(self.registry.stamp_qa("failed"), 2)
        self.assertTrue(all(item["qa_status"] == "failed"
                            for item in self.registry.all()))
        # Stamping again changes nothing.
        self.assertEqual(self.registry.stamp_qa("failed"), 0)

    def test_a_reread_sees_the_stamp(self):
        a_csv(self.project / "a.csv")
        self.registry.register(self.project / "a.csv", "safety")
        self.registry.stamp_qa("passed")
        fresh = ArtifactRegistry(self.settings.job_dir(RUN_ID), RUN_ID, self.project)
        self.assertEqual(fresh.all()[0]["qa_status"], "passed")


class PublicDescriptorTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = temporary_settings(self.root)
        self.project = self.settings.job_dir(RUN_ID) / "project"
        self.project.mkdir(parents=True)
        self.registry = ArtifactRegistry(self.settings.job_dir(RUN_ID), RUN_ID, self.project)

    def tearDown(self):
        self.temp.cleanup()

    def test_no_internal_path_ever_travels(self):
        record = self.registry.register(a_csv(self.project / "summary.csv"), "safety")
        payload = json.dumps(public_descriptor(record), ensure_ascii=False)
        self.assertNotIn("internal_path", payload)
        self.assertNotIn("summary.csv", payload.replace(record["title"], ""))

    def test_the_role_says_what_the_file_is_to_the_reader(self):
        geotiff = describe(a_geotiff(self.project / "dem.tif"), run_id=RUN_ID,
                           stage_id="validate", project_root=self.project)
        table = describe(a_csv(self.project / "s.csv"), run_id=RUN_ID,
                         stage_id="safety", project_root=self.project)
        self.assertEqual(role_for(geotiff), "source_raster")
        self.assertEqual(role_for(table), "source_table")

    def test_available_and_generatable_formats_are_stated_separately(self):
        record = self.registry.register(a_csv(self.project / "summary.csv"), "safety")
        descriptor = public_descriptor(record)
        self.assertEqual(descriptor["format"], "csv")
        self.assertEqual(descriptor["formats"], ["original"])
        self.assertEqual(descriptor["exportable"], ["xlsx"])

    def test_a_raster_is_not_offered_a_workbook(self):
        record = self.registry.register(a_geotiff(self.project / "dem.tif"), "validate")
        descriptor = public_descriptor(record)
        self.assertEqual(descriptor["format"], "geotiff")
        self.assertEqual(descriptor["exportable"], [])


class VisualizationArtifactsTests(unittest.TestCase):
    """The figure-to-file link, over a real job directory."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = temporary_settings(self.root)
        self.service = VisualizationService(self.settings)
        self.exports = ExportService(self.settings)
        self.job_dir = self.settings.job_dir(RUN_ID)
        self.project = self.job_dir / "project"
        self.project.mkdir(parents=True)
        self.registry = ArtifactRegistry(self.job_dir, RUN_ID, self.project)
        write_json(self.job_dir / "status.json", {"run_id": RUN_ID, "status": "succeeded"})

    def tearDown(self):
        self.temp.cleanup()

    def _a_figure(self, *, sources: list[dict], **extra) -> str:
        visualization_id = "viz-abcdef123456"
        record = {
            "visualization_id": visualization_id, "run_id": RUN_ID,
            "kind": "safety_minimum_factor", "stage_id": "safety",
            "title": "最小安全系数", "status": "succeeded", "sources": sources,
            "qa_failed": False, "limitations": [], "image_sha256": "a" * 64,
            **extra,
        }
        write_json(self.service.directory(visualization_id) / "visualization.json", record)
        return visualization_id

    RASTER = "rapid_safety_factor_results/pkg_enhanced_minimum_safety_factor.tif"

    def test_a_source_file_resolves_to_its_artifact(self):
        raster = a_geotiff(self.project / self.RASTER)
        record = self.registry.register(raster, "safety")
        figure = self._a_figure(sources=[{
            "scope": "project", "relative_path": self.RASTER,
            "sha256": record["sha256"], "size_bytes": record["size_bytes"]}])

        payload = self.exports.artifacts_for_visualization(self.service, figure)
        self.assertEqual([item["artifact_id"] for item in payload["artifacts"]],
                         [record["artifact_id"]])
        self.assertEqual(payload["artifacts"][0]["format"], "geotiff")
        self.assertEqual(payload["artifacts"][0]["units"], "dimensionless")
        self.assertEqual(payload["warnings"], [])

    def test_a_file_the_unit_table_does_not_know_carries_no_unit(self):
        """Guessing a unit from an unfamiliar name would be worse than silence."""
        raster = a_geotiff(self.project / "rapid_safety_factor_results/fs.tif")
        record = self.registry.register(raster, "safety")
        descriptor = public_descriptor(record)
        self.assertIsNone(descriptor["units"])

    def test_an_old_record_is_linked_on_first_read(self):
        """Figures rendered before the link existed still offer their files."""
        raster = a_geotiff(self.project / "rapid_safety_factor_results/fs.tif")
        record = self.registry.register(raster, "safety")
        figure = self._a_figure(sources=[{
            "scope": "project",
            "relative_path": "rapid_safety_factor_results/fs.tif",
            "sha256": record["sha256"], "size_bytes": record["size_bytes"]}])

        first = self.service.status(figure)
        self.assertEqual(first["source_artifact_ids"], [record["artifact_id"]])
        # Written back once, so the second read is a plain read.
        stored = json.loads((self.service.directory(figure) / "visualization.json")
                            .read_text(encoding="utf-8"))
        self.assertEqual(stored["source_artifact_ids"], [record["artifact_id"]])

    def test_a_changed_source_file_does_not_resolve(self):
        raster = a_geotiff(self.project / "rapid_safety_factor_results/fs.tif")
        record = self.registry.register(raster, "safety")
        figure = self._a_figure(sources=[{
            "scope": "project",
            "relative_path": "rapid_safety_factor_results/fs.tif",
            "sha256": "0" * 64, "size_bytes": record["size_bytes"]}])
        payload = self.exports.artifacts_for_visualization(self.service, figure)
        self.assertEqual(payload["artifacts"], [])
        self.assertTrue(payload["warnings"])

    def test_a_figure_with_no_registered_source_offers_nothing(self):
        figure = self._a_figure(sources=[])
        payload = self.exports.artifacts_for_visualization(self.service, figure)
        self.assertEqual(payload["artifacts"], [])
        self.assertIn("该图没有可下载的登记成果", payload["warnings"][0])

    def test_a_comparison_resolves_to_the_figures_it_was_built_from(self):
        raster = a_geotiff(self.project / "rapid_safety_factor_results/fs.tif")
        record = self.registry.register(raster, "safety")
        inner = self._a_figure(sources=[{
            "scope": "project",
            "relative_path": "rapid_safety_factor_results/fs.tif",
            "sha256": record["sha256"], "size_bytes": record["size_bytes"]}])
        outer = "viz-bbbbbb123456"
        write_json(self.service.directory(outer) / "visualization.json", {
            "visualization_id": outer, "run_id": RUN_ID, "kind": "comparison_difference",
            "stage_id": "safety", "title": "差值", "status": "succeeded",
            "sources": [{"visualization_id": inner, "image_sha256": "a" * 64}],
        })

        payload = self.exports.artifacts_for_visualization(self.service, outer)
        self.assertEqual([item["artifact_id"] for item in payload["artifacts"]],
                         [record["artifact_id"]])

    def test_the_qa_verdict_is_read_from_the_run_when_the_record_has_none(self):
        raster = a_geotiff(self.project / "rapid_safety_factor_results/fs.tif")
        record = self.registry.register(raster, "safety")
        write_json(self.job_dir / "qa.json", {"passed": False, "failed": ["stability"]})
        figure = self._a_figure(sources=[{
            "scope": "project",
            "relative_path": "rapid_safety_factor_results/fs.tif",
            "sha256": record["sha256"], "size_bytes": record["size_bytes"]}])
        payload = self.exports.artifacts_for_visualization(self.service, figure)
        self.assertEqual(payload["artifacts"][0]["qa_status"], "failed")


class MissingAndUnfinishedTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = temporary_settings(self.root)
        self.service = VisualizationService(self.settings)
        self.exports = ExportService(self.settings)

    def tearDown(self):
        self.temp.cleanup()

    def test_an_unknown_visualization_is_a_clean_error(self):
        from watershed_hydro_mcp.visualization.service import VisualizationError

        with self.assertRaises(VisualizationError):
            self.exports.artifacts_for_visualization(self.service, "viz-000000000000")

    def test_an_unknown_run_has_no_qa_verdict(self):
        self.assertIsNone(self.exports.run_qa_status("hydro-run-000000000000"))
