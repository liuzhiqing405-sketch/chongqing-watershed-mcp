"""The SWMM model bundle: what gets packed, and what the pack admits.

A bundle is a claim: "these files are what this run produced, here is their
digest, and here is what they are". These tests hold the parts of that claim
that are easy to get subtly wrong — a checksum list copied from the registry
instead of computed from the bytes, a second copy of the same file under two
names, a file from a neighbouring run swept in by name, or an incomplete model
presented as complete.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import unittest
import zipfile
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watershed_hydro_mcp import swmm_bundle as bundles  # noqa: E402
from watershed_hydro_mcp.artifact_registry import ArtifactRegistry  # noqa: E402
from watershed_hydro_mcp.export_service import ExportError, ExportService  # noqa: E402
from watershed_hydro_mcp.settings import Settings  # noqa: E402
from watershed_hydro_mcp.state import write_json  # noqa: E402


RUN_ID = "hydro-run-abcdef123456"
MAGIC = 5161141


def temporary_settings(root: Path) -> Settings:
    runs = root / "runs"
    return Settings(root=root, runs_root=runs, hydro_root=runs / "hydro",
                    native_root=root / "native", config_root=Path(
                        __file__).resolve().parents[2] / "config")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def an_inp(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[OPTIONS]\nFLOW_UNITS           CMS\n\n[SUBCATCHMENTS]\nG001  100.0\n",
                    encoding="utf-8")
    return path


def a_report(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "  EPA STORM WATER MANAGEMENT MODEL - VERSION 5.2 (Build 5.2.4)\n"
        "  Analysis begun on:  2016-06-24 00:00:00\n"
        "  ERROR 000: none\n",
        encoding="utf-8")
    return path


def a_swmm_output(path: Path, *, magic: int = MAGIC, tail_magic: int | None = None,
                  periods: int = 299, error_code: int = 0,
                  truncate_bytes: int = 0) -> Path:
    """A minimal but structurally real SWMM binary output.

    ``inspect_swmm_output`` reads the header, the trailing index and the result
    section offset, so a file built the same way is a fair test of it — including
    the damaged variants, which are produced by changing one field rather than by
    writing something that is not an output file at all.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack("<7i", magic, 52004, 0, 3, 2, 2, 0)
    results_pos = len(header) + 64
    start = struct.pack("<di", 42545.0, 300)
    tail = struct.pack("<6i", 0, 0, results_pos, periods, error_code,
                       magic if tail_magic is None else tail_magic)
    body = header + b"\x00" * (results_pos - 12 - len(header)) + start + b"\x11" * 128 + tail
    if truncate_bytes:
        body = body[:-truncate_bytes]
    path.write_bytes(body)
    return path


def a_csv(path: Path, header: str = "name,value", rows: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [header] + [f"r{index},{index}.5" for index in range(rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class BundleFixture(unittest.TestCase):
    """A run directory with the files a real SWMM stage registers."""

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = temporary_settings(self.root)
        self.job = self.settings.job_dir(RUN_ID)
        self.project = self.job / "project"
        self.project.mkdir(parents=True)
        self.registry = ArtifactRegistry(self.job, RUN_ID, self.project)
        self.service = ExportService(self.settings)

    def tearDown(self):
        self.temp.cleanup()

    def register_run(self, *, with_output: bool = True) -> None:
        model = self.project / "swmm/model"
        results = self.project / "swmm/results"
        rainfall = self.project / "inputs/rainfall"
        self.registry.register(an_inp(model / "package.inp"), "build", kind="swmm_input")
        self.registry.register(a_report(results / "package.rpt"), "swmm", kind="swmm_result")
        if with_output:
            self.registry.register(a_swmm_output(results / "package.out"), "swmm",
                                   kind="swmm_result")
        self.registry.register(a_csv(model / "auto_swmm_parameter_review.csv"), "build",
                               kind="swmm_input")
        self.registry.register(a_csv(model / "auto_swmm_rainfall_review.csv"), "build",
                               kind="swmm_input")
        self.registry.register(a_csv(rainfall / "package_active_hourly.csv",
                                     header="datetime_beijing,rain_mm"),
                               "rainfall", kind="rainfall")

    def build(self, **kwargs) -> dict:
        destination = self.project / "exports" / bundles.bundle_name(RUN_ID)
        return bundles.build_bundle(RUN_ID, self.registry, destination=destination,
                                    qa_status="passed", **kwargs)


class MemberTests(BundleFixture):
    def test_only_this_runs_registered_files_become_members(self):
        self.register_run()
        neighbours = self.project / "swmm/model/auto_swmm_parameter_review.csv.bak"
        a_csv(neighbours)  # on disk, never registered
        members = bundles.collect_members(self.registry, RUN_ID)
        paths = {member.path_in_bundle for member in members}
        self.assertIn("model/model.inp", paths)
        self.assertIn("results/model.rpt", paths)
        self.assertIn("forcing/rainfall_hourly.csv", paths)
        self.assertEqual(len(members), len(paths), "one member per bundle path")
        self.assertNotIn("auto_swmm_parameter_review.csv.bak", str(paths))

    def test_the_export_copy_does_not_duplicate_a_member(self):
        """The export stage copies results into final_results and registers them
        again. Both records map to the same bundle path, and one file must not be
        packed twice."""
        self.register_run()
        copied = self.project / "final_results/03_swmm/package.inp"
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_bytes((self.project / "swmm/model/package.inp").read_bytes())
        self.registry.register(copied, "export", kind="export")
        members = bundles.collect_members(self.registry, RUN_ID)
        inps = [item for item in members if item.path_in_bundle == "model/model.inp"]
        self.assertEqual(len(inps), 1)
        # The stage's own copy is kept, not the export duplicate.
        self.assertIn("swmm", str(inps[0].source_path).replace("\\", "/"))

    def test_a_missing_required_file_is_reported_not_invented(self):
        self.register_run(with_output=False)
        (self.project / "swmm/results/package.rpt").unlink()
        members = bundles.collect_members(self.registry, RUN_ID)
        self.assertEqual(bundles.missing_required(members),
                         ["native_report", "native_result"])

    def test_a_missing_native_output_makes_the_model_incomplete(self):
        self.register_run(with_output=False)
        members = bundles.collect_members(self.registry, RUN_ID)
        self.assertEqual(bundles.missing_required(members), ["native_result"])


class ChecksumTests(BundleFixture):
    def test_checksums_describe_the_bytes_that_are_packed(self):
        self.register_run()
        inp = self.project / "swmm/model/package.inp"
        recorded = next(item for item in self.registry.all()
                        if item["title"].endswith(".inp"))["sha256"]
        # A later stage rewrites the file, exactly as the build stage rewrites
        # the rainfall series: the registry's digest is now stale.
        an_inp(inp)
        inp.write_text(inp.read_text(encoding="utf-8") + "\n; rewritten\n", encoding="utf-8")
        result = self.build()
        archive = zipfile.ZipFile(result["bundle_path"])
        listing = archive.read("checksums.sha256").decode("utf-8")
        listed = dict(line.split("  ", 1)[::-1] for line in listing.strip().splitlines())
        packed = archive.read("model/model.inp")
        self.assertEqual(listed["model/model.inp"], hashlib.sha256(packed).hexdigest())
        self.assertNotEqual(listed["model/model.inp"], recorded)
        # And the drift is stated, not hidden.
        manifest = json.loads(archive.read("manifest.json"))
        entry = next(item for item in manifest["members"] if item["path"] == "model/model.inp")
        self.assertEqual(entry["recorded_sha256"], recorded)
        self.assertFalse(entry["sha256_matches_record"])
        self.assertEqual(manifest["drifted_members"], ["model/model.inp"])
        self.assertIn("model/model.inp", archive.read("README.md").decode("utf-8"))

    def test_every_member_verifies_against_the_list(self):
        self.register_run()
        result = self.build()
        archive = zipfile.ZipFile(result["bundle_path"])
        listing = archive.read("checksums.sha256").decode("utf-8")
        for line in listing.strip().splitlines():
            digest, name = line.split("  ", 1)
            self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest, name)


class ArchiveTests(BundleFixture):
    def test_the_archive_has_the_declared_layout(self):
        self.register_run()
        result = self.build()
        archive = zipfile.ZipFile(result["bundle_path"])
        names = set(archive.namelist())
        self.assertLessEqual({"README.md", "manifest.json", "checksums.sha256",
                              "metadata/software_versions.json", "model/model.inp",
                              "results/model.rpt", "results/model.out",
                              "tables/parameter_review.csv", "tables/rainfall_review.csv",
                              "forcing/rainfall_hourly.csv"}, names)
        self.assertTrue(result["complete"])
        self.assertEqual(result["missing_required"], [])

    def test_no_member_path_escapes_the_archive(self):
        self.register_run()
        result = self.build()
        archive = zipfile.ZipFile(result["bundle_path"])
        for name in archive.namelist():
            self.assertFalse(name.startswith("/"), name)
            self.assertNotIn("..", name.split("/"), name)
            self.assertNotIn(":", name, name)
            info = archive.getinfo(name)
            self.assertFalse((info.external_attr >> 16) & 0o170000 == 0o120000,  # symlink
                             f"{name} is a symlink")

    def test_an_incomplete_model_is_named_and_labeled_incomplete(self):
        self.register_run()
        (self.project / "swmm/results/package.rpt").unlink()
        members = bundles.collect_members(self.registry, RUN_ID)
        destination = self.project / "exports" / bundles.bundle_name(RUN_ID, complete=False)
        result = bundles.build_bundle(RUN_ID, self.registry, destination=destination,
                                      qa_status="passed")
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing_required"], ["native_report"])
        self.assertTrue(result["bundle_path"].name.startswith("swmm_model_INCOMPLETE_"))
        archive = zipfile.ZipFile(result["bundle_path"])
        readme = archive.read("README.md").decode("utf-8")
        self.assertIn("不完整", readme)
        self.assertIn("native_report", readme)

    def test_a_bundle_without_out_is_diagnostic_not_complete(self):
        self.register_run(with_output=False)
        destination = self.project / "exports" / bundles.bundle_name(RUN_ID, complete=False)
        result = bundles.build_bundle(RUN_ID, self.registry, destination=destination,
                                      qa_status="passed")
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing_required"], ["native_result"])
        self.assertTrue(result["bundle_path"].name.startswith("swmm_model_INCOMPLETE_"))

    def test_no_bundle_is_written_when_there_is_nothing_to_pack(self):
        with self.assertRaises(bundles.BundleError):
            self.build()

    def test_a_failed_write_leaves_no_partial_archive(self):
        self.register_run()
        destination = self.project / "exports" / bundles.bundle_name(RUN_ID)
        with mock.patch.object(zipfile.ZipFile, "write", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                bundles.build_bundle(RUN_ID, self.registry, destination=destination)
        self.assertFalse(destination.exists())
        self.assertEqual(list(destination.parent.glob("*.partial")), [])


class OutputInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_an_intact_output_is_read_without_reading_the_whole_file(self):
        path = a_swmm_output(self.root / "model.out")
        facts = bundles.inspect_swmm_output(path)
        self.assertTrue(facts["intact"], facts["reason"])
        self.assertEqual(facts["periods"], 299)
        self.assertEqual(facts["report_step_seconds"], 300)
        self.assertEqual(facts["start_datetime"], "2016-06-24T00:00:00")

    def test_a_truncated_output_is_reported_as_damaged_not_raised(self):
        """A download that stopped early must be labelled, not used.

        The reader looks at the header and the trailing index, so a cut file
        shows up as a mismatch between them — either way it is reported as a
        reason rather than raised, because a damaged result is still worth
        handing over as a diagnostic.
        """
        path = a_swmm_output(self.root / "cut.out", truncate_bytes=8)
        facts = bundles.inspect_swmm_output(path)
        self.assertFalse(facts["intact"])
        self.assertTrue(str(facts["reason"]), "a damaged file must say why")
        self.assertTrue("不完整" in facts["reason"] or "截断" in facts["reason"],
                        facts["reason"])

    def test_mismatched_magic_numbers_are_reported(self):
        path = a_swmm_output(self.root / "wrong.out", tail_magic=MAGIC + 1)
        facts = bundles.inspect_swmm_output(path)
        self.assertFalse(facts["intact"])
        self.assertIn("magic", str(facts["reason"]))

    def test_a_simulation_error_code_is_reported(self):
        path = a_swmm_output(self.root / "err.out", error_code=134)
        facts = bundles.inspect_swmm_output(path)
        self.assertFalse(facts["intact"])
        self.assertIn("134", str(facts["reason"]))

    def test_zero_periods_is_reported(self):
        path = a_swmm_output(self.root / "empty.out", periods=0)
        self.assertIn("时段", str(bundles.inspect_swmm_output(path)["reason"]))

    def test_a_file_that_is_not_an_output_at_all(self):
        path = self.root / "not.out"
        path.write_text("hello", encoding="utf-8")
        self.assertFalse(bundles.inspect_swmm_output(path)["intact"])


class PrepareBundleTests(BundleFixture):
    def test_a_bundle_is_reused_until_a_source_changes(self):
        self.register_run()
        first = self.service.prepare_swmm_bundle(RUN_ID)
        self.assertFalse(first["reused"])
        second = self.service.prepare_swmm_bundle(RUN_ID)
        self.assertTrue(second["reused"])
        self.assertEqual(first["artifact"]["artifact_id"], second["artifact"]["artifact_id"])

        inp = self.project / "swmm/model/package.inp"
        inp.write_text(inp.read_text(encoding="utf-8") + "\n; edited\n", encoding="utf-8")
        third = self.service.prepare_swmm_bundle(RUN_ID)
        self.assertFalse(third["reused"])
        self.assertNotEqual(first["artifact"]["sha256"], third["artifact"]["sha256"])
        self.assertNotEqual(first["artifact"]["title"], third["artifact"]["title"])

    def test_warnings_state_the_missing_files_and_the_drift(self):
        self.register_run()
        (self.project / "swmm/results/package.rpt").unlink()
        result = self.service.prepare_swmm_bundle(RUN_ID)
        self.assertFalse(result["complete"])
        self.assertEqual(result["missing_required"], ["native_report"])
        self.assertTrue(any("native_report" in item for item in result["warnings"]))

    def test_no_swmm_files_no_bundle(self):
        with self.assertRaises(ExportError):
            self.service.prepare_swmm_bundle(RUN_ID)

    def test_the_bundle_reports_the_engine_its_own_report_names(self):
        self.register_run()
        result = self.service.prepare_swmm_bundle(RUN_ID)
        self.assertIn("5.2", str(result["swmm_engine"]))
        self.assertEqual(result["native_output"]["periods"], 299)


class LocatorAndBlobTests(BundleFixture):
    def test_the_locator_names_the_file_the_registry_recorded(self):
        self.register_run()
        record = next(item for item in self.registry.all() if item["title"].endswith(".inp"))
        locator = self.service.artifact_locator(RUN_ID, record["artifact_id"])
        self.assertEqual(Path(locator["internal_path"]), self.project / "swmm/model/package.inp")
        self.assertEqual(locator["sha256"], record["sha256"])
        self.assertEqual(locator["size_bytes"], (self.project / "swmm/model/package.inp").stat().st_size)
        self.assertEqual(locator["format"], "inp")

    def test_a_rewritten_file_is_refused_rather_than_located(self):
        self.register_run()
        record = next(item for item in self.registry.all() if item["title"].endswith(".inp"))
        path = self.project / "swmm/model/package.inp"
        path.write_text(path.read_text(encoding="utf-8") + "\n; changed\n", encoding="utf-8")
        with self.assertRaises(ExportError):
            self.service.artifact_locator(RUN_ID, record["artifact_id"])
        with self.assertRaises(ExportError):
            self.service.artifact_blob(RUN_ID, record["artifact_id"])

    def test_an_unknown_artifact_is_refused_by_id(self):
        self.register_run()
        with self.assertRaises(ExportError):
            self.service.artifact_blob(RUN_ID, "artifact-doesnotexist")

    def test_the_blob_carries_the_bytes_and_their_media_type(self):
        self.register_run()
        record = next(item for item in self.registry.all() if item["title"].endswith(".rpt"))
        raw, media_type = self.service.artifact_blob(RUN_ID, record["artifact_id"])
        self.assertEqual(raw, (self.project / "swmm/results/package.rpt").read_bytes())
        self.assertEqual(media_type, "text/plain")

    def test_a_file_over_the_channel_ceiling_is_refused_not_truncated(self):
        self.register_run()
        record = next(item for item in self.registry.all() if item["title"].endswith(".out"))
        with self.assertRaises(ExportError):
            self.service.artifact_blob(RUN_ID, record["artifact_id"], limit_bytes=16)

    def test_an_unregistered_id_cannot_be_reached_by_a_relative_path(self):
        """The id is looked up in the registry; a path-shaped id is not a lookup."""
        self.register_run()
        outside = self.root / "secret.txt"
        outside.write_text("not yours", encoding="utf-8")
        for candidate in ("../../secret.txt", "..\\..\\secret.txt",
                          str(outside), "project/swmm/model/package.inp"):
            with self.assertRaises(ExportError):
                self.service.artifact_blob(RUN_ID, candidate)


class VerdictTests(BundleFixture):
    def test_the_verdict_lists_every_role_with_presence(self):
        self.register_run()
        verdict = self.service.bundle_verdict(RUN_ID)
        roles = {item["role"]: item for item in verdict["members"]}
        self.assertTrue(verdict["complete"])
        self.assertTrue(roles["model_input"]["present"])
        self.assertTrue(roles["native_report"]["present"])
        self.assertTrue(roles["native_result"]["present"])
        self.assertIsNotNone(roles["model_input"]["artifact"]["artifact_id"])
        # Descriptors only: the verdict is what a caller may see.
        self.assertNotIn("internal_path", json.dumps(verdict, ensure_ascii=False))

    def test_a_run_without_an_inp_is_not_complete(self):
        self.registry.register(a_report(self.project / "swmm/results/package.rpt"),
                               "swmm", kind="swmm_result")
        verdict = self.service.bundle_verdict(RUN_ID)
        self.assertFalse(verdict["complete"])
        self.assertIn(".inp", verdict["missing_required"])


class NameTests(unittest.TestCase):
    def test_the_bundle_name_says_what_it_is(self):
        when = datetime(2026, 9, 29, 12, 0, 0)
        self.assertEqual(bundles.bundle_name("hydro-run-abc", when=when),
                         "swmm_model_hydro-run-abc_20260929T120000Z.zip")
        self.assertTrue(bundles.bundle_name("hydro-run-abc", when=when, complete=False)
                        .startswith("swmm_model_INCOMPLETE_"))

    def test_two_bundles_of_different_contents_do_not_share_a_name(self):
        """Same second, different contents: the digest keeps them apart, so the
        second build cannot overwrite the first under a name the first already
        had registered."""
        when = datetime(2026, 9, 29, 12, 0, 0)
        first = bundles.bundle_name("hydro-run-abc", when=when, key="a" * 64)
        second = bundles.bundle_name("hydro-run-abc", when=when, key="b" * 64)
        self.assertNotEqual(first, second)
        self.assertIn("aaaaaaaa", first)


if __name__ == "__main__":
    unittest.main()
