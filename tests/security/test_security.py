"""Attack-shaped inputs must be refused before they reach the filesystem."""

from __future__ import annotations

import json
import os
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import build_package  # noqa: E402

from watershed_exchange.paths import (  # noqa: E402
    PathSafetyError,
    find_secrets,
    resolve_within,
    safe_extract_zip,
    safe_id,
    safe_relative_path,
)


class PathTraversalTests(unittest.TestCase):
    def test_parent_traversal_is_rejected(self):
        for value in ("../secret", "a/../../b", "..", "a/..", "./../x"):
            with self.subTest(value=value), self.assertRaises(PathSafetyError):
                safe_relative_path(value)

    def test_backslash_traversal_is_rejected(self):
        with self.assertRaises(PathSafetyError):
            safe_relative_path("..\\..\\windows\\system32")

    def test_absolute_unix_path_is_rejected(self):
        for value in ("/etc/passwd", "//server/share/x"):
            with self.subTest(value=value), self.assertRaises(PathSafetyError):
                safe_relative_path(value)

    def test_drive_letter_is_rejected(self):
        for value in ("C:/windows/x", "d:\\data\\x", "z:x"):
            with self.subTest(value=value), self.assertRaises(PathSafetyError):
                safe_relative_path(value)

    def test_reserved_device_names_are_rejected(self):
        for value in ("CON", "nul.txt", "dir/COM1.dbf", "lpt9"):
            with self.subTest(value=value), self.assertRaises(PathSafetyError):
                safe_relative_path(value)

    def test_trailing_dot_or_space_is_rejected(self):
        for value in ("name. ", "dir /file", "file."):
            with self.subTest(value=value), self.assertRaises(PathSafetyError):
                safe_relative_path(value)

    def test_forbidden_directories_are_rejected(self):
        for value in ("__pycache__/x", ".git/config", "a/.env"):
            with self.subTest(value=value), self.assertRaises(PathSafetyError):
                safe_relative_path(value)

    def test_ordinary_nested_path_is_accepted(self):
        self.assertEqual(safe_relative_path("a/b/c.tif"), "a/b/c.tif")
        self.assertEqual(safe_relative_path("a\\b\\c.tif"), "a/b/c.tif")


class ResolveWithinTests(unittest.TestCase):
    def test_escape_is_refused(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            root = Path(temp).resolve()
            with self.assertRaises(PathSafetyError):
                resolve_within(root, "../outside.txt")

    def test_inside_path_resolves_under_root(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            root = Path(temp).resolve()
            resolved = resolve_within(root, "nested/file.tif")
            self.assertTrue(str(resolved).startswith(str(root)))


class SymlinkEscapeTests(unittest.TestCase):
    def test_symlink_member_is_flagged(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            root = Path(temp)
            outside = root / "outside"
            outside.mkdir()
            (outside / "target.txt").write_text("secret", encoding="utf-8")
            link = root / "package" / "link.txt"
            link.parent.mkdir(parents=True)
            try:
                os.symlink(outside / "target.txt", link)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is not permitted on this platform")
            from watershed_exchange.paths import contains_link

            self.assertTrue(contains_link(root / "package", "link.txt"))


class ZipSafetyTests(unittest.TestCase):
    def _archive(self, path: Path, members: dict[str, bytes]) -> Path:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, payload in members.items():
                archive.writestr(name, payload)
        return path

    def test_zip_slip_is_rejected(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            archive = self._archive(Path(temp) / "evil.zip",
                                    {"../escaped.txt": b"pwned", "ok.txt": b"fine"})
            with self.assertRaises(PathSafetyError):
                safe_extract_zip(archive, Path(temp) / "out")
            self.assertFalse((Path(temp) / "escaped.txt").exists())

    def test_absolute_member_is_rejected(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            archive = self._archive(Path(temp) / "abs.zip", {"C:/windows/x.txt": b"pwned"})
            with self.assertRaises(PathSafetyError):
                safe_extract_zip(archive, Path(temp) / "out")

    def test_corrupt_zip_is_rejected(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            archive = Path(temp) / "broken.zip"
            archive.write_bytes(b"PK\x03\x04 not really a zip")
            with self.assertRaises(zipfile.BadZipFile):
                safe_extract_zip(archive, Path(temp) / "out")

    def test_member_count_limit(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            archive = self._archive(Path(temp) / "many.zip",
                                    {f"f{index}.txt": b"x" for index in range(20)})
            with self.assertRaises(PathSafetyError):
                safe_extract_zip(archive, Path(temp) / "out", maximum_members=5)

    def test_decompression_bomb_is_rejected(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            archive = Path(temp) / "bomb.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("big.bin", b"\0" * (8 * 1024 * 1024))
            with self.assertRaises(PathSafetyError):
                safe_extract_zip(archive, Path(temp) / "out", maximum_total_bytes=1024 * 1024)

    def test_single_member_size_limit(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            archive = Path(temp) / "large.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
                bundle.writestr("big.bin", b"\0" * (2 * 1024 * 1024))
            with self.assertRaises(PathSafetyError):
                safe_extract_zip(archive, Path(temp) / "out", maximum_member_bytes=1024 * 1024)

    def test_clean_archive_extracts_and_round_trips(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            archive = self._archive(Path(temp) / "good.zip",
                                    {"a/b.txt": b"hello", "c.txt": b"world"})
            extracted = safe_extract_zip(archive, Path(temp) / "out")
            self.assertEqual(sorted(extracted), ["a/b.txt", "c.txt"])
            self.assertEqual((Path(temp) / "out" / "a" / "b.txt").read_bytes(), b"hello")


class IdentifierTests(unittest.TestCase):
    def test_valid_ids(self):
        for value in ("basin-abc123", "hydro-input-0123456789ab", "artifact-ffffffffffff"):
            self.assertEqual(safe_id(value), value)

    def test_invalid_ids(self):
        for value in ("", "a", "..", "a/b", "a b", "x" * 200, "a\\b", "★"):
            with self.subTest(value=value), self.assertRaises(PathSafetyError):
                safe_id(value)


class SecretDetectionTests(unittest.TestCase):
    def test_credential_keys_are_found(self):
        found = find_secrets({"a": {"api_key": "x"}, "list": [{"token": "y"}], "ok": 1})
        self.assertEqual(sorted(found), ["a.api_key", "list[0].token"])

    def test_clean_payload_reports_nothing(self):
        self.assertEqual(find_secrets({"provenance": {"producer": "x"}}), [])


class UnregisteredArtifactTests(unittest.TestCase):
    def test_unregistered_id_is_refused(self):
        from watershed_hydro_mcp.artifact_registry import ArtifactRegistry

        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            registry = ArtifactRegistry(Path(temp), "hydro-run-test", Path(temp))
            with self.assertRaises(ValueError):
                registry.by_id("artifact-doesnotexist")

    def test_path_outside_run_directory_is_refused(self):
        from watershed_hydro_mcp.artifact_registry import ArtifactRegistry

        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            root = Path(temp) / "project"
            root.mkdir()
            registry = ArtifactRegistry(Path(temp), "hydro-run-test", root)
            with self.assertRaises(PathSafetyError):
                registry.resolve({"internal_path": "../outside.tif"})


class InboxTests(unittest.TestCase):
    def test_source_outside_inbox_is_refused(self):
        from watershed_hydro_mcp.input_registry import _resolve_inbox
        from watershed_hydro_mcp.settings import Settings

        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            root = Path(temp)
            settings = Settings(root=root, runs_root=root / "runs", hydro_root=root / "runs" / "hydro",
                                native_root=root / "native", config_root=root / "config")
            with self.assertRaises(PathSafetyError):
                _resolve_inbox(str(root / "elsewhere"), settings)

    def test_source_inside_inbox_is_accepted(self):
        from watershed_hydro_mcp.input_registry import _resolve_inbox
        from watershed_hydro_mcp.settings import Settings

        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            root = Path(temp)
            inbox = root / "exchange" / "imports"
            inbox.mkdir(parents=True)
            settings = Settings(root=root, runs_root=root / "runs", hydro_root=root / "runs" / "hydro",
                                native_root=root / "native", config_root=root / "config")
            self.assertEqual(_resolve_inbox(str(inbox), settings), inbox.resolve())

    def test_package_name_with_separator_is_refused(self):
        from watershed_hydro_mcp.input_registry import find_package_source
        from watershed_hydro_mcp.settings import Settings

        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            root = Path(temp)
            settings = Settings(root=root, runs_root=root / "runs", hydro_root=root / "runs" / "hydro",
                                native_root=root / "native", config_root=root / "config")
            with self.assertRaises(PathSafetyError):
                find_package_source("../secrets", settings)


class PlanSafetyTests(unittest.TestCase):
    def test_client_cannot_name_a_stage_outside_the_whitelist(self):
        from watershed_hydro_mcp.models import HydroRunRequest

        with self.assertRaises(ValueError):
            HydroRunRequest(input_id="hydro-input-abcdef123456", mode="build_run_qa",
                            through="../../evil.py")

    def test_client_cannot_name_an_arbitrary_script(self):
        from watershed_hydro_mcp.models import HydroRunRequest

        with self.assertRaises(ValueError):
            HydroRunRequest(input_id="hydro-input-abcdef123456", mode="build_run_qa",
                            through="watershed_hydro_mcp.core.sync")

    def test_stage_selection_stays_inside_the_whitelist(self):
        from watershed_hydro_mcp.planner import STAGE_IDS, stages_for

        for mode in ("preflight_only", "rainfall_only", "build_run_qa"):
            for stage in stages_for(mode, None):
                self.assertIn(stage.stage_id, STAGE_IDS)

    def test_through_truncates_the_plan(self):
        from watershed_hydro_mcp.planner import stages_for

        full = stages_for("build_run_qa", None)
        partial = stages_for("build_run_qa", "build")
        self.assertLess(len(partial), len(full))
        self.assertEqual(partial[-1].stage_id, "build")

    def test_through_outside_the_mode_is_refused(self):
        from watershed_hydro_mcp.planner import PlanningError, stages_for

        with self.assertRaises(PlanningError):
            stages_for("preflight_only", "safety")


if __name__ == "__main__":
    unittest.main()
