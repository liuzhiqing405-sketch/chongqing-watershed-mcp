"""Model Input Package v1: schemas, hashing and verification."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixtures import build_package, write_dem  # noqa: E402

from watershed_exchange.paths import sha256_file  # noqa: E402
from watershed_exchange.schemas import ModelInputPackage  # noqa: E402
from watershed_exchange.validation import (  # noqa: E402
    PackageValidationError,
    load_manifest,
    read_text_limited,
    verify_package,
)


class ManifestSchemaTests(unittest.TestCase):
    def test_round_trip(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            directory = build_package(Path(temp) / "package")
            package = load_manifest(directory / "manifest.json")
            self.assertEqual(package.schema_version, "1.0")
            self.assertEqual(package.study_area.area_km2, 4.2)
            self.assertEqual(package.missing_roles(), [])
            self.assertEqual(package.roles(), {"dem", "boundary", "rainfall", "landuse", "river"})

    def test_missing_role_is_reported(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            directory = build_package(Path(temp) / "package")
            payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            payload["files"] = [item for item in payload["files"] if item["role"] != "river"]
            directory.joinpath("manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            package = load_manifest(directory / "manifest.json")
            self.assertEqual(package.missing_roles(), ["river"])

    def test_absolute_path_is_rejected(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            directory = build_package(Path(temp) / "package")
            payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            payload["files"][0]["relative_path"] = "C:/windows/system32/dem.tif"
            directory.joinpath("manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(PackageValidationError):
                load_manifest(directory / "manifest.json")

    def test_traversal_path_is_rejected(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            directory = build_package(Path(temp) / "package")
            payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            payload["files"][0]["relative_path"] = "../../secret/dem.tif"
            directory.joinpath("manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(PackageValidationError):
                load_manifest(directory / "manifest.json")

    def test_credential_field_is_rejected(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            directory = build_package(Path(temp) / "package")
            payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            payload["provenance"]["api_key"] = "sk-not-a-real-key"
            directory.joinpath("manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(PackageValidationError):
                load_manifest(directory / "manifest.json")

    def test_short_hash_is_rejected(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            directory = build_package(Path(temp) / "package")
            payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            payload["files"][0]["sha256"] = "abc"
            directory.joinpath("manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(PackageValidationError):
                load_manifest(directory / "manifest.json")


class HashingTests(unittest.TestCase):
    def test_sha256_matches_hashlib(self):
        import hashlib

        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            path = Path(temp) / "payload.bin"
            path.write_bytes(b"watershed" * 5000)
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(sha256_file(path), expected)


class VerificationTests(unittest.TestCase):
    def test_clean_package_verifies(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            directory = build_package(Path(temp) / "package")
            report = verify_package(directory)
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(len(report["files"]), 13)

    def test_tampered_file_is_detected(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            directory = build_package(Path(temp) / "package")
            # Flip bytes without changing the length so only the digest can catch it.
            target = directory / "dem.tif"
            payload = bytearray(target.read_bytes())
            payload[len(payload) // 2] ^= 0xFF
            target.write_bytes(bytes(payload))
            report = verify_package(directory)
            self.assertFalse(report["ok"])
            self.assertTrue(any("SHA-256" in item for item in report["errors"]), report["errors"])

    def test_size_change_is_detected(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            directory = build_package(Path(temp) / "package")
            write_dem(directory / "dem.tif", shape=(32, 32))
            report = verify_package(directory)
            self.assertFalse(report["ok"])
            self.assertTrue(any("大小" in item for item in report["errors"]), report["errors"])

    def test_missing_file_is_detected(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            directory = build_package(Path(temp) / "package")
            (directory / "landuse.dbf").unlink()
            report = verify_package(directory)
            self.assertFalse(report["ok"])
            self.assertTrue(any("缺失" in item or "组件" in item for item in report["errors"]),
                            report["errors"])

    def test_undeclared_file_is_a_warning(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            directory = build_package(Path(temp) / "package")
            (directory / "stray.txt").write_text("leftover", encoding="utf-8")
            report = verify_package(directory)
            self.assertTrue(report["ok"])
            self.assertTrue(any("未在清单声明" in item for item in report["warnings"]))

    def test_virtual_station_caveat_is_required(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            directory = build_package(Path(temp) / "package")
            payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            payload["limitations"] = ["nothing to declare"]
            (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            report = verify_package(directory)
            self.assertTrue(any("虚拟站" in item for item in report["warnings"]), report["warnings"])

    def test_virtual_stations_outside_basin_are_rejected(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            directory = build_package(Path(temp) / "package")
            boundary = directory / "basin.geojson"
            boundary.write_text(json.dumps({
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {}, "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[110.0, 35.0], [110.1, 35.0], [110.1, 35.1],
                                     [110.0, 35.1], [110.0, 35.0]]],
                }}],
            }), encoding="utf-8")
            payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            entry = next(item for item in payload["files"] if item["role"] == "boundary")
            entry["size_bytes"] = boundary.stat().st_size
            entry["sha256"] = sha256_file(boundary)
            (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            report = verify_package(directory)
            self.assertFalse(report["ok"])
            self.assertTrue(any("不在流域范围内" in item for item in report["errors"]))

    def test_empty_manifest_lists_missing_roles(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            directory = build_package(Path(temp) / "package")
            payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            payload["files"] = []
            (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            report = verify_package(directory)
            self.assertFalse(report["ok"])
            self.assertTrue(any("缺少必需文件角色" in item for item in report["errors"]))


class TextReadTests(unittest.TestCase):
    def test_large_file_is_truncated_at_the_limit(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            path = Path(temp) / "big.log"
            path.write_text("x" * 5000, encoding="utf-8")
            payload = read_text_limited(path, 1024)
            self.assertTrue(payload["truncated"])
            self.assertEqual(payload["returned_bytes"], 1024)
            self.assertEqual(payload["size_bytes"], 5000)

    def test_small_file_is_returned_whole(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            path = Path(temp) / "small.log"
            path.write_text("done", encoding="utf-8")
            payload = read_text_limited(path, 1024)
            self.assertFalse(payload["truncated"])
            self.assertEqual(payload["text"], "done")


class SchemaModelTests(unittest.TestCase):
    def test_file_for_role(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as temp:
            directory = build_package(Path(temp) / "package")
            package: ModelInputPackage = load_manifest(directory / "manifest.json")
            self.assertEqual(package.file_for("dem").relative_path, "dem.tif")
            self.assertIsNone(package.file_for("nonexistent"))


if __name__ == "__main__":
    unittest.main()
