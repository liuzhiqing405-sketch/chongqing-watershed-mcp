from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pyproj import CRS, Transformer
from affine import Affine

from chongqing_spatial_mcp.catalog import list_catalog
from chongqing_spatial_mcp.models import RainfallRequest
from chongqing_spatial_mcp.rainfall import (
    _station_value,
    parse_grid_column,
    select_virtual_stations,
)
from chongqing_spatial_mcp.settings import Settings
from chongqing_spatial_mcp.workflow import prepare_inputs
from watershed_exchange.paths import PathSafetyError
from workers.gis_worker import _ring_to_raster_pixels


class CoreTests(unittest.TestCase):
    def test_catalog_contains_four_primary_inputs(self):
        ids = {item["dataset_id"] for item in list_catalog()["datasets"]}
        self.assertTrue({"dem", "river_network", "landuse", "hourly_rainfall"} <= ids)

    def test_grid_column_parser(self):
        self.assertEqual(parse_grid_column("106.25_29.5"), (106.25, 29.5))
        self.assertIsNone(parse_grid_column("not-a-grid"))

    def test_explicit_day_and_period_validation(self):
        day = RainfallRequest(review_id="basin-123456", mode="explicit_period", date="2023-12-23")
        self.assertEqual(day.date, "2023-12-23")
        period = RainfallRequest(
            review_id="basin-123456", mode="explicit_period",
            start_datetime="2023-12-23 02:00", end_datetime="2023-12-23 14:00",
        )
        self.assertIsNone(period.station_count)
        with self.assertRaises(ValueError):
            RainfallRequest(
                review_id="basin-123456", mode="explicit_period",
                start_datetime="2023-12-31 23:00", end_datetime="2024-01-01 02:00",
            )

    def test_virtual_stations_are_spatially_distributed(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}, "geometry": {
                "type": "Polygon",
                "coordinates": [[[106.0, 29.0], [107.0, 29.0], [107.0, 30.0], [106.0, 30.0], [106.0, 29.0]]],
            }}],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "basin.geojson"
            path.write_text(json.dumps(geojson), encoding="utf-8")
            header = ["时间", "106.25_29.25", "106.75_29.25", "106.25_29.75", "106.75_29.75"]
            stations = select_virtual_stations(path, header, 2)
        self.assertEqual(len(stations), 2)
        self.assertTrue(all(item["inside"] for item in stations))
        self.assertNotEqual(stations[0]["name"], stations[1]["name"])
        for station in stations:
            self.assertAlmostEqual(
                sum(item["weight"] for item in station["source_weights"]), 1.0, places=7
            )
            self.assertEqual(station["selection_basis"], "existing_era5_grid")
            self.assertEqual(len(station['source_weights']),1)

    def test_station_value_uses_source_grid_weights(self):
        station = {"source_weights": [
            {"column": 1, "weight": 0.25},
            {"column": 2, "weight": 0.75},
        ]}
        self.assertAlmostEqual(_station_value(["time", "4", "8"], station), 7.0)

    def test_wgs84_boundary_is_reprojected_before_pixel_conversion(self):
        lon, lat = 108.03, 30.44
        x, y = Transformer.from_crs(4326, 32649, always_xy=True).transform(lon, lat)
        transform = Affine(10, 0, x - 100, 0, -10, y + 100)
        columns, rows = _ring_to_raster_pixels(
            [(lon, lat)], CRS.from_epsg(4326), CRS.from_epsg(32649), transform
        )
        self.assertAlmostEqual(columns[0], 10.0, places=5)
        self.assertAlmostEqual(rows[0], 10.0, places=5)

    def test_prepare_inputs_rejects_unsafe_rainfall_package_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = Settings(
                root=root,
                data_root=root / "data",
                runs_root=root / "runs",
                exchange_root=root / "exchange",
                native_root=root / "native",
                defaults={},
            )
            review = settings.watersheds_root / "basin-abcdef" / "review.json"
            review.parent.mkdir(parents=True)
            review.write_text(json.dumps({"status": "confirmed"}), encoding="utf-8")
            with self.assertRaises(PathSafetyError):
                prepare_inputs("basin-abcdef", "../../outside", settings=settings)


if __name__ == "__main__":
    unittest.main()
