"""The siting map behind interactive outlet selection.

What these tests pin down: picking an outlet on the map is only meaningful if
the numbers the browser needs to turn a click back into a coordinate are
honest. So the map payload has to carry a WGS84 box that really contains the
matched river and the candidates, an axes rectangle that is a proper part of
the image (the figure has margins around the plot), and a PNG that decodes.
An unknown river name must come back with no map at all rather than a blank one.
"""

from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyproj
import rasterio
import shapefile
from affine import Affine

from workers.gis_worker import DEFAULT_DELINEATION, outlet_map, _river_outlet_check


#: A synthetic scene in EPSG:32649: a 135 km square of gentle relief with one
#: long river across it. Big enough that the candidate rule "at least 10 km from
#: the DEM edge" is satisfiable well inside the data.
CENTER_LON, CENTER_LAT = 107.40, 30.00
CELL_M, SIZE = 150.0, 900
RIVER_NAME = "测试河"
OTHER_NAME = "别的河"


def _write_scene(root: Path) -> tuple[Path, Path]:
    to_utm = pyproj.Transformer.from_crs(4326, 32649, always_xy=True)
    cx, cy = to_utm.transform(CENTER_LON, CENTER_LAT)
    half = SIZE * CELL_M / 2.0
    transform = Affine.translation(cx - half, cy + half) @ Affine.scale(CELL_M, -CELL_M)
    rows, cols = np.mgrid[0:SIZE, 0:SIZE]
    # A ridge in the middle and a valley along the diagonal: monotone enough that
    # the ranking has a clear low end, without any depressions to fill.
    elevation = 400.0 + 0.004 * np.abs(cols - SIZE / 2) + 0.002 * rows
    dem_path = root / "dem.tif"
    with rasterio.open(
        dem_path, "w", driver="GTiff", width=SIZE, height=SIZE, count=1,
        dtype="int16", crs="EPSG:32649", transform=transform, nodata=0,
    ) as destination:
        destination.write(elevation.astype("int16"), 1)

    # 河面画成沿对角线的细长多边形（与管理图层一样是面要素，不是中心线）。
    to_wgs = pyproj.Transformer.from_crs(32649, 4326, always_xy=True)
    centre_line = []
    for step in range(0, 401):
        t = step / 400.0
        centre_line.append(to_wgs.transform(
            cx - half * 0.6 + t * half * 1.2,
            cy + half * 0.45 - t * half * 0.9,
        ))
    offset_line = []
    other_line = []
    for step in range(0, 401):
        t = step / 400.0
        offset_line.append(to_wgs.transform(
            cx - half * 0.6 + t * half * 1.2,
            cy + half * 0.45 - t * half * 0.9 - 600.0,
        ))
        other_line.append(to_wgs.transform(
            cx - half * 0.6 + t * half * 0.5,
            cy - half * 0.7 + t * half * 0.2,
        ))

    rivers_path = root / "rivers.shp"
    writer = shapefile.Writer(str(rivers_path), shapeType=shapefile.POLYGON, encoding="gbk")
    writer.field("Name", "C", 40)
    writer.field("pyname", "C", 20)
    writer.field("kind", "C", 8)
    writer.poly([centre_line + list(reversed(offset_line))])
    writer.record(RIVER_NAME, "cs", "0123")
    writer.poly([other_line + list(reversed([
        (lon, lat - 0.0005) for lon, lat in other_line]))])
    writer.record(OTHER_NAME, "bd", "0123")
    writer.close()
    (root / "rivers.prj").write_text(
        pyproj.CRS.from_epsg(4326).to_wkt("WKT1_GDAL"), encoding="utf-8")
    return dem_path, rivers_path


class OutletMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temp = tempfile.TemporaryDirectory()
        cls.dem, cls.rivers = _write_scene(Path(cls._temp.name))

    @classmethod
    def tearDownClass(cls):
        cls._temp.cleanup()

    def call(self, name: str, limit: int = 5) -> dict:
        args = SimpleNamespace(dem=self.dem, rivers=self.rivers, name=name, limit=limit,
                               defaults=dict(DEFAULT_DELINEATION))
        return outlet_map(args)

    def test_candidates_come_with_a_decodable_map(self):
        record = self.call(RIVER_NAME)
        self.assertEqual(record["matched_names"], [RIVER_NAME])
        self.assertTrue(record["candidates"])
        payload = record["map"]
        self.assertIsNotNone(payload)
        self.assertEqual(payload["media_type"], "image/png")
        raw = base64.b64decode(payload["image_base64"], validate=True)
        self.assertTrue(raw.startswith(b"\x89PNG"))
        self.assertGreater(payload["image_width"], 0)
        self.assertGreater(payload["image_height"], 0)

    def test_the_box_covers_the_river_and_every_candidate(self):
        record = self.call(RIVER_NAME)
        west, south, east, north = record["map"]["bbox_wgs84"]
        lons = [item["longitude"] for item in record["candidates"]]
        lats = [item["latitude"] for item in record["candidates"]]
        self.assertLessEqual(west, min(lons))
        self.assertGreaterEqual(east, max(lons))
        self.assertLessEqual(south, min(lats))
        self.assertGreaterEqual(north, max(lats))
        # The matched river itself has to be inside the frame, with the padding
        # the user needs in order to click slightly off the channel.
        self.assertGreater(max(lons) - min(lons), 0.0)
        self.assertGreater(east - west, 0.1)

    def test_the_axes_rectangle_is_a_proper_part_of_the_image(self):
        """The figure has margins; a click must be mapped through the plot area."""
        record = self.call(RIVER_NAME)
        left, bottom, width, height = record["map"]["axes_rect"]
        self.assertGreater(left, 0.0)
        self.assertGreater(bottom, 0.0)
        self.assertLess(width, 1.0)
        self.assertLess(height, 1.0)
        self.assertLessEqual(left + width, 1.0 + 1e-9)
        self.assertLessEqual(bottom + height, 1.0 + 1e-9)

    def test_the_box_center_maps_onto_the_axes_center(self):
        """The conversion the browser performs has to be the inverse of the box.

        A click at the centre of the plot area must land at the centre of
        ``bbox_wgs84`` — that is the only reason the box is reported separately
        from the image.
        """
        record = self.call(RIVER_NAME)
        west, south, east, north = record["map"]["bbox_wgs84"]
        left, bottom, width, height = record["map"]["axes_rect"]
        click_fx = left + width / 2.0
        click_fy = 1.0 - (bottom + height / 2.0)  # CSS top-down fraction
        ax = (click_fx - left) / width
        ay = (1.0 - click_fy - bottom) / height
        self.assertAlmostEqual(west + ax * (east - west), (west + east) / 2.0, places=9)
        self.assertAlmostEqual(south + ay * (north - south), (south + north) / 2.0, places=9)

    def test_the_candidate_limit_is_respected(self):
        record = self.call(RIVER_NAME, limit=2)
        self.assertLessEqual(len(record["candidates"]), 2)

    def test_an_unknown_river_gets_no_map(self):
        record = self.call("不存在的河")
        self.assertIsNone(record["map"])
        self.assertFalse(record["candidates"])
        self.assertTrue(record.get("message"))

    def test_a_substring_does_not_select_another_river(self):
        """``别的河`` must not be drawn as if it were the river that was asked for."""
        record = self.call("别")
        self.assertEqual(record["matched_names"], [])
        self.assertIsNone(record["map"])

    def test_manual_map_does_not_run_outlet_recommendation(self):
        from unittest.mock import patch

        with patch("workers.gis_worker.recommend", side_effect=AssertionError("must not rank outlets")):
            record = self.call(RIVER_NAME, limit=0)
        self.assertEqual(record["candidates"], [])
        self.assertEqual(record["matched_names"], [RIVER_NAME])
        self.assertTrue(record["map"]["image_base64"])

    def test_only_the_final_river_suffix_may_be_omitted(self):
        record = self.call("测试", limit=0)
        self.assertEqual(record["matched_names"], [RIVER_NAME])
        self.assertIsNotNone(record["map"])

    def test_unknown_name_in_manual_mode_is_a_normal_no_map_answer(self):
        record = self.call("不存在河", limit=0)
        self.assertEqual(record["matched_names"], [])
        self.assertEqual(record["candidates"], [])
        self.assertIsNone(record["map"])
        self.assertTrue(record["message"])

    def test_unknown_name_with_a_coordinate_centre_produces_only_a_map(self):
        from unittest.mock import patch
        args = SimpleNamespace(dem=self.dem, rivers=self.rivers, name="未知河", limit=5,
                               lon=CENTER_LON, lat=CENTER_LAT, defaults=dict(DEFAULT_DELINEATION))
        with patch("workers.gis_worker.recommend", side_effect=AssertionError("must not rank")):
            result = outlet_map(args)
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["matched_names"], [])
        self.assertEqual(result["confidence"], "coordinate_centre")
        self.assertTrue(result["map"]["image_base64"])
        west, south, east, north = result["map"]["bbox_wgs84"]
        self.assertTrue(west < CENTER_LON < east and south < CENTER_LAT < north)

    def test_valid_basin_area_cannot_substitute_for_named_river_consistency(self):
        tx = pyproj.Transformer.from_crs(4326, 32649, always_xy=True)
        x, y = tx.transform(CENTER_LON, CENTER_LAT)
        nearby = _river_outlet_check(self.rivers, RIVER_NAME, x, y, "EPSG:32649", DEFAULT_DELINEATION)
        far = _river_outlet_check(self.rivers, RIVER_NAME, x, y - 4000, "EPSG:32649", DEFAULT_DELINEATION)
        unknown = _river_outlet_check(self.rivers, "未知河", x, y, "EPSG:32649", DEFAULT_DELINEATION)
        self.assertTrue(nearby["checked"] and nearby["passed"])
        self.assertTrue(far["checked"])
        self.assertFalse(far["passed"])
        self.assertGreater(far["river_distance_m"], far["maximum_river_distance_m"])
        self.assertFalse(unknown["checked"])
        self.assertIsNone(unknown["passed"])
        self.assertIn("人工核对", unknown["note"])

    def test_model_point_outside_dem_is_not_drawn_as_a_valid_location(self):
        args = SimpleNamespace(dem=self.dem, rivers=self.rivers, name="未知河", limit=0,
                               lon=0, lat=0, defaults=dict(DEFAULT_DELINEATION))
        result = outlet_map(args)
        self.assertIsNone(result["map"])
        self.assertIn("覆盖范围", result["message"])

    def test_invalid_or_half_a_coordinate_centre_is_rejected(self):
        for lon, lat in [(CENTER_LON, None), (float("nan"), CENTER_LAT), (181, CENTER_LAT)]:
            with self.subTest(lon=lon, lat=lat), self.assertRaises(ValueError):
                outlet_map(SimpleNamespace(dem=self.dem, rivers=self.rivers, name="未知河", limit=0,
                                           lon=lon, lat=lat, defaults=dict(DEFAULT_DELINEATION)))

    def test_drawn_river_and_image_share_the_browser_geographic_grid(self):
        """A click on a known non-central river vertex must recover that vertex.

        This catches the old projected-raster/geographic-box mismatch; the
        browser's centre-only roundtrip cannot detect it.
        """
        from unittest.mock import patch
        from matplotlib.axes import Axes
        drawn = []
        image_extents = []
        original_fill, original_imshow = Axes.fill, Axes.imshow

        def fill(axis, xs, ys, **kwargs):
            if kwargs.get("color") == "#00d0ff":
                drawn.append((xs, ys))
            return original_fill(axis, xs, ys, **kwargs)

        def imshow(axis, data, **kwargs):
            image_extents.append(kwargs["extent"])
            return original_imshow(axis, data, **kwargs)

        with patch.object(Axes, "fill", fill), patch.object(Axes, "imshow", imshow):
            record = self.call(RIVER_NAME, limit=0)
        west, south, east, north = record["map"]["bbox_wgs84"]
        self.assertTrue(np.allclose(image_extents[0], [west, east, south, north], atol=1e-7))
        with shapefile.Reader(str(self.rivers), encoding="gbk") as reader:
            lon, lat = reader.shape(0).points[0]
        plotted_lon, plotted_lat = drawn[0][0][0], drawn[0][1][0]
        ax = (plotted_lon - west) / (east - west)
        ay = (plotted_lat - south) / (north - south)
        self.assertAlmostEqual(west + ax * (east - west), lon, places=7)
        self.assertAlmostEqual(south + ay * (north - south), lat, places=7)
