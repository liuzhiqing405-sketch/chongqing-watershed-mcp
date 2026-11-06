"""Overlay layers whose CRS is unknown must be reported, not mis-drawn.

The failure this pins down: a model-input package whose ``river.shp`` has no
``.prj``. The layer is in degrees, the DEM is projected, and "reproject with
assume-it-already-matches" then turns every ring into map coordinates *read as
pixels* — the overlay comes out blank (everything lands thousands of pixels off
the canvas) with no error anywhere. The layer is therefore skipped and named in
the figure title instead.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import shapefile
from pyproj import CRS

from workers.gis_worker import _overlay_layers


def _write_layer(root: Path, name: str, *, with_prj: bool) -> None:
    writer = shapefile.Writer(str(root / name), shapeType=shapefile.POLYGON, encoding="gbk")
    writer.field("Name", "C", 40)
    writer.poly([[(107.0, 29.0), (107.1, 29.0), (107.1, 29.1), (107.0, 29.1), (107.0, 29.0)]])
    writer.record("测试河")
    writer.close()
    if with_prj:
        Path(str(root / name).replace(".shp", ".prj")).write_text(
            CRS.from_epsg(4326).to_wkt("WKT1_GDAL"), encoding="utf-8")


class OverlayLayerTests(unittest.TestCase):
    def test_a_layer_without_a_prj_is_reported_instead_of_drawn(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_layer(root, "river.shp", with_prj=False)
            drawable, missing = _overlay_layers(root)
        self.assertEqual(missing, ["river.shp"])
        self.assertEqual(drawable, [])

    def test_a_layer_with_a_prj_is_drawn_with_its_own_crs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_layer(root, "river.shp", with_prj=True)
            drawable, missing = _overlay_layers(root)
        self.assertEqual(missing, [])
        self.assertEqual([Path(path).name for path, _, _ in drawable], ["river.shp"])
        self.assertTrue(drawable[0][2].equals(CRS.from_epsg(4326)))

    def test_an_absent_layer_is_neither_drawn_nor_complained_about(self):
        with tempfile.TemporaryDirectory() as temp:
            drawable, missing = _overlay_layers(Path(temp))
        self.assertEqual(drawable, [])
        self.assertEqual(missing, [])

    def test_a_good_and_a_broken_layer_are_separated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_layer(root, "landuse.shp", with_prj=True)
            _write_layer(root, "river.shp", with_prj=False)
            drawable, missing = _overlay_layers(root)
        self.assertEqual(missing, ["river.shp"])
        self.assertEqual([Path(path).name for path, _, _ in drawable], ["landuse.shp"])
