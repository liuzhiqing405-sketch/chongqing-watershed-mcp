from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
import shapefile
from affine import Affine
from pyproj import CRS

from workers.hydrology_worker import select_outlet
from workers.gis_worker import (CutReport, _named_river_coverage_check,
                                _river_snap_geometry, _write_river_snap_mask)
from slope_evaluation_agent.response_writer import describe_review


class NamedRiverSnapTests(unittest.TestCase):
    def setUp(self):
        self.transform = Affine(30,0,0,0,-30,2400)
        self.grid = np.ones((80,80))
        self.x,self.y = self.transform*(40.5,40.5)
        self.mask = np.zeros(self.grid.shape,dtype=np.uint8)
        self.mask[40,50] = 1

    def test_grows_beyond_five_cells_only_on_named_river(self):
        self.grid[40,44] = 3082  # original false-pass bank catchment
        self.grid[40,50] = 375000  # requested river outside the old radius
        self.grid[42,45] = 9000000  # competing main river must not win
        row,col,check = select_outlet(self.grid,self.transform,self.x,self.y,5,river_mask=self.mask)
        self.assertEqual((row,col),(40,50))
        self.assertTrue(check["passed"])
        self.assertEqual(check["river_channel_area_km2"],337.5)

    def test_large_unrelated_river_does_not_rescue_inconsistent_named_layer(self):
        self.grid[40,44] = 3082
        self.grid[40,50] = 90  # mapped river does not agree with flow grid
        self.grid[42,45] = 9000000
        row,col,check = select_outlet(self.grid,self.transform,self.x,self.y,5,river_mask=self.mask)
        self.assertFalse(check["passed"])
        self.assertEqual(check["reason"],"river_flow_mismatch")
        self.assertNotEqual((row,col),(40,50))

    def test_no_matched_channel_near_click_fails_without_silent_fallback_pass(self):
        self.mask[:] = 0
        self.grid[40,44] = 3082
        _,_,check = select_outlet(self.grid,self.transform,self.x,self.y,5,river_mask=self.mask)
        self.assertFalse(check["passed"])

    def test_cannot_move_past_physical_distance_limit(self):
        self.mask[:] = 0
        self.mask[40,58] = 1
        self.grid[40,58] = 375000
        _,_,check = select_outlet(self.grid,self.transform,self.x,self.y,5,river_mask=self.mask)
        self.assertFalse(check["passed"])

    def test_coordinate_only_outlet_keeps_existing_local_selection(self):
        self.grid[40,44] = 3082
        self.grid[40,50] = 375000
        row,col,check = select_outlet(self.grid,self.transform,self.x,self.y,5)
        self.assertEqual((row,col),(40,44))
        self.assertFalse(check["checked"])

    def test_small_nodata_contact_blocks_named_basin_completeness(self):
        report = CutReport(nodata_contact_cells=2)
        self.assertFalse(_named_river_coverage_check("目标河",report)["passed"])
        self.assertTrue(_named_river_coverage_check("目标河",CutReport())["passed"])
        self.assertIsNone(_named_river_coverage_check(None,report)["passed"])

    def test_failure_reply_addresses_data_instead_of_repeated_outlet_requests(self):
        for field in ["channel_check","coverage_check"]:
            text = describe_review({"qa":{"passed":False,"area_km2":2.77,
                "selected_cells":3082,field:{"passed":False}}})
            self.assertIn("不会进入降雨与稳定性计算",text)
            self.assertIn("核对或补齐空间数据",text)
            self.assertNotIn("请重新打开选址地图调整位置",text)

    def test_mask_excludes_confluence_other_water_and_respects_polygon_hole(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); path=folder/"rivers.shp"
            with shapefile.Writer(str(path)) as writer:
                writer.field("Name","C");writer.field("kind","C")
                writer.poly([[(1000,900),(1000,1600),(1600,1600),(1600,900),(1000,900)],
                             [(1200,1100),(1400,1100),(1400,1300),(1200,1300),(1200,1100)]])
                writer.record("Target","0123")
                writer.poly([[(1550,900),(1550,1600),(1750,1600),(1750,900),(1550,900)]])
                writer.record("Other","0123")
            path.with_suffix(".prj").write_text(CRS.from_epsg(32649).to_wkt(),encoding="utf-8")
            geometry,other=_river_snap_geometry(path,"Target","EPSG:32649",1450,1400,500)
            dem=folder/"dem.tif"
            with rasterio.open(dem,"w",driver="GTiff",height=80,width=80,count=1,dtype="float32",
                               crs="EPSG:32649",transform=self.transform) as dst:
                dst.write(np.ones((80,80),dtype="float32"),1)
            with rasterio.open(_write_river_snap_mask(dem,folder,geometry,other)) as src:
                mask=src.read(1)
                self.assertEqual(mask[src.index(1580,1400)],0)
                self.assertEqual(mask[src.index(1300,1200)],0)
                self.assertEqual(mask[src.index(1100,1400)],1)


if __name__ == "__main__":
    unittest.main()
