"""Adaptive processing ranges: a basin bigger than the range is not a dead end.

The failure these tests exist for: a basin whose longest flow path is longer
than the fixed 20 km half-window is cut by the range edge, the spatial QA
refuses the boundary, and the agent tells the user to adjust the outlet or the
window — neither of which the conversation offers, and the outlet choice is
irrelevant because every candidate on the same river is cut the same way.

The range therefore learns the basin's shape: a side the basin reached grows,
and a side it did not reach only has to cover the basin plus a margin. That
second rule is what keeps an elongated basin from paying for a square's empty
corners with grid resolution.

What these tests pin down: the range grows only when growing can help (never
against the DEM's own edge), it stops for a stated reason (attempts, per-side
ceiling, grid resolution, time budget), and a coarser grid is reported rather
than quietly used.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio

from workers.gis_worker import (
    DEFAULT_DELINEATION,
    CutReport,
    WindowPlan,
    _basin_is_stable,
    _boundary_truncated,
    _grow_all_sides,
    _next_extent,
    _plan_window,
    _window_findings,
)


DEFAULTS = dict(DEFAULT_DELINEATION)
#: A 400 km square in metres, so 20–100 km ranges fit well inside it.
BOUNDS = SimpleNamespace(left=0.0, bottom=0.0, right=400_000.0, top=400_000.0)
CX, CY = 200_000.0, 200_000.0


def square(half_km: float) -> dict[str, float]:
    return {side: half_km for side in ("west", "east", "south", "north")}


def a_plan(half_km: float = 20.0) -> WindowPlan:
    return _plan_window(BOUNDS, CX, CY, square(half_km), DEFAULTS)


class PlanWindowTests(unittest.TestCase):
    def test_a_range_that_fits_is_a_square_at_the_native_grid(self):
        plan = _plan_window(BOUNDS, CX, CY, square(20.0), DEFAULTS)
        self.assertEqual(plan.clamped_sides, ())
        self.assertAlmostEqual(plan.width_km, 40.0, places=3)
        self.assertAlmostEqual(plan.height_km, 40.0, places=3)
        self.assertEqual(plan.resolution_m, 30.0)
        self.assertLessEqual(plan.cells, DEFAULTS["maximum_processing_cells"])

    def test_the_grid_coarsens_only_when_the_cell_budget_is_spent(self):
        """With an 8M-cell budget a 90 km range still reads at 32 m."""
        for half_km, expected in ((20.0, 30.0), (30.0, 30.0), (45.0, 32.0), (100.0, 71.0)):
            plan = _plan_window(BOUNDS, CX, CY, square(half_km), DEFAULTS)
            self.assertEqual(plan.resolution_m, expected, half_km)
            self.assertLessEqual(plan.cells, DEFAULTS["maximum_processing_cells"], half_km)

    def test_a_tall_basin_does_not_pay_for_empty_corners(self):
        """A rectangle is charged for the cells it needs, not for its long side.

        The same 120 km span read as a square is four times the cells and has
        to coarsen the grid; cut to the basin's shape it stays at native
        resolution.
        """
        tall = {"west": 10.0, "east": 10.0, "south": 60.0, "north": 60.0}
        plan = _plan_window(BOUNDS, CX, CY, tall, DEFAULTS)
        self.assertAlmostEqual(plan.width_km, 20.0, places=3)
        self.assertAlmostEqual(plan.height_km, 120.0, places=3)
        self.assertEqual(plan.resolution_m, 30.0)
        as_square = _plan_window(BOUNDS, CX, CY, square(60.0), DEFAULTS)
        self.assertGreater(as_square.cells, plan.cells * 2.5)
        self.assertGreater(as_square.resolution_m, plan.resolution_m)   # 43 m vs 30 m

    def test_an_outlet_near_the_edge_gets_a_clamped_range_not_a_refusal(self):
        """This is what used to raise "too close to the DEM edge" outright."""
        plan = _plan_window(BOUNDS, 15_000.0, CY, square(20.0), DEFAULTS)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.clamped_sides, ("west",))
        self.assertAlmostEqual(plan.width_km, 35.0, places=3)   # 15 km west + 20 km east
        self.assertAlmostEqual(plan.height_km, 40.0, places=3)
        self.assertEqual(plan.resolution_m, 30.0)               # clamping does not sharpen

    def test_two_clamped_sides_are_both_recorded(self):
        plan = _plan_window(BOUNDS, 10_000.0, 390_000.0, square(20.0), DEFAULTS)
        self.assertEqual(sorted(plan.clamped_sides), ["north", "west"])

    def test_an_outlet_outside_the_dem_has_no_range(self):
        self.assertIsNone(_plan_window(BOUNDS, -1.0, CY, square(20.0), DEFAULTS))
        self.assertIsNone(_plan_window(BOUNDS, 400_001.0, CY, square(20.0), DEFAULTS))

    def test_an_outlet_in_a_sliver_of_data_has_no_range(self):
        sliver = SimpleNamespace(left=0.0, bottom=0.0, right=1_000.0, top=400_000.0)
        self.assertIsNone(_plan_window(sliver, 500.0, CY, square(20.0), DEFAULTS))


class NextExtentTests(unittest.TestCase):
    """When the range grows, when it tightens, and every reason to stop."""

    @staticmethod
    def grow(plan: WindowPlan, report: CutReport, *, attempt: int = 1,
             elapsed_s: float = 0.0, defaults: dict | None = None) -> dict[str, float] | None:
        return _next_extent(plan, report, CX, CY, attempt=attempt, elapsed_s=elapsed_s,
                            defaults=defaults or DEFAULTS)

    def test_a_basin_that_fits_is_not_grown(self):
        self.assertIsNone(self.grow(a_plan(), CutReport()))

    def test_a_side_the_basin_reached_grows(self):
        grown = self.grow(a_plan(), CutReport(sides=("north",)))
        self.assertAlmostEqual(grown["north"], 30.0, places=3)     # 20 km × 1.5

    def test_a_side_the_basin_did_not_reach_is_left_alone(self):
        """Never tightened: the flow field depends on the range itself, so a
        range shrunk around a basin can lose headwaters that drain across a
        flat — and then report a smaller, contained, *wrong* basin."""
        report = CutReport(sides=("north",),
                           basin_bounds=(195_000.0, 195_000.0, 205_000.0, 210_000.0))
        grown = self.grow(a_plan(), report)
        self.assertAlmostEqual(grown["north"], 30.0, places=3)     # cut side grows
        for side in ("west", "east", "south"):
            self.assertAlmostEqual(grown[side], 20.0, places=3, msg=side)

    def test_a_side_the_basin_comes_close_to_grows_for_clearance(self):
        """Containment is not enough: the flow field is distorted near the
        range's edges, so a basin that merely fits can be a false pass."""
        # The basin reaches 15 km west of the outlet, inside the 20 km range.
        report = CutReport(sides=("north",),
                           basin_bounds=(185_000.0, 199_000.0, 205_000.0, 201_000.0))
        grown = self.grow(a_plan(), report)
        margin = float(DEFAULTS["window_interior_margin_km"])
        # West: the basin is 15 km out, so clearance needs 25 km (and it grows).
        self.assertAlmostEqual(grown["west"], 15.0 + margin, places=3)
        # East and south: clearance needs 15 and 11 km, both below the current
        # 20 — and the range never tightens, so they stay where they are.
        self.assertAlmostEqual(grown["east"], 20.0, places=3)
        self.assertAlmostEqual(grown["south"], 20.0, places=3)
        self.assertAlmostEqual(grown["north"], 30.0, places=3)           # cut: growth wins

    def test_a_basin_well_inside_is_not_grown_for_clearance(self):
        report = CutReport(sides=("north",),
                           basin_bounds=(199_000.0, 199_000.0, 201_000.0, 201_000.0))
        grown = self.grow(a_plan(), report)
        for side in ("west", "east", "south"):
            self.assertAlmostEqual(grown[side], 20.0, places=3, msg=side)

    def test_growing_one_side_keeps_the_grid_finer_than_growing_all(self):
        """The rectangle is still worth having, even without tightening."""
        grown = self.grow(a_plan(60.0), CutReport(sides=("north",)))
        tall = _plan_window(BOUNDS, CX, CY, grown, DEFAULTS)
        square_all = _plan_window(BOUNDS, CX, CY, square(90.0), DEFAULTS)
        self.assertLess(tall.cells, square_all.cells)
        self.assertLessEqual(tall.resolution_m, square_all.resolution_m)

    def test_touching_only_the_data_edge_is_not_grown(self):
        """No range brings back a headwater the DEM does not contain."""
        plan = _plan_window(BOUNDS, 15_000.0, CY, square(20.0), DEFAULTS)
        self.assertIsNone(self.grow(plan, CutReport(sides=("west",))))

    def test_a_range_side_that_is_also_a_data_edge_still_grows_for_the_others(self):
        plan = _plan_window(BOUNDS, 15_000.0, CY, square(20.0), DEFAULTS)
        grown = self.grow(plan, CutReport(sides=("west", "north")))
        self.assertAlmostEqual(grown["north"], 30.0, places=3)

    def test_growth_stops_at_the_attempt_ceiling(self):
        ceiling = int(DEFAULTS["maximum_window_attempts"])
        self.assertIsNone(self.grow(a_plan(), CutReport(sides=("north",)), attempt=ceiling))

    def test_growth_stops_at_the_per_side_ceiling(self):
        self.assertIsNone(self.grow(a_plan(100.0), CutReport(sides=("north",))))

    def test_growth_stops_when_the_time_budget_is_spent(self):
        budget = float(DEFAULTS["window_time_budget_s"])
        self.assertIsNone(self.grow(a_plan(), CutReport(sides=("north",)), elapsed_s=budget))

    def test_adaptive_ranges_can_be_switched_off(self):
        off = {**DEFAULTS, "adaptive_window": False}
        self.assertIsNone(self.grow(a_plan(), CutReport(sides=("north",)), defaults=off))


class GrowAllSidesTests(unittest.TestCase):
    """The confirmation pass: every side out by a margin, once."""

    def test_every_side_moves_out_by_the_margin(self):
        grown = _grow_all_sides(a_plan(), DEFAULTS)
        margin = float(DEFAULTS["window_confirm_margin_km"])
        for side in ("west", "east", "south", "north"):
            self.assertAlmostEqual(grown[side], 20.0 + margin, places=3, msg=side)

    def test_no_confirmation_when_the_ceiling_is_reached(self):
        ceiling = float(DEFAULTS["maximum_half_window_km"])
        plan = _plan_window(BOUNDS, CX, CY, square(ceiling), DEFAULTS)
        self.assertIsNone(_grow_all_sides(plan, DEFAULTS))


class BasinStabilityTests(unittest.TestCase):
    """Two ranges give the same basin, within the stated tolerances."""

    @staticmethod
    def measured(area: float, bounds=(185_000.0, 195_000.0, 215_000.0, 210_000.0),
                 resolution: float = 30.0) -> dict:
        return {"area_km2": area, "bounds": bounds, "resolution_m": resolution}

    def test_the_same_basin_in_a_bigger_range_is_stable(self):
        self.assertTrue(_basin_is_stable(self.measured(1270.0), self.measured(1270.0), DEFAULTS))

    def test_a_rounding_difference_is_tolerated(self):
        self.assertTrue(_basin_is_stable(self.measured(1270.0), self.measured(1272.0), DEFAULTS))

    def test_a_materially_different_basin_is_not_stable(self):
        """The measured false pass: 252 km² versus 1270 km²."""
        self.assertFalse(_basin_is_stable(self.measured(252.0), self.measured(1270.0), DEFAULTS))

    def test_a_shifted_boundary_is_not_stable(self):
        self.assertFalse(_basin_is_stable(
            self.measured(1270.0, bounds=(180_000.0, 195_000.0, 215_000.0, 210_000.0)),
            self.measured(1270.0), DEFAULTS))

    def test_without_a_previous_attempt_there_is_nothing_to_confirm(self):
        self.assertFalse(_basin_is_stable(self.measured(1270.0), None, DEFAULTS))


class WindowMessageTests(unittest.TestCase):
    """Three different problems must not be reported as one."""

    def plan(self, half_km: float = 30.0, clamped: tuple = ()) -> WindowPlan:
        base = _plan_window(BOUNDS, CX, CY, square(half_km), DEFAULTS)
        return WindowPlan(**{**base.__dict__, "clamped_sides": tuple(clamped)})

    def test_a_grown_range_says_what_was_tried(self):
        errors, notes = _window_findings(self.plan(), CutReport(sides=("north", "east")),
                                         attempts=2, defaults=DEFAULTS)
        self.assertEqual(notes, [])
        self.assertIn("north", errors[0])
        self.assertIn("2", errors[0])            # attempts
        self.assertIn("60", errors[0])           # 30 + 30 km range

    def test_a_data_edge_is_reported_as_a_data_problem(self):
        errors, _ = _window_findings(self.plan(clamped=("west",)),
                                     CutReport(sides=("west",)), attempts=1, defaults=DEFAULTS)
        self.assertIn("DEM 数据范围", errors[0])
        self.assertNotIn("触及处理范围边界", errors[0])

    def test_both_problems_are_reported_separately(self):
        errors, _ = _window_findings(self.plan(clamped=("west",)),
                                     CutReport(sides=("west", "north")),
                                     attempts=2, defaults=DEFAULTS)
        self.assertEqual(len(errors), 2)
        self.assertIn("触及处理范围边界", errors[0])
        self.assertIn("DEM 数据范围", errors[1])

    def test_hitting_the_ceiling_says_what_to_do_instead(self):
        errors, _ = _window_findings(self.plan(half_km=100.0), CutReport(sides=("north",)),
                                     attempts=5, defaults=DEFAULTS, at_ceiling=True)
        self.assertIn("上游", errors[-1])
        self.assertIn("200", errors[-1])         # the 100 + 100 km range it reached

    def test_a_boundary_graze_is_a_note_not_a_refusal(self):
        """0.05% of cells touching no-data is a graze, not a truncation."""
        report = CutReport(nodata_contact_cells=152, basin_cells=310_105)
        errors, notes = _window_findings(self.plan(), report, attempts=2, defaults=DEFAULTS,
                                         confirmed_by_growth=True)
        self.assertEqual(errors, [])
        self.assertEqual(len(notes), 1)
        self.assertIn("152", notes[0])
        self.assertIn("栅格单元", notes[0])

    def test_a_real_cross_boundary_cut_is_an_error(self):
        report = CutReport(nodata_contact_cells=9_000, basin_cells=310_105)  # 2.9%
        errors, notes = _window_findings(self.plan(), report, attempts=2, defaults=DEFAULTS,
                                         confirmed_by_growth=True)
        self.assertEqual(notes, [])
        self.assertIn("跨界截断", errors[0])

    def test_a_confirmed_basin_that_fits_says_nothing(self):
        errors, notes = _window_findings(self.plan(), CutReport(), attempts=2,
                                         defaults=DEFAULTS, confirmed_by_growth=True)
        self.assertEqual((errors, notes), ([], []))

    def test_an_unconfirmed_pass_is_a_note_not_a_silent_pass(self):
        """Fitting inside the range is not proof of completeness."""
        errors, notes = _window_findings(self.plan(), CutReport(), attempts=1,
                                         defaults=DEFAULTS, confirmed_by_growth=False)
        self.assertEqual(errors, [])
        self.assertIn("确认流域稳定", notes[0])


class BoundarySidesTests(unittest.TestCase):
    """Which edge the basin reaches — and where the basin actually sits."""

    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.directory = Path(self._temp.name)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _rasters(self, basin: np.ndarray, window: np.ndarray,
                 *, nodata_value: int = 255) -> tuple[Path, Path]:
        basin_path = self.directory / "basin.tif"
        window_path = self.directory / "window.tif"
        profile = {
            "driver": "GTiff", "height": basin.shape[0], "width": basin.shape[1],
            "count": 1, "dtype": "uint8", "crs": "EPSG:32649", "nodata": nodata_value,
            "transform": rasterio.transform.from_origin(0, basin.shape[0] * 30, 30, 30),
        }
        with rasterio.open(basin_path, "w", **profile) as sink:
            sink.write(basin.astype("uint8"), 1)
        with rasterio.open(window_path, "w", **profile) as sink:
            sink.write(window.astype("uint8"), 1)
        return basin_path, window_path

    def test_the_reached_edges_are_named(self):
        basin = np.zeros((6, 6), dtype="uint8")
        basin[0, 2] = 1                                   # north edge only
        report = _boundary_truncated(*self._rasters(basin, np.ones((6, 6), dtype="uint8")))
        self.assertEqual(report.sides, ("north",))
        self.assertFalse(report.touches_dem_nodata)

    def test_a_corner_basin_names_both_edges(self):
        basin = np.zeros((6, 6), dtype="uint8")
        basin[0, 0] = 1
        report = _boundary_truncated(*self._rasters(basin, np.ones((6, 6), dtype="uint8")))
        self.assertEqual(sorted(report.sides), ["north", "west"])

    def test_an_interior_basin_touches_nothing_and_reports_its_box(self):
        basin = np.zeros((6, 6), dtype="uint8")
        basin[2, 3] = 1
        report = _boundary_truncated(*self._rasters(basin, np.ones((6, 6), dtype="uint8")))
        self.assertEqual(report.sides, ())
        self.assertEqual(report.nodata_contact_cells, 0)
        west, south, east, north = report.basin_bounds
        self.assertEqual((east - west, north - south), (30.0, 30.0))

    def test_dem_no_data_next_to_the_basin_is_counted(self):
        basin = np.zeros((6, 6), dtype="uint8")
        basin[3, 3] = 1
        window = np.ones((6, 6), dtype="uint8")
        window[3, 4] = 255                                # a hole beside the basin
        report = _boundary_truncated(*self._rasters(basin, window))
        self.assertEqual(report.sides, ())
        self.assertEqual(report.nodata_contact_cells, 1)
        self.assertAlmostEqual(report.nodata_fraction, 1.0)

    def test_the_contact_fraction_is_relative_to_the_basin(self):
        """One no-data cell grazes three basin cells, diagonals included."""
        basin = np.zeros((6, 6), dtype="uint8")
        basin[2:5, 2:5] = 1                               # nine cells, one graze
        window = np.ones((6, 6), dtype="uint8")
        window[3, 5] = 255
        report = _boundary_truncated(*self._rasters(basin, window))
        self.assertEqual(report.basin_cells, 9)
        self.assertEqual(report.nodata_contact_cells, 3)
        self.assertAlmostEqual(report.nodata_fraction, 3 / 9, places=4)


if __name__ == "__main__":
    unittest.main()
