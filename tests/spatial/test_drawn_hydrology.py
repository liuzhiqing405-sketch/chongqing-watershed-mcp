import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box
from unittest.mock import patch

from chongqing_spatial_mcp.hydrologic_check import analyze_grid, check_polygon
from chongqing_spatial_mcp.state import require_spatial_quality
from slope_evaluation_agent.response_writer import describe_review
from slope_evaluation_agent.scientific_policy import numeric_audit
from workers.hydrology_worker import build_receivers


def valley():
    rows, cols = np.indices((80, 80))
    # The northern ridge separates external terrain; the inner valley flows S.
    cross = np.where(cols < 15, cols + 10, np.where(cols >= 65, 89 - cols, abs(cols - 39.5)))
    along = np.where(rows < 15, 52 + rows, 80 - rows)
    dem = (4 * cross + along).astype("float32")
    selected = (rows >= 15) & (rows < 60) & (cols >= 15) & (cols < 65)
    return dem, selected


def analyze(dem, selected, **kwargs):
    return analyze_grid(dem, np.isfinite(dem), selected, from_origin(500000, 3200000, 30, 30),
                        "EPSG:32649", min_channel_area_km2=.001, **kwargs)


def test_closed_valley_has_dominant_exit_and_does_not_claim_complete_watershed():
    dem, selected = valley()
    result = analyze(dem, selected)
    assert result["passed"] and result["checked"]
    assert result["external_upstream_cells"] == 0
    assert result["classification"] == "single_outlet_candidate"
    assert result["dominant_outlet_fraction"] > .9
    assert not result["complete_natural_watershed_verified"]
    assert result["outlets"][0]["area_km2"] == pytest.approx(selected.sum() * .0009)


def test_cutting_upstream_is_blocked_even_with_full_selected_dem_coverage():
    dem, selected = valley()
    selected[:35] = False
    result = analyze(dem, selected)
    assert not result["passed"]
    assert result["classification"] == "upstream_truncated"
    assert result["external_upstream_cells"] == 20 * 50
    assert result["inlets"]
    assert "外部入流" in result["errors"][0]
    with pytest.raises(ValueError, match="空间QA"):
        require_spatial_quality({"selection_mode": "drawn_study_area", "qa": {"passed": True, "hydrologic_check": result}})


def test_polygon_masked_dem_cannot_fake_a_closed_basin():
    dem, selected = valley()
    selected[:35] = False
    clipped = np.where(selected, dem, np.nan)
    result = analyze(clipped, selected)
    assert not result["passed"]
    assert result["upstream_context_edge_cells"] > 0


def test_flat_terrain_is_indeterminate_instead_of_fake_outlets():
    _, selected = valley()
    result = analyze(np.ones(selected.shape, dtype="float32"), selected)
    assert not result["checked"] and not result["passed"]
    assert result["classification"] == "indeterminate"
    assert "平坦" in result["errors"][0]


def test_missing_upstream_terrain_blocks_reliable_confirmation():
    dem, selected = valley()
    dem[15:20, 39:41] = np.nan
    selected[:25] = False
    result = analyze(dem, selected)
    assert not result["passed"] and result["upstream_context_edge_cells"] > 0


def test_multi_outlet_domain_is_allowed_as_regional_screening():
    dem, selected = valley()
    # Divide the inner valley into two independent valleys separated by a ridge.
    _, cols = np.indices(dem.shape)
    dem += (35 - abs(cols - 39.5)).clip(min=0) * 6
    result = analyze(dem, selected)
    assert result["passed"]
    assert result["major_outlet_count"] >= 2
    assert result["classification"] == "multiple_or_diffuse_outlets"
    assert any("多个" in warning for warning in result["warnings"])


def test_nonconvex_reentry_does_not_double_count_external_area():
    dem, selected = valley()
    selected[30:40, 35:45] = False  # Gap in selected terrain with throughflow.
    result = analyze(dem, selected)
    assert result["external_upstream_cells"] == 100
    assert sum(p["contributing_cells"] for p in result["inlets"]) <= 100
    assert sum(p["contributing_cells"] for p in result["outlets"]) <= selected.sum()


def test_edge_contributing_to_domain_is_reported_as_lower_bound():
    dem, selected = valley()
    result = analyze(dem[20:], selected[20:])
    assert not result["passed"] and result["upstream_context_edge_cells"] > 0


def write_dem(tmp_path, dem):
    path = tmp_path / "source.tif"
    with rasterio.open(path, "w", driver="GTiff", width=80, height=80, count=1, dtype="float32",
                       crs="EPSG:32649", transform=from_origin(500000, 3200000, 30, 30), nodata=-9999) as dst:
        dst.write(dem, 1)
    return path


def test_surrounding_source_dem_read_keeps_native_resolution_and_rechecks_larger_window(tmp_path):
    dem, _ = valley()
    path = write_dem(tmp_path, dem)
    polygon = box(500000 + 15 * 30, 3200000 - 60 * 30, 500000 + 65 * 30, 3200000 - 15 * 30)
    result = check_polygon(path, polygon, {"initial_buffer_m": 90})
    assert result["passed"]
    assert len(result["context_attempts"]) == 2
    assert result["context_attempts"][1]["cells"] > result["context_attempts"][0]["cells"]
    assert result["resolution_m"] == [30, 30]
    assert result["source_dem"] == str(path)


def test_budget_overflow_refuses_without_resampling(tmp_path):
    dem, _ = valley()
    result = check_polygon(write_dem(tmp_path, dem), box(500450, 3198200, 501950, 3199550), {"maximum_cells": 20})
    assert not result["checked"] and not result["passed"]
    assert "预算" in result["errors"][0]


def test_second_window_over_budget_cannot_silently_certify_first_window(tmp_path):
    dem, _ = valley()
    result = check_polygon(write_dem(tmp_path, dem), box(500450, 3198200, 501950, 3199550),
                           {"initial_buffer_m":90,"maximum_cells":3000})
    assert result["checked"] and not result["passed"]
    assert len(result["context_attempts"]) == 1
    assert any("复核" in error for error in result["errors"])


def test_shifted_outlet_with_same_area_is_not_mistaken_for_stable_context(tmp_path):
    dem, _ = valley()
    calls = []
    def changing_context(*args, **kwargs):
        result = analyze_grid(*args, **kwargs)
        calls.append(result)
        if len(calls) == 2:
            result["_exit_weights"] = {(x + 30, y): weight for (x,y),weight in result["_exit_weights"].items()}
        return result
    with patch("chongqing_spatial_mcp.hydrologic_check.analyze_grid",side_effect=changing_context):
        result = check_polygon(write_dem(tmp_path, dem), box(500450,3198200,501950,3199550), {"initial_buffer_m":90})
    assert not result["passed"] and result["classification"] == "context_unstable"
    assert not result["context_stability_verified"]
    assert "_exit_weights" not in result


def test_small_external_contribution_is_disclosed_even_when_below_tolerance():
    dem, selected = valley()
    selected[15] = False
    result = analyze(dem, selected)
    assert result["passed"]
    assert 0 < result["external_area_fraction"] < .05
    assert any("少量框外" in warning for warning in result["warnings"])


def test_d8_uses_physical_distance_for_rectangular_pixels():
    dem=np.array([[20,20,20],[20,10,9],[20,8,20]],dtype="float32")
    valid=np.ones(dem.shape,dtype=bool)
    parents=np.full(dem.shape,-1,dtype="int64")
    square,_=build_receivers(dem,valid,parents)
    rectangular,_=build_receivers(dem,valid,parents,cell_width=1,cell_height=10)
    assert square[4]==7 and rectangular[4]==5


def test_historical_unchecked_manual_review_is_not_certified():
    with pytest.raises(ValueError, match="水文合理性"):
        require_spatial_quality({"selection_mode": "drawn_study_area", "qa": {"passed": True}})


def test_reply_reports_failed_hydrology_with_auditable_numbers():
    dem, selected = valley()
    selected[:35] = False
    hydrology = analyze(dem, selected)
    record = {"review_id": "basin-test123", "selection_mode": "drawn_study_area",
              "qa": {"passed": False, "area_km2": 1.125, "selected_cells": 1250,
                     "hydrologic_check": hydrology, "errors": hydrology["errors"]}}
    text = describe_review(record)
    assert "水文合理性检查未通过" in text and "框外上游" in text
    assert "回复「确认」" not in text
    assert numeric_audit(text, {"review": record}) == []
