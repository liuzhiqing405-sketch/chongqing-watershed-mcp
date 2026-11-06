"""Regression checks for legacy-compatible defaults through planner and core."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from watershed_hydro_mcp import planner
from watershed_hydro_mcp.core.stability import safety_factor as core
from watershed_hydro_mcp.core.sensitivity import analyze


ROOT = Path(__file__).resolve().parents[2]


def defaults():
    return yaml.safe_load((ROOT / "config/hydro.default.yaml").read_text(encoding="utf-8"))


def test_medium_matches_legacy_yongchuan_and_core_fallback(monkeypatch):
    planned = planner._override_stability(defaults(), "medium", {}, planner.SETTINGS)
    expected = {"deposit_thickness_m": 1.5, "saturated_cohesion_kpa": 6.0,
                "saturated_friction_angle_deg": 16.0, "initial_saturation": .35,
                "natural_density_g_cm3": 1.85, "saturated_density_g_cm3": 2.1,
                "water_density_g_cm3": 1.0, "runoff_density_g_cm3": 1.5,
                "friction_velocity_coefficient_a": .6}
    for configuration in [defaults(), {"stability": planned}, {"stability": {}}]:
        monkeypatch.setattr(core, "CFG", configuration)
        actual = core.effective_stability_params()
        assert {k: actual[k] for k in expected} == expected


def test_manual_overrides_survive_planner_and_core(monkeypatch):
    overrides = {"deposit_thickness_m": 1.8, "saturated_cohesion_kpa": 20.0,
                 "saturated_friction_angle_deg": 24.0}
    planned = planner._override_stability(defaults(), "medium", overrides, planner.SETTINGS)
    monkeypatch.setattr(core, "CFG", {"stability": planned})
    actual = core.effective_stability_params()
    assert {k: actual[k] for k in overrides} == overrides


@pytest.mark.parametrize("preset", ["low", "medium", "high"])
def test_classified_cells_keep_landuse_values_unmatched_cells_use_preset(monkeypatch, preset):
    planned = planner._override_stability(defaults(), preset, {}, planner.SETTINGS)
    monkeypatch.setattr(core, "CFG", {"stability": planned})
    classes = np.array([[1, 2], [0, 0]], dtype="int16")
    monkeypatch.setattr(core, "landuse_class_grid", lambda *args: (classes, {1: "耕地", 2: "有林地"}))
    result = core.stability_parameter_grids((2, 2), np.array([[True, True], [True, False]]), None, None)
    for field, source, known in [
        ("h", "deposit_thickness_m", [1.2, 1.5]),
        ("cohesion", "saturated_cohesion_kpa", [5., 7.]),
        ("phi", "saturated_friction_angle_deg", [15., 18.]),
        ("initial_saturation", "initial_saturation", [.4, .35]),
    ]:
        assert result[field][0] == pytest.approx(known)
        assert result[field][1, 0] == pytest.approx(planned[source])
        assert result[field][1, 1] == core.NODATA


def test_presets_bracket_medium_without_changing_density_or_friction_conversion():
    values = [planner._override_stability(defaults(), p, {}, planner.SETTINGS) for p in ["low", "medium", "high"]]
    assert [p["saturated_cohesion_kpa"] for p in values] == [3., 6., 12.]
    assert [p["saturated_friction_angle_deg"] for p in values] == [14., 16., 18.]
    assert [p["deposit_thickness_m"] for p in values] == [2., 1.5, 1.]
    for field in ["natural_density_g_cm3", "saturated_density_g_cm3", "runoff_density_g_cm3", "friction_velocity_coefficient_a"]:
        assert len({p[field] for p in values}) == 1
    bounds = planner.load_presets()["bounds"]
    for preset in values:
        for field, (lo, hi) in bounds.items():
            if field in preset:
                assert lo <= preset[field] <= hi


def test_sobol_default_matches_legacy_without_reinterpreting_saved_explicit_settings(monkeypatch):
    state = {"cohesion_kpa": np.array([5., 7.]), "friction_angle_deg": np.array([15., 18.])}
    for config, expected in [(defaults(), 2048), ({"analysis": {}}, 2048),
                             ({"analysis": {"sobol_base_samples": 256}}, 256)]:
        monkeypatch.setattr(analyze, "CFG", config)
        _, _, _, count, bootstrap = analyze.analysis_settings(state)
        assert count == expected
        assert bootstrap == 200
