"""Automatic area selection and real rainfall output with manual overrides."""
import json

import pytest
from shapely.geometry import Polygon, MultiPolygon, box, mapping

from chongqing_spatial_mcp import rainfall
from chongqing_spatial_mcp.models import RainfallRequest
from chongqing_spatial_mcp.settings import Settings


def boundary(tmp_path, geometry):
    path = tmp_path / "basin.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {}, "geometry": mapping(geometry)}]}), encoding="utf-8")
    return path


def settings(tmp_path, **rain):
    return Settings(root=tmp_path, data_root=tmp_path / "data", runs_root=tmp_path / "runs",
                    exchange_root=tmp_path / "exchange", native_root=tmp_path / "native",
                    defaults={"rainfall": rain})


@pytest.mark.parametrize("side, expected", [(0.01, 1), (0.1, 2), (0.2, 7), (0.5, 20)])
def test_auto_counts_change_with_area_and_apply_limits(tmp_path, side, expected):
    path = boundary(tmp_path, box(0, 0, side, side))
    policy = rainfall.resolve_station_count(RainfallRequest(review_id="basin-abcdef123456", mode="historical_max"), path)
    assert policy["station_count"] == expected and policy["mode"] == "auto"
    assert policy["capped"] is (side == .5)


def test_holes_subtract_and_opposite_multipart_winding_does_not_cancel(tmp_path):
    geometry = Polygon(box(0, 0, .2, .2).exterior.coords, [list(box(.01, .01, .19, .19).exterior.coords)])
    request = RainfallRequest(review_id="basin-abcdef123456", mode="historical_max")
    selection = rainfall.resolve_station_count(request, boundary(tmp_path, geometry))
    assert 90 < selection["area_km2"] < 100 and selection["station_count"] == 2
    reversed_part = Polygon(list(box(.3, 0, .4, .1).exterior.coords)[::-1])
    selection = rainfall.resolve_station_count(request, boundary(tmp_path, MultiPolygon([box(0, 0, .1, .1), reversed_part])))
    assert 240 < selection["area_km2"] < 250 and selection["station_count"] == 4


def test_density_is_configurable_and_manual_count_wins(tmp_path):
    path = boundary(tmp_path, box(0, 0, .1, .1))
    request = RainfallRequest(review_id="basin-abcdef123456", mode="historical_max")
    configured = settings(tmp_path, automatic_station_area_km2=20., automatic_station_minimum=1,
                          automatic_station_maximum=8)
    assert rainfall.resolve_station_count(request, path, configured)["station_count"] == 7
    for count in (1, 6, 100):
        policy = rainfall.resolve_station_count(request.model_copy(update={"station_count": count}), path, configured)
        assert policy["mode"] == "manual" and policy["station_count"] == count
    stations = rainfall.select_virtual_stations(path, ["time", "0.0_0.0", "0.2_0.0", "0.0_0.2", "0.2_0.2"], 100)
    assert len({(s["lon"], s["lat"]) for s in stations}) == 100
    assert all(s["inside"] for s in stations)


@pytest.mark.parametrize("count", [None, 1, 6, 100])
def test_real_rainfall_zip_and_manifest_have_resolved_count(tmp_path, monkeypatch, count):
    path = boundary(tmp_path, box(0, 0, .1, .1))
    csv = tmp_path / "rain.csv"
    csv.write_text("time,0.0_0.0,0.2_0.0,0.0_0.2,0.2_0.2\n2024-07-01 00:00,1,2,3,4\n2024-07-01 01:00,2,3,4,5\n", encoding="utf-8")
    monkeypatch.setattr(rainfall, "rainfall_files", lambda s: {2024: csv})
    # A bad unrelated review area must not override the submitted geometry.
    monkeypatch.setattr(rainfall, "require_confirmed", lambda r, s: {
        "qa": {"area_km2": 50000}, "artifacts": {"boundary_geojson": str(path)}})
    request = RainfallRequest(review_id="basin-abcdef123456", mode="explicit_period", station_count=count,
                             start_datetime="2024-07-01 00:00", end_datetime="2024-07-01 01:00")
    package = rainfall.build_package(request, settings(tmp_path))
    expected = 2 if count is None else count
    assert len(package["stations"]) == package["station_count"] == expected
    assert package["station_selection"]["mode"] == ("auto" if count is None else "manual")
    assert (tmp_path / "runs" / "rainfall" / package["package_id"] / "station_rainfall.zip").is_file()
    saved = json.loads((tmp_path / "runs" / "rainfall" / package["package_id"] / "rainfall_manifest.json").read_text(encoding="utf-8"))
    assert saved["station_selection"] == package["station_selection"]


@pytest.mark.parametrize("options", [
    {"automatic_station_area_km2": 0}, {"automatic_station_area_km2": float("nan")},
    {"automatic_station_minimum": 21, "automatic_station_maximum": 20},
])
def test_invalid_density_configuration_is_not_silently_used(tmp_path, options):
    with pytest.raises(ValueError):
        rainfall.resolve_station_count(RainfallRequest(review_id="basin-abcdef123456", mode="historical_max"),
                                       boundary(tmp_path, box(0, 0, .1, .1)), settings(tmp_path, **options))
