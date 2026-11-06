"""In-process MCP servers that stand in for the two real ones.

The tool signatures mirror the real servers exactly, and they are registered
the same way (``structured_output=True``, except for the image-returning tools),
so a test exercises the client's real call, validation and error-mapping paths
without paying for a subprocess.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Literal

from mcp.server.mcpserver import Image, MCPServer

from slope_evaluation_agent.config import ServerSettings
from slope_evaluation_agent.mcp_clients import HydroMCPClient, SpatialMCPClient


LOGGER = logging.getLogger("slope_agent_test")

#: A one-pixel PNG, used to check that image bytes pass through untouched.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

#: Parameter lists matching the real servers.
TOOL_SIGNATURES: dict[str, str] = {
    "cleanup_spatial_products": "identifiers: list[str], protected_ids: list[str], dry_run: bool = True",
    "cleanup_hydro_products": "identifiers: list[str], protected_ids: list[str], dry_run: bool = True",
    "list_administrative_areas": "",
    "render_administrative_dem": "administrative_area: str",
    "create_drawn_study_area": "administrative_area: str | None, coordinates: list[list[float]], location_label: str | None = None, assessment_mode: Literal['regional_screening','catchment_check'] = 'regional_screening'",
    # spatial
    "list_datasets": "",
    "inspect_dataset": "dataset_id: str",
    "validate_data_bundle": "",
    "recommend_outlets": "river_name: str, max_candidates: int = 5",
    "render_outlet_candidates": "river_name: str, max_candidates: int = 0, longitude: float | None = None, latitude: float | None = None",
    "delineate_from_outlet": ("longitude: float, latitude: float, "
                              "snap_radius_cells: int = 5, half_window_km: float = 20.0, river_name: str | None = None"),
    "get_watershed_review": "review_id: str",
    "render_process_artifact": ("review_id: str, artifact: str = 'watershed', "
                                "rainfall_package_id: str | None = None, "
                                "model_input_package_id: str | None = None"),
    "confirm_watershed": "review_id: str, confirmed: bool, reviewer_note: str",
    "build_virtual_rainfall": "request: RainfallRequest",
    "prepare_model_inputs": "review_id: str, rainfall_package_id: str",
    # hydro
    "get_hydro_capabilities": "",
    "import_hydro_input_package": "package: str",
    "register_hydro_inputs": "request: dict[str, Any]",
    "validate_hydro_input": "input_id: str",
    "list_hydro_inputs": "",
    "plan_hydro_run": "request: dict[str, Any]",
    "start_hydro_run": "request: dict[str, Any]",
    "get_hydro_run_status": "run_id: str",
    "cancel_hydro_run": "run_id: str",
    "get_hydro_run_result": "run_id: str",
    "recheck_hydro_qa": "run_id: str",
    "inspect_swmm_model": "run_id: str",
    "list_hydro_artifacts": "run_id: str, stage_id: str | None = None",
    "read_hydro_text_artifact": "run_id: str, artifact_id: str, limit_bytes: int = 20480",
    'inspect_hydro_artifact': "run_id: str, artifact_id: str, offset: int = 0, limit: int = 30, search: str = ''",
    'list_hydro_readable_artifacts': 'run_id: str',
    "list_hydro_visualization_options": "run_id: str | None = None, stage_id: str | None = None",
    "create_hydro_visualization": "run_id: str, kind: str",
    "get_hydro_visualization_status": "visualization_id: str",
    "get_hydro_visualization": "visualization_id: str",
    "compare_hydro_visualizations": "visualization_ids: list[str], mode: str | None = None",
    "list_visualization_artifacts": "visualization_id: str",
    "prepare_artifact_export": "run_id: str, artifact_id: str, format: str",
    "prepare_swmm_model_bundle": "run_id: str",
    "locate_artifact": "run_id: str, artifact_id: str",
    "list_swmm_model_artifacts": "run_id: str",
}

#: Tools that answer with an image; the real servers register these without
#: ``structured_output`` because an Image cannot enter ``structuredContent``.
IMAGE_TOOLS = frozenset({"render_process_artifact", "get_hydro_visualization"})


def make_handler(tool_name: str, behaviour):
    """Build a tool function with the real signature and a chosen behaviour."""
    annotation = "Image" if tool_name in IMAGE_TOOLS else "dict[str, Any]"
    from chongqing_spatial_mcp.models import RainfallRequest
    namespace: dict[str, object] = {"_behaviour": behaviour, "Any": Any, "Literal": Literal, "Image": Image,
                                    "RainfallRequest": RainfallRequest}
    call = "_behaviour()"
    if tool_name == "build_virtual_rainfall" and getattr(behaviour, "use_requested_period", False):
        namespace["_requested_rainfall"] = _requested_rainfall
        call = "_requested_rainfall(_behaviour(), request)"
    source = (
        f"def handler({TOOL_SIGNATURES[tool_name]}) -> {annotation}:\n"
        f"    return {call}\n"
    )
    exec(compile(source, f"<stub {tool_name}>", "exec"), namespace)  # noqa: S102
    return namespace["handler"]


def _requested_rainfall(payload, request):
    """Supply a successful stub's actual event metadata from its real request."""
    from copy import deepcopy
    from datetime import datetime, timedelta
    if request.mode != "explicit_period":
        return payload
    result = deepcopy(payload)
    event = result.get("data") or result
    start = datetime.fromisoformat(request.start_datetime or f"{request.date} 00:00:00")
    end = datetime.fromisoformat(request.end_datetime or f"{request.date} 23:00:00")
    event.update(selection_mode=request.mode, event_start=start.isoformat(sep=' '),
                 event_end=end.isoformat(sep=' '), timestamp_count=int((end-start)/timedelta(hours=1))+1)
    return result


def build_server(name: str, handlers: dict[str, Any], *,
                 tools: set[str] | None = None,
                 resources: dict[str, bytes] | None = None) -> MCPServer:
    """An in-process MCP server exposing the given handlers plus inert stubs.

    ``resources`` registers the same artifact-uri template the real hydro
    service serves, backed by a fixed table, so a download test exercises the
    client's resource path (including a uri no table entry matches).
    """
    server = MCPServer(name, version="test")
    catalog = tools if tools is not None else set(TOOL_SIGNATURES)
    for tool_name in sorted(catalog):
        if tool_name.startswith("cleanup_"):
            default = lambda: envelope(data={"external_references": [], "deleted_ids": [], "retained_ids": []})
        else:
            default = lambda: {}
        behaviour = handlers.get(tool_name, default)
        handler = make_handler(tool_name, behaviour)
        if tool_name in IMAGE_TOOLS:
            server.tool(name=tool_name)(handler)
        else:
            server.tool(name=tool_name, structured_output=True)(handler)
    if resources is not None:
        table = dict(resources)

        @server.resource("hydro-artifact://{run_id}/{artifact_id}",
                         mime_type="application/octet-stream")
        def artifact_resource(run_id: str, artifact_id: str) -> bytes:
            uri = f"hydro-artifact://{run_id}/{artifact_id}"
            if uri not in table:
                raise ValueError(f"未登记的桩资源：{uri}")
            return table[uri]
    return server


class StubSpatialClient(SpatialMCPClient):
    """A spatial client whose transport is an in-process server."""

    def __init__(self, server, settings: ServerSettings | None = None) -> None:
        super().__init__(settings or ServerSettings(args=["-m", "unused"]), LOGGER)
        self._server = server

    def _connect_target(self):
        return self._server


class StubHydroClient(HydroMCPClient):
    """A hydro client whose transport is an in-process server."""

    def __init__(self, server, settings: ServerSettings | None = None) -> None:
        super().__init__(settings or ServerSettings(args=["-m", "unused"]), LOGGER)
        self._server = server

    def _connect_target(self):
        return self._server


def spatial_server(handlers: dict[str, Any]) -> MCPServer:
    return build_server("chongqing-watershed-spatial", handlers,
                        tools=SpatialMCPClient.required_tools)


def hydro_server(handlers: dict[str, Any], *,
                 resources: dict[str, bytes] | None = None) -> MCPServer:
    return build_server("watershed-hydro-stability", handlers,
                        tools=HydroMCPClient.required_tools, resources=resources)


# --------------------------------------------------------------- canned payloads


def envelope(**overrides) -> dict[str, Any]:
    payload = {"ok": True, "status": "ok", "summary": "", "facts": {},
               "warnings": [], "limitations": [], "next_actions": [], "data": {}}
    payload.update(overrides)
    return payload


def candidates_payload() -> dict[str, Any]:
    return {
        "river_name": "大宁河",
        "candidates": [
            {"longitude": 109.5, "latitude": 31.4, "elevation_m": 210.0,
             "dem_edge_distance_km": 30.0, "matched_river": "大宁河",
             "confidence": "candidate_only"},
            {"longitude": 109.6, "latitude": 31.5, "elevation_m": 260.0,
             "dem_edge_distance_km": 28.0, "matched_river": "大宁河",
             "confidence": "candidate_only"},
        ],
        "confidence": "candidate_only",
        "requires_delineation_and_human_confirmation": True,
    }


def outlet_map_payload(*, matched: bool = True) -> dict[str, Any]:
    """The siting-map answer: the same candidates, plus a clickable frame."""
    payload = candidates_payload()
    if not matched:
        return {**payload, "river_name": "不存在河", "matched_names": [],
                "candidates": [], "confidence": "none", "map": None,
                "message": "本地水系未找到该名称，无法出图；请换一个河名或直接给出坐标。"}
    payload["matched_names"] = ["大宁河"]
    payload["map"] = {
        "media_type": "image/png",
        "image_base64": base64.b64encode(PNG_BYTES).decode("ascii"),
        "image_width": 1296,
        "image_height": 1026,
        "bbox_wgs84": [109.30, 31.20, 109.90, 31.70],
        "axes_rect": [0.09, 0.07, 0.88, 0.86],
        "crs": "EPSG:4326",
        "note": "bbox_wgs84 是绘图区四边的经纬度。",
    }
    return payload


def review_payload(*, qa_passed: bool = True) -> dict[str, Any]:
    return {
        "review_id": "basin-abcdef123456",
        "status": "awaiting_human_confirmation",
        "outlet": {"requested_longitude": 109.5, "requested_latitude": 31.4,
                   "snapped_longitude": 109.501, "snapped_latitude": 31.401,
                   "shift_m": 120.0},
        "qa": {"passed": qa_passed, "errors": [] if qa_passed else ["流域触及处理窗口边界"],
               "area_km2": 2.15, "selected_cells": 236872,
               "touches_processing_window": not qa_passed, "touches_dem_nodata": False},
        "artifacts": {"boundary_geojson": "boundary.geojson"},
    }


def rainfall_payload(*, mode: str = "historical_max") -> dict[str, Any]:
    payload = {
        "package_id": "rain-abcdef123456",
        "selection_mode": mode,
        "event_start": "2024-07-30 00:00:00",
        "event_end": "2024-07-30 23:00:00",
        "timestamp_count": 24,
        "stations": [{"station_key": "VS01", "lon": 109.5, "lat": 31.4},
                     {"station_key": "VS02", "lon": 109.6, "lat": 31.5}],
        "source_note": "ERA5 gridded historical rainfall represented as virtual stations",
    }
    if mode == "historical_max":
        payload.update({"statistics_years": [2015, 2016, 2017],
                        "selected_day_basin_station_mean_total_mm": 79.29,
                        "selection_rule": "maximum complete calendar-day mean total"})
    else:
        payload["selection_rule"] = "user-specified inclusive hourly period"
    return payload


def model_input_payload() -> dict[str, Any]:
    return {
        "package_id": "package-abcdef123456",
        "status": "ready_for_hydro_preflight",
        "clipping": {"landuse": {"read": 117417, "kept": 6},
                     "river": {"read": 1702, "kept": 1}},
        "exchange": {
            "package_id": "package-abcdef123456",
            "directory": "exchange/exports/package-abcdef123456",
            "archive": "exchange/exports/package-abcdef123456.zip",
            "files": ["dem.tif", "basin.geojson", "station_rainfall.zip",
                      "landuse.shp", "river.shp"],
        },
        "limitations": ["ERA5 virtual stations are not measured gauges."],
    }


def import_payload() -> dict[str, Any]:
    return {
        "input_id": "hydro-input-abcdef123456",
        "origin": "model_input_package",
        "schema_version": "1.0",
        "source_package_id": "package-abcdef123456",
        "event": {"start_datetime": "2024-07-30 00:00:00",
                  "end_datetime": "2024-07-30 23:00:00",
                  "rainfall_source": "ERA5 virtual stations"},
        "study_area": {"area_km2": 2.148, "crs": "OGC:CRS84"},
        "limitations": ["ERA5 virtual stations are not measured gauges."],
    }


def validation_payload(*, ok: bool = True) -> dict[str, Any]:
    return {
        "input_id": "hydro-input-abcdef123456",
        "ok": ok,
        "errors": [] if ok else ["缺少文件：landuse.shp"],
        "warnings": [],
        "spatial": {"dem": {"crs": "EPSG:32649"}},
        "temporal": {"station_count": 4},
        "event": {"start_datetime": "2024-07-30 00:00:00"},
    }


def plan_payload() -> dict[str, Any]:
    """The plan payload as the service shapes it: the input record is nested.

    ``engine.directory`` is present on purpose. The real service returns it, and
    the stub has to carry the same shape as the service it stands in for — an
    earlier version of this stub omitted it, which hid a session that persisted
    a machine path through the plan.
    """
    return {
        "input": {"input_id": "hydro-input-abcdef123456",
                  "origin": "model_input_package"},
        "through": None,
        "stages": [{"stage_id": name, "title": name} for name in
                   ("sync", "validate", "delineate", "rainfall", "build",
                    "rainfall_plot", "swmm", "safety", "analysis", "export", "qa")],
        "parameter_preset": "medium",
        "parameter_overrides": {},
        "parameter_bounds": {"saturated_cohesion_kpa": [0.0, 60.0]},
        "engine": {"ok": True,
                   "files": {"runswmm.exe": True, "swmm5.dll": True},
                   "directory": "D:" + "\\" + "runs" + "\\" + "native" + "\\" + "swmm"},
    }


def run_status_payload(*, status: str = "running",
                       run_id: str = "hydro-run-abcdef123456",
                       completed: list[str] | None = None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": status,
        "summary": f"任务状态：{status}",
        "stages": ["sync", "validate", "delineate"],
        "completed_stages": completed if completed is not None else ["sync", "validate"],
        "current_stage": "delineate" if status == "running" else None,
        "error": None,
        # The real service reports the job's project directory absolutely; the
        # stub does too, so a change that started persisting it would be caught.
        "project": "C:" + "\\" + "runs" + "\\" + "hydro" + "\\" + "jobs" + "\\"
                   + "x" + "\\" + "project",
    }


def run_result_payload(*, status: str = "succeeded", qa_passed: bool = True,
                       safety_available: bool = True) -> dict[str, Any]:
    safety = {
        "available": safety_available,
        "minimum_safety_factor": 0.2752,
        "mean_safety_factor": 1.4295,
        "p95_safety_factor": 2.0,
        "maximum_safety_factor": 2.0,
        "max_unstable_fraction": 0.150457,
        "max_unstable_duration_h": 3.25,
        "max_flow_depth_m": 0.0225,
        "max_velocity_ms": 0.345,
        "max_pore_water_pressure_kpa": 12.5,
        "mechanics_model": "reference_equation_9",
        "parameter_preset": "medium",
        "swmm_dynamic_hydrograph_used": "True",
        "swmm_subcatchments": 6,
    }
    gate_names = ("spatial", "rainfall", "swmm", "stability", "export")
    return {
        "artifact_count": 168,
        "artifacts_by_stage": {"delineate": 13, "safety": 20, "export": 83},
        "swmm_model": {"available": True, "subcatchment_count": 6, "outfall_count": 6,
                       "routing_network_present": False,
                       "limitation": "各子汇水区径流之和不是选定流域出口的路由流量过程。"},
        "safety": safety,
        "rainfall": {"available": True, "station_count": 4, "active_rows": 26,
                     "total_rain_mm": 79.288, "peak_hourly_rain_mm": 6.9361},
        "quality": {"passed": qa_passed,
                    "failed": [] if qa_passed else ["stability", "swmm"],
                    "checked_at": "2026-01-01T00:00:00+00:00",
                    "gates": [{"name": name, "passed": qa_passed,
                               "errors": [] if qa_passed else ["门槛未通过"],
                               "facts": {}} for name in gate_names]},
        "limitations": ["各子汇水区径流之和不是选定流域出口的路由流量过程。"],
        "status": run_status_payload(status=status),
    }


def visualization_artifacts_payload(*, visualization_id: str = "viz-abcdef123456",
                                qa_failed: bool = False) -> dict[str, Any]:
    """The public view of one figure and the registered files behind it.

    Shaped like the real tool's ``data`` block, and deliberately free of any
    server path: the download is addressed by ``artifact_id`` alone.
    """
    return {
        "visualization_id": visualization_id,
        "run_id": "hydro-run-abcdef123456",
        "kind": "safety_minimum_factor",
        "title": "最小安全系数",
        "status": "succeeded",
        "qa_failed": qa_failed,
        "limitations": (["运行未通过质量门槛；图件带有 QA FAILED 标记，不得作为结论使用"]
                        if qa_failed else []),
        "image_sha256": "b" * 64,
        "sources": [{"scope": "project", "sha256": "c" * 64, "size_bytes": 4096}],
        "artifacts": [
            {
                "artifact_id": "artifact-abcdef123456",
                "title": "package_enhanced_minimum_safety_factor.tif",
                "format": "geotiff",
                "role": "source_raster",
                "media_type": "image/tiff",
                "size_bytes": 4096,
                "sha256": "c" * 64,
                "units": "dimensionless",
                "crs": "EPSG:32649",
                "qa_status": "failed" if qa_failed else "passed",
                "stage_id": "safety",
                "ready": True,
                "formats": ["original"],
                "exportable": [],
                "dtype": "float32",
                "nodata": -9999.0,
            },
            {
                "artifact_id": "artifact-abcdef654321",
                "title": "package_enhanced_safety_factor_summary.csv",
                "format": "csv",
                "role": "source_table",
                "media_type": "text/csv",
                "size_bytes": 2048,
                "sha256": "d" * 64,
                "units": None,
                "crs": None,
                "qa_status": "failed" if qa_failed else "passed",
                "stage_id": "safety",
                "ready": True,
                "formats": ["original"],
                "exportable": ["xlsx"],
                "columns": ["datetime_beijing", "representative_fs"],
                "rows": 26,
            },
        ],
        "warnings": [],
    }


def visualization_payload(*, status: str = "succeeded",
                          visualization_id: str = "viz-abcdef123456") -> dict[str, Any]:
    return {
        "visualization_id": visualization_id,
        "run_id": "hydro-run-abcdef123456",
        "kind": "safety_minimum_factor",
        "title": "最小安全系数",
        "stage_id": "safety",
        "status": status,
        "renderer_version": "1.0",
        "sources": [{"scope": "project", "relative_path": "safety.tif",
                     "sha256": "a" * 64, "size_bytes": 100}],
        "qa_failed": False,
        "size_bytes": 89668,
    }


class StubLLM:
    """A language model whose reply the test chooses.

    Shared by the conversational suites. ``reply`` is either the dict the model
    "returns" or an exception it raises, so the same stub covers a well-behaved
    model, a broken one and one that ignores every instruction.
    """

    def __init__(self, reply: dict[str, Any] | Exception, *,
                 calls: list[dict[str, str]] | None = None) -> None:
        self.reply = reply
        self.calls = calls if calls is not None else []

    async def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        self.calls.append({"system": system, "user": user})
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply
