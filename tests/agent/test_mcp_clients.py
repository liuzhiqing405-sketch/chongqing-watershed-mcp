"""Phase 2 acceptance: the MCP client adapter layer.

Two levels, as the plan requires:

* contract tests drive the real servers over stdio — the production path;
* unit tests drive an in-process server, so failure mapping is exercised
  without paying for a subprocess per case.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any, Literal

import pytest
from mcp.server.mcpserver import Image, MCPServer

from slope_evaluation_agent.config import ServerSettings, load_settings
from slope_evaluation_agent.mcp_clients import (
    HydroMCPClient,
    MCPConnectionError,
    MCPProtocolError,
    MCPResponseSchemaError,
    MCPToolRejected,
    MCPToolTimeout,
    SpatialMCPClient,
)
from slope_evaluation_agent.mcp_clients.base import (
    envelope_data,
    envelope_facts,
    is_envelope,
    normalize,
)
from slope_evaluation_agent.models import HydroRunRequestPayload, RainfallRequestPayload


LOGGER = logging.getLogger("slope_agent_test")

#: A one-pixel PNG, used to check that image bytes pass through untouched.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


#: Parameter lists matching the real servers, so a stub server presents the
#: same input schema the production server does. Kept in the test file rather
#: than imported from the servers: these tests must not reach into them.
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
    "build_virtual_rainfall": "request: RainfallRequestPayload",
    "prepare_model_inputs": "review_id: str, rainfall_package_id: str",
    # hydro
    "get_hydro_capabilities": "",
    "import_hydro_input_package": "package: str",
    "register_hydro_inputs": "request: dict",
    "validate_hydro_input": "input_id: str",
    "list_hydro_inputs": "",
    "plan_hydro_run": "request: dict",
    "start_hydro_run": "request: dict",
    "get_hydro_run_status": "run_id: str",
    "cancel_hydro_run": "run_id: str",
    "get_hydro_run_result": "run_id: str",
    "recheck_hydro_qa": "run_id: str",
    "inspect_swmm_model": "run_id: str",
    "list_hydro_artifacts": "run_id: str, stage_id: str | None = None",
    "read_hydro_text_artifact": "run_id: str, artifact_id: str, limit_bytes: int = 20480",
    "list_hydro_visualization_options": "run_id: str | None = None, stage_id: str | None = None",
    "create_hydro_visualization": "run_id: str, kind: str",
    "get_hydro_visualization_status": "visualization_id: str",
    "get_hydro_visualization": "visualization_id: str",
    "compare_hydro_visualizations": "visualization_ids: list[str], mode: str | None = None",
}


def make_handler(tool_name: str, behaviour):
    """Build a tool function with the real signature and a chosen behaviour.

    The return annotation has to match what the behaviour actually returns:
    ``structured_output=True`` (which the real servers use, and which is what
    makes ``structured_content`` arrive) requires a serialisable annotation,
    while an image-returning tool must be annotated with ``Image`` or the SDK's
    result conversion rejects it.
    """
    annotation = "Image" if tool_name in IMAGE_TOOLS else "dict[str, Any]"
    namespace: dict[str, object] = {"_behaviour": behaviour, "Any": Any, "Literal": Literal, "Image": Image,
                                    "RainfallRequestPayload": RainfallRequestPayload}
    source = (
        f"def handler({TOOL_SIGNATURES[tool_name]}) -> {annotation}:\n"
        f"    return _behaviour()\n"
    )
    exec(compile(source, f"<stub {tool_name}>", "exec"), namespace)  # noqa: S102
    return namespace["handler"]


#: Tools that answer with an image rather than a structured payload. The real
#: servers register these without ``structured_output``, and so must the stubs:
#: an Image cannot be serialised into ``structuredContent``.
IMAGE_TOOLS = frozenset({"render_process_artifact", "get_hydro_visualization"})


def build_server(name: str, handlers: dict[str, object], *,
                 tools: set[str] | None = None) -> MCPServer:
    """An in-process MCP server exposing the given handlers plus inert stubs."""
    server = MCPServer(name, version="test")
    catalog = tools if tools is not None else set(TOOL_SIGNATURES)
    for tool_name in sorted(catalog):
        behaviour = handlers.get(tool_name, lambda: {})
        handler = make_handler(tool_name, behaviour)
        if tool_name in IMAGE_TOOLS:
            server.tool(name=tool_name)(handler)
        else:
            server.tool(name=tool_name, structured_output=True)(handler)
    return server


class StubClient(SpatialMCPClient):
    """A spatial client whose transport is an in-process server."""

    def __init__(self, server, settings: ServerSettings | None = None, **kwargs):
        super().__init__(settings or ServerSettings(args=["-m", "unused"]), LOGGER)
        self._server = server
        self.overridden_name = None

    def _connect_target(self):
        return self._server


class StubHydroClient(HydroMCPClient):
    def __init__(self, server, settings: ServerSettings | None = None):
        super().__init__(settings or ServerSettings(args=["-m", "unused"]), LOGGER)
        self._server = server

    def _connect_target(self):
        return self._server


# --------------------------------------------------------------------- contract


class TestStdioContract:
    """The production path: launch both servers and call them."""

    @pytest.mark.asyncio
    async def test_spatial_client_calls_list_datasets_over_stdio(self):
        settings = load_settings()
        async with SpatialMCPClient(settings.mcp.spatial, LOGGER) as client:
            payload = await client.list_datasets()
            assert len(client.available_tools) == 15
        assert payload["ok"] is True
        datasets = payload["data"]["datasets"]
        assert {item["dataset_id"] for item in datasets} >= {"dem", "river_network", "landuse"}

    @pytest.mark.asyncio
    async def test_hydro_client_calls_capabilities_over_stdio(self):
        settings = load_settings()
        async with HydroMCPClient(settings.mcp.hydro, LOGGER) as client:
            payload = await client.get_hydro_capabilities()
            assert len(client.available_tools) == 27
            assert 'inspect_hydro_artifact' in client.available_tools
            assert 'list_hydro_readable_artifacts' in client.available_tools
        assert payload["ok"] is True
        assert payload["data"]["engine"]["ok"] is True
        assert len(payload["data"]["stages"]) == 11

    @pytest.mark.asyncio
    async def test_parameter_bounds_come_from_the_server(self):
        settings = load_settings()
        async with HydroMCPClient(settings.mcp.hydro, LOGGER) as client:
            bounds = await client.parameter_bounds()
        assert "saturated_cohesion_kpa" in bounds
        low, high = bounds["saturated_cohesion_kpa"]
        assert low < high

    @pytest.mark.asyncio
    async def test_validate_data_bundle_over_stdio_reports_a_verdict(self):
        settings = load_settings()
        async with SpatialMCPClient(settings.mcp.spatial, LOGGER) as client:
            payload = await client.validate_data_bundle()
        # A failing bundle is a diagnostic answer, never an exception.
        assert isinstance(payload["ok"], bool)
        assert isinstance(payload["data"]["errors"], list)

    @pytest.mark.asyncio
    async def test_business_rejection_over_stdio_has_no_traceback(self):
        settings = load_settings()
        async with SpatialMCPClient(settings.mcp.spatial, LOGGER) as client:
            with pytest.raises(MCPToolRejected) as excinfo:
                await client.get_watershed_review("basin-doesnotexist")
        message = excinfo.value.user_message()
        assert "Traceback" not in message
        assert 'File "' not in message
        assert excinfo.value.next_step


# ------------------------------------------------------------------ unit level


class TestServiceIdentity:
    @pytest.mark.asyncio
    async def test_old_station_request_schema_requires_service_restart(self,monkeypatch):
        monkeypatch.setitem(TOOL_SIGNATURES,'build_virtual_rainfall','request: dict[str, Any]')
        server=build_server('chongqing-watershed-spatial',{},tools=SpatialMCPClient.required_tools)
        client=StubClient(server)
        with pytest.raises(MCPProtocolError) as error:
            await client.connect()
        assert '按面积自动布站' in str(error.value) and not client.connected

    @pytest.mark.asyncio
    async def test_old_drawn_schema_is_rejected_before_creating_review(self,monkeypatch):
        monkeypatch.setitem(TOOL_SIGNATURES,"create_drawn_study_area","administrative_area: str | None, coordinates: list[list[float]], location_label: str | None = None")
        server=build_server("chongqing-watershed-spatial",{},tools=SpatialMCPClient.required_tools)
        client=StubClient(server)
        with pytest.raises(MCPProtocolError) as error:
            await client.connect()
        assert "旧的手绘协议" in str(error.value)
        assert not client.connected

    @pytest.mark.asyncio
    async def test_stale_map_schema_is_rejected_before_a_tool_call(self, monkeypatch):
        monkeypatch.setitem(TOOL_SIGNATURES, "render_outlet_candidates", "river_name: str, max_candidates: int = 5")
        server = build_server("chongqing-watershed-spatial", {}, tools=SpatialMCPClient.required_tools)
        client = StubClient(server)
        with pytest.raises(MCPProtocolError) as error:
            await client.connect()
        assert "旧的候选推荐协议" in str(error.value)
        assert "重启" in error.value.next_step
        assert not client.connected

    @pytest.mark.asyncio
    async def test_wrong_service_name_is_a_protocol_error(self):
        server = build_server("some-other-service", {"list_datasets": lambda: {"datasets": []}},
                              tools=SpatialMCPClient.required_tools)
        client = StubClient(server)
        with pytest.raises(MCPProtocolError) as excinfo:
            await client.connect()
        assert "身份不符" in str(excinfo.value)
        assert client.connected is False

    @pytest.mark.asyncio
    async def test_missing_required_tool_is_a_protocol_error(self):
        server = build_server("chongqing-watershed-spatial", {}, tools={"list_datasets"})
        client = StubClient(server)
        with pytest.raises(MCPProtocolError) as excinfo:
            await client.connect()
        assert "缺少必需工具" in str(excinfo.value)
        # The detail lists what the server actually offers, and the message
        # names what is missing.
        assert excinfo.value.detail.get("available") == ["list_datasets"]
        assert "recommend_outlets" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_matching_service_connects(self):
        server = build_server("chongqing-watershed-spatial", {},
                              tools=SpatialMCPClient.required_tools)
        client = StubClient(server)
        await client.connect()
        assert client.connected is True
        assert client.available_tools == SpatialMCPClient.required_tools
        await client.close()


class TestFailureMapping:
    def _client(self, handler):
        server = build_server("chongqing-watershed-spatial", {"list_datasets": handler},
                              tools=SpatialMCPClient.required_tools)
        return StubClient(server)

    @pytest.mark.asyncio
    async def test_anticipated_failure_reaches_the_user_verbatim(self):
        # The real servers wrap failures they saw coming in ToolError, which is
        # what carries the message through to the client.
        from mcp.server.mcpserver.exceptions import ToolError

        def boom():
            raise ToolError("数据目录缺少 重庆.tif")

        client = self._client(boom)
        with pytest.raises(MCPToolRejected) as excinfo:
            await client.list_datasets()
        message = excinfo.value.user_message()
        assert "重庆.tif" in message
        assert "Error executing tool" not in message
        assert "Traceback" not in message

    @pytest.mark.asyncio
    async def test_absolute_paths_in_a_server_message_are_scrubbed(self):
        from mcp.server.mcpserver.exceptions import ToolError

        windows_path = "D:" + chr(92) + "lunwen1st" + chr(92) + "runs" + chr(92) + "basin-x"

        def boom():
            raise ToolError("记录不存在：" + windows_path)

        client = self._client(boom)
        with pytest.raises(MCPToolRejected) as excinfo:
            await client.list_datasets()
        message = excinfo.value.user_message()
        assert "lunwen1st" not in message
        assert "<内部路径>" in message
        assert "记录不存在" in message

    @pytest.mark.asyncio
    async def test_an_unexpected_crash_does_not_leak_internals(self):
        # A raw exception is masked by the SDK; the user still gets a usable
        # message rather than an empty one or a traceback.
        def boom():
            raise RuntimeError("secret internal state at C:/private/thing.py")

        client = self._client(boom)
        with pytest.raises(MCPToolRejected) as excinfo:
            await client.list_datasets()
        message = excinfo.value.user_message()
        assert "Traceback" not in message
        assert "C:/private" not in message
        assert "secret internal state" not in message

    @pytest.mark.asyncio
    async def test_ok_false_envelope_becomes_a_rejection(self):
        client = self._client(lambda: {"ok": False, "status": "rejected",
                                       "summary": "参数超出范围",
                                       "next_actions": ["调整参数后重试"],
                                       "facts": {}, "data": {}})
        with pytest.raises(MCPToolRejected) as excinfo:
            await client.list_datasets()
        assert "参数超出范围" in str(excinfo.value)
        assert "调整参数后重试" in excinfo.value.next_step

    @pytest.mark.asyncio
    async def test_tolerated_ok_false_is_returned_not_raised(self):
        client = self._client(lambda: {"ok": False, "status": "failed",
                                       "summary": "x", "facts": {}, "data": {}})
        # A run that failed is an outcome to explain, not a refusal to surface.
        payload = await client._call("list_datasets", expect_ok=False)
        assert payload["ok"] is False
        assert payload["status"] == "failed"

    @pytest.mark.asyncio
    async def test_missing_structured_content_is_a_schema_error(self):
        client = self._client(lambda: "just text")
        with pytest.raises(Exception) as excinfo:
            await client.list_datasets()
        assert "Traceback" not in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_unstartable_command_is_a_connection_error(self):
        settings = ServerSettings(command="definitely-not-a-real-binary-xyz", args=[],
                                  handshake_retries=0)
        client = SpatialMCPClient(settings, LOGGER)
        with pytest.raises(MCPConnectionError) as excinfo:
            await client.connect()
        message = excinfo.value.user_message()
        assert "Traceback" not in message
        assert "definitely-not-a-real-binary-xyz" in message

    @pytest.mark.asyncio
    async def test_tool_outside_the_declared_set_is_unreachable(self):
        client = self._client(lambda: {})
        await client.connect()
        with pytest.raises(MCPProtocolError) as excinfo:
            await client._call("some_other_tool", {})
        assert "不允许调用工具" in str(excinfo.value)
        await client.close()


class TestResponseNormalisation:
    def test_hydro_envelope_is_detected(self):
        payload = {"ok": True, "status": "ok", "summary": "s", "facts": {}, "data": {}}
        assert is_envelope(payload) is True

    def test_spatial_payload_is_not_an_envelope(self):
        assert is_envelope({"datasets": [], "managed_only": True}) is False

    def test_validation_verdict_is_not_an_envelope(self):
        # It has ``ok`` but that is a validation verdict, not an envelope status.
        assert is_envelope({"ok": False, "errors": ["x"], "next_step": "y"}) is False

    def test_wrapping_preserves_the_payload_verbatim(self):
        raw = {"datasets": [{"dataset_id": "dem"}], "managed_only": True}
        wrapped = normalize(raw, tool="list_datasets")
        assert wrapped["ok"] is True
        assert wrapped["data"] == raw

    def test_wrapping_lifts_the_validation_verdict(self):
        raw = {"ok": False, "errors": ["缺少 DEM"], "warnings": ["年份不连续"],
               "next_step": "修复缺失项后重新验证"}
        wrapped = normalize(raw, tool="validate_data_bundle")
        assert wrapped["ok"] is False
        assert wrapped["warnings"] == ["年份不连续"]
        assert wrapped["next_actions"] == ["修复缺失项后重新验证"]

    def test_envelope_passes_through_unchanged(self):
        payload = {"ok": True, "status": "succeeded", "summary": "s",
                   "facts": {"a": 1}, "warnings": [], "limitations": [],
                   "next_actions": [], "data": {"k": "v"}}
        assert normalize(payload, tool="t") is payload

    def test_accessors_tolerate_missing_blocks(self):
        assert envelope_facts({"ok": True}) == {}
        assert envelope_data({"ok": True}) == {}
        assert envelope_data({"data": "not a dict"}) == {}


class TestImagePassthrough:
    @pytest.mark.asyncio
    async def test_image_bytes_are_returned_undecoded(self):
        server = build_server("chongqing-watershed-spatial",
                              {"render_process_artifact": lambda: Image(data=PNG_BYTES,
                                                                        format="png")},
                              tools=SpatialMCPClient.required_tools)
        client = StubClient(server)
        payload = await client.render_process_artifact("basin-abcdef123456", "watershed")
        assert payload["mime_type"] == "image/png"
        assert base64.b64decode(payload["data"]) == PNG_BYTES

    @pytest.mark.asyncio
    async def test_non_image_result_is_a_schema_error(self):
        server = build_server("chongqing-watershed-spatial",
                              {"render_process_artifact": lambda: {"image": "x"}},
                              tools=SpatialMCPClient.required_tools)
        client = StubClient(server)
        with pytest.raises(MCPResponseSchemaError):
            await client.render_process_artifact("basin-abcdef123456")


class TestInputGuards:
    """User text can never reach a tool or a field the agent did not declare."""

    @pytest.mark.asyncio
    async def test_unknown_process_artifact_is_refused_locally(self):
        client = StubClient(build_server("chongqing-watershed-spatial", {}))
        with pytest.raises(ValueError):
            await client.render_process_artifact("basin-abcdef123456", "../../etc/passwd")

    @pytest.mark.asyncio
    async def test_malformed_review_id_is_refused(self):
        client = StubClient(build_server("chongqing-watershed-spatial", {}))
        for bad in ("", "../../x", "a/b", "x" * 200, "basin" + " " * 2):
            with pytest.raises(ValueError):
                await client.get_watershed_review(bad)

    @pytest.mark.asyncio
    async def test_out_of_range_coordinates_are_refused(self):
        client = StubClient(build_server("chongqing-watershed-spatial", {}))
        with pytest.raises(ValueError):
            await client.delineate_from_outlet(999.0, 29.0)
        with pytest.raises(ValueError):
            await client.delineate_from_outlet(106.0, -95.0)
        with pytest.raises(ValueError):
            await client.delineate_from_outlet(106.0, 29.0, half_window_km=5000.0)

    @pytest.mark.asyncio
    async def test_empty_river_name_is_refused(self):
        client = StubClient(build_server("chongqing-watershed-spatial", {}))
        with pytest.raises(ValueError):
            await client.recommend_outlets("   ")

    @pytest.mark.asyncio
    async def test_package_name_with_a_separator_is_refused(self):
        client = StubHydroClient(build_server("watershed-hydro-stability", {}))
        for bad in ("../secret", "a/b", "a\\b", ".."):
            with pytest.raises(ValueError):
                await client.import_hydro_input_package(bad)

    @pytest.mark.asyncio
    async def test_confirmation_requires_a_note(self):
        client = StubClient(build_server("chongqing-watershed-spatial", {}))
        with pytest.raises(ValueError):
            await client.confirm_watershed("basin-abcdef123456", True, "   ")

    @pytest.mark.asyncio
    async def test_hydro_artifact_limit_is_bounded(self):
        client = StubHydroClient(build_server("watershed-hydro-stability", {}))
        with pytest.raises(ValueError):
            await client.read_hydro_text_artifact("hydro-run-abcdef123456",
                                                  "artifact-abcdef123456", 10_000_000)


class TestNamedMethodsOnly:
    def test_there_is_no_generic_call_any_tool(self):
        for cls in (SpatialMCPClient, HydroMCPClient):
            public = [name for name in dir(cls) if not name.startswith("_")]
            assert "call_any_tool" not in public
            assert "call_tool" not in public

    def test_every_declared_tool_has_a_named_method(self):
        for cls in (SpatialMCPClient, HydroMCPClient):
            methods = {name for name in dir(cls) if not name.startswith("_")}
            missing = {tool for tool in cls.required_tools
                       if tool not in methods and tool != "register_hydro_inputs"}
            # register_hydro_inputs takes a prepared payload dict by design.
            missing -= {"register_hydro_inputs"}
            assert not missing, f"{cls.__name__} 缺少具名方法：{sorted(missing)}"

    def test_required_tools_are_sorted_for_readability(self):
        for cls in (SpatialMCPClient, HydroMCPClient):
            assert isinstance(cls.required_tools, frozenset)
            assert len(cls.required_tools) >= 10
