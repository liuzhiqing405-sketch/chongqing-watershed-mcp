"""Contract tests: the two MCP servers are separate services.

Each server is connected to through a real MCP client, so the tool sets,
service identity and structured returns are checked the way a client sees them.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp import Client  # noqa: E402

from chongqing_spatial_mcp.server import mcp as spatial_server  # noqa: E402
from watershed_hydro_mcp.server import mcp as hydro_server  # noqa: E402


SPATIAL_TOOLS = {
    "list_datasets", "inspect_dataset", "validate_data_bundle", "recommend_outlets",
    "delineate_from_outlet", "get_watershed_review", "render_process_artifact",
    "confirm_watershed", "build_virtual_rainfall", "prepare_model_inputs",
}

HYDRO_TOOLS = {
    "get_hydro_capabilities", "import_hydro_input_package", "register_hydro_inputs",
    "validate_hydro_input", "list_hydro_inputs", "plan_hydro_run", "start_hydro_run",
    "get_hydro_run_status", "cancel_hydro_run", "get_hydro_run_result", "recheck_hydro_qa",
    "inspect_swmm_model", "list_hydro_artifacts", "read_hydro_text_artifact",
    "list_hydro_visualization_options", "create_hydro_visualization",
    "get_hydro_visualization_status", "get_hydro_visualization",
    "compare_hydro_visualizations", "list_visualization_artifacts",
    "prepare_artifact_export", "prepare_swmm_model_bundle", "locate_artifact",
    "list_swmm_model_artifacts",
}

#: Tools that must never appear on the other server.
SPATIAL_ONLY = {"recommend_outlets", "delineate_from_outlet", "confirm_watershed",
                "build_virtual_rainfall", "prepare_model_inputs"}
HYDRO_ONLY = {"start_hydro_run", "plan_hydro_run", "inspect_swmm_model",
              "create_hydro_visualization", "import_hydro_input_package"}


async def _tool_names(server) -> set[str]:
    async with Client(server) as client:
        tools = await client.list_tools()
    rows = tools.tools if hasattr(tools, "tools") else (tools[0] if isinstance(tools, tuple) else tools)
    return {tool.name for tool in rows}


class ServiceIdentityTests(unittest.IsolatedAsyncioTestCase):
    async def test_service_names_differ(self):
        async with Client(spatial_server) as client:
            spatial_name = client.server_info.name
        async with Client(hydro_server) as client:
            hydro_name = client.server_info.name
        self.assertNotEqual(spatial_name, hydro_name)
        self.assertIn("spatial", spatial_name)
        self.assertIn("hydro", hydro_name)


class ToolSetTests(unittest.IsolatedAsyncioTestCase):
    async def test_spatial_server_exposes_expected_tools(self):
        names = await _tool_names(spatial_server)
        self.assertTrue(SPATIAL_TOOLS <= names, sorted(SPATIAL_TOOLS - names))

    async def test_hydro_server_exposes_expected_tools(self):
        names = await _tool_names(hydro_server)
        self.assertTrue(HYDRO_TOOLS <= names, sorted(HYDRO_TOOLS - names))

    async def test_spatial_server_has_no_hydro_tools(self):
        names = await _tool_names(spatial_server)
        self.assertEqual(names & HYDRO_ONLY, set())

    async def test_hydro_server_has_no_spatial_tools(self):
        names = await _tool_names(hydro_server)
        self.assertEqual(names & SPATIAL_ONLY, set())

    async def test_tool_sets_are_disjoint(self):
        spatial = await _tool_names(spatial_server)
        hydro = await _tool_names(hydro_server)
        self.assertEqual(spatial & hydro, set())


class StructuredReturnTests(unittest.IsolatedAsyncioTestCase):
    async def test_spatial_tool_returns_structured_content(self):
        async with Client(spatial_server) as client:
            result = await client.call_tool("list_datasets", {})
        self.assertFalse(result.is_error)
        self.assertIn("datasets", result.structured_content)

    async def test_capabilities_returns_envelope(self):
        async with Client(hydro_server) as client:
            result = await client.call_tool("get_hydro_capabilities", {})
        self.assertFalse(result.is_error)
        payload = result.structured_content
        for key in ("ok", "status", "summary", "facts", "warnings", "limitations",
                    "next_actions", "data"):
            self.assertIn(key, payload)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["data"]["independent_of_spatial_mcp"])

    async def test_capabilities_states_the_routing_limitation(self):
        async with Client(hydro_server) as client:
            result = await client.call_tool("get_hydro_capabilities", {})
        limitations = " ".join(result.structured_content["limitations"])
        self.assertIn("路由流量过程", limitations)

    async def test_visualization_options_expose_names_not_code(self):
        async with Client(hydro_server) as client:
            result = await client.call_tool("list_hydro_visualization_options", {})
        options = result.structured_content["data"]["options"]
        self.assertGreater(len(options), 20)
        for option in options:
            self.assertIn("kind", option)
            self.assertIn("stage_id", option)
            # No renderer function name, module path or code is ever exposed.
            self.assertNotIn("renderer", option)
            self.assertNotIn("module", option)

    async def test_invalid_run_id_is_refused_with_an_envelope(self):
        async with Client(hydro_server) as client:
            result = await client.call_tool("get_hydro_run_status", {"run_id": "../escape"})
        payload = result.structured_content
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status"], "failed")

    async def test_unknown_visualization_kind_is_refused(self):
        async with Client(hydro_server) as client:
            result = await client.call_tool(
                "create_hydro_visualization",
                {"run_id": "hydro-run-missing", "kind": "../../etc/passwd"},
            )
        self.assertFalse(result.structured_content["ok"])

    async def test_client_cannot_name_a_script_through_plan(self):
        async with Client(hydro_server) as client:
            result = await client.call_tool("plan_hydro_run", {
                "request": {"input_id": "hydro-input-abcdef123456",
                            "mode": "build_run_qa", "through": "os.system"},
            })
        self.assertTrue(result.is_error or not result.structured_content["ok"])


class FailureReportingTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_input_is_reported_without_a_traceback(self):
        async with Client(hydro_server) as client:
            result = await client.call_tool("validate_hydro_input",
                                            {"input_id": "hydro-input-missing0"})
        payload = result.structured_content
        self.assertFalse(payload["ok"])
        self.assertNotIn("Traceback", payload["summary"])
        self.assertNotIn("File \"", payload["summary"])


if __name__ == "__main__":
    unittest.main()
