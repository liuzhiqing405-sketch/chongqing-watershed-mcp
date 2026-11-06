from __future__ import annotations

import unittest

from mcp import Client

from chongqing_spatial_mcp.server import mcp


class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_lists_expected_tools(self):
        async with Client(mcp) as client:
            tools = await client.list_tools()
        tool_rows = tools.tools if hasattr(tools, "tools") else (tools[0] if isinstance(tools, tuple) else tools)
        names = {tool.name for tool in tool_rows}
        self.assertTrue({
            "list_datasets", "validate_data_bundle", "recommend_outlets",
            "delineate_from_outlet", "confirm_watershed", "build_virtual_rainfall",
            "render_process_artifact", "prepare_model_inputs",
        } <= names)

    async def test_catalog_tool_returns_structured_content(self):
        async with Client(mcp) as client:
            result = await client.call_tool("list_datasets", {})
        self.assertFalse(result.is_error)
        self.assertIn("datasets", result.structured_content)

    async def test_expected_domain_error_is_visible_to_client(self):
        async with Client(mcp) as client:
            result = await client.call_tool(
                "get_watershed_review", {"review_id": "basin-doesnotexist"}
            )
        self.assertTrue(result.is_error)
        text = " ".join(getattr(item, "text", "") for item in result.content)
        self.assertIn("记录不存在", text)


if __name__ == "__main__":
    unittest.main()
