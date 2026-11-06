"""Validate one streamable-HTTP MCP endpoint for local launch scripts.

Prints exactly one JSON object.  A contract mismatch is data (``ok=false``),
not a Python process failure, so Windows PowerShell can always parse the reason.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from mcp import Client
from mcp.client.streamable_http import streamable_http_client


def contract_problems(rows, required: set[str]) -> list[str]:
    """Presence alone cannot distinguish the old five-candidate map contract."""
    if "build_virtual_rainfall" in required:
        from slope_evaluation_agent.station_count import adaptive_station_contract_error
        error = adaptive_station_contract_error(rows)
        if error:
            return [error]
    if "create_drawn_study_area" in required:
        from watershed_exchange.domain_policy import regional_tool_contract_error
        error=regional_tool_contract_error(rows)
        if error:
            return [error]
    if "render_outlet_candidates" not in required:
        return []
    tool = next((item for item in rows if item.name == "render_outlet_candidates"), None)
    if tool is None:
        return []  # Already reported as a missing tool.
    parameter = tool.input_schema.get("properties", {}).get("max_candidates", {})
    if parameter.get("default") != 0 or parameter.get("minimum", 0) > 0:
        return ["render_outlet_candidates requires the manual-map contract (max_candidates=0); restart the spatial MCP with current source"]
    if not {"longitude", "latitude"} <= set(tool.input_schema.get("properties", {})):
        return ["render_outlet_candidates requires coordinate-centred maps; restart the spatial MCP with current source"]
    delineation = next((item for item in rows if item.name == "delineate_from_outlet"), None)
    if "delineate_from_outlet" in required and delineation and "river_name" not in delineation.input_schema.get("properties", {}):
        return ["delineate_from_outlet requires named-river consistency checks; restart the spatial MCP with current source"]
    return []


async def inspect(url: str, expected_name: str, required: set[str]) -> dict:
    try:
        async with Client(streamable_http_client(url)) as client:
            listed = await client.list_tools()
            rows = listed.tools if hasattr(listed, "tools") else listed
            names = {str(item.name) for item in rows}
            actual = str(client.server_info.name)
            missing = sorted(required - names)
            problems = contract_problems(rows, required)
            return {
                "ok": actual == expected_name and not missing and not problems,
                "name": actual,
                "version": str(client.server_info.version),
                "missing": missing,
                "error": "; ".join(problems) or None,
                "tool_count": len(names),
            }
    except Exception as exc:  # noqa: BLE001 - the launcher needs one clear reason
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("expected_name")
    parser.add_argument("required_csv")
    args = parser.parse_args()
    required = {item for item in args.required_csv.split(",") if item}
    print(json.dumps(
        asyncio.run(inspect(args.url, args.expected_name, required)),
        ensure_ascii=True,
    ))


if __name__ == "__main__":
    main()
