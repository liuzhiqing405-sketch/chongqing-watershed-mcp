"""Real MCP clients for the two servers this agent orchestrates."""

from .base import (
    AgentMCPError,
    MCPClientBase,
    MCPConnectionError,
    MCPProtocolError,
    MCPResponseSchemaError,
    MCPToolRejected,
    MCPToolTimeout,
    envelope_data,
    envelope_facts,
    envelope_status,
    envelope_summary,
)
from .hydro import HydroMCPClient
from .spatial import PROCESS_ARTIFACTS, SpatialMCPClient

__all__ = [
    "AgentMCPError",
    "HydroMCPClient",
    "MCPClientBase",
    "MCPConnectionError",
    "MCPProtocolError",
    "MCPResponseSchemaError",
    "MCPToolRejected",
    "MCPToolTimeout",
    "PROCESS_ARTIFACTS",
    "SpatialMCPClient",
    "envelope_data",
    "envelope_facts",
    "envelope_status",
    "envelope_summary",
]
