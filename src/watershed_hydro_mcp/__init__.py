"""Watershed hydrology and slope-stability computation MCP server."""

#: The hydro service's own version, independent of the agent's and of the
#: spatial service's. This is the single source of truth: ``server.py`` reports
#: it to clients, ``get_hydro_capabilities`` states it, and the startup line
#: prints it.
__version__ = "0.1.0"
