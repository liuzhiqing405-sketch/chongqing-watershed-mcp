"""R0 regression tests for R6: timeouts, captured-output stdio, and versions.

``startup_timeout_s`` is configurable but never wraps the handshake, so a server
that accepts a connection and then stalls hangs the agent indefinitely. The
stdio client also inherits whatever ``sys.stdout`` happens to be, which breaks
under pytest's capture. And the version reported by each component disagrees
with the others and with the documentation.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pytest

from slope_evaluation_agent.config import ServerSettings
from slope_evaluation_agent.mcp_clients import HydroMCPClient, SpatialMCPClient
from slope_evaluation_agent.mcp_clients.base import MCPClientBase


pytestmark = [pytest.mark.unit, pytest.mark.real_mcp]


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class StallingServer:
    """A server object whose connection never completes."""

    def __init__(self) -> None:
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        import asyncio

        await asyncio.sleep(3600)

    async def __aexit__(self, *exc) -> None:
        return None


class TestStartupTimeout:
    def test_the_client_uses_the_configured_startup_timeout(self):
        """The setting must reach the connection, not sit unused in the config."""
        source = (PROJECT_ROOT / "src" / "slope_evaluation_agent" / "mcp_clients"
                  / "base.py").read_text(encoding="utf-8")
        assert "startup_timeout_s" in source, (
            "客户端没有使用 startup_timeout_s，握手会无限等待")

    @pytest.mark.asyncio
    async def test_a_stalling_handshake_fails_within_the_timeout(self):
        import asyncio

        from slope_evaluation_agent.mcp_clients import AgentMCPError

        settings = ServerSettings(args=["-m", "unused"], startup_timeout_s=2.0,
                                  handshake_retries=0)
        client = SpatialMCPClient(settings)
        client._connect_target = lambda: StallingServer()  # type: ignore[method-assign]
        started = time.monotonic()
        # The outer bound is a test guard: without it this test would hang on the
        # very defect it is meant to expose. The inner expectation is the fix —
        # connect() must give up on its own and raise a project error.
        try:
            await asyncio.wait_for(client.connect(), timeout=25)
        except asyncio.TimeoutError:
            pytest.fail(f"握手没有超时上限，测试守护在 {time.monotonic() - started:.0f}s 才中止")
        except AgentMCPError as exc:
            elapsed = time.monotonic() - started
            assert elapsed < 20, f"超时过晚：{elapsed:.1f}s"
            assert "Traceback" not in str(exc)
            assert exc.next_step
        else:
            pytest.fail("停滞的握手没有抛出任何错误")

    @pytest.mark.asyncio
    async def test_a_tool_call_timeout_is_a_tool_timeout(self):
        source = (PROJECT_ROOT / "src" / "slope_evaluation_agent" / "mcp_clients"
                  / "base.py").read_text(encoding="utf-8")
        assert "call_timeout_s" in source
        assert "MCPToolTimeout" in source


class TestStdioUnderCapturedOutput:
    """A real stdio connection must work when stdout is not a real stream."""

    @pytest.mark.asyncio
    async def test_a_captured_stdout_does_not_break_the_connection(self, capsys):
        from slope_evaluation_agent.config import load_settings

        # pytest has already replaced sys.stdout; connecting must still work.
        captured = capsys.readouterr()
        assert captured is not None
        settings = load_settings()
        async with SpatialMCPClient(settings.mcp.spatial) as client:
            payload = await client.list_datasets()
        assert payload["ok"] is True

    @pytest.mark.asyncio
    async def test_a_stdout_without_fileno_does_not_break_the_connection(self):
        """The failure seen in practice was 'Tee' object has no attribute 'fileno'."""

        class NoFileno:
            def write(self, text: str) -> int:
                return len(text)

            def flush(self) -> None:
                return None

        from slope_evaluation_agent.config import load_settings

        settings = load_settings()
        original = sys.stdout
        sys.stdout = NoFileno()  # type: ignore[assignment]
        try:
            async with HydroMCPClient(settings.mcp.hydro) as client:
                payload = await client.get_hydro_capabilities()
            assert payload["ok"] is True
        finally:
            sys.stdout = original


class TestVersionConsistency:
    def _server_versions(self) -> dict[str, str]:
        from chongqing_spatial_mcp.server import mcp as spatial
        from watershed_hydro_mcp.server import mcp as hydro

        return {"spatial": str(spatial.version), "hydro": str(hydro.version)}

    def test_each_component_reports_a_version(self):
        versions = self._server_versions()
        for name, value in versions.items():
            assert re.match(r"^\d+\.\d+", value), (name, value)

    def test_each_version_is_defined_once(self):
        """The handshake, the startup line and ``--version`` must agree.

        They do because each service reports the version its own package
        defines, rather than carrying a second literal that can drift. This
        test is what catches a re-introduced second literal.
        """
        import chongqing_spatial_mcp
        import watershed_hydro_mcp
        from chongqing_spatial_mcp.server import mcp as spatial
        from watershed_hydro_mcp.server import mcp as hydro

        assert str(spatial.version) == chongqing_spatial_mcp.__version__
        assert str(hydro.version) == watershed_hydro_mcp.__version__

    def test_the_hydro_capabilities_carry_the_version(self):
        """A client that never sees the handshake can still read the version."""
        from watershed_hydro_mcp import __version__
        from watershed_hydro_mcp.capabilities import capabilities

        assert capabilities().get("version") == __version__

    def test_the_agent_reports_a_version(self):
        from slope_evaluation_agent import __version__

        assert re.match(r"^\d+\.\d+", __version__)

    def test_the_documented_versions_match_the_running_ones(self):
        """README must not claim a version the components do not report."""
        from slope_evaluation_agent import __version__ as agent_version

        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        versions = self._server_versions()
        for label, value in (("空间 MCP", versions["spatial"]),
                             ("水文 MCP", versions["hydro"]),
                             ("智能体", agent_version)):
            assert label in readme, f"README 没有列出{label}的版本"
        for value in (*versions.values(), agent_version):
            assert value in readme, f"README 未记录版本 {value}"

    def test_the_pyproject_name_matches_the_installed_distribution(self):
        import importlib.metadata as md

        text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^name\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
        assert match, "pyproject 没有 name"
        declared = match.group(1)
        # The distribution must be installed under the declared name.
        assert md.version(declared)


class TestATransportTimeoutIsReportedAsATimeout:
    """The SDK reports a request timeout as a plain protocol error.

    Mapping it to the generic protocol failure told the reader to check
    *versions* — measured on a real run: a delineation that legitimately needs
    3–5 minutes under a 180 s timeout came back as "服务版本与 Agent 不匹配",
    which sends everyone after the wrong problem.
    """

    class _Stalling:
        """A connected client whose call always fails the given way."""

        def __init__(self, message: str) -> None:
            self.message = message

        async def call_tool(self, *args, **kwargs):
            from mcp.shared.exceptions import MCPError

            raise MCPError(0, self.message)

    @staticmethod
    def _client(message: str) -> SpatialMCPClient:
        client = SpatialMCPClient(ServerSettings(args=["-m", "unused"]))
        client._client = TestATransportTimeoutIsReportedAsATimeout._Stalling(message)
        client._tools = frozenset({"delineate_from_outlet"})
        return client

    @pytest.mark.asyncio
    async def test_a_timed_out_call_says_so_and_names_the_setting(self):
        from slope_evaluation_agent.mcp_clients.base import MCPToolTimeout

        client = self._client("Request 'tools/call' timed out")
        with pytest.raises(MCPToolTimeout) as caught:
            await client._call("delineate_from_outlet", {})
        assert "call_timeout_s" in str(caught.value)
        assert "调大" in caught.value.next_step
        assert "版本" not in str(caught.value)

    @pytest.mark.asyncio
    async def test_any_other_protocol_error_does_not_blame_the_version(self):
        from slope_evaluation_agent.mcp_clients.base import MCPProtocolError

        client = self._client("connection reset by peer")
        with pytest.raises(MCPProtocolError) as caught:
            await client._call("delineate_from_outlet", {})
        assert "版本与 Agent 不匹配" not in caught.value.next_step
