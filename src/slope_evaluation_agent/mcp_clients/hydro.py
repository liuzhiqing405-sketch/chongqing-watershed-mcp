"""Client for ``watershed-hydro-stability-mcp``.

Only explicitly declared required or optional tools exist as methods.
`get_hydro_run_status` and `get_hydro_run_result` are read with
``expect_ok=False``: a run that finished with ``qa_failed`` reports ``ok: false``
as a legitimate result, and treating that as a refusal would hide exactly the
outcome the workflow must explain.
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from ..models import HydroRunRequestPayload, drop_none
from .base import MCPClientBase, validate_id


#: The service's own run modes.
RunMode = Literal["preflight_only", "rainfall_only", "build_run_qa"]

#: Terminal run states, as the hydro service reports them.
TERMINAL_RUN_STATUSES = frozenset({"succeeded", "qa_failed", "failed", "cancelled"})
ACTIVE_RUN_STATUSES = frozenset({"queued", "running"})
TERMINAL_VISUALIZATION_STATUSES = frozenset({"succeeded", "failed"})


class HydroMCPClient(MCPClientBase):
    service_label = "水文稳定性 MCP"
    expected_name = "watershed-hydro-stability"
    required_tools = frozenset({
        "get_hydro_capabilities", "import_hydro_input_package", "register_hydro_inputs",
        "validate_hydro_input", "list_hydro_inputs", "plan_hydro_run", "start_hydro_run",
        "get_hydro_run_status", "cancel_hydro_run", "get_hydro_run_result",
        "recheck_hydro_qa", "inspect_swmm_model", "list_hydro_artifacts",
        "read_hydro_text_artifact", "list_hydro_visualization_options",
        "create_hydro_visualization", "get_hydro_visualization_status",
        "get_hydro_visualization", "compare_hydro_visualizations",
        "list_visualization_artifacts", "prepare_artifact_export",
        "prepare_swmm_model_bundle", "locate_artifact", "list_swmm_model_artifacts",
        "cleanup_hydro_products",
    })
    optional_tools = frozenset({'inspect_hydro_artifact', 'list_hydro_readable_artifacts'})

    # -------------------------------------------------------------- capability
    async def cleanup_hydro_products(self, identifiers: list[str], protected_ids: list[str],
                                     dry_run: bool = True) -> dict[str, Any]:
        from watershed_exchange.cleanup import product_ids
        return await self._call("cleanup_hydro_products", {
            "identifiers": sorted(product_ids(identifiers)),
            "protected_ids": sorted(product_ids(protected_ids)), "dry_run": dry_run})

    async def get_hydro_capabilities(self) -> dict[str, Any]:
        """What the service can do now: engine, modules, presets, bounds, gates."""
        return await self._call("get_hydro_capabilities")

    async def list_visualization_artifacts(self, visualization_id: str) -> dict[str, Any]:
        """The real registered files behind one figure, by opaque artifact id."""
        return await self._call("list_visualization_artifacts",
                                {"visualization_id": validate_id(visualization_id, "可视化编号")})

    async def prepare_artifact_export(self, run_id: str, artifact_id: str,
                                      export_format: str) -> dict[str, Any]:
        """Ask the service to derive a viewable file from a registered one."""
        return await self._call("prepare_artifact_export", {
            "run_id": validate_id(run_id, "任务编号"),
            "artifact_id": validate_id(artifact_id, "成果编号"),
            "format": export_format,
        })

    async def prepare_swmm_model_bundle(self, run_id: str) -> dict[str, Any]:
        """Ask the service to package this run's SWMM model files for review."""
        return await self._call("prepare_swmm_model_bundle",
                                {"run_id": validate_id(run_id, "任务编号")})

    # ----------------------------------------------------------- downloading
    async def read_artifact(self, run_id: str, artifact_id: str, *,
                            expected_sha256: str | None = None,
                            expected_size: int | None = None) -> dict[str, Any]:
        """Read one registered artifact through the resource channel.

        For files small enough to arrive in one piece; anything larger goes
        through :meth:`locate_artifact`, which the same machine can stream from
        disk. The uri is built here from validated ids — no caller supplies it.
        """
        identifier = validate_id(run_id, "任务编号")
        artifact = validate_id(artifact_id, "成果编号")
        return await self._read_resource(
            f"hydro-artifact://{identifier}/{artifact}",
            expected_sha256=expected_sha256, expected_size=expected_size)

    async def locate_artifact(self, run_id: str, artifact_id: str) -> dict[str, Any]:
        """Where one registered file lives on this machine.

        The answer carries an absolute path and exists for the same-machine
        adapter only: it is never sent to a browser, never stored in a session
        and never logged. Everything a caller may show comes from the artifact's
        public descriptor.
        """
        return await self._call("locate_artifact", {
            "run_id": validate_id(run_id, "任务编号"),
            "artifact_id": validate_id(artifact_id, "成果编号"),
        })

    async def parameter_bounds(self) -> dict[str, tuple[float, float]]:
        """The server's own allowed range for each overridable parameter.

        The agent validates user-requested overrides against this, never
        against a limit of its own invention.
        """
        payload = await self.get_hydro_capabilities()
        data = payload.get("data") or {}
        raw = data.get("parameter_bounds") or {}
        bounds: dict[str, tuple[float, float]] = {}
        for key, value in raw.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                bounds[str(key)] = (float(value[0]), float(value[1]))
        return bounds

    # ------------------------------------------------------------------ inputs
    async def import_hydro_input_package(self, package: str) -> dict[str, Any]:
        """Import a Model Input Package v1 from an inbox root."""
        name = str(package).strip()
        if not name or "/" in name or "\\" in name or ".." in name:
            raise ValueError("包编号或归档名不能包含路径分隔符")
        return await self._call("import_hydro_input_package", {"package": name})

    async def register_hydro_inputs(self, request: dict[str, Any]) -> dict[str, Any]:
        """Register inputs that came from somewhere other than the spatial MCP."""
        return await self._call("register_hydro_inputs", {"request": request})

    async def validate_hydro_input(self, input_id: str) -> dict[str, Any]:
        """Re-check a stored input; a failed validation is a diagnostic result."""
        return await self._call("validate_hydro_input",
                                {"input_id": validate_id(input_id, "输入编号")},
                                expect_ok=False)

    async def list_hydro_inputs(self) -> dict[str, Any]:
        """Managed inputs already registered with the service."""
        return await self._call("list_hydro_inputs")

    # -------------------------------------------------------- plan and execute
    async def plan_hydro_run(self, request: HydroRunRequestPayload) -> dict[str, Any]:
        """Ask what would run, without starting anything."""
        return await self._call("plan_hydro_run", {"request": drop_none(request)})

    async def start_hydro_run(self, request: HydroRunRequestPayload) -> dict[str, Any]:
        """Submit the job; returns immediately with a run id."""
        return await self._call("start_hydro_run", {"request": drop_none(request)})

    async def get_hydro_run_status(self, run_id: str) -> dict[str, Any]:
        """Current state of a job. Never treat a non-succeeded state as success."""
        return await self._call("get_hydro_run_status",
                                {"run_id": validate_id(run_id, "任务编号")},
                                expect_ok=False)

    async def cancel_hydro_run(self, run_id: str) -> dict[str, Any]:
        """Cancel a queued or running job; finished jobs are left untouched."""
        return await self._call("cancel_hydro_run",
                                {"run_id": validate_id(run_id, "任务编号")})

    async def get_hydro_run_result(self, run_id: str) -> dict[str, Any]:
        """The result summary, including the quality-gate verdict."""
        return await self._call("get_hydro_run_result",
                                {"run_id": validate_id(run_id, "任务编号")},
                                expect_ok=False)

    async def recheck_hydro_qa(self, run_id: str) -> dict[str, Any]:
        """Re-run the quality gates over existing artifacts."""
        return await self._call("recheck_hydro_qa",
                                {"run_id": validate_id(run_id, "任务编号")},
                                expect_ok=False)

    # --------------------------------------------------------------- artifacts
    async def inspect_swmm_model(self, run_id: str) -> dict[str, Any]:
        """Structure and continuity of the run's SWMM model."""
        return await self._call("inspect_swmm_model",
                                {"run_id": validate_id(run_id, "任务编号")},
                                expect_ok=False)

    async def list_swmm_model_artifacts(self, run_id: str) -> dict[str, Any]:
        """The run's SWMM model files, by public descriptor only."""
        return await self._call("list_swmm_model_artifacts",
                                {"run_id": validate_id(run_id, "任务编号")})

    async def list_hydro_artifacts(self, run_id: str,
                                   stage_id: str | None = None) -> dict[str, Any]:
        """Registered artifacts; only the returned ``artifact_id`` may be used."""
        arguments: dict[str, Any] = {"run_id": validate_id(run_id, "任务编号")}
        if stage_id:
            arguments["stage_id"] = str(stage_id)
        return await self._call("list_hydro_artifacts", arguments)

    async def read_hydro_text_artifact(self, run_id: str, artifact_id: str,
                                       limit_bytes: int = 20 * 1024) -> dict[str, Any]:
        """Read a text artifact, capped by the server and again here."""
        if not 1024 <= limit_bytes <= 64 * 1024:
            raise ValueError("limit_bytes 必须在 1024 到 65536 之间")
        return await self._call("read_hydro_text_artifact", {
            "run_id": validate_id(run_id, "任务编号"),
            "artifact_id": validate_id(artifact_id, "成果编号"),
            "limit_bytes": int(limit_bytes),
        })

    async def inspect_hydro_artifact(self, run_id: str, artifact_id: str, *,
                                     offset: int = 0, limit: int = 30, search: str = '') -> dict[str, Any]:
        return await self._call('inspect_hydro_artifact', {
            'run_id': validate_id(run_id, '任务编号'),
            'artifact_id': validate_id(artifact_id, '成果编号'),
            'offset': offset, 'limit': limit, 'search': search})

    async def list_hydro_readable_artifacts(self, run_id: str) -> dict[str, Any]:
        return await self._call('list_hydro_readable_artifacts', {
            'run_id': validate_id(run_id, '任务编号')})

    # ---------------------------------------------------------- visualizations
    async def list_hydro_visualization_options(self, run_id: str | None = None,
                                               stage_id: str | None = None) -> dict[str, Any]:
        """The named figures the service can draw; never drawing code."""
        arguments: dict[str, Any] = {}
        if run_id:
            arguments["run_id"] = validate_id(run_id, "任务编号")
        if stage_id:
            arguments["stage_id"] = str(stage_id)
        return await self._call("list_hydro_visualization_options", arguments)

    async def create_hydro_visualization(self, run_id: str, kind: str) -> dict[str, Any]:
        """Submit a rendering job."""
        return await self._call("create_hydro_visualization", {
            "run_id": validate_id(run_id, "任务编号"),
            "kind": str(kind),
        })

    async def get_hydro_visualization_status(self, visualization_id: str) -> dict[str, Any]:
        """Rendering state and source-file digests."""
        return await self._call("get_hydro_visualization_status",
                                {"visualization_id": validate_id(visualization_id, "可视化编号")},
                                expect_ok=False)

    async def get_hydro_visualization(self, visualization_id: str) -> dict[str, Any]:
        """Fetch a finished figure as an image payload."""
        return await self._call_image("get_hydro_visualization",
                                      {"visualization_id": validate_id(visualization_id, "可视化编号")})

    async def compare_hydro_visualizations(self, visualization_ids: Sequence[str],
                                           mode: str | None = None) -> dict[str, Any]:
        """Compare figures: same run? same sources? comparable kind?"""
        ids = [validate_id(item, "可视化编号") for item in visualization_ids]
        if len(ids) < 2:
            raise ValueError("比较至少需要两个可视化编号")
        arguments: dict[str, Any] = {"visualization_ids": ids}
        if mode:
            arguments["mode"] = str(mode)
        return await self._call("compare_hydro_visualizations", arguments)
