"""Client for ``chongqing-spatial-mcp``.

Only the tools declared in :attr:`SpatialMCPClient.required_tools` exist as
methods. Image payloads are passed through untouched; the agent never decodes
or rewrites them.
"""

from __future__ import annotations

from typing import Any, Literal

from ..models import RainfallRequestPayload, drop_none
from .base import MCPClientBase, MCPProtocolError, validate_id


#: Process-figure names the spatial server accepts. Kept as a tuple so a typo
#: fails here rather than as a server-side rejection.
ProcessArtifact = Literal[
    "original_dem", "filled_dem", "fill_depth", "flow_direction",
    "flow_accumulation", "watershed", "virtual_stations", "rainfall_series",
    "rainfall_cumulative", "overlay", "package_extent",
]

PROCESS_ARTIFACTS: tuple[str, ...] = (
    "original_dem", "filled_dem", "fill_depth", "flow_direction",
    "flow_accumulation", "watershed", "virtual_stations", "rainfall_series",
    "rainfall_cumulative", "overlay", "package_extent",
)


class SpatialMCPClient(MCPClientBase):
    service_label = "空间处理 MCP"
    expected_name = "chongqing-watershed-spatial"
    required_tools = frozenset({
        "list_administrative_areas", "render_administrative_dem", "create_drawn_study_area",
        "list_datasets", "inspect_dataset", "validate_data_bundle",
        "recommend_outlets", "render_outlet_candidates", "delineate_from_outlet",
        "get_watershed_review", "render_process_artifact", "confirm_watershed",
        "build_virtual_rainfall", "prepare_model_inputs",
        "cleanup_spatial_products",
    })

    def validate_tool_contracts(self, rows) -> None:
        from ..station_count import adaptive_station_contract_error
        if adaptive_station_contract_error(rows):
            raise MCPProtocolError("空间处理 MCP 未加载按面积自动布站和自定义站数协议。",
                                   next_step="请重启本项目空间 MCP 服务后重试。")
        tool = next(item for item in rows if item.name == "render_outlet_candidates")
        parameter = tool.input_schema.get("properties", {}).get("max_candidates", {})
        if parameter.get("default") != 0 or parameter.get("minimum", 0) > 0:
            raise MCPProtocolError(
                "空间处理 MCP 仍使用旧的候选推荐协议，无法加载地图点选。",
                next_step="请重启本项目空间 MCP 服务以加载当前代码，然后重试地图选址。",
            )
        if not {"longitude", "latitude"} <= set(tool.input_schema.get("properties", {})):
            raise MCPProtocolError(
                "空间处理 MCP 未加载按坐标定位地图的协议。",
                next_step="请重启本项目空间 MCP 服务以加载当前代码，再重试地图选址。")
        delineation = next(item for item in rows if item.name == "delineate_from_outlet")
        if "river_name" not in delineation.input_schema.get("properties", {}):
            raise MCPProtocolError("空间处理 MCP 未加载目标河流一致性检查。",
                                   next_step="请重启本项目空间 MCP 服务后重试。")
        from watershed_exchange.domain_policy import regional_tool_contract_error
        if regional_tool_contract_error(rows):
            raise MCPProtocolError("空间处理 MCP 仍运行旧的手绘协议，尚未加载区域评价与水文诊断。",
                                   next_step="请更新空间 MCP 服务，再重新提交已保存的边界；原边界可继续使用。")

    # ---------------------------------------------------------------- catalog
    async def cleanup_spatial_products(self, identifiers: list[str], protected_ids: list[str],
                                       dry_run: bool = True) -> dict[str, Any]:
        from watershed_exchange.cleanup import product_ids
        return await self._call("cleanup_spatial_products", {
            "identifiers": sorted(product_ids(identifiers)),
            "protected_ids": sorted(product_ids(protected_ids)), "dry_run": dry_run})

    async def list_datasets(self) -> dict[str, Any]:
        """Managed datasets and whether they are present."""
        return await self._call("list_datasets")

    async def inspect_dataset(self, dataset_id: str) -> dict[str, Any]:
        """Metadata for one managed dataset."""
        if dataset_id not in {
            "dem", "river_network", "landuse", "administrative_boundary", "hourly_rainfall",
        }:
            raise ValueError(f"未知数据集：{dataset_id}")
        return await self._call("inspect_dataset", {"dataset_id": dataset_id})

    async def validate_data_bundle(self) -> dict[str, Any]:
        """Check the four base layers, shapefile components and rainfall years.

        Returns the envelope as-is: a bundle that fails validation is a normal
        diagnostic outcome, not a refusal.
        """
        return await self._call("validate_data_bundle", expect_ok=False)

    # ---------------------------------------------------------------- outlets
    async def list_administrative_areas(self) -> dict[str, Any]:
        return await self._call("list_administrative_areas")

    async def render_administrative_dem(self, administrative_area: str) -> dict[str, Any]:
        return await self._call("render_administrative_dem",{"administrative_area":administrative_area})

    async def create_drawn_study_area(self, administrative_area: str | None, coordinates: list[list[float]],
                                    location_label: str | None = None,
                                    assessment_mode: str = "regional_screening") -> dict[str, Any]:
        return await self._call("create_drawn_study_area",{
            "administrative_area":administrative_area,"coordinates":coordinates,"location_label":location_label,
            "assessment_mode":assessment_mode})

    async def recommend_outlets(self, river_name: str,
                                max_candidates: int = 5) -> dict[str, Any]:
        """Candidate outlets for a river name. Candidates, never a conclusion."""
        name = str(river_name).strip()
        if not name:
            raise ValueError("河流名称不能为空")
        if not 1 <= max_candidates <= 10:
            raise ValueError("max_candidates 必须在 1 到 10 之间")
        return await self._call("recommend_outlets",
                                {"river_name": name, "max_candidates": max_candidates})

    async def render_outlet_candidates(self, river_name: str,
                                       max_candidates: int = 0,
                                       longitude: float | None = None,
                                       latitude: float | None = None) -> dict[str, Any]:
        """Clickable siting map, without recommendations when max_candidates=0."""
        name = str(river_name).strip()
        if not name:
            raise ValueError("河流名称不能为空")
        if not 0 <= max_candidates <= 10:
            raise ValueError("max_candidates 必须在 0 到 10 之间")
        arguments = {"river_name": name, "max_candidates": max_candidates}
        if (longitude is None) != (latitude is None):
            raise ValueError("地图中心必须同时提供经度和纬度")
        if longitude is not None:
            if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
                raise ValueError("地图中心经纬度超出范围")
            arguments.update(longitude=longitude, latitude=latitude)
        return await self._call("render_outlet_candidates", arguments)

    async def delineate_from_outlet(
        self,
        longitude: float,
        latitude: float,
        snap_radius_cells: int = 5,
        half_window_km: float = 20.0,
        river_name: str | None = None,
    ) -> dict[str, Any]:
        """Delineate the upstream watershed of a WGS84 outlet."""
        if not -180.0 <= float(longitude) <= 180.0:
            raise ValueError("经度超出范围")
        if not -90.0 <= float(latitude) <= 90.0:
            raise ValueError("纬度超出范围")
        if not 1 <= snap_radius_cells <= 20:
            raise ValueError("snap_radius_cells 必须在 1 到 20 之间")
        if not 10.0 <= half_window_km <= 100.0:
            raise ValueError("half_window_km 必须在 10 到 100 之间")
        arguments = {
            "longitude": float(longitude),
            "latitude": float(latitude),
            "snap_radius_cells": int(snap_radius_cells),
            "half_window_km": float(half_window_km),
        }
        if river_name:
            arguments["river_name"] = river_name
        return await self._call("delineate_from_outlet", arguments)

    async def get_watershed_review(self, review_id: str) -> dict[str, Any]:
        """The review record for a delineation: outlet, area, QA, artifacts."""
        return await self._call("get_watershed_review",
                                {"review_id": validate_id(review_id, "流域审核编号")})

    async def render_process_artifact(
        self,
        review_id: str,
        artifact: str = "watershed",
        rainfall_package_id: str | None = None,
        model_input_package_id: str | None = None,
    ) -> dict[str, Any]:
        """Render one process figure and return its image payload.

        Some figures need a rainfall package or a model input package; the
        server enforces which, so the agent passes what it has and surfaces the
        server's message when something is missing.
        """
        if artifact not in PROCESS_ARTIFACTS:
            raise ValueError(f"不支持的过程图件：{artifact}；允许值：{list(PROCESS_ARTIFACTS)}")
        arguments: dict[str, Any] = {
            "review_id": validate_id(review_id, "流域审核编号"),
            "artifact": artifact,
        }
        if rainfall_package_id:
            arguments["rainfall_package_id"] = validate_id(rainfall_package_id, "降雨包编号")
        if model_input_package_id:
            arguments["model_input_package_id"] = validate_id(model_input_package_id, "输入包编号")
        return await self._call_image("render_process_artifact", arguments)

    # ------------------------------------------------------------ confirmation
    async def confirm_watershed(self, review_id: str, confirmed: bool,
                                reviewer_note: str) -> dict[str, Any]:
        """Record the human decision on a delineated watershed."""
        note = str(reviewer_note).strip()
        if not note:
            raise ValueError("确认或拒绝必须写明理由")
        return await self._call("confirm_watershed", {
            "review_id": validate_id(review_id, "流域审核编号"),
            "confirmed": bool(confirmed),
            "reviewer_note": note[:2000],
        })

    # ---------------------------------------------------------------- rainfall
    async def build_virtual_rainfall(self, request: RainfallRequestPayload) -> dict[str, Any]:
        """Build the ERA5 virtual-station rainfall package for a confirmed basin."""
        return await self._call("build_virtual_rainfall", {"request": drop_none(request)})

    # ------------------------------------------------------------- model input
    async def prepare_model_inputs(
        self,
        review_id: str,
        rainfall_package_id: str,
    ) -> dict[str, Any]:
        """Clip the inputs and export the Model Input Package v1."""
        return await self._call("prepare_model_inputs", {
            "review_id": validate_id(review_id, "流域审核编号"),
            "rainfall_package_id": validate_id(rainfall_package_id, "降雨包编号"),
        })
