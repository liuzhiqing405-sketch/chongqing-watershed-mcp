from __future__ import annotations

from typing import Annotated, Any
from threading import RLock
from watershed_exchange.domain_policy import AssessmentMode

from pydantic import Field

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import __version__
from .catalog import inspect, list_catalog, validate_bundle
from .models import RainfallRequest
from .cleanup import cleanup_spatial_products as clean_products
from .rainfall import build_package
from .study_area import districts, area_map, create_area_review
from .workflow import (confirm, delineate, get_review, outlet_map, prepare_inputs,
                       recommend, render_artifact)


mcp = MCPServer(
    "chongqing-watershed-spatial",
    version=__version__,
    instructions=(
        "受控的DEM覆盖区任意中小流域数据预处理工具，不绑定大宁河或任何固定流域。"
        "出口推荐不是结论；划分后必须由用户审核并显式调用"
        "confirm_watershed。不得把ERA5虚拟站表述为实测站，不得把历史样本最大日表述为重现期设计暴雨。"
    ),
)


PRODUCT_LOCK = RLock()


def _expected(call):
    """Expose domain rejections to the agent without labelling them as crashes."""
    try:
        with PRODUCT_LOCK:
            return call()
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(structured_output=True)
def list_datasets() -> dict[str, Any]:
    """列出本服务允许访问的DEM、水系、土地利用、边界和逐时降雨数据。"""
    return list_catalog()


@mcp.tool(structured_output=True)
def list_administrative_areas() -> dict[str, Any]:
    """列出可沿行政边界裁切 DEM 的区县；模型计算仍采用用户确认的评价边界。"""
    return _expected(districts)


@mcp.tool(structured_output=True)
def render_administrative_dem(administrative_area: str) -> dict[str, Any]:
    """按区县多边形裁切原分辨率 DEM，渲染 WGS84 底图供用户绘制评价边界。"""
    return _expected(lambda: area_map(administrative_area))


@mcp.tool(structured_output=True)
def create_drawn_study_area(administrative_area: str | None, coordinates: list[list[float]],
                          location_label: str | None = None,
                          assessment_mode: AssessmentMode = "regional_screening") -> dict[str, Any]:
    """检查手绘 WGS84 边界、DEM/土地利用覆盖和框外 DEM 汇流，保存审核范围。

    不吸附出口，不自动改变边界，不宣称人工选区是完整自然流域。
    默认区域评价只计算框内降雨，框外来水与窗口截断作为提示；数据缺口仍阻断。
    catchment_check 保留严格上游检查，但不证明完整自然流域或包含外部入流。
    必须再显式 confirm_watershed 才能准备降雨或模型输入。
    """
    return _expected(lambda: create_area_review(administrative_area,coordinates,location_label=location_label,
                                               assessment_mode=assessment_mode))


@mcp.tool(structured_output=True)
def inspect_dataset(dataset_id: str) -> dict[str, Any]:
    """读取一个受管数据集的空间、字段或时间元数据；不接受任意文件路径。"""
    return _expected(lambda: inspect(dataset_id))


@mcp.tool(structured_output=True)
def validate_data_bundle() -> dict[str, Any]:
    """检查四类基础数据、Shapefile组件、降雨年份连续性和本机GIS运行时。"""
    return validate_bundle()


@mcp.tool(structured_output=True)
def recommend_outlets(river_name: str, max_candidates: int = 5) -> dict[str, Any]:
    """按当前水系库中的任意河流名称给出出口候选；候选必须划分并人工审核。"""
    if not 1 <= max_candidates <= 10:
        raise ValueError("max_candidates 必须在1到10之间")
    return _expected(lambda: recommend(river_name, max_candidates))


@mcp.tool(structured_output=True)
def delineate_from_outlet(
    longitude: float,
    latitude: float,
    snap_radius_cells: int = 5,
    half_window_km: float = 20.0,
    river_name: str | None = None,
) -> dict[str, Any]:
    """对DEM范围内任意WGS84出口填洼、计算汇流并划分待人工审核的上游流域。

    `half_window_km` 是**初始**处理窗口的半宽：流域若顶到窗口边缘，服务会按配置
    （`delineation.adaptive_window` 等）自动放大窗口重划，够用即止，并把每次尝试、
    最终窗口与栅格分辨率记入审核记录。窗口放大受四个上限约束：尝试次数、半窗口
    上限、栅格分辨率上限与时间预算；只有 DEM 数据范围本身不够时才会明确报告
    "数据范围不足"，而不是把截断的边界当成完整流域。出水口贴近 DEM 边界时窗口会
    被夹到数据范围内（其余方向照常放大）。
    """
    if not 1 <= snap_radius_cells <= 20:
        raise ValueError("snap_radius_cells 必须在1到20之间")
    if not 10 <= half_window_km <= 100:
        raise ValueError("half_window_km 必须在10到100之间")
    return _expected(lambda: delineate(longitude, latitude, snap_radius_cells, half_window_km,
                                      river_name=river_name))


@mcp.tool(structured_output=True)
def render_outlet_candidates(
    river_name: str, max_candidates: Annotated[int, Field(ge=0, le=10)] = 0,
    longitude: Annotated[float | None, Field(ge=-180, le=180)] = None,
    latitude: Annotated[float | None, Field(ge=-90, le=90)] = None,
) -> dict[str, Any]:
    """按河名生成可点选的出水口选址底图，默认不推荐地点。

    max_candidates=0 时只生成底图；正值保留低层 MCP 的兼容候选功能。
    返回 `map`：PNG（base64）、绘图区经纬度范围 `bbox_wgs84`、图像尺寸与
    `axes_rect`。前端据此把用户在地图上的点击线性换算成经纬度，再作为出水口调用
    delineate_from_outlet。候选与点选结果都只是待审对象，必须划分后由用户显式确认。
    """
    if not 0 <= max_candidates <= 10:
        raise ValueError("max_candidates 必须在0到10之间；0表示只生成选址底图")
    return _expected(lambda: outlet_map(river_name, max_candidates,
                                       longitude=longitude, latitude=latitude))


@mcp.tool(structured_output=True)
def get_watershed_review(review_id: str) -> dict[str, Any]:
    """读取出水口吸附、流域面积、边界路径和QA，供用户确认前检查。"""
    return _expected(lambda: get_review(review_id))


@mcp.tool()
def render_process_artifact(
    review_id: str,
    artifact: str = "watershed",
    rainfall_package_id: str | None = None,
    model_input_package_id: str | None = None,
) -> Image:
    """按需渲染并直接返回过程PNG。

    可用图件：
    original_dem / filled_dem / fill_depth / flow_direction / flow_accumulation /
    watershed（只需 review_id）；
    virtual_stations / rainfall_series / rainfall_cumulative（还需 rainfall_package_id）；
    overlay / package_extent（还需 model_input_package_id）。
    """
    return _expected(lambda: Image(path=render_artifact(
        review_id, artifact,
        rainfall_package_id=rainfall_package_id,
        model_input_package_id=model_input_package_id,
    )))


@mcp.tool(structured_output=True)
def confirm_watershed(review_id: str, confirmed: bool, reviewer_note: str) -> dict[str, Any]:
    """记录用户对划分后流域边界的明确确认或拒绝；这是后续工具的强制门控。"""
    if not reviewer_note.strip():
        raise ValueError("必须记录人工审核说明")
    return _expected(lambda: confirm(review_id, confirmed, reviewer_note))


@mcp.tool(structured_output=True)
def build_virtual_rainfall(request: RainfallRequest) -> dict[str, Any]:
    """在流域内布设虚拟站并插值ERA5逐时雨量，支持历史最大日或指定时段。"""
    return _expected(lambda: build_package(request))


@mcp.tool(structured_output=True)
def prepare_model_inputs(review_id: str, rainfall_package_id: str) -> dict[str, Any]:
    """仅对已确认流域裁切DEM、土地利用和水系，并绑定对应虚拟雨站包。"""
    return _expected(lambda: prepare_inputs(review_id, rainfall_package_id))


@mcp.tool(structured_output=True)
def cleanup_spatial_products(identifiers: list[str], protected_ids: list[str],
                             dry_run: bool = True) -> dict[str, Any]:
    """删除明确指定的会话产物；保护被引用产物，始终保留原始数据和公共地图缓存。"""
    def call():
        try:
            result = clean_products(identifiers, protected_ids, dry_run=dry_run)
        except OSError as exc:
            raise RuntimeError("空间产物清理失败；对话已保留，请检查文件占用后重试删除。") from exc
        return {"ok": True, "status": "planned" if dry_run else "deleted", "data": result,
                "summary": "会话空间产物清理", "facts": {}}
    return _expected(call)


def main(argv: list[str] | None = None) -> int:
    """Start the spatial MCP.

    ``--transport streamable-http`` makes it a persistent local service, which
    is what the agent uses so that a long-running companion task is not tied to
    the agent's own lifetime.
    """
    from watershed_exchange.serve import serve

    return serve(mcp, "chongqing-spatial-mcp", default_port=8765, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
