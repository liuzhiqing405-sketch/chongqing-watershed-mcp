"""Visualizations the spatial MCP can render on demand.

`render_process_artifact` accepts any name from this catalog. Each entry says
which inputs it needs, so the tool can report a clear reason when a figure is
requested before its inputs exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SpatialVisualization:
    artifact: str
    title: str
    description: str
    #: ``review`` always exists; ``rainfall`` and ``model_input`` are optional.
    requires: tuple[str, ...] = ("review",)


CATALOG: tuple[SpatialVisualization, ...] = (
    SpatialVisualization("original_dem", "原始DEM",
                         "处理窗口内的原始高程栅格。"),
    SpatialVisualization("filled_dem", "填洼DEM",
                         "Priority-Flood 填洼后的高程栅格。"),
    SpatialVisualization("fill_depth", "填洼深度",
                         "填洼前后高程差，用于判断洼地分布。"),
    SpatialVisualization("flow_direction", "流向",
                         "D8 流向编码栅格。"),
    SpatialVisualization("flow_accumulation", "汇流累积量",
                         "上游汇流像元数，对数色标。"),
    SpatialVisualization("watershed", "流域边界",
                         "DEM 上的流域边界与吸附后的出水口位置。"),
    SpatialVisualization("virtual_stations", "虚拟雨站位置",
                         "选定 ERA5 虚拟雨站与研究区边界的位置关系。",
                         ("review", "rainfall")),
    SpatialVisualization("rainfall_series", "逐时降雨过程",
                         "各虚拟雨站的逐时降雨过程线。",
                         ("review", "rainfall")),
    SpatialVisualization("rainfall_cumulative", "累积降雨",
                         "各虚拟雨站与流域面平均的累积降雨曲线。",
                         ("review", "rainfall")),
    SpatialVisualization("overlay", "DEM与土地利用水系叠加",
                         "DEM 为底、叠加土地利用、水系与流域边界。",
                         ("review", "model_input")),
    SpatialVisualization("package_extent", "模型输入包覆盖范围",
                         "模型输入包中 DEM 与流域边界的覆盖范围。",
                         ("review", "model_input")),
)


def catalog() -> list[dict[str, Any]]:
    return [
        {"artifact": item.artifact, "title": item.title,
         "description": item.description, "requires": list(item.requires)}
        for item in CATALOG
    ]


def names() -> list[str]:
    return [item.artifact for item in CATALOG]


def by_name(artifact: str) -> SpatialVisualization:
    for item in CATALOG:
        if item.artifact == artifact:
            return item
    raise ValueError(f"不支持的过程图件：{artifact}；允许值：{names()}")


def requirements(artifact: str) -> tuple[str, ...]:
    return by_name(artifact).requires
