"""Named visualizations per pipeline stage.

The catalog is the contract: a client picks a name from this list, never a
drawing command. Each entry knows which produced files it draws, so a figure
can always be traced back to the artifacts and their SHA-256 digests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Visualization:
    kind: str
    stage_id: str
    title: str
    description: str
    renderer: str
    source_roles: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)


CATALOG: tuple[Visualization, ...] = (
    # -------------------------------------------------------------------- sync
    Visualization(
        "sync_run_manifest", "sync", "运行配置快照",
        "本次运行的输入、阶段、参数预设与已冻结输入校验摘要。",
        "run_manifest", ("configuration",),
    ),
    # ---------------------------------------------------------------- validate
    Visualization(
        "validate_input_coverage", "validate", "输入覆盖与分辨率",
        "DEM 覆盖范围、像元分辨率与无数据像元比例，以及土地利用/边界/降雨的相对位置。",
        "input_coverage", ("dem", "boundary", "landuse"),
    ),
    # --------------------------------------------------------------- delineate
    Visualization(
        "delineate_subcatchments", "delineate", "子汇水区划分",
        "DEM 派生的 SWMM 子汇水区边界，附带异常小分区标注。",
        "subcatchments", ("partition",),
    ),
    Visualization(
        "delineate_area_histogram", "delineate", "子汇水区面积分布",
        "各子汇水区面积直方图，用于判断面积阈值是否合适。",
        "area_histogram", ("partition",),
    ),
    Visualization(
        "delineate_coverage_gap", "delineate", "划分覆盖缺口",
        "研究区内未被任何子汇水区覆盖的部分。",
        "coverage_gap", ("partition",),
    ),
    # ---------------------------------------------------------------- rainfall
    Visualization(
        "rainfall_station_series", "rainfall", "逐时降雨过程",
        "各虚拟雨站的逐时降雨过程线。",
        "rainfall_series", ("rainfall",),
    ),
    Visualization(
        "rainfall_cumulative", "rainfall", "累积降雨",
        "各站点与流域面平均的累积降雨曲线。",
        "rainfall_cumulative", ("rainfall",),
    ),
    Visualization(
        "rainfall_station_map", "rainfall", "降雨源分布",
        "虚拟雨站与研究区边界的空间关系。",
        "rainfall_station_map", ("rainfall", "boundary"),
    ),
    Visualization(
        "rainfall_station_weights", "rainfall", "站点空间权重",
        "子汇水区与雨站的 IDW 权重分配热力图。",
        "station_weights", ("rainfall", "partition"),
    ),
    # ------------------------------------------------------------ rainfall plot
    Visualization(
        "rainfall_plot_series", "rainfall_plot", "降雨绘图阶段成果",
        "降雨绘图阶段输出的多站逐时降雨过程，便于单独检查该阶段。",
        "rainfall_series", ("figure", "rainfall"),
    ),
    # ------------------------------------------------------------------- build
    Visualization(
        "build_model_structure", "build", "SWMM 结构",
        "区分紫色降雨来源点（ERA5代理点）、红色倒三角分区出口，并单列分区插值雨量计数量。",
        "model_structure", ("swmm_input",),
    ),
    Visualization(
        "build_area_impervious", "build", "面积与不透水率",
        "各子汇水区面积与不透水率分布。",
        "area_impervious", ("swmm_input",),
    ),
    Visualization(
        "build_infiltration", "build", "渗透参数",
        "各子汇水区渗透参数取值分布。",
        "infiltration", ("swmm_input",),
    ),
    # -------------------------------------------------------------------- swmm
    Visualization(
        "swmm_runoff_hydrograph", "swmm", "径流过程",
        "流域总径流与各子汇水区径流汇总过程线。",
        "swmm_runoff", ("swmm_result",),
    ),
    Visualization(
        "swmm_peak_runoff", "swmm", "峰值径流",
        "各子汇水区峰值径流对比。",
        "swmm_peak", ("swmm_result",),
    ),
    Visualization(
        "swmm_cumulative_runoff", "swmm", "累积径流",
        "累积径流随时间的增长。",
        "swmm_cumulative", ("swmm_result",),
    ),
    Visualization(
        "swmm_continuity", "swmm", "连续性误差",
        "径流与演算连续性误差仪表盘。",
        "swmm_continuity", ("swmm_result",),
    ),
    # ------------------------------------------------------------------ safety
    Visualization(
        "safety_minimum_factor", "safety", "最小安全系数",
        "事件期内每个像元的最小安全系数。",
        "stability_raster", ("stability",), {"metric": "minimum_safety_factor"},
    ),
    Visualization(
        "safety_max_depth", "safety", "最大水深",
        "事件期内每个像元的最大水深。",
        "stability_raster", ("stability",), {"metric": "maximum_flow_depth_m"},
    ),
    Visualization(
        "safety_max_velocity", "safety", "最大流速",
        "事件期内每个像元的最大流速。",
        "stability_raster", ("stability",), {"metric": "maximum_velocity_ms"},
    ),
    Visualization(
        "safety_max_pore_pressure", "safety", "最大孔压",
        "事件期内每个像元的最大孔隙水压力。",
        "stability_raster", ("stability",), {"metric": "maximum_pore_water_pressure_kpa"},
    ),
    Visualization(
        "safety_unstable_duration", "safety", "失稳持续时间",
        "各像元处于失稳状态的累计小时数。",
        "stability_raster", ("stability",), {"metric": "unstable_duration_h"},
    ),
    Visualization(
        "safety_time_series", "safety", "安全系数时序",
        "流域平均安全系数与失稳面积比例随时间的变化。",
        "safety_time_series", ("stability",),
    ),
    # ---------------------------------------------------------------- analysis
    Visualization(
        "analysis_sensitivity_ranking", "analysis", "参数敏感性排序",
        "单参数 OAT 敏感性排序。",
        "sensitivity_ranking", ("analysis",),
    ),
    Visualization(
        "analysis_sobol_contribution", "analysis", "参数贡献",
        "Sobol 方差分解得到的参数贡献。",
        "sobol_contribution", ("analysis",),
    ),
    Visualization(
        "analysis_representative_series", "analysis", "关键位置时序",
        "代表点的安全系数时间序列。",
        "representative_series", ("analysis",),
    ),
    Visualization(
        "analysis_stage_comparison", "analysis", "等级面积统计",
        "不同阶段的稳定性等级面积统计。",
        "stage_comparison", ("analysis",),
    ),
    # ---------------------------------------------------------------------- qa
    Visualization(
        "qa_dashboard", "qa", "QA 仪表盘",
        "各质量门槛的通过情况与关键指标。",
        "qa_dashboard", ("partition", "stability"),
    ),
    Visualization(
        "qa_spatial_coverage", "qa", "空间覆盖检查",
        "划分覆盖与单像元分区的诊断图。",
        "qa_coverage", ("partition",),
    ),
    # ------------------------------------------------------------------ export
    Visualization(
        "export_index", "export", "成果索引",
        "最终成果清单与各自的空间范围。",
        "export_index", ("export",),
    ),
)


def catalog() -> list[dict[str, Any]]:
    return [
        {
            "kind": item.kind,
            "stage_id": item.stage_id,
            "title": item.title,
            "description": item.description,
            "source_roles": list(item.source_roles),
            "options": item.options,
        }
        for item in CATALOG
    ]


def for_stage(stage_id: str | None = None) -> list[Visualization]:
    if stage_id is None:
        return list(CATALOG)
    return [item for item in CATALOG if item.stage_id == stage_id]


def by_kind(kind: str) -> Visualization:
    for item in CATALOG:
        if item.kind == kind:
            return item
    raise ValueError(f"未知可视化类型：{kind}")
