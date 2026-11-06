"""Build the evidence object, and say what it supports.

The evidence object is assembled only from what ``get_hydro_run_result`` and
its quality block returned. It is written to ``evidence.json`` beside the
session and is what every later sentence is checked against.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from .models import AgentSession
from .response_writer import sanitize
from .scientific_policy import (
    AssessmentStatus,
    audit_reply,
    decide_status,
    interpretations,
    limitations,
)


def build_assessment(session: AgentSession, result: dict[str, Any],
                     quality: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the assessment structure for a finished run."""
    status_block = result.get("status") or {}
    run_state = str(status_block.get("status") or session.hydro_run_status or "")
    safety = result.get("safety") or {}
    rainfall = result.get("rainfall") or {}
    swmm_model = result.get("swmm_model") or {}
    domain_policy = result.get("domain_policy") or {}
    quality = quality or result.get("quality") or {}
    qa_passed = quality.get("passed")
    if qa_passed is None:
        qa_passed = session.qa_passed

    decision = decide_status(qa_passed=qa_passed, run_state=run_state,
                             safety_available=bool(safety.get("available")))

    factors = {
        "rainfall": {
            "total_rain_mm": rainfall.get("total_rain_mm"),
            "peak_hourly_rain_mm": rainfall.get("peak_hourly_rain_mm"),
            "station_count": rainfall.get("station_count"),
            "active_rows": rainfall.get("active_rows"),
            "source": "ERA5 virtual stations",
        },
        "safety_factor": {
            "minimum": safety.get("minimum_safety_factor"),
            "mean": safety.get("mean_safety_factor"),
            "p95": safety.get("p95_safety_factor"),
            "maximum": safety.get("maximum_safety_factor"),
            "units": "dimensionless",
        },
        "instability_extent_duration": {
            "max_unstable_fraction": safety.get("max_unstable_fraction"),
            "max_unstable_duration_h": safety.get("max_unstable_duration_h"),
            "units": {"max_unstable_fraction": "fraction of extent",
                      "max_unstable_duration_h": "hour"},
        },
        "hydraulic_pore_pressure": {
            "max_flow_depth_m": safety.get("max_flow_depth_m"),
            "max_velocity_ms": safety.get("max_velocity_ms"),
            "max_pore_water_pressure_kpa": safety.get("max_pore_water_pressure_kpa"),
        },
    }
    if safety.get('spatial_distribution'):
        factors['spatial_distribution'] = safety['spatial_distribution']

    warnings = list(quality.get("warnings") or [])
    if decision.status is not AssessmentStatus.VALID_PRELIMINARY:
        warnings.extend(decision.reasons)

    statements = (interpretations(safety, rainfall, factors["instability_extent_duration"])
                  if decision.status is not AssessmentStatus.UNAVAILABLE else [])

    assessment = {
        "domain_policy": domain_policy,
        "assessment_status": decision.status.value,
        "qa_passed": qa_passed,
        "hydro_run_id": session.hydro_run_id,
        "event": _event_block(rainfall, result),
        "model_context": {
            "parameter_preset": safety.get("parameter_preset") or session.parameter_preset,
            "mechanics_model": safety.get("mechanics_model"),
            "swmm_dynamic_hydrograph_used": safety.get("swmm_dynamic_hydrograph_used"),
            "subcatchment_count": swmm_model.get("subcatchment_count"),
            "outfall_count": swmm_model.get("outfall_count"),
            "routing_network_present": swmm_model.get("routing_network_present"),
            # The plan payload does not carry the mode; the session recorded the
            # one the agent asked for when the plan was generated.
            "run_mode": session.run_mode or (session.last_plan or {}).get("mode"),
            'simulation_window': swmm_model.get('simulation_window') or {},
        },
        "factors": factors,
        "interpretations": statements,
        "warnings": warnings,
        # Not an invention of the agent's: the hydro service's own recorded
        # limitations are carried through unchanged.
        "limitations": list(dict.fromkeys([*(domain_policy.get("boundary_assumptions") or []), *limitations(preset=safety.get("parameter_preset") or session.parameter_preset,
                                   rainfall_source=_rainfall_source(session),
                                   extra=list(result.get("limitations") or []))])),
        "evidence_sources": _sources(result),
        "quality_gates": (quality.get("gates") if isinstance(quality, dict) else None) or [],
        # The compact factors are convenient for summaries, but are not the
        # full set of facts returned by the current run (area, overlays,
        # errors, task version, etc.). Preserve those facts for read-only QA.
        "result_details": deepcopy(result),
        "derived_evidence": _report_conversions(result),
    }
    return assessment


def _report_conversions(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Exact reporting conversions with their source paths, never new physics.

    Ratios are defined only for explicit count/area numerator-denominator
    fields. In particular an instantaneous fraction is not turned into an
    event-wide footprint, and overlapping areas are not silently unioned.
    """
    derived: list[dict[str, Any]] = []

    def number(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value) if math.isfinite(value) else None

    def walk(node, path="", depth=0):
        if depth > 12 or not isinstance(node, dict):
            return
        for key, value in node.items():
            source = f"{path}.{key}".strip(".")
            if isinstance(value, dict):
                walk(value, source, depth + 1)
            elif str(key).endswith("_km2") and (area := number(value)) is not None and area >= 0:
                converted = area * 1_000_000
                if math.isfinite(converted):
                    derived.append({"name": source[:-4] + "_m2", "value": converted,
                                    "unit": "m2", "operation": "km2_to_m2", "sources": [source]})
        for numerator, denominator, name in (
            ("intersected_polygon_count", "total_polygon_count", "polygon_count_fraction"),
            ("overlap_landslide_area_km2", "total_landslide_area_km2", "landslide_area_fraction"),
        ):
            a, b = number(node.get(numerator)), number(node.get(denominator))
            if a is not None and b is not None and b > 0 and 0 <= a <= b:
                derived.append({"name": f"{path}.{name}".strip("."), "value": a / b,
                                "unit": "fraction", "operation": "divide",
                                "sources": [f"{path}.{numerator}".strip("."),
                                            f"{path}.{denominator}".strip(".")]})

    walk(result)
    return derived


def describe_result_summary(assessment: dict[str, Any]) -> str:
    """A short usable result; detailed provenance stays in the saved assessment."""
    if assessment.get('assessment_status') != AssessmentStatus.VALID_PRELIMINARY.value:
        return describe_assessment(assessment)
    factors = assessment.get('factors') or {}
    distribution = factors.get('spatial_distribution') or {}
    minimum = (factors.get('safety_factor') or {}).get('minimum')
    instant = (factors.get('instability_extent_duration') or {}).get('max_unstable_fraction')
    parts = ['计算已完成，质量检查通过。']
    if minimum is not None:
        parts.append(f'最小安全系数为 {minimum:.4f}。')
    if distribution.get('available'):
        area = distribution.get('below_one_area_km2')
        denominator = distribution.get('calculation_area_km2')
        if area is not None and denominator is not None:
            parts.append(f'事件内曾出现FS<1的面积为 {area:.4f} km²；'
                         f'面积分母为计算域有效像元面积 {denominator:.4f} km²。')
        parts.append(f'事件最小Fs栅格中，Fs<1区域占 {distribution["below_one_fraction"]:.2%}；'
                     f'其余区域占 {distribution["at_or_above_one_fraction"]:.2%}。')
        region = distribution.get('largest_below_one_sector')
        if region:
            parts.append(f'按方位统计，{region}的Fs<1面积最多。')
    elif instant is not None:
        parts.append(f'同一时刻最大失稳比例为 {instant:.2%}。')
    parts.append('这是本次模拟的初步评价。')
    return ''.join(parts)


def _rainfall_source(session: AgentSession) -> str | None:
    request = session.rainfall_request or {}
    return request.get("mode")


def _event_block(rainfall: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "rainfall_total_mm": rainfall.get("total_rain_mm"),
        "rainfall_peak_mm_h": rainfall.get("peak_hourly_rain_mm"),
        "timestamps": rainfall.get("active_rows"),
        "station_count": rainfall.get("station_count"),
        "artifacts": result.get("artifact_count"),
    }


def _sources(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Where each factor came from, so a reader can trace it back."""
    return [
        {"factor": "rainfall", "field": "rainfall.total_rain_mm",
         "source": "get_hydro_run_result"},
        {"factor": "safety_factor", "field": "safety.minimum_safety_factor",
         "source": "get_hydro_run_result"},
        {"factor": "instability_extent_duration",
         "field": "safety.max_unstable_fraction", "source": "get_hydro_run_result"},
        {"factor": "hydraulic_pore_pressure", "field": "safety.max_flow_depth_m",
         "source": "get_hydro_run_result"},
        {"factor": "model_context", "field": "swmm_model",
         "source": "get_hydro_run_result"},
    ]


# --------------------------------------------------------------------- wording


def describe_assessment(assessment: dict[str, Any]) -> str:
    """Render the assessment, honouring how far it may be taken."""
    status = assessment.get("assessment_status")
    qa_passed = assessment.get("qa_passed")
    facts = assessment.get("factors") or {}
    safety = facts.get("safety_factor") or {}
    extent = facts.get("instability_extent_duration") or {}
    hydraulic = facts.get("hydraulic_pore_pressure") or {}

    if status == AssessmentStatus.UNAVAILABLE.value:
        lines = ["这次运行没有可用于评价的结果。"]
        lines += [f"- {item}" for item in (assessment.get("warnings") or [])[:4]]
        if qa_passed is None:
            lines.append("可以先回复「状态」确认任务是否仍在执行；失败的任务可重新生成计划后运行。")
        return sanitize("\n".join(lines))

    if status == AssessmentStatus.DIAGNOSTIC_ONLY.value:
        lines = ["计算已完成，但质量门槛未通过：下面只能作为诊断信息，不构成稳定性结论。"]
        failed = assessment.get("quality_gates") or []
        failing = [item.get("name") for item in failed
                   if isinstance(item, dict) and not item.get("passed")]
        if failing:
            lines.append("未通过的门槛：" + "、".join(str(item) for item in failing))
        for item in (assessment.get("warnings") or [])[:4]:
            lines.append(f"- {item}")
        for statement in assessment.get("interpretations") or []:
            lines.append(f"可参考的数值：{statement['text']}")
        lines.append("请检查数据与参数后重新运行，不要据此判断坡体是否稳定。")
        return sanitize("\n".join(lines))

    lines = ["运行通过全部质量门槛，以下是初步评价。"]
    for statement in assessment.get("interpretations") or []:
        lines.append(f"- {statement['text']}")
    hydraulic_line = []
    if hydraulic.get("max_flow_depth_m") is not None:
        hydraulic_line.append(f"最大水深 {hydraulic['max_flow_depth_m']:.4f} m")
    if hydraulic.get("max_velocity_ms") is not None:
        hydraulic_line.append(f"最大流速 {hydraulic['max_velocity_ms']:.4f} m/s")
    if hydraulic.get("max_pore_water_pressure_kpa") is not None:
        hydraulic_line.append(f"最大孔隙水压力 {hydraulic['max_pore_water_pressure_kpa']:.2f} kPa")
    if hydraulic_line:
        lines.append("水力与孔压：" + "；".join(hydraulic_line))
    model = assessment.get("model_context") or {}
    lines.append(
        f"模型上下文：参数预设 {model.get('parameter_preset')}；"
        f"力学模型 {model.get('mechanics_model')}；"
        f"SWMM 动态过程 {'已使用' if model.get('swmm_dynamic_hydrograph_used') else '未使用'}；"
        f"子汇水区 {model.get('subcatchment_count')} 个。"
    )
    for item in (assessment.get("limitations") or [])[:5]:
        lines.append(f"限制：{item}")
    lines.append("可以继续追问某个数值的来源，或回复「看最小安全系数的图」查看图件。")
    return sanitize("\n".join(lines))


def describe_diagnostics(assessment: dict[str, Any]) -> str:
    """The shorter, blunter form used when a gate failed."""
    return describe_assessment(assessment)


# ------------------------------------------------------------------- questions


TOPICS = {
    "safety_factor": (
        "安全系数由力学模型逐像元计算：抗滑力与下滑力之比。"
        "本次结果是每个像元在事件期内的最小值，再对全域取的最小、均值、P95 与最大值。"
        "最小值只说明存在局部像元进入 FS<1，不等于整个流域失稳；均值用于看整体水平。"
    ),
    "pore_pressure": (
        "孔隙水压力由降雨入渗与坡面径流的水深换算，反映水压对有效应力的削弱。"
        "本模型用简化力学模型计算，未做场地勘察标定，因此孔压绝对值只适用于相对比较。"
    ),
    "unstable": (
        "失稳比例是某一时刻模型判定 FS<1 的像元占计算域的比例，是瞬时值，不是全时段累计。"
        "失稳持续时间是单个像元连续处于失稳状态的最长小时数。"
    ),
    "qa": (
        "质量门槛检查空间划分、降雨、SWMM 连续性、稳定性结果和成果导出五方面。"
        "只有全部门槛通过，结果才会被当作初步评价；未通过时只能给出诊断信息。"
    ),
    "rainfall": (
        "降雨来自 ERA5 再分析网格，按虚拟雨站插值到流域。"
        "这不是地面实测雨量；若流域小于一个网格，站点代表性会明显下降。"
    ),
    "model": (
        "模型为独立子汇水区出水口配置，没有管网汇流："
        "各子汇水区径流之和不是选定流域出口的路由流量过程。"
    ),
    "default": (
        "我可以解释本次结果的构成：安全系数的含义、失稳比例与持续时间的区别、"
        "降雨来源与虚拟雨站限制、质量门槛的作用，以及模型的适用范围。"
    ),
}

TOPIC_KEYWORDS = (
    ("safety_factor", ("安全系数", "fs", "系数")),
    ("pore_pressure", ("孔压", "孔隙水压力", "水压")),
    ("unstable", ("失稳", "不稳定", "持续时间", "比例")),
    ("qa", ("质量门槛", "qa", "门槛", "通过")),
    ("rainfall", ("降雨", "雨量", "era5", "虚拟站", "雨站")),
    ("model", ("模型", "汇流", "子汇水区", "出口")),
)


def explain(session: AgentSession, parsed, text: str,
            assessment: dict[str, Any] | None = None) -> str:
    """Answer a term or "why" question from the current session's evidence.

    Every optional field is checked on its own: a summary that carries a minimum
    but no mean is normal, and must not break the answer. When the session has no
    result yet, only the term is explained — no value is invented for a basin
    that has not been computed.
    """
    lowered = str(text).lower()
    topic = "default"
    for name, keywords in TOPIC_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            topic = name
            break

    # The user's own words are not echoed back into the reply. Doing so put
    # whatever phrasing they used — including the very terms the scientific
    # audit exists to keep out of replies — inside a text that is then audited,
    # so asking *about* a forbidden conclusion produced an internal-consistency
    # failure instead of an answer. It also read as a machine quoting the input
    # rather than as a reply.
    lines = [TOPICS[topic]]

    status = (assessment or {}).get("assessment_status")
    if not assessment:
        lines.append("当前会话还没有可引用的计算结果；先完成一次运行，我才能结合本次结果说明。")
        return sanitize("\n".join(lines))

    if status == AssessmentStatus.DIAGNOSTIC_ONLY.value:
        lines.append("注意：本次运行未通过质量门槛，下面的数值只能用于诊断，"
                     "不能作为坡体稳定性的结论。")
    elif status == AssessmentStatus.UNAVAILABLE.value:
        lines.append("本次运行没有产生可用于评价的结果，因此不引用任何数值。")
        return sanitize("\n".join(lines))

    factors = assessment.get("factors") or {}
    quoted = _quote_current_values(factors)
    if quoted:
        lines.append("本次结果：" + "；".join(quoted) + "。")

    # "Why is it low?" may only point at inputs and companion signals, never at
    # a cause: this release has no attribution analysis behind it.
    if topic == "safety_factor" and any(word in str(text) for word in ("为什么", "为何", "怎么会")):
        companions = _companion_signals(factors)
        if companions:
            lines.append("与低安全系数同时出现、可以一并核对的有：" + "；".join(companions) + "。")
        lines.append("本次结果只说明模型在这些输入下判定的像元状态，"
                     "不能据此把其中某个因素认定为失稳的原因。")

    lines.append(_scope_line(assessment))
    return sanitize("\n".join(lines))


def _quote_current_values(factors: dict[str, Any]) -> list[str]:
    """Each available metric, formatted on its own; missing ones are skipped."""
    quoted: list[str] = []
    safety = factors.get("safety_factor") or {}
    if safety.get("minimum") is not None:
        quoted.append(f"最小安全系数 {float(safety['minimum']):.3f}")
    if safety.get("mean") is not None:
        quoted.append(f"均值 {float(safety['mean']):.3f}")
    if safety.get("p95") is not None:
        quoted.append(f"P95 {float(safety['p95']):.3f}")
    extent = factors.get("instability_extent_duration") or {}
    if extent.get("max_unstable_fraction") is not None:
        quoted.append(f"最大失稳面积比例 {float(extent['max_unstable_fraction']) * 100:.2f}%")
    if extent.get("max_unstable_duration_h") is not None:
        quoted.append(f"最长失稳持续时间 {float(extent['max_unstable_duration_h']):.2f} 小时")
    hydraulic = factors.get("hydraulic_pore_pressure") or {}
    if hydraulic.get("max_flow_depth_m") is not None:
        quoted.append(f"最大水深 {float(hydraulic['max_flow_depth_m']):.4f} m")
    if hydraulic.get("max_velocity_ms") is not None:
        quoted.append(f"最大流速 {float(hydraulic['max_velocity_ms']):.4f} m/s")
    if hydraulic.get("max_pore_water_pressure_kpa") is not None:
        quoted.append(f"最大孔隙水压力 {float(hydraulic['max_pore_water_pressure_kpa']):.2f} kPa")
    return quoted


def _companion_signals(factors: dict[str, Any]) -> list[str]:
    """Inputs and companion quantities a reader can check alongside the result."""
    signals: list[str] = []
    rainfall = factors.get("rainfall") or {}
    if rainfall.get("total_rain_mm") is not None:
        signals.append(f"累计降雨 {float(rainfall['total_rain_mm']):.2f} mm")
    if rainfall.get("peak_hourly_rain_mm") is not None:
        signals.append(f"峰值小时雨量 {float(rainfall['peak_hourly_rain_mm']):.2f} mm/h")
    hydraulic = factors.get("hydraulic_pore_pressure") or {}
    if hydraulic.get("max_pore_water_pressure_kpa") is not None:
        signals.append(f"孔隙水压力峰值 {float(hydraulic['max_pore_water_pressure_kpa']):.2f} kPa")
    extent = factors.get("instability_extent_duration") or {}
    if extent.get("max_unstable_fraction") is not None:
        signals.append(f"失稳面积比例峰值 {float(extent['max_unstable_fraction']) * 100:.2f}%")
    return signals


def _scope_line(assessment: dict[str, Any]) -> str:
    """What the quoted values do and do not support."""
    model = assessment.get("model_context") or {}
    preset = model.get("parameter_preset") or "未记录"
    return (f"适用范围：区域尺度、降雨事件驱动的初步筛查与诊断；"
            f"本次参数预设为 {preset}，最小安全系数只说明存在局部像元进入该状态，"
            f"不等于整个流域的稳定或失稳。")


def audit(assessment: dict[str, Any], text: str) -> list[str]:
    """Check a finished reply against the evidence it came from."""
    return audit_reply(text, assessment)
