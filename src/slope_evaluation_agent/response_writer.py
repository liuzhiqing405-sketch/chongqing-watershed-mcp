"""Turn evidence into language.

Two layers, as the plan requires: this module never invents a number. Every
figure it prints is taken from the evidence object, which is itself built from
MCP responses. The deterministic templates here are always available; the
optional rewriting pass in phase 8 may only rephrase, never recompute.

Wording rules that matter scientifically:

* a candidate outlet is called a candidate, never an outlet;
* the historical maximum day is a sample maximum over the data period, never a
  design storm;
* ERA5 grid points are virtual stations, never measured gauges;
* a local minimum safety factor says nothing about the whole watershed;
* a run that failed the quality gates produces no stability verdict at all;
* subcatchment runoff is never described as the outlet hydrograph.
"""

from __future__ import annotations

import re
from typing import Any

from . import DISCLAIMER_ZH, SCOPE_STATEMENT
from .intent import Intent
from .models import AgentSession
from .state_machine import WorkflowGateError, pending_action


#: Phrases a reply must never contain. Checked by the phase-8 audit and used
#: here to keep the templates honest.
FORBIDDEN_PHRASES = (
    "绝对安全", "保证不会失稳", "不会发生滑坡", "实测雨量", "实测降雨",
    "出口流量过程", "工程安全", "设计暴雨", "预报", "泥石流易发性",
)

#: Replacements applied when a phrase would otherwise slip through.
SOFTENINGS = {
    "绝对安全": "在该次模拟中未出现失稳像元",
    "保证不会失稳": "该次模拟未判定失稳",
    "实测雨量": "虚拟雨站雨量",
    "实测降雨": "虚拟雨站降雨",
    "设计暴雨": "资料期内样本最大日",
}

#: Words that make a phrase a *correct* statement rather than a forbidden one.
#: "不是地面实测雨量站" must survive; rewriting it would produce nonsense and
#: would also destroy the very caveat the wording rules exist to protect.
NEGATIONS = ("不是", "并非", "非", "未", "无", "没有", "不属于", "不代表", "不同于",
             "不能作", "不能用于", "不可作", "不可用于", "不作")

#: How far back to look for a negation before a phrase.
_NEGATION_WINDOW = 6


def _is_negated(text: str, start: int) -> bool:
    window = text[max(0, start - _NEGATION_WINDOW):start]
    return any(negation in window for negation in NEGATIONS)


def _fmt(value: Any, unit: str = "", *, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}{unit}"
    return f"{value}{unit}"


def _text(value: Any, placeholder: str = "—") -> str:
    """Never show a user the word ``None``."""
    return placeholder if value is None or value == "" else str(value)


def _size(value: Any) -> str:
    """A size in megabytes, written so the number audit can read it.

    "MB" is not one of the units the audit knows, and its unit table matches a
    leading "m" — so "3.0 MB" would be read as three metres and flagged as a
    unit mismatch. The Chinese form carries the same meaning and no false unit.
    """
    return _fmt(value, " 兆字节", digits=1)


def _pct(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def sanitize(text: str) -> str:
    """Replace a forbidden phrase with an accurate one, unless it is negated.

    "不是地面实测雨量站" is a correct and required caveat, so a phrase inside a
    negation is left alone. Used by every deterministic template and by the
    phase-8 audit, so a phrase cannot reach a user through either path.
    """
    result = str(text)
    for phrase, replacement in SOFTENINGS.items():
        if phrase not in result:
            continue
        pieces: list[str] = []
        cursor = 0
        while True:
            found = result.find(phrase, cursor)
            if found == -1:
                pieces.append(result[cursor:])
                break
            pieces.append(result[cursor:found])
            pieces.append(phrase if _is_negated(result, found) else replacement)
            cursor = found + len(phrase)
        result = "".join(pieces)
    return result


def find_forbidden(text: str) -> list[str]:
    """Forbidden phrases still present after sanitizing, for the audit.

    A phrase inside a negation is a correct statement, not a violation.
    """
    violations: list[str] = []
    for phrase in FORBIDDEN_PHRASES:
        cursor = 0
        while True:
            found = text.find(phrase, cursor)
            if found == -1:
                break
            if not _is_negated(text, found):
                violations.append(phrase)
                break
            cursor = found + len(phrase)
    return violations


# --------------------------------------------------------------------- errors


def describe_gate_error(error: WorkflowGateError, session: AgentSession) -> str:
    return sanitize(f"{error}\n\n当前可以做的事：{pending_action(session.state)}")


def describe_mcp_error(error: Exception) -> str:
    """A tool failure, as a user should see it."""
    message = getattr(error, "user_message", None)
    text = message() if callable(message) else str(error)
    if "Traceback" in text or 'File "' in text:
        text = "调用 MCP 服务时发生错误。"
    return sanitize(text)


# ------------------------------------------------------------------- spatial


def describe_data_check(payload: dict[str, Any]) -> str:
    data = payload.get("data") or {}
    errors = list(data.get("errors") or [])
    warnings = list(data.get("warnings") or [])
    years = data.get("rainfall_years") or []
    if payload.get("ok"):
        lines = ["基础数据检查通过。"]
        if years:
            lines.append(f"逐时降雨资料覆盖 {years[0]}–{years[-1]} 年，共 {len(years)} 年。")
        for warning in warnings:
            lines.append(f"注意：{warning}")
        lines.append("下一步可以给出河流名称，或直接提供出水口经纬度。")
        return sanitize("\n".join(lines))
    lines = ["基础数据检查未通过："]
    lines += [f"- {item}" for item in errors[:6]]
    if len(errors) > 6:
        lines.append(f"- 另有 {len(errors) - 6} 项")
    lines.append("请先补齐缺失或被占用的数据，再继续。")
    return sanitize("\n".join(lines))


def describe_outlet_candidates(payload: dict[str, Any], river_name: str) -> str:
    data = payload.get("data") or {}
    candidates = list(data.get("candidates") or [])
    matched = data.get("matched_names") or []
    if not candidates:
        note = data.get("message") or "没有找到可用候选点。"
        return sanitize(
            f"未能在水系中找到「{river_name}」的可用候选出口。{note}\n"
            "可以换一个河流名称，或直接提供出水口经纬度。"
        )
    lines = [f"为「{river_name}」找到 {len(candidates)} 个候选出口。"
             "这些只是候选点，不是唯一的水文出口。"]
    for index, item in enumerate(candidates, start=1):
        lines.append(
            f"{index}. 经纬度 {item.get('longitude')}, {item.get('latitude')}；"
            f"高程 {_fmt(item.get('elevation_m'), ' m', digits=1)}；"
            f"距 DEM 边缘 {_fmt(item.get('dem_edge_distance_km'), ' km', digits=1)}；"
            f"所在河流 {item.get('matched_river')}"
        )
    if matched:
        lines.append(f"匹配到的名称：{'、'.join(matched)}")
    lines.append("请回复「第 N 个」选择一个候选，或直接给出经纬度。")
    return sanitize("\n".join(lines))


def describe_review(review: dict[str, Any], *, brief: bool = False) -> str:
    qa = review.get("qa") or {}
    if brief:
        from watershed_exchange.domain_policy import drawn_modeling_allowed
        allowed = (drawn_modeling_allowed(qa, review.get('assessment_mode'))
                   if review.get('selection_mode') == 'drawn_study_area' else qa.get('passed'))
        if allowed:
            return f'评价范围已保存，面积 {_fmt(qa.get("area_km2"), " km²", digits=3)}。请核对地图后回复「确认」。'
        errors = qa.get('errors') or (qa.get('modeling_check') or {}).get('errors') or ['请调整范围后重新检查。']
        return sanitize('边界检查未通过，不能确认或计算：' + '；'.join(str(e) for e in errors[:3]))
    if review.get("selection_mode")=="drawn_study_area":
        from watershed_exchange.domain_policy import drawn_modeling_allowed
        policy=qa.get("modeling_check") or {}
        lines=[f"已保存用户绘制的评价范围，审核编号 {review.get('review_id')}。",
               f"面积 {_fmt(qa.get('area_km2'),' km²',digits=3)}，像元数 {_fmt(qa.get('selected_cells'),digits=0)}。",
               "该范围由你绘制，区域内部将在建模阶段按地形划分子汇水区；它尚未被验证为完整自然流域。"]
        hydro=qa.get("hydrologic_check") or {}
        if policy:
            lines.append("评价方式：" + ("区域评价（按框内降雨建模）。" if policy.get("assessment_mode")=="regional_screening" else "上游来水严格检查。"))
            lines.extend(policy.get("boundary_assumptions") or [])
        if hydro.get("checked"):
            lines.append(f"水文合理性检查{'通过' if hydro.get('passed') else '未通过'}："
                         f"识别到 {_fmt(hydro.get('major_outlet_count'),digits=0)} 处主要出流位置，"
                         f"框外上游贡献面积至少 {_fmt(hydro.get('external_upstream_area_km2'),' km²',digits=3)}。")
            lines.append("图中黑星为 DEM 推测的主要出流位置，红三角为明显的框外入流位置；这些点用于核对，不会自动改变你的边界或选定出水口。")
            if not policy:
                lines.extend(f"提示：{warning}" for warning in hydro.get("warnings") or [])
        elif not hydro:
            lines.append("该审核记录缺少水文诊断，可能由旧空间服务生成；请更新服务后重新提交已保存的边界。")
        lines.extend(f"提示：{warning}" for warning in policy.get("warnings") or [])
        if drawn_modeling_allowed(qa,review.get("assessment_mode")):
            lines.append("DEM 和土地利用覆盖与建模条件检查通过。请核对边界、诊断提示和上述边界假设后回复「确认」或「拒绝」；确认后再选择降雨并建模。")
        else:
            lines.extend(f"- {error}" for error in qa.get("errors") or [])
            lines.append("当前范围不能确认或计算，请调整绘制边界或补齐数据。")
        return sanitize("\n".join(lines))
    outlet = review.get("outlet") or {}
    processing = review.get("processing") or {}
    lines = [
        f"{'流域已划分' if qa.get('passed') else '本次划分未通过质量检查，仅供排查'}，审核编号 {review.get('review_id')}。",
        f"面积 {_fmt(qa.get('area_km2'), ' km²', digits=3)}，"
        f"像元数 {_fmt(qa.get('selected_cells'), digits=0)}。",
        f"吸附后的出水口：{outlet.get('snapped_longitude')}, {outlet.get('snapped_latitude')}；"
        f"与请求点偏移 {_fmt(outlet.get('shift_m'), ' m', digits=1)}。",
    ]
    window_km = processing.get("window_km")
    resolution = processing.get("resolution_m")
    attempts = processing.get("window_attempts") or 0
    if window_km is not None:
        line = f"处理窗口 {_fmt(window_km, ' km（半宽）', digits=1)}，栅格 {_fmt(resolution, ' m', digits=0)}"
        if attempts and attempts > 1:
            line += f"，自适应放大 {attempts} 次"
        lines.append(line + "。")
        # A coarser grid is not a detail: the boundary and the slopes that feed
        # the stability model were traced on it, and saying so is the difference
        # between a limitation and a surprise.
        if processing.get("grid_coarsened"):
            native = processing.get("native_cell_size_m")
            lines.append(
                f"注意：本次为容纳完整流域把栅格降到 {_fmt(resolution, ' m', digits=0)}"
                f"（原生 {_fmt(native, ' m', digits=0)}），边界与坡度在该分辨率下计算。")
    errors = list(qa.get("errors") or [])
    if qa.get("passed"):
        lines.append("空间质量检查通过。")
        lines.append("该边界表示所选出水口以上的汇水区；仍需核对是否覆盖你要分析的河段及上游范围。")
    else:
        lines.append("空间质量检查未通过：")
        lines += [f"- {item}" for item in errors]
        if (qa.get("coverage_check") or {}).get("passed") is False:
            lines.append("当前数据不足以确认上游完整范围。需补齐上游地形数据，不能靠换点或扩大现有窗口消除数据缺口。")
        elif (qa.get("channel_check") or {}).get("passed") is False:
            lines.append("当前水系图层与 DEM 汇流河道无法对应。需核对图层配准、地形数据及水库或平坦水面的流向处理，再重新划分；不能直接吸附到附近最大的河道。")
        elif qa.get("touched_sides") and set(qa["touched_sides"]) <= set(
                processing.get("window_clamped_sides") or []):
            lines.append("该边界的缺口来自 DEM 数据范围本身，改出水口或放大窗口都无法补全。")
        elif (qa.get("river_check") or {}).get("passed") is False:
            lines.append("当前出水口与指定河流不一致，面积超过下限也不能把其他支流当作目标流域。请在高亮目标河段上重新选点。")
        elif qa.get("area_km2", float("inf")) < qa.get("minimum_area_km2", 0):
            lines.append("当前点只汇集了很小的局部坡面，未形成有效的目标流域。请在地图河道上重新选出水口；单纯放大处理窗口不能证明选点正确。")
        else:
            lines.append("该边界不能确认；可以换一个出水口位置，或（若数据允许）用更大的处理窗口重新划分。")
    # Notes are observations that do not block confirmation — a boundary graze
    # is worth knowing about, but it must not read like the reason the boundary
    # was refused.
    for note in qa.get("notes") or []:
        lines.append(f"提示：{note}")
    if qa.get("passed"):
        lines.append("请查看流域图后，明确回复「确认」或「拒绝」。")
    elif ((qa.get("coverage_check") or {}).get("passed") is False
          or (qa.get("channel_check") or {}).get("passed") is False):
        lines.append("该结果不能确认，也不会进入降雨与稳定性计算。请先核对或补齐空间数据，再重新划分。")
    else:
        lines.append("该结果不能确认，也不会进入降雨与稳定性计算。请重新打开选址地图调整位置。")
    return sanitize("\n".join(lines))


def describe_confirmation(review: dict[str, Any], *, confirmed: bool, brief: bool = False) -> str:
    if brief:
        return ('边界已确认。请指定降雨日期、时段，或使用历史样本最大日。' if confirmed
                else '已拒绝当前边界，请在地图上修改评价范围。')
    if confirmed:
        return sanitize(
            f"已记录确认：{review.get('review_id')}。\n"
            "接下来请说明降雨事件：指定日期或时段，或者使用资料期内的样本最大日。"
        )
    return sanitize(
        f"已记录拒绝：{review.get('review_id')}。审核记录保留。\n"
        + ("请按区县、河名或经纬度重新定位地图，并绘制评价边界。"
           if review.get("selection_mode")=="drawn_study_area" else "请重新打开地图选择出水口，或给出经纬度。")
    )


def describe_rainfall_package(package: dict[str, Any], *, brief: bool = False) -> str:
    if brief:
        start, end = package.get('event_start'), package.get('event_end')
        period = f'：{start} 至 {end}' if start and end else ''
        total = package.get('selected_day_basin_station_mean_total_mm')
        extra = f'，所选雨量源日均值 {_fmt(total, " mm")}' if total is not None else ''
        label = '历史样本最大日降雨' if package.get('selection_mode') == 'historical_max' else '降雨'
        return sanitize(f'ERA5 {label}已准备{period}{extra}。')
    mode = package.get("selection_mode")
    stations = package.get("stations") or []
    lines = [f"降雨包已生成：{_text(package.get('package_id'))}。",
             f"事件窗口 {_text(package.get('event_start'))} 至 "
             f"{_text(package.get('event_end'))}，"
             f"共 {_text(package.get('timestamp_count'))} 个时次。",
             f"降雨输入共 {len(stations)} 个雨量源。"]
    if package.get('source_selection_policy'):
        lines.append(f"优先使用现有 ERA5 格点 {package.get('existing_grid_count',0)} 个，"
                     f"不足部分补充虚拟雨站 {package.get('interpolated_station_count',0)} 个。")
        lines.append('优先雨量源使用对应ERA5源列；若真实格点中心在区外，模型代理点设在格网与选区交集内部，保留真实来源坐标。所有模型雨量点均在区内，分区雨量沿用原来的IDW计算。')
    selection = package.get("station_selection") or {}
    if selection.get("mode") == "auto":
        lines.append(f"站数按选区面积自动确定：面积 {_fmt(selection.get('area_km2'), ' km²')}。"
                     f"每 {_text(selection.get('area_per_station_km2'))} km² 配置1个，向上取整，"
                     f"自动数量范围为 {_text(selection.get('minimum'))}～{_text(selection.get('maximum'))} 个。")
        if selection.get("capped"):
            lines.append("面积公式所得站数超过自动上限，本次采用上限；如需增加可明确指定站数。")
    elif selection.get("mode") == "manual":
        lines.append("站数采用用户明确指定的数量，覆盖按面积自动选择。")
    if mode == "historical_max":
        years = package.get("statistics_years") or []
        span = f"{years[0]}–{years[-1]}" if years else "指定年限"
        lines.append(
            f"选取规则：{span} 内所选雨量源平均日累计量最大的完整日，"
            f"该日均值为 {_fmt(package.get('selected_day_basin_station_mean_total_mm'), ' mm')}。"
        )
        lines.append("这是资料期内样本最大值，不是重现期设计暴雨。")
    else:
        lines.append(f"选取规则：{package.get('selection_rule') or '用户指定的时段'}。")
    lines.append("这些站点是 ERA5 网格代理点，不是地面实测雨量站。")
    if len(stations) == 1:
        lines.append("单雨站输入在区域内使用相同降雨，不能表达降雨空间差异。")
    warning = package.get("warning_station_distance") or package.get("warning")
    if warning:
        lines.append(f"注意：{warning}")
    lines.append("下一步将裁切 DEM、土地利用与水系，并导出模型输入包。")
    return sanitize("\n".join(lines))


def describe_model_inputs(package: dict[str, Any], *, brief: bool = False) -> str:
    if brief:
        return '地形、土地利用、水系与降雨输入已准备。'
    exchange = package.get("exchange") or {}
    files = exchange.get("files") or []
    lines = [
        f"模型输入包已生成：{package.get('package_id')}。",
        f"包含 {len(files)} 个文件：{'、'.join(files[:6])}"
        + ("…" if len(files) > 6 else ""),
    ]
    clipping = package.get("clipping") or {}
    landuse = clipping.get("landuse") or {}
    river = clipping.get("river") or {}
    if landuse or river:
        lines.append(
            f"裁切结果：土地利用 {landuse.get('kept', '—')}/{landuse.get('read', '—')} 个图斑，"
            f"水系 {river.get('kept', '—')}/{river.get('read', '—')} 个要素。"
        )
    for item in (package.get("limitations") or [])[:2]:
        lines.append(f"限制：{item}")
    lines.append("下一步把该输入包导入水文稳定性服务。")
    return sanitize("\n".join(lines))


# --------------------------------------------------------------- hydro stages


def describe_import(record: dict[str, Any], *, brief: bool = False) -> str:
    if brief:
        return '计算输入已导入，接着校验。'
    event = record.get("event") or {}
    area = record.get("study_area") or {}
    lines = [f"输入包已导入并复制到受管目录：{_text(record.get('input_id'))}。"]
    if event:
        lines.append(f"事件窗口：{_text(event.get('start_datetime'))} 至 "
                     f"{_text(event.get('end_datetime'))}；"
                     f"降雨来源：{_text(event.get('rainfall_source'))}。")
    if area:
        lines.append(f"研究区面积 {_fmt(area.get('area_km2'), ' km²', digits=3)}。")
    lines.append("下一步先验证输入，再生成运行计划。")
    return sanitize("\n".join(lines))


def describe_input_validation(report: dict[str, Any], *, brief: bool = False) -> str:
    if brief and report.get('ok'):
        return '输入校验通过，可以生成计算方案。'
    if report.get("ok"):
        lines = [f"输入验证通过：{_text(report.get('input_id'))}。"]
        for warning in (report.get("warnings") or [])[:3]:
            lines.append(f"注意：{warning}")
        lines.append("下一步生成运行计划。")
        return sanitize("\n".join(lines))
    lines = [f"输入验证未通过：{_text(report.get('input_id'))}。"]
    lines += [f"- {item}" for item in (report.get("errors") or [])[:6]]
    lines.append("请修正后重新导入，或检查数据来源。")
    return sanitize("\n".join(lines))


def describe_plan(plan: dict[str, Any], *, preset: str, brief: bool = False) -> str:
    if brief:
        label = {'medium': '默认', 'conservative': '保守', 'failure_prone': '易失稳情景'}.get(preset, preset)
        overrides = plan.get('parameter_overrides') or {}
        changed = ('本次参数：' + '；'.join(f'{key}={value}' for key, value in overrides.items()) + '。') if overrides else ''
        return sanitize(f'水文与稳定性计算方案已就绪，采用{label}参数。{changed}回复「开始运行」即可提交计算。')
    stages = plan.get("stages") or []
    titles = [item.get("stage_id", item) if isinstance(item, dict) else item for item in stages]
    overrides = plan.get("parameter_overrides") or {}
    lines = [
        f"运行计划已生成，共 {len(titles)} 个阶段：{'、'.join(titles)}。",
        f"参数预设：{preset}。",
    ]
    bounds = plan.get("parameter_bounds") or {}
    if overrides:
        described = []
        for key, value in overrides.items():
            limit = bounds.get(key)
            described.append(f"{key}={value}" + (f"（允许范围 {limit[0]}–{limit[1]}）" if limit else ""))
        lines.append("参数覆盖：" + "；".join(described))
    else:
        lines.append("未覆盖任何参数，全部使用预设值。")
    # The service nests the input record under ``input``; older payloads carried
    # the id at the top level, so accept either.
    source = plan.get("input") if isinstance(plan.get("input"), dict) else {}
    input_id = source.get("input_id") or plan.get("input_id")
    mode = plan.get("mode") or "build_run_qa"
    lines.append(f"输入编号：{_text(input_id)}；模式：{_text(mode)}。")
    lines.append("确认无误后回复「开始运行」，任务将在后台执行。")
    return sanitize("\n".join(lines))


def describe_run_started(status: dict[str, Any], *, brief: bool = False) -> str:
    if brief:
        return '已提交后台计算，右侧会显示运行进度。'
    return sanitize(
        f"任务已提交：{status.get('run_id')}，当前状态 {status.get('status')}。\n"
        f"共 {len(status.get('stages') or [])} 个阶段将在后台执行。\n"
        "可以随时回复「状态」查询进度，或回复「取消」终止任务。"
    )


def describe_run_status(status: dict[str, Any], *, brief: bool = False) -> str:
    if brief and status.get('status') in {'queued', 'running', 'succeeded'}:
        state = status.get('status')
        if state == 'succeeded':
            return '计算已完成，质量检查通过，可以查看结果。'
        stage = status.get('current_stage')
        labels = {'delineate': '划分子汇水区', 'rainfall': '准备降雨', 'build': '构建模型',
                  'swmm': '水文计算', 'safety': '稳定性计算', 'analysis': '结果分析',
                  'export': '整理成果', 'qa': '质量检查', 'validate': '校验输入', 'sync': '同步输入'}
        return ('任务正在排队。' if state == 'queued' else
                f'正在{labels.get(stage, "计算")}，右侧可查看进度。')
    state = status.get("status")
    completed = status.get("completed_stages") or []
    lines = [f"任务 {_text(status.get('run_id'))} 当前状态：{_text(state)}。"]
    current = status.get("current_stage")
    if current:
        lines.append(f"正在执行：{current}。")
    lines.append(f"已完成 {len(completed)} 个阶段：{'、'.join(completed) or '无'}。")
    if state == "qa_failed":
        lines.append("计算已完成，但质量门槛未通过：结果只能用于诊断，不能作为稳定性结论。")
        for error in status.get("qa_errors") or []:
            lines.append(f"- {_text(error)}")
    elif state == "failed":
        lines.append(f"失败原因：{status.get('summary') or '见任务日志'}。")
    elif state == "cancelled":
        lines.append("任务已取消，可以重新生成计划并运行。")
    elif state == "succeeded":
        lines.append("运行通过质量门槛，可以查看结果与初步评价。")
    return sanitize("\n".join(lines))


def describe_cancellation(status: dict[str, Any]) -> str:
    completed = status.get("completed_stages") or []
    return sanitize(
        f"任务 {_text(status.get('run_id'))} 状态：{_text(status.get('status'))}。"
        f"已完成 {len(completed)} 个阶段。\n取消后可以重新生成计划并运行。"
    )


def describe_visualization_started(record: dict[str, Any]) -> str:
    return sanitize(
        f"图件渲染已提交：{record.get('visualization_id')}（{record.get('title')}）。\n"
        "完成后会显示图片。"
    )


def describe_visualization_ready(record: dict[str, Any], *, brief: bool = False) -> str:
    if brief:
        warning = '本次运行未通过质量检查，该图仅供诊断。' if record.get('qa_failed') else ''
        return sanitize(f'已生成{_text(record.get("title"), "图件")}。{warning}')
    sources = record.get("sources") or []
    lines = [f"图件已生成：{_text(record.get('title'))}。",
             f"渲染器版本 {_text(record.get('renderer_version'))}，来源文件 {len(sources)} 个。"]
    if record.get("qa_failed"):
        lines.append("该图来自未通过质量门槛的运行，图上带有 QA FAILED 标记，不能作为结论使用。")
    return sanitize("\n".join(lines))


def describe_catalog(options: list[dict[str, Any]]) -> str:
    if not options:
        return "当前没有可用的图件类型。"
    lines = ["可用的图件类型（按阶段）："]
    by_stage: dict[str, list[str]] = {}
    for item in options:
        by_stage.setdefault(str(item.get("stage_id")), []).append(str(item.get("kind")))
    for stage, kinds in sorted(by_stage.items()):
        lines.append(f"- {stage}: {'、'.join(sorted(kinds))}")
    lines.append("回复图件名称即可生成，例如「看最小安全系数的图」。")
    return sanitize("\n".join(lines))


def describe_artifact_ready(descriptor: dict[str, Any], *,
                            figure_title: str | None = None,
                            derived: bool = False,
                            alternatives: list[dict[str, Any]] | None = None) -> str:
    """One prepared download: what it is, where it came from, what else exists.

    The size is given in megabytes from a value the caller also puts in the
    evidence, so the number in this text is one the reply can account for.
    """
    lines: list[str] = []
    if figure_title:
        lines.append(f"图件：{_text(figure_title)}。")
    label = "派生文件" if derived else "成果文件"
    lines.append(f"{label}：{_text(descriptor.get('title'))}"
                 f"（格式 {_text(descriptor.get('format'))}，"
                 f"{_size(descriptor.get('size_mb'))}）。")
    if descriptor.get("units"):
        lines.append(f"单位：{_text(descriptor.get('units'))}。")
    # The CRS is deliberately not printed: an EPSG code is a number, and a
    # number in this text must be one the evidence accounts for. It travels in
    # the descriptor, which is where the interface reads it from.
    if str(descriptor.get("qa_status") or "").lower() in {"failed", "qa_failed"}:
        lines.append("该成果来自未通过质量门槛的运行，仅供诊断，不能作为结论使用。")
    if derived:
        lines.append("派生文件由原始成果转换而来；引用数据时请以原始成果为准。")
    if alternatives:
        names = "、".join(str(item.get("title")) for item in alternatives[:4])
        lines.append(f"这张图的其他成果：{names}。")
    return sanitize("\n".join(lines))


def describe_no_figure_artifacts() -> str:
    return sanitize(
        "这个会话还没有可下载的图件成果。先让我画出图件（例如「看最小安全系数的图」），"
        "再下载它对应的原始成果；也可以直接下载本次运行的 SWMM 建模成果。"
    )


def describe_artifact_choices(figure_title: str, artifacts: list[dict[str, Any]]) -> str:
    """No file matched what was asked for: say what this figure does offer."""
    lines = [f"图件「{_text(figure_title)}」没有这种格式的成果。可下载的成果："]
    for item in artifacts[:8]:
        lines.append(f"- {_text(item.get('title'))}（{_text(item.get('format'))}，"
                     f"{_size(item.get('size_mb'))}）")
    lines.append("请说明要哪一种，例如「下载这张图的 GeoTIFF」。")
    return sanitize("\n".join(lines))


def describe_swmm_model(model: dict[str, Any]) -> str:
    """What the run's SWMM model files are, and whether the set is complete."""
    artifacts = list(model.get("artifacts") or [])
    lines = [f"本次运行的 SWMM 建模成果 {len(artifacts)} 项："]
    for item in artifacts[:12]:
        lines.append(f"- {_text(item.get('title'))}"
                     f"（{_text(item.get('role') or item.get('format'))}）")
    if not model.get("complete", True):
        missing = "、".join(model.get("missing_required") or [])
        lines.append(f"缺少必需文件：{_text(missing)}；这些成果不完整，不能按完整模型成果使用。")
    lines.append("需要一次拿到全部文件（含清单与校验和）时，可以下载 SWMM 模型包。")
    return sanitize("\n".join(lines))


def describe_swmm_bundle_ready(bundle: dict[str, Any], *,
                               complete: bool, reused: bool = False) -> str:
    lines = [("已复用" if reused else "已生成") + " SWMM 模型包："
             f"{_text(bundle.get('title'))}"
             f"（{_size(bundle.get('size_mb'))}）。"]
    if not complete:
        missing = "、".join(bundle.get("missing_required") or [])
        lines.append(f"该包缺少必需文件：{_text(missing)}；只能作为诊断件，不是完整模型包。")
    else:
        lines.append("包内含 INP、RPT、OUT、审查表与雨量序列，附清单、校验和与说明。")
    return sanitize("\n".join(lines))


def describe_comparison(result: dict[str, Any]) -> str:
    lines = [f"比较了 {len(result.get('visualization_ids') or [])} 张图件。",
             f"来自运行：{'、'.join(result.get('run_ids') or [])}。"]
    if result.get("same_sources"):
        lines.append("来源文件一致，可以直接比较。")
    else:
        for item in result.get("differences") or []:
            lines.append(f"差异：{item}")
        lines.append("存在差异时不能直接排名。")
    if result.get("qa_failed_count"):
        lines.append(f"其中 {result['qa_failed_count']} 张来自未通过质量门槛的运行。")
    return sanitize("\n".join(lines))


def describe_capabilities(payload: dict[str, Any]) -> str:
    data = payload.get("data") or {}
    engine = (data.get("engine") or {}).get("ok")
    modules = (data.get("modules") or {}).get("ok")
    lines = [
        f"服务：{data.get('service')}。",
        f"SWMM 引擎：{'可用' if engine else '不可用'}；"
        f"计算依赖：{'齐全' if modules else '缺失'}。",
        f"支持 {len(data.get('stages') or [])} 个阶段，"
        f"参数预设：{'、'.join(data.get('parameter_presets') or [])}。",
        f"已登记输入：{data.get('registered_inputs', 0)} 个。",
    ]
    for item in (data.get("limitations") or [])[:3]:
        lines.append(f"限制：{item}")
    return sanitize("\n".join(lines))


# ------------------------------------------------------------------- fallback


def describe_unknown(session: AgentSession, text: str) -> str:
    return sanitize(
        "我没有理解这个请求。\n"
        f"当前可以做的事：{session.pending_action()}\n"
        f"（原始输入：{text[:60]}）"
    )


def describe_scope() -> str:
    return sanitize(f"{SCOPE_STATEMENT}。\n{DISCLAIMER_ZH}")


def describe_missing_fields(intent: Intent, session: AgentSession) -> str:
    """One question, for the single most important missing piece."""
    if intent is Intent.SELECT_OUTLET:
        return sanitize("请给出河流名称、区县或经纬度来定位地图，再绘制闭合评价边界。")
    if intent is Intent.SET_RAINFALL_PERIOD:
        return sanitize("请说明降雨事件：指定日期（例如 2024-07-30 全天）或时段，"
                        "也可以回复「用历史最大日」。")
    return sanitize(f"请补充信息。当前可以做的事：{session.pending_action()}")


def stamp_limitations(lines: list[str]) -> list[str]:
    """Always-true limitations, appended to a result summary."""
    return [
        "结果为区域尺度、降雨事件驱动的初步筛查与诊断，不是工程级安全鉴定。",
        "降雨来自 ERA5 虚拟雨站，不是地面实测站。",
        "当前模型为独立子汇水区出水口配置，各子汇水区径流之和不是流域出口的路由流量过程。",
        *lines,
    ]


def scrub_paths(text: str) -> str:
    """Remove absolute machine paths that a server message may have carried."""
    pattern = re.compile(
        r"(?:[A-Za-z]:[\\/](?:[^\\/\s\"']+[\\/])*[^\\/\s\"']*)"      # Windows
        r"|(?:\\\\[^\\/\s]+[\\/][^\\/\s]+(?:[\\/][^\\/\s]+)*)"       # UNC
        r"|(?:/(?:usr|home|opt|var|etc|tmp|mnt|data)(?:/[^/\s\"']+)+)"  # POSIX root
    )
    return pattern.sub("<内部路径>", str(text))
