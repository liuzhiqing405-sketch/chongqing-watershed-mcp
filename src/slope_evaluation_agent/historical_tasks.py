"""Read saved task revisions without restoring or mutating the current task."""
from __future__ import annotations

import re

from .task_workspace import revision_rows


def _reference_text(text):
    """Only positive references select a revision, not a protected branch."""
    return '，'.join(clause for clause in re.split(r'[，,。；;\n]', text)
                    if not re.match(r'\s*(?:请)?(?:不要|不能|不得|不必|勿)', clause))


def _labels(text):
    return [match.group(1) or match.group(2) for match in re.finditer(
        r'(?:标记|标签|名为|叫做|方案)\s*[为是:：]?\s*["“「]?([A-Za-z][A-Za-z0-9_-]*)'
        r'|(?<![A-Za-z0-9_-])([A-Za-z][A-Za-z0-9_-]*)\s*方案', text)]


def is_history_question(text):
    reference = _reference_text(text)
    # A saved file/log is current-run evidence; the word "历史" in a
    # prohibition on inventing history does not select a previous task.
    return bool(_labels(reference) or re.search(
        r'(?:历史|已保存|旧|原)(?:的)?(?:任务|方案|版本)|'
        r'第[一二三四五六七八九十\d]+(?:个|套)?方案|revision-', reference))


def _multiple_requested(text):
    return bool(re.search(r'区分.*(?:套|方案|记录)|(?:各|全部|所有)(?:套|个)?(?:方案|版本)|'
                          r'(?:比较|对比).*方案', _reference_text(text)))


def select_revisions(store, session_id, text):
    text = _reference_text(text)
    rows = revision_rows(store, session_id)
    ids = set(re.findall(r"(?:revision|(?:hydro-)?run)-[A-Za-z0-9_-]+", text))
    if ids:
        rows = [row for row in rows if row["revision_id"] in ids
                or row["snapshot"].get("hydro_run_id") in ids]
    dates = set(re.findall(r"\d{4}-\d{2}-\d{2}", text))
    if dates:
        rows = [row for row in rows if dates & {
            (row["snapshot"].get("rainfall_request") or {}).get("date"),
            str((row["snapshot"].get("rainfall_context") or {}).get("event_start", ""))[:10]}]
    labels = _labels(text)
    if labels and not (_multiple_requested(text) and len(labels) == 1):
        rows = [row for row in rows if any(re.search(r"(?<![A-Za-z0-9_-])" + re.escape(token)
                         + r"(?![A-Za-z0-9_-])", row["label"]) for token in labels)]
    # An explicitly named region is another constraint, not a reason to switch.
    places = {str(row["snapshot"].get("administrative_area") or
                  row["snapshot"].get("river_name") or
                  row["snapshot"].get("selection_context", {}).get("location_label") or "") for row in rows}
    named_region = re.search(r"(?:区域|选区)\s*[A-Za-z0-9]+", text)
    if named_region:
        rows = [row for row in rows if named_region.group().replace(" ", "") in {
            str(row["snapshot"].get("administrative_area") or "").replace(" ", ""),
            str(row["snapshot"].get("selection_context", {}).get("location_label") or "").replace(" ", "")}]
    cited = {place for place in places if place and place in text}
    if cited:
        rows = [row for row in rows if (row["snapshot"].get("administrative_area") or
                row["snapshot"].get("river_name") or
                row["snapshot"].get("selection_context", {}).get("location_label")) in cited]
    # Many checkpoints can describe the same run. Prefer the most recent saved
    # evidence for that run, while keeping different runs distinct.
    unique = {}
    for row in rows:
        snapshot = row["snapshot"]
        key = snapshot.get("hydro_run_id") or row["revision_id"]
        if key not in unique or row.get("assessment") or row.get("evidence"):
            unique[key] = row
    rows = list(unique.values())
    ordinal = re.search(r"第([一二三四五六七八九十\d]+)(?:个|套)?方案", text)
    if ordinal:
        from .temporal_inputs import _integer
        index = _integer(ordinal.group(1)) - 1
        # A scenario/solution means a submitted run, not every intermediate save.
        runs = [row for row in rows if row["snapshot"].get("hydro_run_id")]
        return runs[index:index+1] if index >= 0 else []
    if not labels and not dates:
        runs = [row for row in rows if row["snapshot"].get("hydro_run_id")]
        if runs:
            rows = runs
    return rows


def history_answer(store, session, text):
    rows = select_revisions(store, session.session_id, text)
    if not rows:
        return {"text": "本会话没有找到与该编号、标记或日期对应的已保存方案。请提供本会话的版本编号或明确标记。当前任务保持不变。",
                "kind": "question", "ok": False, "facts": {"historical_revisions": []}}
    if len(rows) != 1 and not _multiple_requested(text):
        choices = [{"revision_id": row["revision_id"], "label": row["label"],
                    "run_id": row["snapshot"].get("hydro_run_id"),
                    "rainfall": row["snapshot"].get("rainfall_request")} for row in rows]
        lines = [f"- {row['revision_id']}：{row['label']}，降雨 {row['rainfall']}" for row in choices]
        return {"text": "找到多个已保存版本，请指定版本编号：\n" + "\n".join(lines) + "\n当前任务保持不变。",
                "kind": "question", "ok": True, "facts": {"historical_revisions": choices}}
    if len(rows) > 1:
        answers = [_render_revision(row, session) for row in rows]
        return {"text": '\n\n'.join(answers), "kind": "result", "ok": True,
                "facts": {"historical_revisions": rows}}
    row = rows[0]
    return {"text": _render_revision(row, session), "kind": "result", "ok": True,
            "facts": {"historical_revision": row}}


def _render_revision(row, session):
    snapshot = row["snapshot"]
    rainfall = snapshot.get("rainfall_request") or {}
    period = rainfall.get("date") or ((snapshot.get("rainfall_context") or {}).get("event_start"))
    if not period:
        period = (f"历史最大日筛选：截至{rainfall['end_year']}年" if rainfall.get("end_year") else
                  "历史最大日筛选，未保存截止年份") if rainfall.get("mode") == "historical_max" else "未保存具体日期"
    lines = [f"已保存方案「{row['label']}」（{row['revision_id']}）。", f"降雨：{period}。",
             f"参数预设：{snapshot.get('parameter_preset') or '未保存'}。"]
    from .models import OVERRIDABLE_PARAMETERS
    labels = {"saturated_cohesion_kpa": "饱和黏聚力", "deposit_thickness_m": "堆积层厚度",
              "natural_density_g_cm3": "天然密度", "saturated_density_g_cm3": "饱和密度",
              "saturated_friction_angle_deg": "饱和内摩擦角", "initial_saturation": "初始饱和度",
              "friction_velocity_coefficient_a": "剪切速度系数", "runoff_density_g_cm3": "径流密度",
              "calculation_resolution_m": "计算分辨率"}
    for key, value in (snapshot.get("parameter_overrides") or {}).items():
        lines.append(f"{labels.get(key, key)}：{value} {OVERRIDABLE_PARAMETERS.get(key, '')}。")
    assessment = row.get("assessment") or row.get("evidence") or {}
    if assessment.get("hydro_run_id") != snapshot.get("hydro_run_id"):
        assessment = {}
    if snapshot.get("run_input_signature"):
        from .models import AgentSession
        from .input_binding import input_signature
        saved_session = AgentSession.model_validate({"session_id": session.session_id, **snapshot})
        if saved_session.run_input_signature != input_signature(saved_session):
            assessment = {}
    minimum = (assessment.get("factors", {}).get("safety_factor") or {}).get("minimum")
    if minimum is not None and assessment.get("assessment_status") == "valid_preliminary":
        lines.append(f"该方案已保存结果的最小安全系数为 {minimum}。")
    else:
        lines.append("该版本未保存通过质量检查的最小安全系数，不能用当前任务数值代替。")
    lines.append("以上来自该版本的保存记录，未恢复、重算或切换当前任务。")
    return "\n".join(lines)
