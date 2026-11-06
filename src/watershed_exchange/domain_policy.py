"""Shared, data-only contract for user-drawn modelling scopes."""
from typing import Literal

AssessmentMode = Literal["regional_screening", "catchment_check"]
ASSESSMENT_MODES = ("regional_screening", "catchment_check")
LOCAL_SCOPE = "仅计算所选区域内部的降雨产流与坡面响应；不计框外上游入流，不能解释为包含全部上游来水的河道流量或水位。"


def regional_tool_contract_error(rows) -> str | None:
    """Reject a named but outdated tool before it can create a bad review."""
    tool=next((row for row in rows if row.name=="create_drawn_study_area"),None)
    if tool is None:
        return "create_drawn_study_area missing"
    mode=tool.input_schema.get("properties",{}).get("assessment_mode",{})
    if mode.get("default")!="regional_screening" or set(mode.get("enum") or [])!=set(ASSESSMENT_MODES):
        return "create_drawn_study_area requires regional assessment modes and hydrologic diagnostics; restart the spatial MCP with current source"
    return None


def modeling_check(diagnostic: dict, mode: str) -> dict:
    if mode not in ASSESSMENT_MODES:
        raise ValueError("未知评价方式，请选择区域评价或上游来水严格检查")
    errors = list(diagnostic.get("errors") or [])
    warnings = list(diagnostic.get("warnings") or [])
    if mode == "regional_screening":
        blocking = errors if diagnostic.get("blocking_input_error") else []
        if not diagnostic.get("checked"):
            warnings.append("框外水文诊断未完成，不能判断自然流域完整性。" + "；".join(errors))
        elif errors:
            warnings.append("框外汇水或汇流边界存在未解决问题；本次仍按框内降雨边界假设开展区域评价。")
        if diagnostic.get("classification") == "upstream_truncated":
            warnings.append("识别到明显框外上游汇水，这些来水未纳入本次计算。")
        if diagnostic.get("upstream_context_edge_cells"):
            warnings.append("框外上游地形范围不完整，贡献面积可能被低估；本次评价范围保持用户原边界。")
    else:
        blocking = errors
        if not diagnostic.get("checked") or not diagnostic.get("passed"):
            blocking = blocking or ["上游来水严格检查未通过，请扩大地形范围或调整边界"]
    return {"version": 1, "assessment_mode": mode, "passed": not blocking,
            "output_scope": "local_rainfall_response", "external_inflow": "excluded",
            "boundary_assumptions": [LOCAL_SCOPE], "errors": blocking,
            "warnings": list(dict.fromkeys(warnings))}


def drawn_modeling_allowed(qa: dict, requested_mode: str | None = None) -> bool:
    """Use explicit policy for new reviews; old reviews keep their strict gate."""
    if qa.get("passed") is not True:
        return False
    diagnostic = qa.get("hydrologic_check") or {}
    policy = qa.get("modeling_check")
    if not isinstance(diagnostic, dict):
        return False
    if policy is None:
        return bool(diagnostic.get("checked") and diagnostic.get("passed"))
    if not isinstance(policy, dict) or not isinstance(diagnostic, dict):
        return False
    mode = policy.get("assessment_mode")
    if (policy.get("version") != 1 or mode not in ASSESSMENT_MODES or
            (requested_mode is not None and requested_mode != mode) or
            policy.get("passed") is not True or policy.get("errors") or
            policy.get("output_scope") != "local_rainfall_response" or
            policy.get("external_inflow") != "excluded" or
            policy.get("boundary_assumptions") != [LOCAL_SCOPE] or
            "checked" not in diagnostic or diagnostic.get("blocking_input_error")):
        return False
    return mode == "regional_screening" or bool(diagnostic.get("checked") and diagnostic.get("passed"))
