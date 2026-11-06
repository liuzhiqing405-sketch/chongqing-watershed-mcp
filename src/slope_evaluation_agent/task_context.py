"""Public task facts for knowledge answers and the interactive workspace."""
from __future__ import annotations

from copy import deepcopy
import re


def public_spatial_qa(qa):
    fields=("passed","errors","notes","modeling_check","area_km2","selected_cells","calculation_area_km2",
            "dem_coverage_fraction","landuse_coverage_fraction")
    result={key:deepcopy(qa[key]) for key in fields if key in qa}
    hydro=qa.get("hydrologic_check") or {}
    result["hydrologic_check"]={key:deepcopy(hydro[key]) for key in (
        "checked","passed","classification","errors","warnings","limitations","outlets","inlets",
        "major_outlet_count","external_upstream_area_km2","external_area_fraction","dominant_outlet_fraction",
        "upstream_context_edge_cells","context_stability_verified","resolution_m","thresholds",
        "complete_natural_watershed_verified","blocking_input_error") if key in hydro}
    return result


def task_background(session):
    from .session_store import _scrub, _strip_paths
    context=session.selection_context
    points=session.study_area_coordinates
    extent=([min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)]
            if points else None)
    return _strip_paths(_scrub({"scope":"当前任务背景；只有已提供输入与已保存检查，不能代替计算结果",
            "location":{"administrative_area":session.administrative_area,"river_name":session.river_name,
                        "label":context.get("location_label"),"map_center":context.get("map_center"),
                        "approximate_location_unverified":bool(session.location_hint)},
            "workflow_state":session.state,"next_action":session.pending_action(),
            "boundary":{"drawn":bool(session.study_area_coordinates),"confirmed":session.watershed_confirmed,
                        "assessment_mode":session.assessment_mode,
                        "qa_passed":session.watershed_qa_passed,"checks":deepcopy(session.study_area_qa),
                        "review_id":session.review_id,
                        "bbox_wgs84":extent,"unsubmitted_changes":boundary_draft_changed(session)},
            "rainfall_inputs":deepcopy(session.rainfall_request),
            "prepared_rainfall":deepcopy(session.rainfall_context),
            "parameters":{"preset":session.parameter_preset,"overrides":dict(session.parameter_overrides)},
            "run":{"submitted":bool(session.hydro_run_id),"id":session.hydro_run_id,"status":session.hydro_run_status,
                   "quality_passed":session.qa_passed,"detail":deepcopy(session.run_status_detail)},
            "delivered_visualizations":[{key:deepcopy(figure[key]) for key in
                ('visualization_id','kind','title','run_id','status','qa_failed') if key in figure}
                for figure in session.visualizations if figure.get('run_id') == session.hydro_run_id],
            "pending_task":deepcopy(session.pending_workflow),
            "authority":"只读背景，不允许修改、确认边界、恢复版本、提交或取消计算。"}))


def task_fact_reply(question, session):
    """A read-only fallback for task facts, never a question/task router.

    Called only after a turn has been routed to knowledge chat. Keep saved
    evidence useful when generation is unavailable or rejected.
    """
    from .engineering_knowledge import asks_general_concept
    from .session_store import sanitize_user_text
    if asks_general_concept(question):
        return None
    if (re.search(r'哪些.*复用|复用.*哪些|可.*复用', question)
            and session.hydro_run_id and session.hydro_run_status == 'succeeded'
            and session.qa_passed is True and session.run_input_signature):
        from .input_binding import input_signature
        if session.run_input_signature == input_signature(session):
            return {'topic':'task_reuse', 'text':
                f'当前评价边界、已导入输入和通过的输入验证均已保留；本次参数更新已有完成运行 {session.hydro_run_id} 的结果。'
                '这些输入可继续沿用，已完成的结果仅对应当前保存的参数。若再改边界、降雨或参数，须重新核对受影响的计划和成果；'
                '本轮没有新增输入改动，因此未重复提交计算，也未把旧结果作为新结果。'}
    background = task_background(session)
    run = background['run']
    boundary = background['boundary']
    if re.search(r'qa|质量|门槛|检查.*(?:失败|通过)', question, re.IGNORECASE):
        errors = list((run['detail'] or {}).get('qa_errors') or [])
        if run['submitted'] and (errors or run['status'] == 'qa_failed'):
            text = '当前对话引用的运行未通过质量检查。'
            text += '\n' + ('\n'.join('- '+error for error in dict.fromkeys(map(str, errors)))
                            if errors else '会话尚未保存具体失败原因，需要刷新任务状态后查看。')
            text += '\n计算结果目前只能用于诊断。上述原因来自该次运行已保存的检查记录。'
            return {'text':sanitize_user_text(text),'topic':'task_quality'}
        checks = boundary['checks']
        if checks.get('errors'):
            return {'text':sanitize_user_text('当前选区检查未通过：\n'+'\n'.join(
                '- '+str(e) for e in checks['errors'])),'topic':'task_quality'}
    refers = any(cue in question for cue in ('这个', '当前', '本次', '这次', '这里', '所选', '我选', '该流域', '这个任务'))
    if refers and re.search(r'哪(?:个|一)?(?:区|县|地方|位置)|在哪|哪里|行政|位置|属于', question):
        location = background['location']
        parts = []
        if location['administrative_area']:
            parts.append('当前任务选择的行政区是'+location['administrative_area']+'。')
        if location['river_name']:
            parts.append('任务记录的河流名称是'+location['river_name']+'。')
        if boundary['drawn']:
            parts.append('实际计算范围是你在地图上绘制的区域。')
        if not parts:
            parts.append('当前任务尚未保存可确认的位置记录。')
        return {'text':sanitize_user_text(''.join(parts)),'topic':'task_location'}
    return None


def workspace_view(session):
    background=task_background(session)
    running=session.state=="hydro_running" or session.hydro_run_status in {"queued","running"}
    dirty=boundary_draft_changed(session)
    return {"background":background,"location":deepcopy(session.selection_context),
            "coordinates":deepcopy(session.study_area_coordinates),"spatial_qa":deepcopy(session.study_area_qa),
            "draft":deepcopy(session.study_area_draft),
            "assessment_mode":session.assessment_mode,
            "rainfall":deepcopy(session.rainfall_request),"parameters":background["parameters"],
            "run_detail":deepcopy(session.run_status_detail),"last_error":deepcopy(session.last_error),
            "actions":{"edit_boundary":not running and bool(session.selection_context),
                       "edit_location":not running,"edit_rainfall":not running and session.watershed_confirmed,
                       "edit_parameters":not running,"restore":not running,
                       "confirm":not dirty and session.state=="awaiting_watershed_confirmation" and session.watershed_qa_passed is True,
                       "start":session.state=="hydro_planned" and session.hydro_planned and not boundary_draft_changed(session),
                       "prepare":not dirty and not running and session.watershed_confirmed and bool(session.rainfall_request) and session.state in {
                           "watershed_confirmed","rainfall_ready","model_inputs_ready","hydro_input_imported"},
                       "plan":not dirty and not running and session.hydro_input_validated and (not session.hydro_planned or bool(session.hydro_run_id)),
                       "refresh":bool(session.hydro_run_id),"cancel":running},
            "explanation":("后台计算尚未结束。可刷新进度或取消；结束后可修改并恢复任务版本。" if running else
                           "修改边界或位置会撤销原确认及下游输入；修改降雨保留边界；修改参数保留验证过的输入并重新展示计划。")}


def boundary_draft_changed(session):
    if not session.study_area_draft:
        return False
    def opened(points):
        return points[:-1] if len(points)>1 and points[0]==points[-1] else points
    return (session.study_area_draft.get("assessment_mode",session.assessment_mode)!=session.assessment_mode or
            not session.study_area_draft.get("closed") or
            opened(session.study_area_draft.get("coordinates",[]))!=opened(session.study_area_coordinates))
