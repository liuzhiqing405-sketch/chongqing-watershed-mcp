"""Reversible task edits, with explicit invalidation of derived products."""
from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from .models import AgentSession
from .session_store import read_json, write_json_atomic, utc_now, _scrub, _strip_paths
from .state_machine import WorkflowError


TASK_FIELDS = (
    "state", "intent", "river_name", "administrative_area", "selection_mode", "assessment_mode", "selection_context",
    "study_area_coordinates", "study_area_draft", "study_area_qa", "rainfall_context", "requested_outlet", "location_hint", "rainfall_request",
    "parameter_preset", "parameter_overrides", "outlet_candidates", "review_id", "watershed_confirmed",
    "watershed_qa_passed", "reviewer_note", "rainfall_package_id", "model_input_package_id",
    "hydro_input_id", "hydro_input_validated", "hydro_run_id", "hydro_run_status", "run_status_detail",
    "qa_passed", "hydro_planned", "last_plan", "run_mode", "planned_input_signature",
    "run_input_signature", "visualizations", "process_artifacts",
    "swmm_model", "pending_workflow", "pending_question", "last_clarification", "last_error",
)


def task_snapshot(session):
    data = session.model_dump(mode="json")
    return {key: data[key] for key in TASK_FIELDS}


def revision_rows(store, session_id):
    path = store.directory(session_id) / "task_revisions.json"
    return read_json(path).get("revisions", []) if path.is_file() else []


def save_revision(store, session, label):
    rows = revision_rows(store, session.session_id)
    snapshot = _strip_paths(_scrub(task_snapshot(session)))
    assessment = store.load_assessment(session.session_id)
    evidence = store.load_evidence(session.session_id)
    if (rows and rows[-1]["snapshot"] == snapshot and rows[-1].get("assessment") == assessment
            and rows[-1].get("evidence") == evidence and rows[-1].get("label") == label):
        return rows[-1]["revision_id"]
    row = {"revision_id": "revision-" + uuid4().hex[:12], "at": utc_now(), "label": label,
           "snapshot": snapshot, "assessment": assessment, "evidence": evidence}
    rows.append(row)
    write_json_atomic(store.directory(session.session_id) / "task_revisions.json", _strip_paths(_scrub({"revisions": rows})))
    return row["revision_id"]


def public_revisions(store, session_id):
    return [{"revision_id": row["revision_id"], "at": row["at"], "label": row["label"],
             "state": row["snapshot"]["state"], "location": row["snapshot"].get("administrative_area")
             or row["snapshot"].get("river_name") or row["snapshot"].get("selection_context", {}).get("location_label"),
             "rainfall": row["snapshot"].get("rainfall_request"),
             "has_boundary": bool(row["snapshot"].get("study_area_coordinates")),
             "has_result": bool(row["snapshot"].get("hydro_run_id"))} for row in reversed(revision_rows(store, session_id))]


class TaskWorkspaceMixin:
    async def ensure_task_editable(self, session):
        if session.hydro_run_id and (session.state == "hydro_running"
                                     or session.hydro_run_status in {None, "queued", "running"}):
            status = await self.refresh_run_status(session)
            if status.get("status") not in {"succeeded", "qa_failed", "failed", "cancelled", "interrupted"}:
                raise WorkflowError("当前计算仍在运行或状态无法确认。请先查看状态并取消任务，结束后可在本任务修改或恢复。",
                                    next_step="刷新运行状态；需要修改时先取消当前计算。")

    def checkpoint_task(self, session, label):
        return save_revision(self.store, session, label)

    async def prepare_task_edit(self, session, scope, *, previous=None):
        old = previous or session
        await self.ensure_task_editable(old)
        self.checkpoint_task(old, {"location": "修改位置前", "boundary": "修改边界前",
                                   "rainfall": "修改降雨前", "parameters": "修改参数前"}[scope])
        session.reset_for_new_run()
        session.run_status_detail = {}
        session.pending_question = None
        session.last_clarification = None
        session.pending_workflow = {}
        self.store.clear_artifacts(session.session_id, ("evidence.json", "final_assessment.json"))
        if scope in {"location", "boundary", "rainfall"}:
            session.rainfall_context = {}
            session.rainfall_package_id = None
            session.model_input_package_id = None
            session.hydro_input_id = None
            session.hydro_input_validated = False
        if scope in {"location", "boundary"}:
            session.review_id = None
            session.watershed_confirmed = False
            session.watershed_qa_passed = None
            session.reviewer_note = None
            session.study_area_qa = {}
            if session.rainfall_request:
                session.rainfall_request = {key:value for key,value in session.rainfall_request.items() if key != "review_id"}
            session.state = "outlet_candidates_ready" if session.selection_context else "needs_location"
        if scope == "location":
            session.selection_context = {}
            session.study_area_coordinates = []
            session.study_area_draft = {}
            session.state = "needs_location"
        elif scope == "rainfall":
            session.state = ("watershed_confirmed" if session.watershed_confirmed else
                             "awaiting_watershed_confirmation" if session.review_id else "needs_location")
        elif scope == "parameters" and session.hydro_input_validated:
            session.state = "hydro_input_validated"
        self._events(session).append("task_edited", scope=scope, invalidated="downstream_products")
        session.refresh_summary()
        self.store.save(session)

    async def restore_task_revision(self, session, revision_id=None):
        await self.ensure_task_editable(session)
        rows = revision_rows(self.store, session.session_id)
        current = task_snapshot(session)
        chosen = next((row for row in reversed(rows) if
                       (row["revision_id"] == revision_id if revision_id else row["snapshot"] != current)), None)
        if chosen is None:
            raise WorkflowError("当前任务没有可恢复的对应版本；请先保存或修改任务。")
        candidate = session.model_copy(deep=True)
        for field in TASK_FIELDS:
            setattr(candidate, field, deepcopy(chosen["snapshot"].get(field,getattr(candidate,field))))
        candidate.run_input_signature = chosen["snapshot"].get("run_input_signature")
        candidate.planned_input_signature = chosen["snapshot"].get("planned_input_signature")
        if candidate.review_id:
            payload = await self._call(session, self.spatial, "get_watershed_review", review_id=candidate.review_id)
            record = payload.get("data") or payload
            qa = record.get("qa") or {}
            hydro = qa.get("hydrologic_check") or {}
            from .task_context import public_spatial_qa
            candidate.study_area_qa=public_spatial_qa(qa)
            from watershed_exchange.domain_policy import drawn_modeling_allowed
            candidate.watershed_qa_passed = bool(qa.get("passed") and
                (candidate.selection_mode != "drawn_study_area" or drawn_modeling_allowed(qa,candidate.assessment_mode)))
            candidate.watershed_confirmed = bool(candidate.watershed_qa_passed and record.get("status") == "confirmed")
            if not candidate.watershed_confirmed:
                candidate.rainfall_context = {}
                candidate.rainfall_package_id = candidate.model_input_package_id = candidate.hydro_input_id = None
                candidate.hydro_input_validated = False
                candidate.reset_for_new_run()
                candidate.state = "awaiting_watershed_confirmation" if candidate.watershed_qa_passed else "watershed_qa_failed"
        if candidate.hydro_run_id:
            payload = await self._call(session, self.hydro, "get_hydro_run_status", run_id=candidate.hydro_run_id)
            status = payload.get("data") or payload
            if status.get("status") not in {"queued", "running", "succeeded", "qa_failed", "failed", "cancelled", "interrupted"}:
                raise WorkflowError("历史计算的真实状态无法确认，未恢复该版本；可以恢复计算之前的版本。")
            candidate.hydro_run_status = status["status"]
            candidate.run_status_detail = self.public_run_detail(status)
            if status["status"]!="succeeded":
                candidate.qa_passed=False if status["status"]=="qa_failed" else None
            candidate.state = {"queued":"hydro_running","running":"hydro_running","succeeded":"hydro_succeeded",
                               "qa_failed":"hydro_qa_failed","failed":"hydro_failed","cancelled":"hydro_cancelled",
                               "interrupted":"hydro_failed"}[status["status"]]
        self.checkpoint_task(session, "恢复版本前")
        if candidate.selection_context:
            candidate.selection_context["map_id"]=uuid4().hex
            if candidate.study_area_draft:
                candidate.study_area_draft["map_id"]=candidate.selection_context["map_id"]
        for field in TASK_FIELDS:
            setattr(session, field, deepcopy(getattr(candidate, field)))
        # A plan must be shown again before a new submission. Restore never runs.
        session.hydro_planned = False
        session.planned_input_signature = None
        session.last_plan = None
        if session.state == "hydro_planned":
            session.state = "hydro_input_validated"
        self.store.clear_artifacts(session.session_id, ("evidence.json", "final_assessment.json"))
        if candidate.hydro_run_id and candidate.hydro_run_status == "succeeded":
            if chosen.get("evidence"):
                self.store.save_evidence(session.session_id, chosen["evidence"])
            if chosen.get("assessment"):
                self.store.save_assessment(session.session_id, chosen["assessment"])
        session.pending_question = session.last_clarification = None
        session.refresh_summary()
        self.store.save(session)
        self._events(session).append("task_restored", revision_id=chosen["revision_id"])
        return await self._finish(session, f"已在当前任务恢复「{chosen['label']}」保存的版本。\n{session.pending_action()}",
                                  kind="question", facts={"restored_revision":chosen["revision_id"]})

    @staticmethod
    def public_run_detail(status):
        from .session_store import _scrub, _strip_paths
        return _strip_paths(_scrub({key:status[key] for key in
            ("status", "stage", "current_stage", "stages", "completed_stages", "progress", "progress_percent", "message",
             "error", "errors", "warnings", "qa_passed", "qa_rechecked_at", "qa_failed_gates", "qa_errors",
             "started_at", "updated_at", "finished_at") if key in status}))
