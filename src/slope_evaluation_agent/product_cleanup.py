"""Conversation product ownership and coordinated MCP cleanup.

Called only by explicit session deletion. The LLM does not route to it.
"""
from __future__ import annotations

import json

from watershed_exchange.cleanup import PRODUCT_ID, linked, metadata, product_ids, references, safe_target

from .session_store import SESSION_ID_PATTERN, SessionError, write_json_atomic
from .state_machine import WorkflowError
from .task_workspace import TASK_FIELDS


CREATION_KEYS = {
    "delineate_from_outlet": "review_id",
    "create_drawn_study_area": "review_id",
    "build_virtual_rainfall": "package_id",
    "prepare_model_inputs": "package_id",
    "import_hydro_input_package": "input_id",
    "register_hydro_inputs": "input_id",
    "start_hydro_run": "run_id",
    "create_hydro_visualization": "visualization_id",
    "compare_hydro_visualizations": "comparison_visualization_id",
}


def record_created_product(store, session_id, method, payload):
    key = CREATION_KEYS.get(method)
    if not key or payload.get("ok") is False:
        return
    data = payload.get("data") or payload
    identifier = data.get(key)
    if not isinstance(identifier, str) or not PRODUCT_ID.fullmatch(identifier):
        return
    path = store.directory(session_id) / "products.json"
    ids = product_ids(metadata(path).get("owned_ids", []))
    ids.update(product_ids([identifier]))
    # Persist before further response formatting/state transitions can fail.
    write_json_atomic(path, {"schema_version": 1, "owned_ids": sorted(ids)})


def session_products(store, session_id):
    directory = store.directory(session_id)
    safe_target(store.root, directory, inspect_tree=False)
    current = metadata(directory / "session.json")
    ids = references({key: current.get(key) for key in TASK_FIELDS})
    history = metadata(directory / "task_revisions.json")
    for row in history.get("revisions", []):
        ids.update(references(row.get("snapshot", {})))
    ids.update(product_ids(metadata(directory / "products.json").get("owned_ids", [])))
    ids.update(product_ids(metadata(directory / "cleanup_pending.json").get("identifiers", [])))
    # Legacy sessions did not have ownership ledgers. Read the complete event
    # log to recover ids that a task edit/retry no longer stores as current.
    events = directory / "events.jsonl"
    if linked(events):
        raise SessionError("会话事件文件含链接，拒绝清理")
    if events.exists():
        with events.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # A torn final append contains no confirmed ids.
                if event.get("event") == "tool_call":
                    ids.update(references(event.get("arguments", {})))
    return ids


class ProductCleanupMixin:
    async def cleanup_session_products(self, session):
        identifier = session.session_id
        try:
            ids = session_products(self.store, identifier)
            safe_target(self.store.root, self.store.directory(identifier))
            protected = set()
            for directory in self.store.root.iterdir():
                if SESSION_ID_PATTERN.fullmatch(directory.name) and directory.name != identifier:
                    protected.update(session_products(self.store, directory.name))
        except (OSError, ValueError) as exc:
            raise WorkflowError("无法安全读取会话产物记录；对话已保留，请修复记录后重试删除。") from exc
        if not ids:
            if session.state == "hydro_running" or session.hydro_run_status in {"queued", "running"}:
                raise WorkflowError("该对话仍有任务运行；请先取消任务，确认停止后再删除。")
            return {"deleted_ids": [], "retained_ids": []}
        # Preflight both services before deleting anything. The hydro preview
        # reports spatial dependencies of retained inputs in all other runs.
        hydro_ids = sorted(x for x in ids if x.startswith(("hydro-", "viz-")))
        spatial_ids = sorted(x for x in ids if x.startswith(("basin-", "rain-", "package-")))
        keep = sorted(protected)
        hydro_plan = None
        # Spatial-only deletion still checks retained hydro dependencies, so
        # an imported/shared package cannot disappear while a job uses it.
        if hydro_ids or spatial_ids:
            hydro_plan = (await self.hydro.cleanup_hydro_products(hydro_ids, keep, dry_run=True))["data"]
            keep = sorted(protected | product_ids(hydro_plan["external_references"]))
            ids.update(product_ids(hydro_plan.get("related_external_ids", [])))
            spatial_ids = sorted(x for x in ids if x.startswith(("basin-", "rain-", "package-")))
        if spatial_ids:
            spatial_plan = (await self.spatial.cleanup_spatial_products(spatial_ids, keep, dry_run=True))["data"]
            ids.update(product_ids(spatial_plan.get("planned_ids", [])))
        if hydro_plan:
            ids.update(product_ids(hydro_plan.get("planned_ids", [])))
        # Keep retry identifiers if one service is interrupted after committing.
        write_json_atomic(self.store.directory(identifier) / "cleanup_pending.json",
                          {"identifiers": sorted(ids)})
        deleted, retained = set(), set()
        if hydro_ids:
            result = (await self.hydro.cleanup_hydro_products(hydro_ids, sorted(protected), dry_run=False))["data"]
            deleted.update(result["deleted_ids"])
            retained.update(result["retained_ids"])
            keep = sorted(protected | product_ids(result["external_references"]))
        if spatial_ids:
            result = (await self.spatial.cleanup_spatial_products(spatial_ids, keep, dry_run=False))["data"]
            deleted.update(result["deleted_ids"])
            retained.update(result["retained_ids"])
        return {"deleted_ids": sorted(deleted), "retained_ids": sorted(retained)}
