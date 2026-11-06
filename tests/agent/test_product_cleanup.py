"""Real managed directories behind WebUI deletion, isolated from user runs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chongqing_spatial_mcp.cleanup import cleanup_spatial_products
from chongqing_spatial_mcp.settings import Settings as SpatialSettings
from slope_evaluation_agent.agent import SlopeEvaluationAgent
from slope_evaluation_agent.config import AgentSettings
from slope_evaluation_agent.mcp_clients.base import MCPConnectionError
from slope_evaluation_agent.product_cleanup import record_created_product, session_products
from slope_evaluation_agent.session_store import write_json_atomic
from slope_evaluation_agent.web import create_app
from watershed_exchange.cleanup import Product, cleanup_products
from watershed_exchange.paths import PathSafetyError
from watershed_hydro_mcp.cleanup import cleanup_hydro_products
from watershed_hydro_mcp.job_manager import JobStore
from watershed_hydro_mcp.settings import Settings as HydroSettings
from watershed_hydro_mcp.visualization.service import VisualizationService


def identifier(prefix, n):
    return f"{prefix}-{n:012x}"


def put(root, name, manifest, payload):
    path = root / name
    path.mkdir(parents=True)
    write_json_atomic(path / manifest, payload)
    (path / "product.bin").write_bytes(b"isolated science output")
    return path


class SpatialAdapter:
    def __init__(self, settings):
        self.settings = settings
        self.fail_preview = False
        self.fail_commit = False

    async def connect(self):
        pass

    async def close(self):
        pass

    async def cleanup_spatial_products(self, identifiers, protected_ids, dry_run=True):
        if self.fail_preview or (self.fail_commit and not dry_run):
            raise MCPConnectionError("隔离测试模拟连接中断")
        result = cleanup_spatial_products(identifiers, protected_ids, dry_run=dry_run, settings=self.settings)
        return {"ok": True, "data": result}


class HydroAdapter(SpatialAdapter):
    def __init__(self, settings):
        super().__init__(settings)
        self.store = JobStore(settings)
        self.visualizations = VisualizationService(settings)

    async def cleanup_hydro_products(self, identifiers, protected_ids, dry_run=True):
        try:
            data = cleanup_hydro_products(identifiers, protected_ids, dry_run=dry_run,
                                          store=self.store, visualizations=self.visualizations)
        except RuntimeError as exc:
            from slope_evaluation_agent.mcp_clients.base import MCPToolRejected
            raise MCPToolRejected(str(exc)) from exc
        return {"ok": True, "data": data}


@pytest.fixture
def system(tmp_path):
    spatial_settings = SpatialSettings(root=tmp_path, data_root=tmp_path / "data",
        runs_root=tmp_path / "runs", exchange_root=tmp_path / "exchange", native_root=tmp_path / "native")
    hydro_settings = HydroSettings(root=tmp_path, runs_root=tmp_path / "runs",
        hydro_root=tmp_path / "runs" / "hydro", native_root=tmp_path / "native", config_root=tmp_path / "config")
    settings = AgentSettings.model_validate({"sessions": {"root": str(tmp_path / "runs" / "agent" / "sessions")}})
    spatial, hydro = SpatialAdapter(spatial_settings), HydroAdapter(hydro_settings)
    agent = SlopeEvaluationAgent(settings, spatial=spatial, hydro=hydro)
    return agent, spatial, hydro, create_app(settings=settings, agent=agent)


def products(system, n=1, *, source_package=None, run_status="succeeded"):
    agent, spatial, hydro, _ = system
    review, rain, package, input_id, run, viz = [identifier(p, n) for p in
        ("basin", "rain", "package", "hydro-input", "hydro-run", "viz")]
    s, h = spatial.settings, hydro.settings
    paths = [
        put(s.watersheds_root, review, "review.json", {"review_id": review}),
        put(s.rainfall_runs_root, rain, "rainfall_manifest.json", {"review_id": review, "package_id": rain}),
        put(s.model_inputs_root, package, "input_manifest.json",
            {"review_id": review, "rainfall_package_id": rain, "package_id": package}),
        put(s.spatial_visualizations_root, review, "figure.json", {}),
        put(s.exports_root, package, "manifest.json", {"package_id": package}),
        put(h.inputs_root, input_id, "input_record.json", {"input_id": input_id, "source_package_id": source_package or package}),
        put(h.jobs_root, run, "job.json", {"input_id": input_id, "run_id": run}),
        put(h.visualizations_root, viz, "visualization.json", {"run_id": run, "status": "succeeded"}),
    ]
    archive = s.exports_root / f"{package}.zip"
    archive.write_bytes(b"export archive")
    paths.append(archive)
    write_json_atomic(h.jobs_root / run / "status.json", {"run_id": run, "status": run_status})
    return {"review_id": review, "rainfall_package_id": rain, "model_input_package_id": package,
            "hydro_input_id": input_id, "hydro_run_id": run}, paths


def task(agent, fields):
    session = agent.new_session()
    for key, value in fields.items():
        setattr(session, key, value)
    session.state = "hydro_succeeded"
    session.hydro_run_status = "succeeded"
    agent.store.save(session)
    return session


def test_web_delete_cleans_current_history_and_old_events_but_preserves_sources(system):
    agent, spatial, hydro, app = system
    current, current_paths = products(system)
    old, old_paths = products(system, 2)
    forgotten, forgotten_paths = products(system, 3)
    unrelated, unrelated_paths = products(system, 4)
    session = task(agent, current)
    directory = agent.store.directory(session.session_id)
    write_json_atomic(directory / "task_revisions.json", {"revisions": [{"snapshot": old}]})
    agent.store.events(session.session_id).append("tool_call", tool="get_hydro_run_status",
                                                 arguments={"run_id": forgotten["hydro_run_id"]})
    for _ in range(250):
        agent.store.events(session.session_id).append("test_padding")
    # Hydro ancestry discovers the forgotten input/export. Record the spatial
    # packages explicitly, as real prepare/import calls do.
    write_json_atomic(directory / "products.json", {"owned_ids": [forgotten["model_input_package_id"]]})
    source = spatial.settings.data_root / "raw_dem.tif"
    source.parent.mkdir()
    source.write_bytes(b"original source")
    cache = spatial.settings.runs_root / "administrative_dem" / "shared.tif"
    cache.parent.mkdir()
    cache.write_bytes(b"public map")
    image = directory.parent.parent / "web_images" / f"{session.session_id}-{'a'*16}.png"
    image.parent.mkdir()
    image.write_bytes(b"preview")
    with TestClient(app) as client:
        response = client.delete(f"/api/v1/sessions/{session.session_id}")
    assert response.status_code == 200, response.text
    assert not agent.store.exists(session.session_id)
    assert all(not p.exists() for p in current_paths + old_paths + forgotten_paths)
    assert all(p.exists() for p in unrelated_paths)
    assert source.read_bytes() == b"original source" and cache.exists() and not image.exists()


def test_other_session_and_other_run_protect_shared_products(system):
    agent, spatial, hydro, app = system
    fields, paths = products(system)
    session = task(agent, fields)
    other = task(agent, {"hydro_run_id": fields["hydro_run_id"]})
    with TestClient(app) as client:
        response = client.delete(f"/api/v1/sessions/{session.session_id}")
    assert response.status_code == 200, response.text
    assert fields["hydro_run_id"] in response.json()["cleanup"]["retained_ids"]
    assert all(p.exists() for p in paths)
    assert agent.store.exists(other.session_id)


@pytest.mark.parametrize('protect_unselected', [False, True])
def test_batch_deletes_last_selected_reference_but_protects_unselected_tasks(system,protect_unselected):
    agent, _, _, app = system
    fields, paths = products(system)
    first, second = task(agent,fields), task(agent,fields)
    remaining = task(agent,fields) if protect_unselected else None
    with TestClient(app) as client:
        response=client.post('/api/v1/sessions/batch-delete',json={
            'session_ids':[first.session_id,second.session_id]})
    assert response.status_code==200, response.text
    result=response.json()
    assert result['deleted']==[first.session_id,second.session_id]
    assert result['ok'] and not result['failed']
    assert not agent.store.exists(first.session_id) and not agent.store.exists(second.session_id)
    assert all(p.exists()==protect_unselected for p in paths)
    if remaining:
        assert agent.store.exists(remaining.session_id)
        assert fields['hydro_run_id'] in result['cleanup']['retained_ids']
    else:
        assert not result['cleanup']['retained_ids']


def test_batch_removes_history_and_previews_but_keeps_failed_and_unselected_sessions(system):
    agent, spatial, _, app=system
    fields, paths=products(system)
    old, old_paths=products(system,2)
    active, active_paths=products(system,3,run_status='queued')
    other, other_paths=products(system,4)
    selected=task(agent,fields)
    write_json_atomic(agent.store.directory(selected.session_id)/'task_revisions.json',
                      {'revisions':[{'snapshot':old}]})
    running, unselected=task(agent,active),task(agent,other)
    empty=agent.new_session()
    images=agent.store.root.parent/'web_images'
    images.mkdir()
    image=images/f"{selected.session_id}-{'a'*16}.png"
    image.write_bytes(b'preview')
    source=spatial.settings.data_root/'raw_dem.tif'
    source.parent.mkdir();source.write_bytes(b'original')
    with TestClient(app) as client:
        result=client.post('/api/v1/sessions/batch-delete',json={'session_ids':[
            selected.session_id,running.session_id,empty.session_id]}).json()
    assert not result['ok'] and result['deleted']==[selected.session_id,empty.session_id]
    assert result['failed'][0]['session_id']==running.session_id
    assert result['failed'][0]['status_code']==409
    assert all(not p.exists() for p in paths+old_paths) and not image.exists()
    assert all(p.exists() for p in active_paths+other_paths)
    assert agent.store.exists(running.session_id) and agent.store.exists(unselected.session_id)
    assert source.read_bytes()==b'original'


def test_batch_validates_entire_selection_and_deduplicates_before_deleting(system):
    agent, _, _, app=system
    fields, paths=products(system)
    session=task(agent,fields)
    with TestClient(app) as client:
        for ids in ([session.session_id,'../not-a-session'],[],[session.session_id]*201):
            response=client.post('/api/v1/sessions/batch-delete',json={'session_ids':ids})
            assert response.status_code==422
            assert agent.store.exists(session.session_id) and all(p.exists() for p in paths)
        result=client.post('/api/v1/sessions/batch-delete',json={
            'session_ids':[session.session_id,session.session_id]}).json()
    assert result['ok'] and result['deleted']==[session.session_id] and not result['failed']
    assert all(not p.exists() for p in paths)


def test_batch_partial_cleanup_can_retry_and_missing_session_does_not_stop_others(system):
    agent, spatial, _, app=system
    fields, paths=products(system)
    session=task(agent,fields)
    missing=identifier('session',999)
    empty=agent.new_session()
    with TestClient(app) as client:
        spatial.fail_commit=True
        result=client.post('/api/v1/sessions/batch-delete',json={
            'session_ids':[missing,session.session_id,empty.session_id]}).json()
        assert result['deleted']==[empty.session_id]
        assert [r['status_code'] for r in result['failed']]==[404,409]
        assert agent.store.exists(session.session_id) and paths[0].exists() and not paths[6].exists()
        spatial.fail_commit=False
        retry=client.post('/api/v1/sessions/batch-delete',json={'session_ids':[session.session_id]}).json()
    assert retry['ok'] and not agent.store.exists(session.session_id)
    assert all(not p.exists() for p in paths)


def test_shared_spatial_package_protected_even_without_another_session(system):
    agent, spatial, hydro, app = system
    fields, paths = products(system)
    # An independent hydro import may not have a WebUI conversation.
    extra_input = identifier("hydro-input", 9)
    put(hydro.settings.inputs_root, extra_input, "input_record.json",
        {"input_id": extra_input, "source_package_id": fields["model_input_package_id"]})
    session = task(agent, fields)
    with TestClient(app) as client:
        response = client.delete(f"/api/v1/sessions/{session.session_id}")
    assert response.status_code == 200, response.text
    assert fields["model_input_package_id"] in response.json()["cleanup"]["retained_ids"]
    assert all(p.exists() for p in paths[:5] + paths[-1:])
    assert all(not p.exists() for p in paths[5:8])


@pytest.mark.parametrize("active", ["run", "figure"])
def test_active_old_versions_block_before_any_deletion(system, active):
    agent, spatial, hydro, app = system
    fields, paths = products(system)
    old, old_paths = products(system, 2, run_status="queued" if active == "run" else "succeeded")
    if active == "figure":
        write_json_atomic(hydro.settings.visualizations_root / identifier("viz", 2) / "visualization.json",
                          {"run_id": old["hydro_run_id"], "status": "running"})
    session = task(agent, fields)
    write_json_atomic(agent.store.directory(session.session_id) / "task_revisions.json",
                      {"revisions": [{"snapshot": old}]})
    with TestClient(app) as client:
        response = client.delete(f"/api/v1/sessions/{session.session_id}")
    assert response.status_code == 409 and "先取消任务" in response.text
    assert agent.store.exists(session.session_id)
    assert all(p.exists() for p in paths + old_paths)


def test_preview_failure_keeps_all_products_and_commit_failure_can_retry(system):
    agent, spatial, hydro, app = system
    fields, paths = products(system)
    session = task(agent, fields)
    with TestClient(app) as client:
        spatial.fail_preview = True
        assert client.delete(f"/api/v1/sessions/{session.session_id}").status_code == 409
        assert all(p.exists() for p in paths)
        spatial.fail_preview, spatial.fail_commit = False, True
        assert client.delete(f"/api/v1/sessions/{session.session_id}").status_code == 409
        assert agent.store.exists(session.session_id)
        assert not paths[6].exists() and paths[0].exists()
        blocked = client.post(f"/api/v1/sessions/{session.session_id}/messages", json={"text": "继续计算"})
        assert blocked.status_code == 409 and "重试删除" in blocked.text
        spatial.fail_commit = False
        response = client.delete(f"/api/v1/sessions/{session.session_id}")
    assert response.status_code == 200, response.text
    assert all(not p.exists() for p in paths)


def test_ledger_survives_state_failure_and_ignores_free_form_text(system):
    agent, _, _, _ = system
    session = agent.new_session()
    run_id = identifier("hydro-run", 10)
    record_created_product(agent.store, session.session_id, "start_hydro_run", {"data": {"run_id": run_id}})
    session.record_turn("user", identifier("hydro-run", 999))
    agent.store.save(session)
    assert session_products(agent.store, session.session_id) == {run_id}


def test_cross_run_comparison_protects_source_run_and_its_input(system):
    agent, spatial, hydro, _ = system
    fields, paths = products(system)
    other, _ = products(system, 2)
    put(hydro.settings.visualizations_root, identifier("viz", 7), "visualization.json",
        {"run_id": other["hydro_run_id"], "source_run_ids": [fields["hydro_run_id"], other["hydro_run_id"]],
         "source_visualization_ids": [identifier("viz", 1), identifier("viz", 2)], "status": "succeeded"})
    import asyncio
    result = asyncio.run(hydro.cleanup_hydro_products([fields["hydro_run_id"]], [], dry_run=False))["data"]
    assert fields["hydro_run_id"] in result["retained_ids"]
    assert all(p.exists() for p in paths)


def test_unsafe_ids_junctions_and_cycles_refused_before_changes(tmp_path, monkeypatch):
    name = identifier("basin", 1)
    path = put(tmp_path, name, "review.json", {})
    catalog = {name: Product(name, [(tmp_path, path)])}
    for unsafe in ["../data", str(tmp_path), "basin-../../data"]:
        with pytest.raises(ValueError):
            cleanup_products(catalog, [unsafe], [], dry_run=False)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "is_junction", lambda p: p == path)
        with pytest.raises(PathSafetyError):
            cleanup_products(catalog, [name], [], dry_run=False)
    nested = path / "nested"
    nested.mkdir()
    with monkeypatch.context() as patch:
        patch.setattr(Path, "is_junction", lambda p: p == nested)
        with pytest.raises(PathSafetyError):
            cleanup_products(catalog, [name], [], dry_run=False)
    second = identifier("rain", 1)
    catalog[name].dependencies.add(second)
    catalog[second] = Product(second, [], {name})
    with pytest.raises(ValueError, match="循环"):
        cleanup_products(catalog, [name, second], [], dry_run=True)
    assert path.exists()


def test_stale_ui_running_status_does_not_block_a_finished_run(system):
    agent, _, _, app = system
    fields, paths = products(system)
    session = task(agent, fields)
    session.state, session.hydro_run_status = "hydro_running", "running"
    agent.store.save(session)
    with TestClient(app) as client:
        response = client.delete(f"/api/v1/sessions/{session.session_id}")
    assert response.status_code == 200, response.text
    assert all(not p.exists() for p in paths)


@pytest.mark.asyncio
async def test_real_mcp_clients_delete_only_isolated_web_conversation(system, monkeypatch):
    import httpx
    from slope_evaluation_agent.config import ServerSettings
    from slope_evaluation_agent.mcp_clients import SpatialMCPClient, HydroMCPClient

    agent, spatial, hydro, app = system
    fields, paths = products(system)
    _, unrelated_paths = products(system, 2)
    session = task(agent, fields)
    monkeypatch.setenv("CQ_RUNS_ROOT", str(spatial.settings.runs_root))
    monkeypatch.setenv("CQ_EXCHANGE_ROOT", str(spatial.settings.exchange_root))
    monkeypatch.setenv("CQ_DATA_ROOT", str(spatial.settings.data_root))
    monkeypatch.setenv("WHS_HYDRO_ROOT", str(hydro.settings.hydro_root))
    agent.spatial = SpatialMCPClient(ServerSettings(args=["-m", "chongqing_spatial_mcp.server"]), agent.logger)
    agent.hydro = HydroMCPClient(ServerSettings(args=["-m", "watershed_hydro_mcp.server"]), agent.logger)
    async with agent:
        assert agent.spatial.connected and agent.hydro.connected
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://isolated") as client:
            response = await client.delete(f"/api/v1/sessions/{session.session_id}")
    assert response.status_code == 200, response.text
    assert all(not p.exists() for p in paths)
    assert all(p.exists() for p in unrelated_paths)
    assert not agent.store.exists(session.session_id)


@pytest.mark.parametrize("nested", [False, True])
def test_real_windows_junction_never_deletes_its_source(tmp_path, nested):
    winapi = pytest.importorskip("_winapi")
    name = identifier("basin", 1)
    source = tmp_path / "original_data"
    source.mkdir()
    sentinel = source / "source.tif"
    sentinel.write_bytes(b"preserve original source")
    path = tmp_path / name
    if nested:
        path.mkdir()
    link = path / "nested_link" if nested else path
    winapi.CreateJunction(str(source), str(link))
    try:
        assert link.is_junction()
        with pytest.raises(PathSafetyError):
            cleanup_products({name: Product(name, [(tmp_path, path)])}, [name], [], dry_run=False)
        assert sentinel.read_bytes() == b"preserve original source"
    finally:
        # Remove only the junction entry, never recurse into its target.
        link.rmdir()


def test_damaged_other_session_blocks_before_product_deletion(system):
    agent, _, _, app = system
    fields, paths = products(system)
    session = task(agent, fields)
    other = agent.new_session()
    agent.store.session_path(other.session_id).write_text("{broken", encoding="utf-8")
    with TestClient(app) as client:
        response = client.delete(f"/api/v1/sessions/{session.session_id}")
    assert response.status_code == 409
    assert agent.store.exists(session.session_id)
    assert all(p.exists() for p in paths)
