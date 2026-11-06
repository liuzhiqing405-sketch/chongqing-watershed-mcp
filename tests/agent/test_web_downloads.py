"""The download endpoints: what a browser may fetch, and what it must not learn.

Everything a page receives about a file comes from the agent's descriptor, and
the URL contains nothing but opaque ids. These tests drive the real app and
assert on the bytes: the right headers, the right body, a 404 that does not
distinguish "not yours" from "does not exist", and a refusal for anything shaped
like a path.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest

from slope_evaluation_agent.agent import SlopeEvaluationAgent
from slope_evaluation_agent.config import AgentSettings
from slope_evaluation_agent.intent import IntentParser
from slope_evaluation_agent.session_store import SessionStore
from slope_evaluation_agent.web import content_disposition, create_app

from helpers import Calls, hydro_handlers, spatial_handlers
from stubs import StubHydroClient, StubSpatialClient, envelope, hydro_server, spatial_server
from test_artifact_downloads import (
    CSV_BYTES,
    CSV_ID,
    RUN_ID,
    TIFF_BYTES,
    TIFF_ID,
    XLSX_BYTES,
    XLSX_ID,
    artifacts_payload,
    hydro_with_downloads,
    sha256_of,
)


pytestmark = [pytest.mark.unit]

VIZ_ID = "viz-abcdef123456"


def web_with_downloads(tmp_path: Path, calls: Calls, **overrides):
    handlers, table = hydro_with_downloads(calls, **overrides)
    settings = AgentSettings.model_validate({"sessions": {"root": str(tmp_path / "sessions")}})
    agent = SlopeEvaluationAgent(
        settings,
        spatial=StubSpatialClient(spatial_server(spatial_handlers(calls))),
        hydro=StubHydroClient(hydro_server(handlers, resources=table)),
        store=SessionStore(settings.sessions),
        parser=IntentParser())
    return create_app(settings=settings, agent=agent), agent


def a_client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                             base_url="http://agent.test")


async def a_session(client: httpx.AsyncClient) -> str:
    created = await client.post("/api/v1/sessions", json={"language": "zh"})
    assert created.status_code == 201, created.text
    return created.json()["session_id"]


def a_finished_run(agent: SlopeEvaluationAgent, session_id: str, *,
                   run_status: str = "succeeded") -> None:
    """Put the stored conversation past its run, so downloads are in scope."""
    session = agent.store.load(session_id)
    session.hydro_run_id = RUN_ID
    session.hydro_run_status = run_status
    session.state = "assessment_ready" if run_status == "succeeded" else "hydro_qa_failed"
    agent.store.save(session)


async def show_a_figure(client: httpx.AsyncClient, session_id: str) -> dict:
    response = await client.post(f"/api/v1/sessions/{session_id}/messages",
                                 json={"text": "看最小安全系数的图"})
    assert response.status_code == 200, response.text
    return response.json()["reply"]


class TestDispositionHeader:
    def test_a_non_ascii_name_survives_the_header(self):
        header = content_disposition("虚拟雨站_逐时雨量.csv")
        assert header.startswith("attachment; ")
        assert "filename*=UTF-8''" in header
        assert "\n" not in header

    def test_a_name_cannot_break_out_of_the_header(self):
        header = content_disposition('a"; x=1\r\nX-Evil: 1')
        assert "\r" not in header and "\n" not in header
        assert header.count('"') == 2


class TestDownloadEndpoint:
    @pytest.mark.asyncio
    async def test_a_session_downloads_its_own_artifact(self, tmp_path, calls):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            reply = await show_a_figure(client, session_id)
            artifact = reply["prepared_artifact"] or \
                reply["visualizations"][0]["artifacts"][0]
            assert artifact["artifact_id"] == TIFF_ID

            response = await client.get(
                f"/api/v1/sessions/{session_id}/artifacts/{TIFF_ID}/download")
        assert response.status_code == 200, response.text
        assert response.content == TIFF_BYTES
        assert response.headers["content-type"] == "image/tiff"
        assert response.headers["content-length"] == str(len(TIFF_BYTES))
        assert response.headers["etag"] == f'"{sha256_of(TIFF_BYTES)}"'
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "attachment" in response.headers["content-disposition"]

    @pytest.mark.asyncio
    async def test_the_descriptor_lists_what_can_be_downloaded(self, tmp_path, calls):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            reply = await show_a_figure(client, session_id)
        figure = reply["visualizations"][0]
        assert figure["preview"]["url"].endswith(f"/visualizations/{VIZ_ID}/image")
        formats = {item["format"] for item in figure["artifacts"]}
        assert formats == {"geotiff", "csv"}
        csv_row = next(item for item in figure["artifacts"] if item["format"] == "csv")
        assert csv_row["export_formats"] == []
        assert csv_row["format"] == "csv"
        assert csv_row["size_text"] == f"{len(CSV_BYTES)} B"
        assert csv_row["sha256_short"] == sha256_of(CSV_BYTES)[:12]
        assert csv_row["download_url"] == \
            f"/api/v1/sessions/{session_id}/artifacts/{CSV_ID}/download"

    @pytest.mark.asyncio
    async def test_another_sessions_artifact_is_a_404(self, tmp_path, calls):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            await show_a_figure(client, session_id)
            other = await a_session(client)
            a_finished_run(agent, other)
            response = await client.get(
                f"/api/v1/sessions/{other}/artifacts/{TIFF_ID}/download")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_an_unknown_artifact_looks_exactly_like_a_forbidden_one(
            self, tmp_path, calls):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            await show_a_figure(client, session_id)
            missing = await client.get(
                f"/api/v1/sessions/{session_id}/artifacts/artifact-ffffffffffff/download")
            forbidden = await client.get(
                f"/api/v1/sessions/{session_id}/artifacts/artifact-abcdef000000/download")
        assert missing.status_code == forbidden.status_code == 404
        assert missing.json() == forbidden.json()

    @pytest.mark.parametrize("candidate", [
        "../../../../etc/passwd",
        "..%2F..%2Fsecret.txt",
        "..\\..\\windows\\win.ini",
        "D:",
        "C:\\Windows\\win.ini",
        "artifact-" + "a" * 400,
        "package_enhanced_minimum_safety_factor.tif",
    ])
    @pytest.mark.asyncio
    async def test_anything_shaped_like_a_path_is_refused(self, tmp_path, calls, candidate):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            await show_a_figure(client, session_id)
            response = await client.get(
                f"/api/v1/sessions/{session_id}/artifacts/{candidate}/download")
        assert response.status_code in {404, 422}

    @pytest.mark.asyncio
    async def test_an_unfinished_file_is_not_downloadable(self, tmp_path, calls):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            await show_a_figure(client, session_id)
            session = agent.store.load(session_id)
            entry = session.visualizations[0]
            entry["artifacts"][0]["ready"] = False
            entry["artifacts"][0]["formats"] = []
            session.visualizations = [entry]
            agent.store.save(session)
            response = await client.get(
                f"/api/v1/sessions/{session_id}/artifacts/{TIFF_ID}/download")
        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_a_qa_failed_file_downloads_with_its_verdict(self, tmp_path, calls):
        app, agent = web_with_downloads(
            tmp_path, calls,
            list_visualization_artifacts=envelope(data=artifacts_payload(qa_failed=True)))
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id, run_status="qa_failed")
            reply = await show_a_figure(client, session_id)
            row = reply["visualizations"][0]["artifacts"][0]
            assert row["diagnostic_only"] is True
            response = await client.get(
                f"/api/v1/sessions/{session_id}/artifacts/{row['artifact_id']}/download")
        assert response.status_code == 200
        assert response.headers["x-artifact-qa"] == "failed"

    @pytest.mark.asyncio
    async def test_no_response_carries_a_path(self, tmp_path, calls):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            reply = await show_a_figure(client, session_id)
            page = await client.get(f"/api/v1/sessions/{session_id}")
            listing = await client.get(f"/api/v1/sessions/{session_id}/artifacts")
        for body in (json.dumps(reply), page.text, listing.text):
            assert "internal_path" not in body
            assert "D:\\\\" not in body and "D:/" not in body
            assert "/project/" not in body

    @pytest.mark.asyncio
    async def test_a_deleted_session_loses_its_downloads_immediately(self, tmp_path, calls):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            await show_a_figure(client, session_id)
            deleted = await client.delete(f"/api/v1/sessions/{session_id}")
            assert deleted.status_code == 200
            response = await client.get(
                f"/api/v1/sessions/{session_id}/artifacts/{TIFF_ID}/download")
        assert response.status_code == 404
        # Cleanup is stubbed here; real product deletion is verified separately.
        assert not (agent.store.root / session_id).exists()


class TestStyle:
    @pytest.mark.asyncio
    async def test_a_large_file_streams_and_never_enters_the_session(self, tmp_path, calls):
        """A big file is served from disk in chunks; nothing about it is stored.

        The proof has two halves: the response body is the file, and the session
        file — which is what a browser reload reads — contains no base64 of it.
        """
        payload = tmp_path / "big" / "package.out"
        payload.parent.mkdir(parents=True, exist_ok=True)
        raw = bytes(range(256)) * 8192  # 2 MiB
        payload.write_bytes(raw)
        app, agent = web_with_downloads(
            tmp_path, calls,
            locate_artifact=envelope(data={
                "artifact_id": TIFF_ID, "run_id": RUN_ID, "title": "package.out",
                "filename": "package.out", "internal_path": str(payload),
                "sha256": sha256_of(raw), "size_bytes": len(raw),
                "media_type": "application/octet-stream", "format": "out",
                "role": "native_result", "qa_status": "passed", "ready": True}))
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            await show_a_figure(client, session_id)
            session = agent.store.load(session_id)
            entry = session.visualizations[0]
            entry["artifacts"][0].update({"sha256": sha256_of(raw), "size_bytes": len(raw),
                                          "media_type": "application/octet-stream",
                                          "format": "out", "role": "native_result"})
            session.visualizations = [entry]
            agent.store.save(session)
            agent.settings.download.resource_read_limit_bytes = 1024

            download = await agent.download_artifact(session, TIFF_ID)
            assert download.channel == "local"
            assert not isinstance(download.chunks, (bytes, bytearray))

            response = await client.get(
                f"/api/v1/sessions/{session_id}/artifacts/{TIFF_ID}/download")
        assert response.status_code == 200
        assert response.content == raw
        assert response.headers["content-length"] == str(len(raw))
        stored = (agent.store.root / session_id / "session.json").read_text(encoding="utf-8")
        assert base64.b64encode(raw[:32]).decode() not in stored
        assert str(payload) not in stored


class TestExportEndpoint:
    @pytest.mark.asyncio
    async def test_excel_is_prepared_on_request_and_then_downloadable(
            self, tmp_path, calls):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            await show_a_figure(client, session_id)
            response = await client.post(
                f"/api/v1/sessions/{session_id}/artifacts/{CSV_ID}/exports",
                json={"format": "xlsx"})
            assert response.status_code == 200, response.text
            artifact = response.json()["artifact"]
            downloaded = await client.get(artifact["download_url"])
        assert artifact["artifact_id"] == XLSX_ID
        assert artifact["format"] == "xlsx"
        assert downloaded.status_code == 200, downloaded.text
        assert downloaded.content == XLSX_BYTES

    @pytest.mark.asyncio
    async def test_a_swmm_csv_explicitly_converted_to_excel_is_authorized(
            self, tmp_path, calls):
        swmm_csv_id = "artifact-acdeff123456"
        swmm_csv = b"parameter,value\nwidth,12.5\n"
        resources = {
            f"hydro-artifact://{RUN_ID}/{TIFF_ID}": TIFF_BYTES,
            f"hydro-artifact://{RUN_ID}/{CSV_ID}": CSV_BYTES,
            f"hydro-artifact://{RUN_ID}/{swmm_csv_id}": swmm_csv,
            f"hydro-artifact://{RUN_ID}/{XLSX_ID}": XLSX_BYTES,
        }
        swmm = envelope(data={
            "run_id": RUN_ID, "complete": False,
            "missing_required": [".inp", ".rpt", ".out"],
            "artifacts": [{
                "artifact_id": swmm_csv_id,
                "title": "auto_swmm_parameter_review.csv",
                "format": "csv", "role": "source_table", "media_type": "text/csv",
                "size_bytes": len(swmm_csv), "sha256": sha256_of(swmm_csv),
                "qa_status": "passed", "stage_id": "build", "ready": True,
                "formats": ["original"], "exportable": ["xlsx"],
            }],
        })
        app, agent = web_with_downloads(
            tmp_path, calls, resources=resources,
            list_swmm_model_artifacts=swmm)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            listed = await client.get(f"/api/v1/sessions/{session_id}/artifacts")
            assert listed.status_code == 200
            response = await client.post(
                f"/api/v1/sessions/{session_id}/artifacts/{swmm_csv_id}/exports",
                json={"format": "xlsx"})
            assert response.status_code == 200, response.text
            downloaded = await client.get(response.json()["artifact"]["download_url"])
        assert downloaded.status_code == 200, downloaded.text
        assert downloaded.content == XLSX_BYTES

    @pytest.mark.asyncio
    async def test_all_process_sources_are_listed_and_csv_downloads_as_csv(
            self, tmp_path, calls):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            listing = await client.get(f"/api/v1/sessions/{session_id}/artifacts")
            assert listing.status_code == 200, listing.text
            rows = listing.json()["process_artifacts"]
            csv_row = next(item for item in rows if item["artifact_id"] == CSV_ID)
            downloaded = await client.get(csv_row["download_url"])
        assert csv_row["format"] == "csv"
        assert csv_row["stage_id"] == "safety"
        assert csv_row["export_formats"] == []
        assert downloaded.status_code == 200
        assert downloaded.content == CSV_BYTES
        assert downloaded.headers["content-type"].startswith("text/csv")
        assert ".csv" in downloaded.headers["content-disposition"].lower()

    @pytest.mark.asyncio
    async def test_a_format_the_artifact_cannot_produce_is_rejected(self, tmp_path, calls):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            await show_a_figure(client, session_id)
            response = await client.post(
                f"/api/v1/sessions/{session_id}/artifacts/{CSV_ID}/exports",
                json={"format": "csv"})
        assert response.status_code == 422  # the request body names no such format

    @pytest.mark.asyncio
    async def test_exporting_an_artifact_of_another_session_is_a_404(self, tmp_path, calls):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            await show_a_figure(client, session_id)
            other = await a_session(client)
            response = await client.post(
                f"/api/v1/sessions/{other}/artifacts/{CSV_ID}/exports",
                json={"format": "xlsx"})
        assert response.status_code == 404


class TestSwmmBundleEndpoint:
    @pytest.mark.asyncio
    async def test_the_bundle_is_created_for_this_sessions_run(self, tmp_path, calls):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            response = await client.post(
                f"/api/v1/sessions/{session_id}/runs/{RUN_ID}/swmm-bundle")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["artifact"]["artifact_id"] == "artifact-bundle00001"
        assert body["artifact"]["format"] == "zip"
        assert body["swmm"]["bundle"]["artifact_id"] == "artifact-bundle00001"

    @pytest.mark.asyncio
    async def test_another_runs_id_is_refused(self, tmp_path, calls):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            response = await client.post(
                f"/api/v1/sessions/{session_id}/runs/hydro-run-ffffffffffff/swmm-bundle")
        assert response.status_code == 404


class TestRestoreEndpoints:
    @pytest.mark.asyncio
    async def test_a_reloaded_page_can_list_its_figures_and_files(self, tmp_path, calls):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            await show_a_figure(client, session_id)
            response = await client.get(f"/api/v1/sessions/{session_id}/artifacts")
        assert response.status_code == 200, response.text
        body = response.json()
        figure = body["visualizations"][0]
        assert figure["visualization_id"] == VIZ_ID
        assert [item["artifact_id"] for item in figure["artifacts"]] == [TIFF_ID, CSV_ID]
        # The preview is fetched from the service again, not stored in the page.
        assert figure["preview"]["url"] == \
            f"/api/v1/sessions/{session_id}/visualizations/{VIZ_ID}/image"

    @pytest.mark.asyncio
    async def test_the_preview_comes_back_after_a_reload(self, tmp_path, calls):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            await show_a_figure(client, session_id)
            first = await client.get(
                f"/api/v1/sessions/{session_id}/visualizations/{VIZ_ID}/image")
            second = await client.get(
                f"/api/v1/sessions/{session_id}/visualizations/{VIZ_ID}/image")
        assert first.status_code == second.status_code == 200
        assert first.headers["content-type"] == "image/png"
        assert first.content.startswith(b"\x89PNG")
        # Cached under a name derived from the figure: a reload does not
        # accumulate copies.
        assert len(list(agent.store.root.parent.joinpath("web_images").iterdir())) == 1

    @pytest.mark.asyncio
    async def test_a_figure_of_another_session_is_not_served(self, tmp_path, calls):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            await show_a_figure(client, session_id)
            other = await a_session(client)
            response = await client.get(
                f"/api/v1/sessions/{other}/visualizations/{VIZ_ID}/image")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_the_listing_includes_the_swmm_model_section(self, tmp_path, calls):
        app, agent = web_with_downloads(tmp_path, calls)
        async with agent, a_client(app) as client:
            session_id = await a_session(client)
            a_finished_run(agent, session_id)
            await show_a_figure(client, session_id)
            response = await client.get(f"/api/v1/sessions/{session_id}/artifacts")
        swmm = response.json()["swmm"]
        assert swmm is not None
        assert {item["format"] for item in swmm["artifacts"]} == {"inp", "out"}
        assert swmm["complete"] is True
