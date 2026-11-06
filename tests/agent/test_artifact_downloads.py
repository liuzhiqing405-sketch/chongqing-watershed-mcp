"""Downloads: what a conversation may reach, and how it gets it.

The session is the authority. These tests drive a real agent over in-process
stub servers and check three things that decide whether the download feature is
safe: an artifact id the session was never given is refused; a file travels
without its location ever entering the reply, the session file or a browser
payload; and what the service says about a file (its digest, its size, its
quality verdict) is what the caller is held to.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from slope_evaluation_agent import artifacts as artifacts_module
from slope_evaluation_agent.config import AgentSettings
from slope_evaluation_agent.intent import Intent
from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.session_store import SessionStore
from slope_evaluation_agent.agent import SlopeEvaluationAgent
from slope_evaluation_agent.intent import IntentParser

from helpers import PRELUDE, Calls, agent_with, converse, hydro_handlers, prepared_agent
from stubs import (
    hydro_server,
    StubHydroClient,
    envelope,
    run_result_payload,
    run_status_payload,
    visualization_artifacts_payload,
    visualization_payload,
)


pytestmark = [pytest.mark.unit]


RUN_ID = "hydro-run-abcdef123456"
VIZ_ID = "viz-abcdef123456"
TIFF_ID = "artifact-abcdef123456"
CSV_ID = "artifact-abcdef654321"
XLSX_ID = "artifact-derived0001"

#: A tiny "raster" and "table". The bytes are arbitrary on purpose: what the
#: agent must preserve is the identity of the file, not its content.
TIFF_BYTES = b"II*\x00" + bytes(range(64))
CSV_BYTES = b"datetime_beijing,representative_fs\n2024-07-30 00:00:00,0.2752\n"
XLSX_BYTES = b"PK\x03\x04" + bytes(range(32))


def sha256_of(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def artifacts_payload(*, qa_failed: bool = False, tiff_sha: str | None = None,
                      csv_sha: str | None = None) -> dict:
    """The public listing of one figure's files, with digests that match bytes."""
    payload = visualization_artifacts_payload(qa_failed=qa_failed)
    payload["artifacts"][0]["sha256"] = tiff_sha or sha256_of(TIFF_BYTES)
    payload["artifacts"][0]["size_bytes"] = len(TIFF_BYTES)
    payload["artifacts"][1]["sha256"] = csv_sha or sha256_of(CSV_BYTES)
    payload["artifacts"][1]["size_bytes"] = len(CSV_BYTES)
    payload["run_id"] = RUN_ID
    return payload


def hydro_with_downloads(calls: Calls, *, resources: dict[str, bytes] | None = None,
                         **overrides):
    """Stub hydro handlers that can answer every download question."""
    # The two files the stub figures point at are readable by default: a test
    # that wants a missing or tampered resource passes its own table.
    table = dict(resources) if resources is not None else {
        f"hydro-artifact://{RUN_ID}/{TIFF_ID}": TIFF_BYTES,
        f"hydro-artifact://{RUN_ID}/{CSV_ID}": CSV_BYTES,
        f"hydro-artifact://{RUN_ID}/{XLSX_ID}": XLSX_BYTES}
    tiff_sha = sha256_of(table.get(f"hydro-artifact://{RUN_ID}/{TIFF_ID}", TIFF_BYTES))
    csv_sha = sha256_of(table.get(f"hydro-artifact://{RUN_ID}/{CSV_ID}", CSV_BYTES))
    handlers = hydro_handlers(calls, **{
        "list_visualization_artifacts": envelope(
            data=artifacts_payload(tiff_sha=tiff_sha, csv_sha=csv_sha)),
        "list_hydro_artifacts": envelope(data={
            "artifacts": artifacts_payload(tiff_sha=tiff_sha,
                                             csv_sha=csv_sha)["artifacts"]}),
        "prepare_artifact_export": envelope(data={
            "reused": False,
            "artifact": {
                "artifact_id": XLSX_ID,
                "title": "package_enhanced_safety_factor_summary.xlsx",
                "format": "xlsx", "role": "spreadsheet",
                "media_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "size_bytes": len(XLSX_BYTES), "sha256": sha256_of(XLSX_BYTES),
                "qa_status": "passed",
                "stage_id": "export", "ready": True, "formats": ["original"],
                "exportable": [], "producer": "watershed_hydro_mcp.workbook"},
            "source_artifact_id": CSV_ID}),
        "list_swmm_model_artifacts": envelope(data={
            "run_id": RUN_ID, "complete": True, "missing_required": [],
            "artifacts": [
                {"artifact_id": "artifact-inp00000001", "title": "package.inp",
                 "format": "inp", "role": "model_input", "media_type": "text/plain",
                 "size_bytes": 1024, "sha256": "f" * 64, "qa_status": "passed",
                 "stage_id": "build", "ready": True, "formats": ["original"],
                 "exportable": []},
                {"artifact_id": "artifact-out00000001", "title": "package.out",
                 "format": "out", "role": "native_result",
                 "media_type": "application/octet-stream",
                 "size_bytes": 2048, "sha256": "0" * 64, "qa_status": "passed",
                 "stage_id": "swmm", "ready": True, "formats": ["original"],
                 "exportable": []}]}),
        "prepare_swmm_model_bundle": envelope(data={
            "reused": False, "complete": True, "missing_required": [],
            "artifact": {
                "artifact_id": "artifact-bundle00001",
                "title": "swmm_model_hydro-run-abcdef123456_20260929T120000Z_1234abcd.zip",
                "format": "zip", "role": "package", "media_type": "application/zip",
                "size_bytes": 8192, "sha256": "1" * 64, "qa_status": "passed",
                "stage_id": "export", "ready": True, "formats": ["original"],
                "exportable": []},
            "members": [], "warnings": []}),
        **overrides,
    })
    return handlers, table


def an_agent(build_agent, calls, *, resources=None, **overrides):
    """An agent over the download-capable stubs, and a finished conversation."""
    handlers, table = hydro_with_downloads(calls, resources=resources, **overrides)
    agent = build_agent(hydro=hydro_server(handlers, resources=table))
    return agent, a_ready_session(agent)


async def say(agent, session, *texts):
    """Drive turns on an agent the test holds open.

    The in-process stub servers can only be connected once per event loop, so a
    test that both talks and downloads keeps one connection for the whole
    scenario — the way the CLI and the Web app hold theirs.
    """
    replies = []
    for text in texts:
        replies.append(await agent.handle(session, text))
    return replies


async def show_a_figure(agent, session):
    """Render the safety-factor figure, on an open agent."""
    return await say(agent, session, "看最小安全系数的图")


def a_ready_session(agent, *, run_status: str = "succeeded") -> AgentSession:
    """A conversation whose run has finished, so figures and downloads are open."""
    session = agent.new_session()
    session.hydro_run_id = RUN_ID
    session.hydro_run_status = run_status
    session.state = "assessment_ready" if run_status == "succeeded" else "hydro_qa_failed"
    return session


class TestIntent:
    """The rules must place a download without any model involved."""

    @pytest.mark.asyncio
    async def test_download_phrasing_is_recognised(self):
        parser = IntentParser()
        session = AgentSession(session_id="session-abcdef123456", state="new")
        parsed = await parser.parse("下载刚才的安全系数 TIFF", session)
        assert parsed.intent is Intent.DOWNLOAD_ARTIFACT
        assert parsed.artifact_format == "geotiff"

    @pytest.mark.asyncio
    async def test_excel_phrasing_asks_for_a_derived_file(self):
        parser = IntentParser()
        session = AgentSession(session_id="session-abcdef123456", state="new")
        parsed = await parser.parse("导出这张图的数据到 Excel", session)
        assert parsed.intent is Intent.DOWNLOAD_ARTIFACT
        assert parsed.artifact_format == "xlsx"

    @pytest.mark.asyncio
    async def test_a_question_about_downloading_is_not_a_download(self):
        parser = IntentParser()
        session = AgentSession(session_id="session-abcdef123456", state="new")
        parsed = await parser.parse("怎么下载 Excel？", session)
        assert parsed.intent is not Intent.DOWNLOAD_ARTIFACT

    @pytest.mark.asyncio
    async def test_asking_for_a_figure_is_still_asking_for_a_figure(self):
        parser = IntentParser()
        session = AgentSession(session_id="session-abcdef123456", state="new")
        parsed = await parser.parse("看最小安全系数的图", session)
        assert parsed.intent is Intent.SHOW_VISUALIZATION


class TestFigureDescriptors:
    @pytest.mark.asyncio
    async def test_a_rendered_figure_carries_its_real_files(self, build_agent, calls):
        agent, session = an_agent(build_agent, calls)
        async with agent:
            replies = await show_a_figure(agent, session)
        figure = replies[-1].visualizations[0]
        assert figure.visualization_id == VIZ_ID
        assert [item.artifact_id for item in figure.artifacts] == [TIFF_ID, CSV_ID]
        assert figure.artifacts[0].format == "geotiff"
        assert figure.artifacts[0].units == "dimensionless"
        # The reply still carries the image for every existing client.
        assert len(replies[-1].images) == 1
        # And the session remembers the same descriptors, ids only.
        stored = agent.visualizations(session)[0]
        assert [item.artifact_id for item in stored.artifacts] == [TIFF_ID, CSV_ID]

    @pytest.mark.asyncio
    async def test_nothing_about_the_download_holds_a_path(self, build_agent, calls):
        agent, session = an_agent(build_agent, calls)
        async with agent:
            await show_a_figure(agent, session)
        serialized = json.dumps(session.model_dump(mode="json"), ensure_ascii=False)
        assert "internal_path" not in serialized
        assert "D:\\" not in serialized and "D:/" not in serialized

    @pytest.mark.asyncio
    async def test_a_quality_gate_failure_marks_the_download(self, build_agent, calls):
        handlers, table = hydro_with_downloads(calls)
        handlers["list_visualization_artifacts"] = lambda: envelope(
            data=artifacts_payload(qa_failed=True))
        agent = build_agent(hydro=hydro_server(handlers, resources=table))
        session = a_ready_session(agent)
        async with agent:
            await show_a_figure(agent, session)
        figure = agent.visualizations(session)[0]
        assert figure.qa_failed is True
        assert all(item.diagnostic_only for item in figure.artifacts)

    @pytest.mark.asyncio
    async def test_a_failed_listing_does_not_lose_the_figure(self, build_agent, calls):
        handlers, table = hydro_with_downloads(
            calls, list_visualization_artifacts=lambda: (_ for _ in ()).throw(
                AssertionError("never")))
        handlers["list_visualization_artifacts"] = envelope(data={})
        agent = build_agent(hydro=hydro_server(handlers, resources=table))
        session = a_ready_session(agent)
        async with agent:
            replies = await show_a_figure(agent, session)
        assert len(replies[-1].images) == 1  # the picture still arrives
        assert replies[-1].visualizations[0].warnings


class TestAuthorisation:
    @pytest.mark.asyncio
    async def test_a_session_cannot_download_an_artifact_it_was_never_given(
            self, build_agent, calls):
        agent, session = an_agent(build_agent, calls)
        async with agent:
            await show_a_figure(agent, session)
            # A second conversation on the same run that was never shown this
            # figure: the id alone is not enough.
            stranger = a_ready_session(agent)
            with pytest.raises(artifacts_module.DownloadError) as error:
                await agent.download_artifact(stranger, TIFF_ID)
        assert error.value.code == "not_found" and error.value.http_status == 404

    @pytest.mark.asyncio
    async def test_an_artifact_of_another_run_is_refused(self, build_agent, calls):
        agent, session = an_agent(build_agent, calls)
        async with agent:
            await show_a_figure(agent, session)
            session.hydro_run_id = "hydro-run-ffffffffffff"
            with pytest.raises(artifacts_module.DownloadError) as error:
                await agent.download_artifact(session, TIFF_ID)
        assert error.value.code == "not_found"

    @pytest.mark.asyncio
    async def test_a_file_that_is_not_ready_cannot_be_downloaded(self, build_agent, calls):
        agent, session = an_agent(build_agent, calls)
        async with agent:
            await show_a_figure(agent, session)
            entry = session.visualizations[0]
            entry["artifacts"][0]["ready"] = False
            entry["artifacts"][0]["formats"] = []
            session.visualizations = [entry]
            with pytest.raises(artifacts_module.DownloadError) as error:
                await agent.download_artifact(session, TIFF_ID)
        assert error.value.code == "not_ready" and error.value.http_status == 409

    @pytest.mark.asyncio
    async def test_a_path_shaped_id_is_not_a_lookup(self, build_agent, calls):
        agent, session = an_agent(build_agent, calls)
        await show_a_figure(agent, session)
        for candidate in ("../../etc/passwd", "D:\\secret.txt", "artifact-..\\..\\x"):
            with pytest.raises(artifacts_module.DownloadError):
                await agent.download_artifact(session, candidate)


class TestChannels:
    @pytest.mark.asyncio
    async def test_a_small_file_travels_through_the_resource_channel(self, build_agent, calls):
        resources = {f"hydro-artifact://{RUN_ID}/{TIFF_ID}": TIFF_BYTES,
                     f"hydro-artifact://{RUN_ID}/{CSV_ID}": CSV_BYTES}
        agent, session = an_agent(build_agent, calls, resources=resources)
        async with agent:
            await show_a_figure(agent, session)
            download = await agent.download_artifact(session, TIFF_ID)
        assert download.channel == "resource"
        assert download.media_type == "image/tiff"
        assert download.filename == "package_enhanced_minimum_safety_factor.tif"
        served = b"".join([chunk async for chunk in download.chunks])
        assert served == TIFF_BYTES
        assert download.size_bytes == len(TIFF_BYTES)
        assert download.sha256 == sha256_of(TIFF_BYTES)

    @pytest.mark.asyncio
    async def test_bytes_that_do_not_match_the_digest_are_refused(self, build_agent, calls):
        """A resource that returns different bytes is not the artifact."""
        resources = {f"hydro-artifact://{RUN_ID}/{TIFF_ID}": TIFF_BYTES + b"tampered"}
        agent, session = an_agent(build_agent, calls, resources=resources)
        async with agent:
            await show_a_figure(agent, session)
            with pytest.raises(Exception) as error:  # noqa: B017 - a mapped MCP error
                await agent.download_artifact(session, TIFF_ID)
        assert "摘要" in str(error.value) or "长度" in str(error.value)

    @pytest.mark.asyncio
    async def test_a_large_file_streams_from_disk_with_a_locator(self, build_agent, calls,
                                                                 tmp_path):
        """Above the resource ceiling the same-machine adapter is used, and the
        bytes still arrive — the path never leaves the agent."""
        payload = tmp_path / "big.out"
        raw = b"\x00" * 4096 + b"tail"
        payload.write_bytes(raw)
        resources = {f"hydro-artifact://{RUN_ID}/{TIFF_ID}": TIFF_BYTES}
        agent, session = an_agent(
            build_agent, calls, resources=resources,
            locate_artifact=envelope(data={
                "artifact_id": TIFF_ID, "run_id": RUN_ID, "title": "package.out",
                "filename": "package.out", "internal_path": str(payload),
                "sha256": sha256_of(raw), "size_bytes": len(raw),
                "media_type": "application/octet-stream", "format": "out",
                "role": "native_result", "qa_status": "passed", "ready": True}))
        async with agent:
            await show_a_figure(agent, session)
            # Point the descriptor at the same digests the locator will report,
            # and make it too large for the resource channel.
            entry = session.visualizations[0]
            entry["artifacts"][0].update({"sha256": sha256_of(raw),
                                          "size_bytes": len(raw),
                                          "ready": True, "formats": ["original"]})
            session.visualizations = [entry]
            agent.settings.download.resource_read_limit_bytes = 1024
            download = await agent.download_artifact(session, TIFF_ID)
            assert download.channel == "local"
            assert b"".join([chunk async for chunk in download.chunks]) == raw
        serialized = json.dumps(session.model_dump(mode="json"), ensure_ascii=False)
        assert str(payload) not in serialized

    @pytest.mark.asyncio
    async def test_a_locator_pointing_at_another_file_is_refused(self, build_agent, calls,
                                                                 tmp_path):
        payload = tmp_path / "elsewhere.bin"
        payload.write_bytes(b"different bytes entirely")
        agent, session = an_agent(
            build_agent, calls,
            locate_artifact=envelope(data={
                "artifact_id": TIFF_ID, "run_id": RUN_ID, "title": "x.tif",
                "filename": "x.tif", "internal_path": str(payload),
                "sha256": "9" * 64, "size_bytes": payload.stat().st_size,
                "media_type": "image/tiff", "format": "geotiff",
                "role": "source_raster", "qa_status": "passed", "ready": True}))
        async with agent:
            await show_a_figure(agent, session)
            agent.settings.download.resource_read_limit_bytes = 0  # force the adapter
            with pytest.raises(artifacts_module.DownloadError) as error:
                await agent.download_artifact(session, TIFF_ID)
        assert error.value.code == "integrity"

    @pytest.mark.asyncio
    async def test_a_file_over_the_policy_ceiling_is_refused_before_reading(
            self, build_agent, calls):
        agent, session = an_agent(build_agent, calls)
        async with agent:
            await show_a_figure(agent, session)
            agent.settings.download.max_file_bytes = 64
            with pytest.raises(artifacts_module.DownloadError) as error:
                await agent.download_artifact(session, TIFF_ID)
        assert error.value.code == "too_large" and error.value.http_status == 413


class TestDownloadTurns:
    @pytest.mark.asyncio
    async def test_the_deterministic_route_downloads_without_any_model(
            self, build_agent, calls):
        resources = {f"hydro-artifact://{RUN_ID}/{TIFF_ID}": TIFF_BYTES,
                     f"hydro-artifact://{RUN_ID}/{CSV_ID}": CSV_BYTES}
        agent, session = an_agent(build_agent, calls, resources=resources)
        async with agent:
            replies = await say(agent, session, "看最小安全系数的图",
                                "下载刚才的安全系数 TIFF")
            assert replies[-1].kind == "result", replies[-1].text
            artifact = replies[-1].facts["artifact"]
            assert artifact["artifact_id"] == TIFF_ID
            assert "package_enhanced_minimum_safety_factor.tif" in replies[-1].text
            # The download really works from the recorded session.
            download = await agent.download_artifact(replies[-1].session, TIFF_ID)
            assert b"".join([chunk async for chunk in download.chunks]) == TIFF_BYTES

    @pytest.mark.asyncio
    async def test_excel_is_derived_and_then_remembered(self, build_agent, calls):
        agent, session = an_agent(build_agent, calls)
        async with agent:
            replies = await say(agent, session, "看最小安全系数的图",
                                "导出这张图的数据到 Excel")
        assert "xlsx" in replies[-1].facts["artifact"]["title"]
        assert replies[-1].facts["derived_from"] == CSV_ID
        ids = [item.artifact_id for item in agent.visualizations(session)[0].artifacts]
        assert "artifact-derived0001" in ids
        # Which means it can be downloaded like anything else the session holds.
        descriptor = agent.artifacts.authorize(session, "artifact-derived0001")
        assert descriptor.ready

    @pytest.mark.asyncio
    async def test_a_format_nothing_has_is_explained_not_guessed(self, build_agent, calls):
        """A ZIP exists only once a package has been made; before that the reply
        says what there is instead of offering a file that does not exist."""
        agent, session = an_agent(build_agent, calls)
        async with agent:
            await show_a_figure(agent, session)
            reply = await agent.handle(session, "下载这张图的 ZIP 压缩包")
        assert reply.kind == "question"
        assert "package.inp" in reply.text or "package.out" in reply.text

    @pytest.mark.asyncio
    async def test_no_figure_yet_is_a_question_not_a_failure(self, build_agent, calls):
        agent, session = an_agent(build_agent, calls)
        async with agent:
            reply = await agent.handle(session, "下载刚才的成果")
        assert reply.kind == "question"
        assert reply.ok

    @pytest.mark.asyncio
    async def test_a_download_before_a_run_is_blocked_by_the_gate(self, build_agent, calls):
        agent, _ready = an_agent(build_agent, calls)
        async with agent:
            reply = await agent.handle(agent.new_session(), "下载刚才的安全系数 TIFF")
        assert reply.kind == "question"
        assert not reply.ok
        assert "还没有可下载的成果" in reply.text


class TestSwmmSection:
    @pytest.mark.asyncio
    async def test_a_shown_result_learns_the_model_files(self, build_agent, calls):
        agent, session = an_agent(build_agent, calls)
        async with agent:
            reply = await agent.show_result(session)
        model = session.swmm_model or {}
        assert [item["artifact_id"] for item in model["artifacts"]] == \
            ["artifact-inp00000001", "artifact-out00000001"]
        assert model["complete"] is True
        # The result text says the model files are there, and the section is in
        # the session as descriptors, not paths.
        assert reply.facts['swmm']['artifacts']
        assert len(reply.text) < 250  # model files remain in the collapsible UI section
        assert "internal_path" not in json.dumps(model, ensure_ascii=False)

    @pytest.mark.asyncio
    async def test_a_model_file_can_be_downloaded_by_format(self, build_agent, calls):
        agent, session = an_agent(build_agent, calls)
        async with agent:
            await agent.refresh_swmm_model(session)
            reply = await agent.handle(session, "下载这次运行的 OUT 文件")
        assert reply.facts["artifact"]["artifact_id"] == "artifact-out00000001"

    @pytest.mark.asyncio
    async def test_the_model_package_is_prepared_and_remembered(self, build_agent, calls):
        agent, session = an_agent(build_agent, calls)
        async with agent:
            reply = await agent.handle(session, "下载 SWMM 模型包")
        assert reply.facts["artifact"]["artifact_id"] == "artifact-bundle00001"
        assert (session.swmm_model or {}).get("bundle", {}).get("artifact_id") == \
            "artifact-bundle00001"
        assert session.swmm_model["bundle_complete"] is True

    @pytest.mark.asyncio
    async def test_the_package_is_downloadable_once_it_exists(self, build_agent, calls,
                                                              tmp_path):
        agent, session = an_agent(build_agent, calls)
        async with agent:
            await agent.handle(session, "下载 SWMM 模型包")
        descriptor = agent.artifacts.authorize(session, "artifact-bundle00001")
        assert descriptor.format == "zip" and descriptor.ready


class TestOlderSessions:
    """A conversation written by the previous release must still open.

    It stored only figure ids, and its model section does not exist at all. The
    page must render it, and it must simply have nothing to download — not an
    error, and never a guess at what those ids once pointed to.
    """

    @pytest.mark.asyncio
    async def test_a_session_without_the_new_fields_opens_and_lists_nothing(
            self, build_agent, calls):
        agent, _session = an_agent(build_agent, calls)
        legacy = {
            "schema_version": 2, "session_id": "session-abcdef123456", "state": "new",
            "created_at": "2026-09-01T00:00:00+00:00",
            "updated_at": "2026-09-01T00:00:00+00:00",
            # Only the ids: no descriptors, no model section.
            "visualizations": [{"visualization_id": VIZ_ID, "kind": "safety_minimum_factor"}],
        }
        directory = agent.store.root / "session-abcdef123456"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "session.json").write_text(json.dumps(legacy), encoding="utf-8")
        session = agent.store.load("session-abcdef123456")

        kept = agent.visualizations(session)
        assert [item.visualization_id for item in kept] == [VIZ_ID]
        assert kept[0].artifacts == []          # the id survives, the files are unknown
        assert agent.artifacts.descriptors(session) == []
        with pytest.raises(artifacts_module.DownloadError) as error:
            agent.artifacts.authorize(session, TIFF_ID)
        assert error.value.code == "not_found"

    @pytest.mark.asyncio
    async def test_a_figure_learned_before_the_download_release_can_still_be_filled_in(
            self, build_agent, calls):
        """Asking for a figure's files refreshes an entry that has none."""
        agent, session = an_agent(build_agent, calls)
        session.visualizations = [{"visualization_id": VIZ_ID,
                                   "kind": "safety_minimum_factor"}]
        async with agent:
            reply = await agent.handle(session, "下载这张图的 GeoTIFF")
        assert reply.facts["artifact"]["artifact_id"] == TIFF_ID


class TestCommandLineDelivery:
    """The CLI has no button to click, so it must deliver the file itself."""

    @pytest.mark.asyncio
    async def test_a_download_turn_saves_the_file_beside_the_session(self, build_agent, calls):
        from slope_evaluation_agent.cli import save_download

        resources = {f"hydro-artifact://{RUN_ID}/{TIFF_ID}": TIFF_BYTES,
                     f"hydro-artifact://{RUN_ID}/{CSV_ID}": CSV_BYTES}
        agent, session = an_agent(build_agent, calls, resources=resources)
        async with agent:
            await say(agent, session, "看最小安全系数的图")
            reply = await agent.handle(session, "下载刚才的安全系数 TIFF")
            path = await save_download(agent, session, reply, agent.settings)
        assert path, "命令行下这句请求必须真的落盘"
        assert Path(path).read_bytes() == TIFF_BYTES

    @pytest.mark.asyncio
    async def test_an_ordinary_reply_saves_nothing(self, build_agent, calls):
        from slope_evaluation_agent.cli import save_download

        agent, session = an_agent(build_agent, calls)
        async with agent:
            reply = await agent.handle(session, "看最小安全系数的图")
            assert await save_download(agent, session, reply, agent.settings) is None


class TestRewriteSafety:
    @pytest.mark.asyncio
    async def test_an_unverifiable_number_in_a_rewrite_is_dropped(self, build_agent, calls):
        """The reply text is audited like every other: a size the evidence does
        not contain cannot be stated."""
        from slope_evaluation_agent.scientific_policy import numeric_audit

        agent, session = an_agent(build_agent, calls)
        async with agent:
            await say(agent, session, "看最小安全系数的图")
            reply = await agent.handle(session, "下载刚才的安全系数 TIFF")
        assert numeric_audit(reply.text, reply.facts) == []
