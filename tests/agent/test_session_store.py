"""Phase 3 acceptance: session persistence, recovery and the event log."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slope_evaluation_agent.config import SessionSettings
from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.session_store import (
    SessionError,
    SessionStore,
    new_session_id,
    validate_session_id,
    write_json_atomic,
)
from slope_evaluation_agent.state_machine import WorkflowState, advance


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return SessionStore(SessionSettings(root=str(tmp_path / "sessions")))


class TestIdentifiers:
    def test_generated_id_has_the_expected_shape(self):
        identifier = new_session_id()
        assert identifier.startswith("session-")
        assert validate_session_id(identifier) == identifier

    def test_malformed_ids_are_refused(self):
        for bad in ("", "session", "session-XYZ", "../../etc", "session-ABCDEF123456"):
            with pytest.raises(SessionError):
                validate_session_id(bad)


class TestLifecycle:
    def test_create_writes_a_loadable_session(self, store):
        created = store.create()
        assert created.state == WorkflowState.NEW.value
        loaded = store.load(created.session_id)
        assert loaded.session_id == created.session_id
        assert loaded.state == "new"

    def test_create_refuses_a_duplicate(self, store):
        created = store.create()
        with pytest.raises(SessionError):
            store.create(session_id=created.session_id)

    def test_loading_a_missing_session_is_a_clean_error(self, store):
        with pytest.raises(SessionError) as excinfo:
            store.load("session-ffffffffffff")
        assert "Traceback" not in excinfo.value.user_message()

    def test_a_corrupt_session_file_is_reported_not_raised_raw(self, store):
        created = store.create()
        store.session_path(created.session_id).write_text("{not json", encoding="utf-8")
        with pytest.raises(SessionError) as excinfo:
            store.load(created.session_id)
        assert "JSON" in str(excinfo.value)

    def test_a_session_with_an_unknown_state_fails_to_load(self, store):
        created = store.create()
        payload = json.loads(store.session_path(created.session_id).read_text(encoding="utf-8"))
        payload["state"] = "teleported"
        store.session_path(created.session_id).write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SessionError):
            store.load(created.session_id)

    def test_list_is_ordered_by_most_recent(self, store):
        first = store.create()
        second = store.create()
        advance(second, WorkflowState.NEEDS_LOCATION)
        store.save(second)
        rows = store.list_sessions()
        assert rows[0]["session_id"] == second.session_id
        assert {row["session_id"] for row in rows} == {first.session_id, second.session_id}

    def test_latest_session_id(self, store):
        assert store.latest_session_id() is None
        created = store.create()
        assert store.latest_session_id() == created.session_id

    def test_list_carries_a_safe_history_title_and_message_count(self, store):
        session = store.create()
        session.record_turn("user", "我想看看大宁河那边怎么样")
        session.record_turn("agent", "可以，先确定出口位置。")
        session.river_name = "大宁河"
        session.refresh_summary()
        store.save(session)

        row = store.list_sessions()[0]

        assert row["title"] == "河流 大宁河"
        assert row["conversation_summary"] == "河流 大宁河"
        assert row["message_count"] == 2

    def test_delete_removes_exactly_one_session(self, store):
        keep = store.create()
        removed = store.create()
        store.save_evidence(removed.session_id, {"x": 1})

        store.delete(removed.session_id)

        assert store.exists(removed.session_id) is False
        assert store.exists(keep.session_id) is True
        with pytest.raises(SessionError):
            store.load(removed.session_id)

    def test_delete_rejects_missing_and_malformed_ids(self, store):
        with pytest.raises(SessionError):
            store.delete("session-ffffffffffff")
        with pytest.raises(SessionError):
            store.delete("../../outside")

    def test_delete_refuses_a_symlink_instead_of_following_it(self, store, tmp_path):
        target = tmp_path / "must-survive"
        target.mkdir()
        marker = target / "marker.txt"
        marker.write_text("keep", encoding="utf-8")
        link = store.root / "session-aaaaaaaaaaaa"
        store.root.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            pytest.skip("当前 Windows 权限不允许创建符号链接")

        with pytest.raises(SessionError):
            store.delete("session-aaaaaaaaaaaa")

        assert marker.read_text(encoding="utf-8") == "keep"


class TestRecovery:
    def test_identifiers_survive_a_restart(self, tmp_path):
        """Plan §12 phase 3: review_id and hydro_run_id must come back."""
        root = tmp_path / "sessions"
        store = SessionStore(SessionSettings(root=str(root)))
        session = store.create()
        session.review_id = "basin-abcdef123456"
        session.hydro_run_id = "hydro-run-abcdef123456"
        session.hydro_input_id = "hydro-input-abcdef123456"
        session.rainfall_package_id = "rain-abcdef123456"
        store.save(session)

        # A fresh store, as after a process restart.
        reopened = SessionStore(SessionSettings(root=str(root))).load(session.session_id)
        assert reopened.review_id == "basin-abcdef123456"
        assert reopened.hydro_run_id == "hydro-run-abcdef123456"
        assert reopened.hydro_input_id == "hydro-input-abcdef123456"
        assert reopened.rainfall_package_id == "rain-abcdef123456"

    def test_state_survives_a_restart(self, tmp_path):
        root = tmp_path / "sessions"
        store = SessionStore(SessionSettings(root=str(root)))
        session = store.create()
        advance(session, WorkflowState.NEEDS_LOCATION)
        advance(session, WorkflowState.OUTLET_CANDIDATES_READY)
        store.save(session)
        reopened = SessionStore(SessionSettings(root=str(root))).load(session.session_id)
        assert reopened.state == "outlet_candidates_ready"

    def test_structured_state_is_kept_while_conversation_is_trimmed(self, store):
        session = store.create()
        session.review_id = "basin-abcdef123456"
        for index in range(60):
            session.record_turn("user", f"第 {index} 轮", maximum_turns=20)
        assert len(session.conversation) == 20
        assert session.conversation[-1].text == "第 59 轮"
        # The id is still there even though the turn that produced it is gone.
        assert session.review_id == "basin-abcdef123456"

    def test_long_turn_text_is_clipped(self, store):
        session = store.create()
        session.record_turn("user", "x" * 5000, maximum_chars=200)
        assert len(session.conversation[0].text) == 200


class TestAtomicWrites:
    def test_no_temporary_file_is_left_behind(self, tmp_path):
        target = tmp_path / "out.json"
        write_json_atomic(target, {"a": 1})
        assert target.is_file()
        assert [item.name for item in tmp_path.iterdir()] == ["out.json"]

    def test_a_failed_write_leaves_the_previous_file_intact(self, tmp_path, monkeypatch):
        target = tmp_path / "out.json"
        write_json_atomic(target, {"version": 1})

        def explode(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(json, "dump", explode)
        with pytest.raises(OSError):
            write_json_atomic(target, {"version": 2})
        assert json.loads(target.read_text(encoding="utf-8")) == {"version": 1}

    def test_a_failed_save_does_not_destroy_the_session(self, store, monkeypatch):
        session = store.create()
        before = store.session_path(session.session_id).read_text(encoding="utf-8")
        monkeypatch.setattr(json, "dump", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
        with pytest.raises(OSError):
            store.save(session)
        assert store.session_path(session.session_id).read_text(encoding="utf-8") == before

    def test_a_save_survives_a_reader_holding_the_file_open(self, store):
        """``status`` and ``inspect`` read while ``chat`` writes.

        On Windows ``os.replace`` fails with a sharing violation while another
        handle holds the destination open, so without a retry the same
        read-while-writing overlap that broke the hydro worker would break a
        session save.
        """
        import threading
        import time

        session = store.create()
        path = store.session_path(session.session_id)
        holding = threading.Event()

        def hold_open() -> None:
            with path.open("r", encoding="utf-8") as stream:
                stream.read()
                holding.set()
                time.sleep(0.4)

        reader = threading.Thread(target=hold_open, daemon=True)
        reader.start()
        assert holding.wait(5.0), "读者线程没能打开文件"
        try:
            session.river_name = "大宁河"
            store.save(session)
        finally:
            reader.join(5.0)

        assert store.load(session.session_id).river_name == "大宁河"
        assert [item.name for item in path.parent.iterdir() if item.name.startswith(".tmp-")] == []


class TestEventLog:
    def test_creation_is_logged(self, store):
        session = store.create()
        events = store.read_events(session.session_id)
        assert events[0]["event"] == "session_created"

    def test_transitions_are_logged_in_order(self, store):
        session = store.create()
        log = store.events(session.session_id)
        advance(session, WorkflowState.NEEDS_LOCATION, reason="需给出位置", events=log)
        advance(session, WorkflowState.OUTLET_CANDIDATES_READY, reason="已推荐出口", events=log)
        events = [item for item in store.read_events(session.session_id)
                  if item["event"] == "state_transition"]
        assert [(item["from"], item["to"]) for item in events] == [
            ("new", "needs_location"),
            ("needs_location", "outlet_candidates_ready"),
        ]

    def test_tool_calls_can_be_logged_around_a_call(self, store):
        session = store.create()
        log = store.events(session.session_id)
        log.append("tool_call", tool="recommend_outlets", arguments={"river_name": "大宁河"})
        log.append("tool_result", tool="recommend_outlets", ok=True, candidates=3)
        events = [item for item in store.read_events(session.session_id)
                  if item["event"].startswith("tool_")]
        assert events[0]["tool"] == "recommend_outlets"
        assert events[1]["candidates"] == 3

    def test_credentials_never_reach_the_log(self, store):
        session = store.create()
        store.events(session.session_id).append(
            "tool_call", tool="x", api_key="sk-real-key-value",
            nested={"token": "another-secret", "safe": 1},
        )
        raw = store.events_path(session.session_id).read_text(encoding="utf-8")
        assert "sk-real-key-value" not in raw
        assert "another-secret" not in raw
        assert '"safe": 1' in raw

    def test_an_oversized_record_is_truncated_not_written_whole(self, store):
        session = store.create()
        store.events(session.session_id).append("tool_call", tool="x", blob="y" * 100000)
        line = store.events_path(session.session_id).read_text(encoding="utf-8").splitlines()[-1]
        assert len(line) <= store.settings.max_event_payload_chars + 200
        assert json.loads(line)["truncated"] is True

    def test_a_corrupt_line_does_not_break_reading(self, store):
        session = store.create()
        with store.events_path(session.session_id).open("a", encoding="utf-8") as stream:
            stream.write("{broken\n")
        assert store.read_events(session.session_id)


class TestPendingWatershed:
    def test_nothing_pending_at_the_start(self, store):
        assert store.create().has_pending_watershed() is False

    def test_pending_only_when_awaiting_a_decision(self, store):
        session = store.create()
        advance(session, WorkflowState.NEEDS_LOCATION)
        advance(session, WorkflowState.WATERSHED_DELINEATED)
        advance(session, WorkflowState.AWAITING_WATERSHED_CONFIRMATION)
        # Still nothing to confirm until a review actually exists.
        assert session.has_pending_watershed() is False
        session.review_id = "basin-abcdef123456"
        assert session.has_pending_watershed() is True

    def test_not_pending_once_confirmed(self, store):
        session = store.create()
        advance(session, WorkflowState.NEEDS_LOCATION)
        advance(session, WorkflowState.WATERSHED_DELINEATED)
        advance(session, WorkflowState.AWAITING_WATERSHED_CONFIRMATION)
        session.review_id = "basin-abcdef123456"
        session.watershed_confirmed = True
        assert session.has_pending_watershed() is False

    def test_not_pending_in_an_unrelated_state(self, store):
        session = store.create()
        session.review_id = "basin-abcdef123456"
        assert session.has_pending_watershed() is False


class TestDerivedArtifacts:
    def test_evidence_round_trips(self, store):
        session = store.create()
        store.save_evidence(session.session_id, {"factors": {"rainfall": {"total_mm": 79.3}}})
        assert store.load_evidence(session.session_id)["factors"]["rainfall"]["total_mm"] == 79.3

    def test_assessment_round_trips(self, store):
        session = store.create()
        store.save_assessment(session.session_id, {"assessment_status": "diagnostic_only"})
        assert store.load_assessment(session.session_id)["assessment_status"] == "diagnostic_only"

    def test_missing_artifacts_read_as_none(self, store):
        session = store.create()
        assert store.load_evidence(session.session_id) is None
        assert store.load_assessment(session.session_id) is None

    def test_clear_artifacts_keeps_the_session_and_log(self, store):
        session = store.create()
        store.save_evidence(session.session_id, {"x": 1})
        store.save_assessment(session.session_id, {"y": 2})
        store.clear_artifacts(session.session_id, ["evidence.json", "final_assessment.json"])
        assert store.load_evidence(session.session_id) is None
        assert store.load(session.session_id).session_id == session.session_id
        assert store.read_events(session.session_id)


class TestSessionFacts:
    def test_reset_for_new_run_clears_only_run_scoped_facts(self, store):
        session = store.create()
        session.review_id = "basin-abcdef123456"
        session.hydro_input_id = "hydro-input-abcdef123456"
        session.hydro_run_id = "hydro-run-abcdef123456"
        session.qa_passed = True
        session.hydro_planned = True
        session.last_plan = {"stages": ["sync"]}
        session.visualizations = [{"visualization_id": "viz-abcdef123456"}]
        session.reset_for_new_run()
        # The basin and input are still valid; only the run is forgotten.
        assert session.review_id == "basin-abcdef123456"
        assert session.hydro_input_id == "hydro-input-abcdef123456"
        assert session.hydro_run_id is None
        assert session.qa_passed is None
        assert session.hydro_planned is False
        assert session.visualizations == []

    def test_note_error_is_recorded_without_a_traceback(self, store):
        session = store.create()
        session.note_error(code="MCPToolRejected", message="输入校验未通过",
                           next_step="重新导入")
        assert session.last_error["code"] == "MCPToolRejected"
        assert "Traceback" not in json.dumps(session.last_error, ensure_ascii=False)
        session.clear_error()
        assert session.last_error is None

    def test_state_assignment_is_validated(self, store):
        session = store.create()
        with pytest.raises(Exception):
            session.state = "not_a_state"
