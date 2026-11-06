"""Phase 3 acceptance: the data models the agent reasons about.

Two families live here: the request payloads that are allowed to reach a server,
and the session object that carries workflow state.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from slope_evaluation_agent.models import (
    OVERRIDABLE_PARAMETERS,
    SESSION_SCHEMA_VERSION,
    AgentSession,
    ArtifactDescriptor,
    ArtifactFilter,
    ConversationTurn,
    HydroRunRequestPayload,
    RainfallRequestPayload,
    VisualizationRequest,
    drop_none,
)
from slope_evaluation_agent.state_machine import WorkflowState


REVIEW_ID = "basin-abcdef123456"
INPUT_ID = "hydro-input-abcdef123456"
RUN_ID = "hydro-run-abcdef123456"


def test_legacy_artifact_metadata_is_normalised_from_the_original_filename():
    raster = ArtifactDescriptor.model_validate({
        "artifact_id": "artifact-abcdef123456",
        "title": "package_enhanced_maximum_velocity_ms.tif",
        "media_type": "application/octet-stream",
        "ready": True,
        "formats": ["original"],
    })
    report = ArtifactDescriptor.model_validate({
        "artifact_id": "artifact-abcdef654321",
        "title": "model.rpt",
        "media_type": "application/octet-stream",
        "ready": True,
        "formats": ["original"],
    })
    assert raster.media_type == "image/tiff"
    assert raster.units == "m/s"
    assert report.media_type == "text/plain"


def a_session(**overrides) -> AgentSession:
    payload = {"session_id": "session-abcdef123456", "state": "new", **overrides}
    return AgentSession.model_validate(payload)


class TestRainfallRequestPayload:
    def test_historical_max_needs_only_a_lookback(self):
        payload = RainfallRequestPayload(review_id=REVIEW_ID, mode="historical_max",
                                         lookback_years=10)
        assert payload.lookback_years == 10
        assert payload.end_year is None

    def test_historical_max_rejects_a_concrete_date(self):
        with pytest.raises(ValidationError):
            RainfallRequestPayload(review_id=REVIEW_ID, mode="historical_max",
                                   date="2024-07-30")

    def test_an_explicit_day_is_accepted(self):
        payload = RainfallRequestPayload(review_id=REVIEW_ID, mode="explicit_period",
                                         date="2024-07-30")
        assert payload.date == "2024-07-30"

    def test_an_explicit_window_is_accepted(self):
        payload = RainfallRequestPayload(
            review_id=REVIEW_ID, mode="explicit_period",
            start_datetime="2024-07-30 02:00", end_datetime="2024-07-30 14:00")
        assert payload.station_count is None

    def test_mixing_date_and_window_is_refused(self):
        with pytest.raises(ValidationError):
            RainfallRequestPayload(review_id=REVIEW_ID, mode="explicit_period",
                                   date="2024-07-30",
                                   start_datetime="2024-07-30 01:00",
                                   end_datetime="2024-07-30 02:00")

    def test_a_half_given_window_is_refused(self):
        with pytest.raises(ValidationError):
            RainfallRequestPayload(review_id=REVIEW_ID, mode="explicit_period",
                                   start_datetime="2024-07-30 01:00")

    def test_a_cross_month_window_is_refused(self):
        with pytest.raises(ValidationError):
            RainfallRequestPayload(review_id=REVIEW_ID, mode="explicit_period",
                                   start_datetime="2024-07-31 23:00",
                                   end_datetime="2024-08-01 02:00")

    def test_a_reversed_window_is_refused(self):
        with pytest.raises(ValidationError):
            RainfallRequestPayload(review_id=REVIEW_ID, mode="explicit_period",
                                   start_datetime="2024-07-30 10:00",
                                   end_datetime="2024-07-30 02:00")

    def test_a_window_beyond_the_station_format_is_refused(self):
        with pytest.raises(ValidationError):
            RainfallRequestPayload(review_id=REVIEW_ID, mode="explicit_period",
                                   start_datetime="2024-07-01 00:00",
                                   end_datetime="2024-07-20 00:00")

    def test_station_count_is_bounded(self):
        for value in (0, 101, True, 6.5):
            with pytest.raises(ValidationError):
                RainfallRequestPayload(review_id=REVIEW_ID, mode="historical_max",
                                       station_count=value)

    def test_the_default_timezone_matches_the_data(self):
        assert RainfallRequestPayload(review_id=REVIEW_ID,
                                      mode="historical_max").source_timezone == "Asia/Shanghai"


class TestHydroRunRequestPayload:
    def test_defaults_match_the_documented_contract(self):
        payload = HydroRunRequestPayload(input_id=INPUT_ID)
        assert payload.mode == "build_run_qa"
        assert payload.parameter_preset == "medium"
        assert payload.through is None
        assert payload.parameter_overrides == {}

    def test_the_agent_does_not_invent_overrides(self):
        assert HydroRunRequestPayload(input_id=INPUT_ID).parameter_overrides == {}

    def test_unknown_parameters_are_refused(self):
        with pytest.raises(ValidationError) as excinfo:
            HydroRunRequestPayload(input_id=INPUT_ID,
                                   parameter_overrides={"os.system": 1.0})
        assert "不支持的参数" in str(excinfo.value)

    def test_a_known_parameter_is_accepted(self):
        payload = HydroRunRequestPayload(
            input_id=INPUT_ID, parameter_overrides={"saturated_cohesion_kpa": 15.0})
        assert payload.parameter_overrides == {"saturated_cohesion_kpa": 15.0}

    def test_every_overridable_parameter_is_named_and_united(self):
        assert set(OVERRIDABLE_PARAMETERS) == {
            "deposit_thickness_m", "natural_density_g_cm3", "saturated_density_g_cm3",
            "saturated_cohesion_kpa", "saturated_friction_angle_deg", "initial_saturation",
            "friction_velocity_coefficient_a", "runoff_density_g_cm3",
            "calculation_resolution_m",
        }
        assert all(isinstance(unit, str) and unit for unit in OVERRIDABLE_PARAMETERS.values())

    def test_notes_are_bounded(self):
        payload = HydroRunRequestPayload(input_id=INPUT_ID, notes="x" * 5000)
        assert len(payload.notes) == 2000

    def test_the_mode_is_constrained(self):
        with pytest.raises(ValidationError):
            HydroRunRequestPayload(input_id=INPUT_ID, mode="everything")

    def test_drop_none_omits_unset_optional_fields(self):
        wire = drop_none(HydroRunRequestPayload(input_id=INPUT_ID))
        assert "through" not in wire
        assert wire["mode"] == "build_run_qa"
        assert wire["parameter_overrides"] == {}


class TestVisualizationRequest:
    def test_a_catalog_name_is_accepted(self):
        request = VisualizationRequest(run_id=RUN_ID, kind="safety_minimum_factor")
        assert request.kind == "safety_minimum_factor"

    def test_a_path_like_kind_is_refused(self):
        for bad in ("../../etc/passwd", "Safety Minimum", "drop;--", ""):
            with pytest.raises(ValidationError):
                VisualizationRequest(run_id=RUN_ID, kind=bad)

    def test_the_agent_does_not_decide_which_kinds_exist(self):
        # A well-formed name that the server does not offer still validates here:
        # the server's catalog is the single source of truth.
        assert VisualizationRequest(run_id=RUN_ID, kind="not_a_real_kind")


class TestArtifactFilter:
    def test_optional_stage_filter(self):
        assert ArtifactFilter(run_id=RUN_ID).stage_id is None

    def test_a_path_like_stage_is_refused(self):
        with pytest.raises(ValidationError):
            ArtifactFilter(run_id=RUN_ID, stage_id="../etc")


class TestAgentSession:
    def test_defaults_describe_a_brand_new_session(self):
        session = a_session()
        # Version 4 adds all registered process artifacts to the download side
        # of a session. Still ids and public metadata only — no
        # path and no image bytes may be persisted. The version is pinned here
        # so a silent change to the shape is caught.
        assert session.schema_version == SESSION_SCHEMA_VERSION == 4
        assert session.state == WorkflowState.NEW.value
        assert session.watershed_confirmed is False
        assert session.hydro_input_validated is False
        assert session.hydro_planned is False
        assert session.conversation == []
        assert session.parameter_preset == "medium"

    def test_the_session_id_must_look_like_one(self):
        with pytest.raises(ValidationError):
            AgentSession(session_id="abc", state="new")

    def test_an_unknown_state_cannot_be_constructed(self):
        with pytest.raises(ValidationError):
            AgentSession(session_id="session-abcdef123456", state="nowhere")

    def test_an_unknown_state_cannot_be_assigned(self):
        session = a_session()
        with pytest.raises(ValidationError):
            session.state = "nowhere"

    def test_the_preset_is_constrained(self):
        with pytest.raises(ValidationError):
            a_session(parameter_preset="extreme")

    def test_recording_turns_updates_updated_at(self):
        session = a_session()
        before = session.updated_at
        session.record_turn("user", "分析大宁河")
        assert session.conversation[0].role == "user"
        assert session.updated_at >= before

    def test_turns_carry_the_tools_they_triggered(self):
        session = a_session()
        session.record_turn("agent", "已推荐出口", tools=["recommend_outlets"])
        assert session.conversation[-1].tools == ["recommend_outlets"]

    def test_recent_conversation_returns_the_tail(self):
        session = a_session()
        for index in range(10):
            session.record_turn("user", f"第 {index} 轮")
        assert [turn.text for turn in session.recent_conversation(3)] == [
            "第 7 轮", "第 8 轮", "第 9 轮"]

    def test_pending_action_prefers_an_explicit_question(self):
        session = a_session(state="outlet_candidates_ready")
        assert "绘制闭合评价边界" in session.pending_action()
        session.pending_question = "请问用哪一天？"
        assert session.pending_action() == "请问用哪一天？"

    def test_json_round_trip_preserves_everything(self):
        session = a_session(state=WorkflowState.HYDRO_PLANNED.value,
                            hydro_input_id=INPUT_ID, hydro_input_validated=True,
                            hydro_planned=True, last_plan={"stages": ["sync", "swmm"]})
        restored = AgentSession.model_validate_json(session.model_dump_json())
        assert restored == session

    def test_a_session_never_serialises_a_credential_field(self):
        payload = json.dumps(a_session().model_dump(mode="json"), ensure_ascii=False)
        for marker in ("api_key", "sk-", "token"):
            assert marker not in payload
