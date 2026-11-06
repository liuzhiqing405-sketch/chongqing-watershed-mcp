"""Phase 3 acceptance: the workflow's transition table and gates.

The prohibitions in the plan are asserted here one by one, because they are the
rules that keep a rainfall package from being built for an unconfirmed basin and
keep an unvalidated input from ever reaching a run.
"""

from __future__ import annotations

import pytest

from slope_evaluation_agent.models import AgentSession
from slope_evaluation_agent.state_machine import (
    ALLOWED_TRANSITIONS,
    GATES,
    RETRYABLE_HYDRO_STATES,
    TERMINAL_STATES,
    IllegalTransition,
    Operation,
    WorkflowGateError,
    WorkflowState,
    advance,
    can_operate,
    can_transition,
    gate_for,
    is_known_state,
    next_states,
    pending_action,
    require_operation,
    require_transition,
)


def session_at(state: WorkflowState, **facts) -> AgentSession:
    """A session parked in ``state`` with the given facts recorded."""
    payload = {
        "session_id": "session-abcdef123456",
        "state": state.value,
        **facts,
    }
    return AgentSession.model_validate(payload)


class TestStateCatalogue:
    def test_every_state_in_the_table_is_known(self):
        for state in ALLOWED_TRANSITIONS:
            assert is_known_state(state.value)

    def test_every_known_state_has_an_entry(self):
        for state in WorkflowState:
            assert state in ALLOWED_TRANSITIONS, state

    def test_every_state_has_a_pending_action_message(self):
        for state in WorkflowState:
            message = pending_action(state.value)
            assert message and message.endswith(("。", "！", "?"))

    def test_unknown_state_is_reported(self):
        assert is_known_state("not_a_state") is False
        assert next_states("not_a_state") == []
        assert "未知" in pending_action("not_a_state")

    def test_terminal_states_only_allow_retry_or_assessment(self):
        for state in TERMINAL_STATES:
            edges = ALLOWED_TRANSITIONS[state]
            assert edges <= {WorkflowState.HYDRO_PLANNED, WorkflowState.ASSESSMENT_READY,
                             WorkflowState.HYDRO_SUCCEEDED, WorkflowState.HYDRO_QA_FAILED}, state

    def test_assessment_can_be_revoked_when_quality_is_rechecked(self):
        assert ALLOWED_TRANSITIONS[WorkflowState.ASSESSMENT_READY] == frozenset({WorkflowState.HYDRO_QA_FAILED})


class TestTransitions:
    def test_allowed_edge_passes(self):
        require_transition(WorkflowState.NEW.value, WorkflowState.NEEDS_LOCATION.value)
        assert can_transition("new", "needs_location") is True

    def test_unknown_state_is_refused(self):
        with pytest.raises(IllegalTransition):
            require_transition("not_a_state", "new")

    def test_delineated_cannot_jump_straight_to_rainfall_ready(self):
        """Plan §6: watershed_delineated must not skip human confirmation."""
        with pytest.raises(IllegalTransition) as excinfo:
            require_transition(WorkflowState.WATERSHED_DELINEATED.value,
                               WorkflowState.RAINFALL_READY.value)
        assert "不允许" in str(excinfo.value)
        assert "awaiting_watershed_confirmation" in excinfo.value.next_step

    def test_confirmed_requires_the_review_step(self):
        assert can_transition("watershed_delineated", "awaiting_watershed_confirmation")
        assert not can_transition("watershed_delineated", "watershed_confirmed")

    def test_new_cannot_skip_to_the_hydro_stages(self):
        for target in (WorkflowState.HYDRO_PLANNED, WorkflowState.HYDRO_RUNNING,
                       WorkflowState.ASSESSMENT_READY):
            assert not can_transition("new", target.value), target

    def test_qa_failed_cannot_reach_assessment_ready(self):
        """Plan §6: a failed quality gate never yields a formal assessment."""
        assert not can_transition("hydro_qa_failed", "assessment_ready")
        with pytest.raises(IllegalTransition):
            require_transition("hydro_qa_failed", "assessment_ready")

    def test_failed_and_cancelled_cannot_reach_assessment_ready(self):
        for state in ("hydro_failed", "hydro_cancelled"):
            assert not can_transition(state, "assessment_ready"), state

    def test_only_succeeded_reaches_assessment_ready(self):
        sources = [state.value for state in ALLOWED_TRANSITIONS
                   if can_transition(state.value, "assessment_ready")]
        assert sources == ["hydro_succeeded"]

    def test_retry_is_possible_from_a_failed_run(self):
        for state in RETRYABLE_HYDRO_STATES:
            assert can_transition(state.value, "hydro_planned"), state

    def test_rejected_watershed_can_be_redone(self):
        for target in ("outlet_candidates_ready", "needs_location", "watershed_delineated"):
            assert can_transition("watershed_rejected", target), target

    def test_advance_moves_the_session_and_logs(self):
        session = session_at(WorkflowState.NEW)
        logged: list[dict] = []

        class Sink:
            @staticmethod
            def append(event, **fields):
                logged.append({"event": event, **fields})

        advance(session, WorkflowState.NEEDS_LOCATION, reason="需给出位置", events=Sink())
        assert session.state == "needs_location"
        assert logged[0]["from"] == "new"
        assert logged[0]["to"] == "needs_location"

    def test_advance_refuses_an_illegal_target(self):
        session = session_at(WorkflowState.WATERSHED_DELINEATED)
        with pytest.raises(IllegalTransition):
            advance(session, WorkflowState.RAINFALL_READY)
        assert session.state == "watershed_delineated"


class TestGates:
    def test_validate_data_is_available_at_the_start(self):
        session = session_at(WorkflowState.NEW)
        assert can_operate(session, Operation.VALIDATE_DATA)

    def test_rainfall_cannot_be_built_before_confirmation(self):
        """Plan §6: watershed_confirmed == false blocks build_virtual_rainfall."""
        for state in (WorkflowState.WATERSHED_DELINEATED,
                      WorkflowState.AWAITING_WATERSHED_CONFIRMATION):
            session = session_at(state, review_id="basin-abcdef123456")
            with pytest.raises(WorkflowGateError):
                require_operation(session, Operation.BUILD_RAINFALL)

    def test_rainfall_cannot_be_built_with_the_flag_but_no_review(self):
        session = session_at(WorkflowState.WATERSHED_CONFIRMED, watershed_confirmed=True)
        with pytest.raises(WorkflowGateError) as excinfo:
            require_operation(session, Operation.BUILD_RAINFALL)
        assert "流域审核记录" in str(excinfo.value)

    def test_rainfall_is_available_once_confirmed(self):
        session = session_at(WorkflowState.WATERSHED_CONFIRMED,
                             watershed_confirmed=True, review_id="basin-abcdef123456")
        require_operation(session, Operation.BUILD_RAINFALL)

    def test_confirm_requires_a_pending_review(self):
        empty = session_at(WorkflowState.NEW)
        with pytest.raises(WorkflowGateError):
            require_operation(empty, Operation.CONFIRM_WATERSHED)

    def test_confirm_cannot_be_repeated(self):
        session = session_at(WorkflowState.AWAITING_WATERSHED_CONFIRMATION,
                             review_id="basin-abcdef123456", watershed_confirmed=True)
        with pytest.raises(WorkflowGateError) as excinfo:
            require_operation(session, Operation.CONFIRM_WATERSHED)
        assert "已经确认" in str(excinfo.value)

    def test_start_requires_a_validated_input(self):
        """Plan §6: an unvalidated input must not be able to start a run."""
        session = session_at(WorkflowState.HYDRO_PLANNED,
                             hydro_input_id="hydro-input-abcdef123456",
                             hydro_planned=True, hydro_input_validated=False)
        with pytest.raises(WorkflowGateError) as excinfo:
            require_operation(session, Operation.START_RUN)
        assert "通过验证的输入" in str(excinfo.value)

    def test_start_requires_a_plan(self):
        """Plan §6: without plan_hydro_run a run must not start."""
        session = session_at(WorkflowState.HYDRO_PLANNED,
                             hydro_input_id="hydro-input-abcdef123456",
                             hydro_input_validated=True, hydro_planned=False)
        with pytest.raises(WorkflowGateError) as excinfo:
            require_operation(session, Operation.START_RUN)
        assert "运行计划" in str(excinfo.value)

    def test_start_is_available_with_both_facts(self):
        session = session_at(WorkflowState.HYDRO_PLANNED,
                             hydro_input_id="hydro-input-abcdef123456",
                             hydro_input_validated=True, hydro_planned=True)
        require_operation(session, Operation.START_RUN)

    def test_start_is_impossible_from_any_earlier_state(self):
        early = [state for state in WorkflowState
                 if state not in {WorkflowState.HYDRO_PLANNED}]
        for state in early:
            session = session_at(state, hydro_input_id="hydro-input-abcdef123456",
                                 hydro_input_validated=True, hydro_planned=True)
            with pytest.raises(WorkflowGateError):
                require_operation(session, Operation.START_RUN)

    def test_assessment_is_refused_while_qa_has_not_passed(self):
        for state in (WorkflowState.HYDRO_QA_FAILED, WorkflowState.HYDRO_FAILED,
                      WorkflowState.HYDRO_CANCELLED):
            session = session_at(state, hydro_run_id="hydro-run-abcdef123456")
            with pytest.raises(WorkflowGateError):
                require_operation(session, Operation.ASSESS)

    def test_assessment_is_refused_when_qa_is_unknown(self):
        session = session_at(WorkflowState.HYDRO_SUCCEEDED,
                             hydro_run_id="hydro-run-abcdef123456", qa_passed=None)
        with pytest.raises(WorkflowGateError) as excinfo:
            require_operation(session, Operation.ASSESS)
        assert "质量门槛" in str(excinfo.value)

    def test_assessment_is_available_for_a_passed_run(self):
        session = session_at(WorkflowState.HYDRO_SUCCEEDED,
                             hydro_run_id="hydro-run-abcdef123456", qa_passed=True)
        require_operation(session, Operation.ASSESS)

    def test_result_and_visualisation_are_available_for_a_failed_run(self):
        session = session_at(WorkflowState.HYDRO_QA_FAILED,
                             hydro_run_id="hydro-run-abcdef123456")
        # Diagnostics must remain reachable, or the failure cannot be explained.
        require_operation(session, Operation.SHOW_RESULT)

    def test_cancel_needs_a_running_job(self):
        planned = session_at(WorkflowState.HYDRO_PLANNED,
                             hydro_input_id="hydro-input-abcdef123456")
        with pytest.raises(WorkflowGateError):
            require_operation(planned, Operation.CANCEL_RUN)

    def test_unknown_operation_is_refused(self):
        with pytest.raises(WorkflowGateError):
            gate_for("delete_everything")

    def test_every_operation_has_a_gate(self):
        assert set(GATES) == set(Operation)

    def test_every_gate_names_at_least_one_state(self):
        for operation, gate in GATES.items():
            assert gate.states, operation

    def test_gate_next_steps_are_user_facing(self):
        for operation, gate in GATES.items():
            if gate.next_step:
                assert not gate.next_step.startswith("File ")
                assert "Traceback" not in gate.next_step
