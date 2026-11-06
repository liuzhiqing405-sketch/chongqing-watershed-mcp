"""The deterministic workflow: which states exist, and what may happen next.

This module is the only authority on workflow order. The intent parser cannot
advance a session, and a language model's prose never decides whether a tool
may run — :func:`require_operation` does, from the session's own recorded facts.

Two independent things are checked for every operation:

* the session must be in a state where the operation makes sense (the
  transition table below), and
* the facts the operation depends on must actually be recorded (the gates).

Keeping them separate means a state reached by an odd route still cannot
unlock, say, ``start_hydro_run`` without a validated input and a plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable


class WorkflowState(StrEnum):
    """Every state a session can be in."""

    NEW = "new"
    NEEDS_LOCATION = "needs_location"
    OUTLET_CANDIDATES_READY = "outlet_candidates_ready"
    WATERSHED_DELINEATED = "watershed_delineated"
    WATERSHED_QA_FAILED = "watershed_qa_failed"
    AWAITING_WATERSHED_CONFIRMATION = "awaiting_watershed_confirmation"
    WATERSHED_REJECTED = "watershed_rejected"
    WATERSHED_CONFIRMED = "watershed_confirmed"
    RAINFALL_READY = "rainfall_ready"
    MODEL_INPUTS_READY = "model_inputs_ready"
    HYDRO_INPUT_IMPORTED = "hydro_input_imported"
    HYDRO_INPUT_VALIDATED = "hydro_input_validated"
    HYDRO_PLANNED = "hydro_planned"
    HYDRO_RUNNING = "hydro_running"
    HYDRO_SUCCEEDED = "hydro_succeeded"
    HYDRO_QA_FAILED = "hydro_qa_failed"
    HYDRO_FAILED = "hydro_failed"
    HYDRO_CANCELLED = "hydro_cancelled"
    ASSESSMENT_READY = "assessment_ready"


#: States from which a run may be planned again, so a user can retry with other
#: settings without starting a whole new session.
RETRYABLE_HYDRO_STATES = frozenset({
    WorkflowState.HYDRO_SUCCEEDED,
    WorkflowState.HYDRO_QA_FAILED,
    WorkflowState.HYDRO_FAILED,
    WorkflowState.HYDRO_CANCELLED,
})

#: Terminal states: nothing advances from here without a deliberate retry.
TERMINAL_STATES = frozenset({
    WorkflowState.ASSESSMENT_READY,
}) | RETRYABLE_HYDRO_STATES


#: The transition table. A session may only move along these edges.
ALLOWED_TRANSITIONS: dict[WorkflowState, frozenset[WorkflowState]] = {
    WorkflowState.NEW: frozenset({
        WorkflowState.NEEDS_LOCATION,
        WorkflowState.OUTLET_CANDIDATES_READY,
        WorkflowState.WATERSHED_DELINEATED,
        # Mode C: the user already has a Model Input Package.
        WorkflowState.HYDRO_INPUT_IMPORTED,
    }),
    WorkflowState.NEEDS_LOCATION: frozenset({
        WorkflowState.OUTLET_CANDIDATES_READY,
        WorkflowState.WATERSHED_DELINEATED,
    }),
    WorkflowState.OUTLET_CANDIDATES_READY: frozenset({
        WorkflowState.WATERSHED_DELINEATED,
        WorkflowState.NEEDS_LOCATION,
    }),
    WorkflowState.WATERSHED_DELINEATED: frozenset({
        # The only way forward: a human has to look at it.
        WorkflowState.AWAITING_WATERSHED_CONFIRMATION,
        WorkflowState.WATERSHED_QA_FAILED,
    }),
    WorkflowState.AWAITING_WATERSHED_CONFIRMATION: frozenset({
        WorkflowState.WATERSHED_CONFIRMED,
        WorkflowState.WATERSHED_REJECTED,
        WorkflowState.WATERSHED_QA_FAILED,
    }),
    WorkflowState.WATERSHED_QA_FAILED: frozenset({
        WorkflowState.OUTLET_CANDIDATES_READY,
        WorkflowState.NEEDS_LOCATION,
        WorkflowState.WATERSHED_DELINEATED,
        WorkflowState.WATERSHED_REJECTED,
    }),
    WorkflowState.WATERSHED_REJECTED: frozenset({
        WorkflowState.OUTLET_CANDIDATES_READY,
        WorkflowState.NEEDS_LOCATION,
        WorkflowState.WATERSHED_DELINEATED,
    }),
    WorkflowState.WATERSHED_CONFIRMED: frozenset({
        WorkflowState.RAINFALL_READY,
    }),
    WorkflowState.RAINFALL_READY: frozenset({
        WorkflowState.MODEL_INPUTS_READY,
    }),
    WorkflowState.MODEL_INPUTS_READY: frozenset({
        WorkflowState.HYDRO_INPUT_IMPORTED,
    }),
    WorkflowState.HYDRO_INPUT_IMPORTED: frozenset({
        WorkflowState.HYDRO_INPUT_VALIDATED,
        # A failed validation keeps the session here so the user can retry.
        WorkflowState.HYDRO_INPUT_IMPORTED,
    }),
    WorkflowState.HYDRO_INPUT_VALIDATED: frozenset({
        WorkflowState.HYDRO_PLANNED,
    }),
    WorkflowState.HYDRO_PLANNED: frozenset({
        WorkflowState.HYDRO_RUNNING,
    }),
    WorkflowState.HYDRO_RUNNING: frozenset({
        WorkflowState.HYDRO_SUCCEEDED,
        WorkflowState.HYDRO_QA_FAILED,
        WorkflowState.HYDRO_FAILED,
        WorkflowState.HYDRO_CANCELLED,
    }),
    WorkflowState.HYDRO_SUCCEEDED: frozenset({
        WorkflowState.ASSESSMENT_READY,
        WorkflowState.HYDRO_QA_FAILED,
        # A deliberate re-run with other settings supersedes this result; the
        # finished run stays on disk in the hydro service either way.
        WorkflowState.HYDRO_PLANNED,
    }),
    # A run that ended without a usable result can only be planned again.
    WorkflowState.HYDRO_QA_FAILED: frozenset({WorkflowState.HYDRO_PLANNED, WorkflowState.HYDRO_SUCCEEDED}),
    WorkflowState.HYDRO_FAILED: frozenset({WorkflowState.HYDRO_PLANNED}),
    WorkflowState.HYDRO_CANCELLED: frozenset({WorkflowState.HYDRO_PLANNED}),
    WorkflowState.ASSESSMENT_READY: frozenset({WorkflowState.HYDRO_QA_FAILED}),
}


class WorkflowError(RuntimeError):
    """Base for workflow rule violations."""

    next_step = "Follow the instructions for the current step."

    def __init__(self, message: str, *, next_step: str | None = None) -> None:
        super().__init__(message)
        if next_step:
            self.next_step = next_step

    def user_message(self) -> str:
        return f"{self}\nNext: {self.next_step}"


class IllegalTransition(WorkflowError):
    """The session is not in a state that permits this move."""

    next_step = "Finish the current step or start a new session."


class WorkflowGateError(WorkflowError):
    """The state allows the move, but a required fact is not recorded yet."""

    next_step = "Complete the required steps and retry."


def is_known_state(state: str) -> bool:
    return state in set(WorkflowState)


def can_transition(current: str, target: str) -> bool:
    """True when ``current -> target`` is an allowed edge."""
    if not is_known_state(current) or not is_known_state(target):
        return False
    return WorkflowState(target) in ALLOWED_TRANSITIONS.get(WorkflowState(current), frozenset())


def require_transition(current: str, target: str) -> None:
    """Raise unless ``current -> target`` is allowed."""
    if isinstance(current, WorkflowState):
        current = current.value
    if isinstance(target, WorkflowState):
        target = target.value
    if not is_known_state(current):
        raise IllegalTransition(f"未知会话状态：{current}")
    if not is_known_state(target):
        raise IllegalTransition(f"未知目标状态：{target}")
    if not can_transition(current, target):
        allowed = sorted(item.value for item in ALLOWED_TRANSITIONS[WorkflowState(current)])
        raise IllegalTransition(
            f"不允许从 {current} 直接进入 {target}",
            next_step=(f"当前允许进入：{allowed}；" if allowed else "当前状态已是终点；")
                      + "按工作流顺序继续，或开始新会话。",
        )


def next_states(current: str) -> list[str]:
    """The states reachable in one step, for user-facing hints."""
    if not is_known_state(current):
        return []
    return sorted(item.value for item in ALLOWED_TRANSITIONS[WorkflowState(current)])


# ------------------------------------------------------------------------ gates


class Operation(StrEnum):
    """Every action the workflow can take. A user request maps onto these."""

    VALIDATE_DATA = "validate_data"
    RECOMMEND_OUTLETS = "recommend_outlets"
    DELINEATE = "delineate"
    REVIEW_WATERSHED = "review_watershed"
    CONFIRM_WATERSHED = "confirm_watershed"
    REJECT_WATERSHED = "reject_watershed"
    BUILD_RAINFALL = "build_rainfall"
    PREPARE_MODEL_INPUTS = "prepare_model_inputs"
    IMPORT_PACKAGE = "import_package"
    VALIDATE_INPUT = "validate_input"
    PLAN_RUN = "plan_run"
    START_RUN = "start_run"
    CHECK_STATUS = "check_status"
    CANCEL_RUN = "cancel_run"
    SHOW_RESULT = "show_result"
    SHOW_VISUALIZATION = "show_visualization"
    DOWNLOAD_ARTIFACT = "download_artifact"
    ASSESS = "assess"


@dataclass(frozen=True)
class Gate:
    """Where an operation is allowed to happen, and what it depends on."""

    operation: Operation
    states: frozenset[WorkflowState]
    #: Session fields that must be truthy, with the phrase used to explain them.
    requires: tuple[tuple[str, str], ...] = ()
    #: Session fields that must be absent/empty, guarding against double use.
    forbids: tuple[tuple[str, str], ...] = ()
    next_step: str = ""


def _states(*items: WorkflowState) -> frozenset[WorkflowState]:
    return frozenset(items)


GATES: dict[Operation, Gate] = {
    Operation.VALIDATE_DATA: Gate(
        Operation.VALIDATE_DATA,
        _states(WorkflowState.NEW, WorkflowState.NEEDS_LOCATION,
                WorkflowState.OUTLET_CANDIDATES_READY),
        next_step="Then specify the outlet by river name or coordinates.",
    ),
    Operation.RECOMMEND_OUTLETS: Gate(
        Operation.RECOMMEND_OUTLETS,
        _states(WorkflowState.NEW, WorkflowState.NEEDS_LOCATION,
                WorkflowState.OUTLET_CANDIDATES_READY, WorkflowState.WATERSHED_REJECTED,
                WorkflowState.WATERSHED_QA_FAILED),
        next_step="Choose an outlet on the map or enter longitude and latitude.",
    ),
    Operation.DELINEATE: Gate(
        Operation.DELINEATE,
        _states(WorkflowState.NEW, WorkflowState.NEEDS_LOCATION,
                WorkflowState.OUTLET_CANDIDATES_READY, WorkflowState.WATERSHED_REJECTED,
                WorkflowState.WATERSHED_QA_FAILED),
        next_step="Review and confirm the delineated boundary.",
    ),
    Operation.REVIEW_WATERSHED: Gate(
        Operation.REVIEW_WATERSHED,
        _states(WorkflowState.AWAITING_WATERSHED_CONFIRMATION,
                WorkflowState.WATERSHED_DELINEATED,
                WorkflowState.WATERSHED_QA_FAILED,
                WorkflowState.WATERSHED_CONFIRMED, WorkflowState.WATERSHED_REJECTED),
        requires=(("review_id", "已划分并保存的流域审核记录"),),
        next_step="查看审核信息与流域图后，明确确认或拒绝该边界。",
    ),
    Operation.CONFIRM_WATERSHED: Gate(
        Operation.CONFIRM_WATERSHED,
        _states(WorkflowState.AWAITING_WATERSHED_CONFIRMATION),
        requires=(("review_id", "待确认的流域审核记录"),),
        forbids=(("watershed_confirmed", "该流域已经确认过"),),
        next_step="After confirmation, prepare the rainfall package and model inputs.",
    ),
    Operation.REJECT_WATERSHED: Gate(
        Operation.REJECT_WATERSHED,
        _states(WorkflowState.AWAITING_WATERSHED_CONFIRMATION, WorkflowState.WATERSHED_QA_FAILED),
        requires=(("review_id", "待审核的流域审核记录"),),
        next_step="可以重新加载选址地图或改用坐标划分。",
    ),
    Operation.BUILD_RAINFALL: Gate(
        Operation.BUILD_RAINFALL,
        _states(WorkflowState.WATERSHED_CONFIRMED),
        # The watershed must be confirmed by a human before any rainfall is built.
        requires=(("watershed_confirmed", "人工确认的流域边界"),
                  ("review_id", "已确认的流域审核记录")),
        next_step="随后用 prepare_model_inputs 生成模型输入包。",
    ),
    Operation.PREPARE_MODEL_INPUTS: Gate(
        Operation.PREPARE_MODEL_INPUTS,
        _states(WorkflowState.RAINFALL_READY),
        requires=(("watershed_confirmed", "人工确认的流域边界"),
                  ("rainfall_package_id", "已生成的降雨包")),
        next_step="随后把输入包导入水文服务。",
    ),
    Operation.IMPORT_PACKAGE: Gate(
        Operation.IMPORT_PACKAGE,
        _states(WorkflowState.NEW, WorkflowState.MODEL_INPUTS_READY,
                WorkflowState.HYDRO_INPUT_IMPORTED, WorkflowState.HYDRO_INPUT_VALIDATED,
                WorkflowState.HYDRO_PLANNED),
        next_step="导入后需要先验证输入。",
    ),
    Operation.VALIDATE_INPUT: Gate(
        Operation.VALIDATE_INPUT,
        _states(WorkflowState.HYDRO_INPUT_IMPORTED, WorkflowState.HYDRO_INPUT_VALIDATED,
                WorkflowState.MODEL_INPUTS_READY),
        requires=(("hydro_input_id", "已导入或注册的输入"),),
        next_step="验证通过后生成运行计划。",
    ),
    Operation.PLAN_RUN: Gate(
        Operation.PLAN_RUN,
        _states(WorkflowState.HYDRO_INPUT_VALIDATED) | set(RETRYABLE_HYDRO_STATES),
        requires=(("hydro_input_id", "已导入或注册的输入"),
                  ("hydro_input_validated", "通过验证的输入")),
        next_step="Review the plan before starting the run.",
    ),
    Operation.START_RUN: Gate(
        Operation.START_RUN,
        _states(WorkflowState.HYDRO_PLANNED),
        # Both are required: a validated input AND a plan the user has seen.
        requires=(("hydro_input_validated", "通过验证的输入"),
                  ("hydro_planned", "已生成并展示的运行计划")),
        next_step="The job runs in the background. Check its status to follow progress.",
    ),
    Operation.CHECK_STATUS: Gate(
        Operation.CHECK_STATUS,
        # Only once a run has actually been submitted; a plan alone has no job.
        _states(WorkflowState.HYDRO_RUNNING,WorkflowState.ASSESSMENT_READY) | set(RETRYABLE_HYDRO_STATES),
        requires=(("hydro_run_id", "已提交的运行任务"),),
        next_step="",
    ),
    Operation.CANCEL_RUN: Gate(
        Operation.CANCEL_RUN,
        _states(WorkflowState.HYDRO_RUNNING),
        requires=(("hydro_run_id", "已提交的运行任务"),),
        next_step="取消后可以重新计划并运行。",
    ),
    Operation.SHOW_RESULT: Gate(
        Operation.SHOW_RESULT,
        _states(WorkflowState.HYDRO_SUCCEEDED, WorkflowState.HYDRO_QA_FAILED,
                WorkflowState.HYDRO_FAILED, WorkflowState.HYDRO_CANCELLED,
                WorkflowState.ASSESSMENT_READY),
        requires=(("hydro_run_id", "已提交的运行任务"),),
        next_step="",
    ),
    Operation.SHOW_VISUALIZATION: Gate(
        Operation.SHOW_VISUALIZATION,
        _states(WorkflowState.HYDRO_RUNNING, WorkflowState.HYDRO_SUCCEEDED, WorkflowState.HYDRO_QA_FAILED,
                WorkflowState.HYDRO_FAILED, WorkflowState.ASSESSMENT_READY),
        requires=(("hydro_run_id", "已提交的运行任务"),),
        next_step="",
    ),
    Operation.DOWNLOAD_ARTIFACT: Gate(
        Operation.DOWNLOAD_ARTIFACT,
        # The same states that may show a figure: a download is the figure's
        # real file, and a run whose quality gate failed may still be examined
        # (marked as diagnostic) rather than hidden.
        _states(WorkflowState.HYDRO_SUCCEEDED, WorkflowState.HYDRO_QA_FAILED,
                WorkflowState.HYDRO_FAILED, WorkflowState.ASSESSMENT_READY),
        requires=(("hydro_run_id", "已提交的运行任务"),),
        next_step="先让我画出图件（例如「看最小安全系数的图」），再下载它背后的成果。",
    ),
    Operation.ASSESS: Gate(
        Operation.ASSESS,
        # ``qa_failed`` and the failure states are deliberately absent: they may
        # produce a diagnostic summary but never a formal assessment.
        _states(WorkflowState.HYDRO_SUCCEEDED, WorkflowState.ASSESSMENT_READY),
        requires=(("hydro_run_id", "已提交的运行任务"),
                  ("qa_passed", "通过的质量门槛")),
        next_step="",
    ),
}


def gate_for(operation: Operation | str) -> Gate:
    if isinstance(operation, str):
        try:
            operation = Operation(operation)
        except ValueError as exc:
            raise WorkflowGateError(f"未知工作流操作：{operation}") from exc
    return GATES[operation]


def require_operation(session: Any, operation: Operation | str) -> Gate:
    """Raise unless the session permits this operation right now.

    ``session`` is an :class:`~slope_evaluation_agent.models.AgentSession`; it is
    typed loosely so the state machine stays independent of the model module.
    """
    gate = gate_for(operation)
    from .input_binding import input_signature
    binding = (getattr(session, "planned_input_signature", None) if operation == Operation.START_RUN
               else getattr(session, "run_input_signature", None) if operation in {
                   Operation.SHOW_RESULT, Operation.SHOW_VISUALIZATION,
                   Operation.DOWNLOAD_ARTIFACT, Operation.ASSESS} else None)
    if binding and binding != input_signature(session):
        raise WorkflowError("当前输入已改变，原计划或结果不再对应当前任务。",
                            next_step="请按当前输入重新准备和展示计划；历史结果可通过已保存版本只读查询。")
    state = session.state
    if not is_known_state(state):
        raise IllegalTransition(f"未知会话状态：{state}")
    if WorkflowState(state) not in gate.states:
        allowed = sorted(item.value for item in gate.states)
        raise WorkflowGateError(
            f"当前状态（{state}）不能执行「{operation.value}」",
            next_step=f"该操作只在 {allowed} 中可用。"
                      + (gate.next_step or "Follow the instructions for the current step."),
        )
    for attribute, description in gate.requires:
        if not getattr(session, attribute, None):
            raise WorkflowGateError(
                f"缺少{description}，不能执行「{operation.value}」",
                next_step=gate.next_step or "请先完成前一步。",
            )
    for attribute, description in gate.forbids:
        if getattr(session, attribute, None):
            raise WorkflowGateError(
                f"{description}，不能重复执行「{operation.value}」",
                next_step=gate.next_step or "请继续下一步。",
            )
    return gate


def can_operate(session: Any, operation: Operation | str) -> bool:
    try:
        require_operation(session, operation)
        return True
    except WorkflowError:
        return False


def advance(session: Any, target: WorkflowState | str, *, reason: str = "",
            events: Any = None) -> str:
    """Move a session to ``target``, refusing anything the table forbids.

    The session object validates its own ``state`` on assignment, so an illegal
    value cannot be written even if this function were bypassed — but the check
    here is what produces a message a user can act on.
    """
    target_value = target.value if isinstance(target, WorkflowState) else str(target)
    require_transition(session.state, target_value)
    previous = session.state
    session.state = target_value
    if events is not None:
        events.append("state_transition", **{"from": previous, "to": target_value,
                                             "reason": reason})
    return target_value


def pending_action(state: str) -> str:
    """The one thing the workflow is waiting for, in the user's terms."""
    if not is_known_state(state):
        return "Unknown session state. Start a new session."
    return PENDING_ACTIONS.get(WorkflowState(state), "Describe what you would like to do.")


PENDING_ACTIONS: dict[WorkflowState, str] = {
    WorkflowState.NEW: "Select a district, enter a river name or coordinates, then draw the study area. You can also import an existing model input package.",
    WorkflowState.NEEDS_LOCATION: "Enter a river name, district or map-centre coordinates.",
    WorkflowState.OUTLET_CANDIDATES_READY: "Draw a closed boundary on the map and submit it for coverage checks.",
    WorkflowState.WATERSHED_DELINEATED: "Review the map and boundary checks, then confirm or reject the boundary.",
    WorkflowState.WATERSHED_QA_FAILED: "The watershed checks failed. Choose a new outlet on the river before proceeding.",
    WorkflowState.AWAITING_WATERSHED_CONFIRMATION: "Confirm or reject the current watershed boundary.",
    WorkflowState.WATERSHED_REJECTED: "Choose a location by district, river name or coordinates, then draw the boundary again.",
    WorkflowState.WATERSHED_CONFIRMED: "Specify a rainfall date or time window, or request the maximum day in the available record.",
    WorkflowState.RAINFALL_READY: "Create the model input package.",
    WorkflowState.MODEL_INPUTS_READY: "Import the model input package into the hydrology service.",
    WorkflowState.HYDRO_INPUT_IMPORTED: "Validate the imported inputs.",
    WorkflowState.HYDRO_INPUT_VALIDATED: "Build the run plan and review the parameters.",
    WorkflowState.HYDRO_PLANNED: "Start the run after reviewing the plan, or edit parameters and rebuild it.",
    WorkflowState.HYDRO_RUNNING: "The job is running. You can check progress or cancel it.",
    WorkflowState.HYDRO_SUCCEEDED: "The run passed the quality checks. View the results or assessment.",
    WorkflowState.HYDRO_QA_FAILED: "The run failed quality checks. Review diagnostics before adjusting inputs and rerunning.",
    WorkflowState.HYDRO_FAILED: "The run failed. Review the error before building a new plan.",
    WorkflowState.HYDRO_CANCELLED: "The run was cancelled. You can build a new plan and run it.",
    WorkflowState.ASSESSMENT_READY: "The preliminary assessment is ready. View the figures or ask a follow-up question.",
}


def summarise_path(states: Iterable[str]) -> str:
    """A readable rendering of a transition path, for diagnostics."""
    return " -> ".join(states)


@dataclass
class TransitionRecord:
    """One recorded state change, for the event log."""

    from_state: str
    to_state: str
    reason: str = ""
    at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
