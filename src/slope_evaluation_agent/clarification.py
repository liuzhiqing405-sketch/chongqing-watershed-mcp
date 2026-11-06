"""Asking one question at a time.

A clarification is what the agent does instead of guessing, and instead of
answering a fuzzy request with a mechanical refusal. Two rules shape it:

* **One question.** A wall of questions is a form, and a user who wanted a form
  would have used one. The most important missing thing is asked about, and the
  rest waits — even when several things are missing.
* **Deterministic.** Every word here is written by hand, so asking works with
  no language model at all. The model may rephrase a reply elsewhere; it never
  decides what has to be asked.

The examples offered are examples. They are phrased the way a person would say
them, and nothing in the reply implies the user has to type one of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .intent import Intent
from .state_machine import WorkflowState


#: The order in which missing things are asked about, most fundamental first.
#: A place comes before everything: with no place there is nothing to
#: delineate, so every later question would be premature.
TOPIC_ORDER = ("place", "watershed_confirmation", "rainfall", "parameters", "start")

#: Which topics each entity field belongs to.
FIELD_TOPICS: dict[str, str] = {
    "administrative_area": "place",
    "river_name": "place",
    "longitude": "place",
    "latitude": "place",
    "candidate_index": "place",
    "rainfall_mode": "rainfall",
    "date": "rainfall",
    "start_datetime": "rainfall",
    "end_datetime": "rainfall",
    "lookback_years": "rainfall",
    "end_year": "rainfall",
    "station_count": "rainfall",
    "station_count_mode": "rainfall",
}

#: The question, and a couple of ways a person might answer it.
TOPICS: dict[str, dict[str, Any]] = {
    "place": {
        "question": "Which location should I assess? Choose a district or enter a river name or coordinates to open the DEM and draw a boundary.",
        # Digit-free on purpose: a clarification is a reply, and every reply
        # goes through the numeric audit. An example coordinate here is not
        # evidence and the audit rightly refuses it — and an example that
        # *looks* like a real coordinate invites copying it, which would run
        # an analysis somewhere the user never chose.
        "examples": ["105.90, 29.35", "Use the previously selected river"],
    },
    "watershed_confirmation": {
        "question": "The boundary is ready for review. After checking the map, reply “confirm boundary” "
                    "or “reject boundary”.",
        "examples": ["confirm boundary", "reject boundary"],
    },
    "rainfall": {
        "question": "Which rainfall event should I use? Specify a date or time window, "
                    "or select the maximum day in the available record.",
        "examples": ["2009-08-04", "historical maximum"],
    },
    "parameters": {
        "question": "The run uses the default preset. To override it, provide a parameter name and value. "
                    "Otherwise, review the plan and start the run.",
        "examples": ["start run", "Use the default parameters"],
    },
    "start": {
        "question": "运行计划已经生成。确认无误就回复「start run」，任务会在后台执行，"
                    "You can check progress while it runs.",
        "examples": ["start run"],
    },
    "restart": {
        "question": "Edit the location, boundary, rainfall or parameters in the current task. "
                    "You can restore earlier revisions. Check and confirm an edited boundary before reviewing the new run plan.",
        "examples": ["Change rainfall date", "restore previous revision"],
    },
    "help": {
        "question": "Choose a location to start a preliminary assessment, or "
                    "ask a question about slope stability. "
                    "",
        "examples": ["105.90, 29.35", "What is the factor of safety?"],
    },
}


@dataclass(frozen=True)
class Clarification:
    """The one question being asked, and what prompted it."""

    topic: str
    question: str
    #: Ways to answer, shown as examples rather than as a menu.
    examples: list[str] = field(default_factory=list)
    #: What the agent understood from this turn before asking, in the user's terms.
    recorded: list[str] = field(default_factory=list)
    #: The gate that blocked the request, when one did.
    blocked_by: str | None = None

    def render(self) -> str:
        lines: list[str] = []
        if self.recorded:
            lines.append("Recorded: " + "；".join(self.recorded) + "。")
        lines.append(self.question)
        if self.examples:
            lines.append("For example: " + "、".join(f"「{item}」" for item in self.examples) + "。")
        return "\n".join(lines)

    def log_safe(self) -> dict[str, Any]:
        return {"topic": self.topic, "blocked_by": self.blocked_by,
                "examples": len(self.examples), "recorded": len(self.recorded)}


#: How a recorded entity reads back to the user. Only the fields a person would
#: recognise are named; the rest stay internal.
RECORDED_PHRASES: dict[str, str] = {
    "administrative_area": "District: {value}",
    "river_name": "River: {value}",
    "longitude": "Longitude: {value}",
    "latitude": "Latitude: {value}",
    "date": "Date: {value}",
    "start_datetime": "Start: {value}",
    "end_datetime": "End: {value}",
    "rainfall_mode": "Rainfall: {value}",
    "lookback_years": "Lookback: {value} years",
    "end_year": "Record end year: {value}",
    "station_count": "Virtual gauge count: {value}",
}

RAINFALL_MODE_WORDS = {"historical_max": "maximum day in the record",
                       "explicit_period": "specified date or time window"}


def describe_recorded(entities: dict[str, Any]) -> list[str]:
    """What to say back about the values this turn established."""
    spoken: list[str] = []
    for name, value in entities.items():
        if value in (None, ""):
            continue
        if name == "rainfall_mode":
            spoken.append(f"Rainfall: {RAINFALL_MODE_WORDS.get(str(value), value)}")
            continue
        template = RECORDED_PHRASES.get(name)
        if template:
            spoken.append(template.format(value=value))
    return spoken


def _rainfall_specified(session: Any) -> bool:
    request = getattr(session, "rainfall_request", None) or {}
    if request.get("mode") == "historical_max" or request.get("date"):
        return True
    return bool(request.get("start_datetime") and request.get("end_datetime"))


def next_topic(session: Any, *, missing_fields: list[str] | None = None,
               intent: Intent = Intent.UNKNOWN) -> str:
    """The single most useful thing to ask about now.

    A place comes first: with no place there is nothing to delineate. A pending
    boundary decision comes next, because it blocks every later step whatever
    else is missing. The rest follow the order a person would work through them.
    """
    topics = [FIELD_TOPICS[name] for name in (missing_fields or []) if name in FIELD_TOPICS]

    if "place" in topics:
        return "place"
    if (session.state == WorkflowState.AWAITING_WATERSHED_CONFIRMATION.value
            and not session.watershed_confirmed):
        return "watershed_confirmation"
    if "rainfall" in topics or (session.watershed_confirmed
                                and not _rainfall_specified(session)):
        return "rainfall"
    if session.state == WorkflowState.HYDRO_PLANNED.value and not session.hydro_run_id:
        return "start"
    if "parameters" in topics:
        return "parameters"
    return "help"


def ask(session: Any, *, topic: str | None = None,
        missing_fields: list[str] | None = None,
        intent: Intent = Intent.UNKNOWN, ambiguity: str | None = None,
        recorded: list[str] | None = None, blocked_by: str | None = None) -> Clarification:
    """Build the one question to ask this turn."""
    if intent is Intent.SHOW_VISUALIZATION:
        question=(ambiguity if ambiguity and ambiguity.rstrip().endswith(('？','?')) else
                  'Would you like the modelled subcatchments or the study boundary you drew?')
        return Clarification(topic='visualization',question=question,
                             examples=['Show subcatchment map','Show study area map'],
                             recorded=list(recorded or []),blocked_by=blocked_by)
    resolved = topic if topic in TOPICS else next_topic(
        session, missing_fields=missing_fields, intent=intent)
    spec = TOPICS[resolved]
    question = spec["question"]
    examples = list(spec["examples"])
    if resolved == "place" and getattr(session, "administrative_area", None):
        question = (f"已定位「{session.administrative_area}」。请在 DEM 上绘制闭合评价边界并提交检查；"
                    "如果要换位置，请给出新的区县、河名或经纬度。")
        examples = []
    elif resolved == "place" and getattr(session, "river_name", None):
        question = (f"已记录「{session.river_name}」。请补充所在区县、乡镇或附近地标，"
                    "以便确定具体河段后手绘边界；也可以给经纬度定位。")
        examples = []
    if resolved == "help" and ambiguity:
        question = f"{question}\n(Clarification: {ambiguity})"
    return Clarification(topic=resolved, question=question, examples=examples,
                         recorded=list(recorded or []), blocked_by=blocked_by)


def blocked(session: Any, *, next_step: str, recorded: list[str] | None = None,
            topic: str | None = None) -> Clarification:
    """The question to ask when the state machine refused an operation.

    A refusal is not an error the user caused: it means something else has to
    happen first. Saying which thing, in the user's words, is a question — and
    it keeps the internal state and operation names out of the reply.
    """
    resolved = topic or next_topic(session)
    if resolved == "help":
        resolved = "watershed_confirmation" if (
            session.state == WorkflowState.AWAITING_WATERSHED_CONFIRMATION.value
            and not session.watershed_confirmed) else resolved
    spec = TOPICS.get(resolved, TOPICS["help"])
    question = next_step.strip() or spec["question"]
    return Clarification(topic=resolved, question=question,
                         examples=list(spec["examples"]),
                         recorded=list(recorded or []), blocked_by="state")
