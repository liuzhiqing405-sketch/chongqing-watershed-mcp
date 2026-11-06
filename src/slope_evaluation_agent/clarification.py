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
        "question": "你想评价哪个位置？可以给区县、具体河名或经纬度，我会打开 DEM 供你绘制边界。",
        # Digit-free on purpose: a clarification is a reply, and every reply
        # goes through the numeric audit. An example coordinate here is not
        # evidence and the audit rightly refuses it — and an example that
        # *looks* like a real coordinate invites copying it, which would run
        # an analysis somewhere the user never chose.
        "examples": ["分析大宁河", "就看前面说的那条河"],
    },
    "watershed_confirmation": {
        "question": "流域边界已经划分好了。看过图之后，确认就回复「确认该边界」，"
                    "我会接着准备降雨和模型输入；不满意就回复「拒绝」，我们换个位置重来。",
        "examples": ["确认该边界", "拒绝，换一个位置"],
    },
    "rainfall": {
        "question": "接下来是降雨事件。你想用哪一场雨？指定某一天或某个时段都可以；"
                    "如果不确定，也可以直接用资料期内的样本最大日。",
        "examples": ["2024-07-30 全天", "用历史最大日"],
    },
    "parameters": {
        "question": "运行参数目前用默认预设。想改哪个参数，直接说参数名和取值就行；"
                    "不改的话，直接开始运行。",
        "examples": ["开始运行", "用默认参数就行"],
    },
    "start": {
        "question": "运行计划已经生成。确认无误就回复「开始运行」，任务会在后台执行，"
                    "可以随时问进度。",
        "examples": ["开始运行"],
    },
    "restart": {
        "question": "可以在当前任务修改位置、边界、降雨或参数。相关下游输入会重新准备，"
                    "修改前版本可恢复；新的边界仍需检查和确认，运行前须核对新计划。",
        "examples": ["修改降雨日期", "恢复上一步"],
    },
    "help": {
        "question": "我没能确定你想做什么。这个智能体做的是区域尺度、降雨事件驱动的"
                    "坡面稳定性初步筛查与诊断：给一条河或一个坐标，我会带你走完"
                    "降雨、建模和评价；也可以直接问我某个概念。",
        "examples": ["分析大宁河", "安全系数是什么意思？"],
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
            lines.append("好的，" + "；".join(self.recorded) + "。")
        lines.append(self.question)
        if self.examples:
            lines.append("比如：" + "、".join(f"「{item}」" for item in self.examples) + "。")
        return "\n".join(lines)

    def log_safe(self) -> dict[str, Any]:
        return {"topic": self.topic, "blocked_by": self.blocked_by,
                "examples": len(self.examples), "recorded": len(self.recorded)}


#: How a recorded entity reads back to the user. Only the fields a person would
#: recognise are named; the rest stay internal.
RECORDED_PHRASES: dict[str, str] = {
    "administrative_area": "区县记为 {value}",
    "river_name": "河流记为 {value}",
    "longitude": "经度记为 {value}",
    "latitude": "纬度记为 {value}",
    "date": "日期记为 {value}",
    "start_datetime": "开始时间记为 {value}",
    "end_datetime": "结束时间记为 {value}",
    "rainfall_mode": "降雨改用{value}",
    "lookback_years": "回看年数记为 {value}",
    "end_year": "资料期截止年份记为 {value}年",
    "station_count": "虚拟雨站数量记为 {value} 个",
}

RAINFALL_MODE_WORDS = {"historical_max": "资料期内样本最大日",
                       "explicit_period": "指定日期或时段"}


def describe_recorded(entities: dict[str, Any]) -> list[str]:
    """What to say back about the values this turn established."""
    spoken: list[str] = []
    for name, value in entities.items():
        if value in (None, ""):
            continue
        if name == "rainfall_mode":
            spoken.append(f"降雨改用{RAINFALL_MODE_WORDS.get(str(value), value)}")
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
                  '我理解你想查看当前任务的图件。想看建模子汇水区划分，还是你绘制的评价边界？')
        return Clarification(topic='visualization',question=question,
                             examples=['查看子汇水区划分图','查看评价边界图'],
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
        question = f"{question}\n（这次没接上的地方：{ambiguity}）"
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
