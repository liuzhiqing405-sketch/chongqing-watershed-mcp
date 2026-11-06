"""The conversational routing contract.

The agent has two channels and they must not blur: understanding what the user
wants, and doing something about it. This module defines the first — a
:class:`DialogueDecision` that a language model may produce, and the
:class:`EntityCandidate` it may attach to it. Nothing here executes anything.

Three properties are enforced by construction rather than by asking the model
nicely:

* **Nowhere to put a tool call.** The decision has no field for a tool name, a
  command, a script, a working directory, an MCP method or a free-form argument
  map, and every model forbids unknown fields. A model that tries to return one
  does not have it ignored — the whole response fails validation and the caller
  falls back to the deterministic path.
* **A candidate must be grounded.** Every candidate carries the user's own
  ``evidence_text``, and a candidate whose evidence is not a verbatim fragment
  of the message is dropped before anything is validated. A model cannot invent
  a coordinate and have it reach a server.
* **A route never decides.** ``intent`` may not be ``CONFIRM_WATERSHED``,
  ``REJECT_WATERSHED`` or ``CANCEL_RUN``: those are the user's decisions,
  recorded by deterministic rules. A decision the model could emit would make
  the confirmation gate, and the user's control over a long computation,
  meaningless.

These are contracts only. The router that fills them in, and the checks that
turn them into workflow actions, live in the stages that follow.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from . import domain_knowledge
from .engineering_knowledge import is_engineering_question
from .intent import (
    CANCEL_PHRASES,
    CONFIRM_PHRASES,
    DECISION_INTENTS,
    Intent,
    IntentParser,
    LLMClient,
    ParsedIntent,
    RuleParser,
    REJECT_PHRASES,
)


#: The confidence at which the rules are trusted to have understood a request
#: without help. Mirrors the parser's own threshold so the two levels cannot
#: disagree about what "the rules are sure" means.
RULE_CERTAINTY: float = IntentParser.RULE_CERTAINTY


def requests_plan_preparation(text: str) -> bool:
    """Shared by routing and dispatch: this request can only prepare a plan."""
    if re.search(r'如果|假如|假设|只问|只是询问|'
                 r'(?:不要|先别|不必|无需)[^，。；;]{0,20}(?:生成|展示|更新|修改|调整|更改|改为)', text):
        return False
    return bool(re.search(
        r'(?:生成|展示|更新|匹配).*计划|先.*计划|完成.*(?:后续|剩余|建模|流程)|继续.*(?:建模|准备|流程)|'
        r'(?:参数|黏聚力|粘聚力|摩擦角|厚度|密度|饱和度).*(?:改|调|设)', text))


#: Intents that name a conversation rather than an action. The rules place them
#: confidently, but the route is still conversation: explaining something must
#: not move the workflow, and with general chat switched off the request falls
#: back to the same bounded explainer it used before V2.
CONVERSATIONAL_INTENTS: frozenset[Intent] = frozenset({Intent.ASK_EXPLANATION})


#: Markers that tie a message to one particular event or place. "这场降雨会不
#: 会不稳定" mentions 不稳定 but is about a case, not about the concept, so it
#: is not answered from a glossary. "为什么要我确认边界" mentions a workflow
#: noun but asks about the subject in general, so it is.
SPECIFIC_CASE = ("这场", "这次", "那次", "那场", "最近", "刚才", "前面说的",
                 "这个流域", "这条河", "那条河", "本次", "目前", "当前", "这轮",
                 "上一次", "本次")

#: Shapes that make a message a question about a concept.
QUESTION_SHAPES = ("吗", "呢", "是什么", "什么是", "什么意思", "怎么理解", "为什么",
                   "能不能", "可以吗", "？", "?")


def deterministic_knowledge_topic(text: str) -> str | None:
    """A conversational topic recognised without a model, or ``None``.

    Greetings, capability questions and concept questions are cheap to spot and
    harmless to answer, so they work with no model at all. A concept word alone
    is not enough: "最近这场雨会不会不稳定" contains 不稳定 but is plainly a
    request for an analysis, and answering that from a glossary would be worse
    than asking about it.
    """
    topic = domain_knowledge.match_topic(text)
    if topic is None:
        return None
    if topic in {"greeting", "capability"}:
        return topic
    if any(cue in text for cue in SPECIFIC_CASE):
        return None
    if not any(shape in text for shape in QUESTION_SHAPES) and not re.match(
        r"\s*(?:what|why|how|can|could|does|do|explain)\b", text, re.I
    ):
        return None
    return topic


class DecisionRefused(ValueError):
    """A model returned a decision it is not allowed to make.

    Confirming, rejecting or cancelling on the model's word would defeat the
    human confirmation gate, so the whole reply is refused and the caller falls
    back to the rules.
    """


class RouteKind(StrEnum):
    """What the agent should do with this turn."""

    #: A real workflow request with the entities it needs; goes to the state machine.
    WORKFLOW_ACTION = "workflow_action"
    #: The goal is clear but something essential is missing; ask one question.
    CLARIFICATION = "clarification"
    #: A question about a concept, the agent's abilities, or the current result.
    KNOWLEDGE_CHAT = "knowledge_chat"
    #: Nothing to do with this project; say so briefly and offer the way back.
    OUT_OF_SCOPE = "out_of_scope"


#: The entity fields a model may propose. Deliberately absent: tool names,
#: commands, scripts, paths, MCP method names, run identifiers and any
#: free-form argument map. Identifiers are the servers' to mint and the rules'
#: to read back out of the user's text.
EntityField = Literal[
    "river_name",
    "administrative_area",
    "longitude",
    "latitude",
    "date",
    "start_datetime",
    "end_datetime",
    "rainfall_mode",
    "station_count",
    "station_count_mode",
    "lookback_years",
    "end_year",
    "candidate_index",
    "visualization_kind",
]

ENTITY_FIELDS: frozenset[str] = frozenset(EntityField.__args__)  # type: ignore[attr-defined]

#: Routes that may change the workflow state. Only a workflow action may; the
#: other three are conversation and must leave the state machine untouched.
STATE_CHANGING_ROUTES: frozenset[RouteKind] = frozenset({RouteKind.WORKFLOW_ACTION})

# Inspecting existing work cannot submit a scientific run or replace inputs.
# Workflow execution still retains its separate confidence and human gates.
INSPECTION_INTENTS=frozenset({Intent.CHECK_STATUS,Intent.SHOW_RESULT,Intent.SHOW_VISUALIZATION})


#: Bounds on what a candidate may carry. A value longer than this is not an
#: entity a user typed, it is a model padding its answer.
_MAX_VALUE_CHARS = 200
_MAX_EVIDENCE_CHARS = 400
_MAX_TOPIC_CHARS = 200


#: Intents a model may never produce.
#:
#: Confirmation and rejection are a human decision recorded by deterministic
#: rules; a model that could emit one would make the confirmation gate
#: meaningless. Cancelling is here for a different reason: the plan groups it
#: with confirm/reject under "确定性安全解析", and it destroys work that may have
#: been running for a long time. Reading "别管这个了" as a cancel is a risk the
#: user's own words should carry, not the model's inference. The rules still
#: recognise an explicit "取消"; the model simply cannot propose one.
MODEL_FORBIDDEN_INTENTS: frozenset[Intent] = DECISION_INTENTS | {Intent.CANCEL_RUN}


class EntityCandidate(BaseModel):
    """One entity the model believes it read out of the user's message."""

    model_config = ConfigDict(extra="forbid")

    field: EntityField
    value: str | int | float
    #: A verbatim fragment of the user's message. Required, and checked against
    #: the message before the value is validated or used.
    evidence_text: Annotated[str, Field(min_length=1, max_length=_MAX_EVIDENCE_CHARS)]

    @field_validator("value", mode="before")
    @classmethod
    def _simple_value(cls, value: Any) -> Any:
        # ``bool`` is an ``int`` in Python, so this has to run *before* pydantic
        # coerces ``True`` into ``1``; afterwards the distinction is gone.
        if isinstance(value, bool):
            raise ValueError("实体取值不能是布尔值")
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("实体取值不能为空")
            if len(text) > _MAX_VALUE_CHARS:
                raise ValueError(f"实体取值过长（上限 {_MAX_VALUE_CHARS} 字符）")
            return text
        return value

    @field_validator("evidence_text")
    @classmethod
    def _real_evidence(cls, value: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("证据片段不能为空：实体必须能在用户原文中找到")
        return text

    def is_grounded_in(self, message: str) -> bool:
        """Whether the evidence really occurs in the user's message."""
        evidence = self.evidence_text.strip()
        return (bool(evidence) and evidence in str(message)
                and (self.field not in {"river_name","administrative_area"} or str(self.value) in evidence))

    def as_text(self) -> str:
        return str(self.value)


class DialogueDecision(BaseModel):
    """What the router decided about one turn."""

    model_config = ConfigDict(extra="forbid")

    route: RouteKind
    intent: Intent = Intent.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    candidates: list[EntityCandidate] = Field(default_factory=list)
    #: Entity fields the model believes the request still needs.
    missing_fields: list[str] = Field(default_factory=list)
    #: For knowledge chat: what the question is about. Free text, never a value.
    topic: str | None = None
    #: Why the request could not be placed, when it could not.
    ambiguity: str | None = None

    @classmethod
    def from_model(cls, payload: Any) -> "DialogueDecision":
        """Validate what a model returned.

        This is the only door a model's answer comes through, so this is where
        the decisions it may not make are refused. The restriction lives here
        rather than in a field validator because the router carries the *rules*'
        verdicts in the same type — and the rules are exactly the component
        allowed to record a confirmation.
        """
        decision = cls.model_validate(payload)
        if decision.intent in MODEL_FORBIDDEN_INTENTS:
            raise DecisionRefused(
                f"模型不得产生「{decision.intent.value}」：确认、拒绝与取消只能来自用户的明确表达")
        return decision

    @field_validator("missing_fields")
    @classmethod
    def _known_fields_only(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - ENTITY_FIELDS)
        if unknown:
            raise ValueError(f"未知字段：{unknown}；允许值：{sorted(ENTITY_FIELDS)}")
        # Order-preserving de-duplication: the first mention is the priority.
        return list(dict.fromkeys(value))

    @field_validator("topic", "ambiguity")
    @classmethod
    def _bounded_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text[:_MAX_TOPIC_CHARS] or None

    @model_validator(mode="after")
    def _workflow_actions_carry_an_intent(self) -> "DialogueDecision":
        if self.route is RouteKind.WORKFLOW_ACTION and self.intent is Intent.UNKNOWN:
            raise ValueError("workflow_action 必须给出一个工作流意图")
        return self

    # ------------------------------------------------------------------ checks
    def grounded_candidates(self, message: str) -> list[EntityCandidate]:
        """Only the candidates the user's own words support."""
        return [item for item in self.candidates if item.is_grounded_in(message)]

    def ungrounded_candidates(self, message: str) -> list[EntityCandidate]:
        """Candidates dropped for having no evidence in the message."""
        return [item for item in self.candidates if not item.is_grounded_in(message)]

    @staticmethod
    def conflicts(candidates: list[EntityCandidate]) -> dict[str, list[str]]:
        """Fields proposed with two or more different values.

        A model that offers two coordinates for one field has not decided, and
        the agent must ask rather than pick one.
        """
        seen: dict[str, set[str]] = {}
        for item in candidates:
            seen.setdefault(item.field, set()).add(item.as_text())
        return {field: sorted(values) for field, values in seen.items() if len(values) > 1}

    @property
    def changes_workflow_state(self) -> bool:
        return self.route in STATE_CHANGING_ROUTES

    def log_safe(self) -> dict[str, Any]:
        """A view for the event log: no prompt text, no evidence, no values."""
        return {
            "route": self.route.value,
            "intent": self.intent.value,
            "confidence": round(self.confidence, 3),
            "candidate_fields": sorted({item.field for item in self.candidates}),
            "missing_fields": list(self.missing_fields),
            "has_topic": self.topic is not None,
            "has_ambiguity": self.ambiguity is not None,
        }


# --------------------------------------------------------------- requirements

#: What each workflow intent needs before it can be executed. The outer tuple is
#: "any one of these", the inner tuple is "all of these": a start request is
#: satisfied by a river name *or* by a coordinate pair, and a rainfall request
#: by a mode, a date, or an explicit window.
REQUIRED_ENTITIES: dict[Intent, tuple[tuple[str, ...], ...]] = {
    Intent.START_ASSESSMENT: (("administrative_area",), ("river_name",), ("longitude", "latitude")),
    Intent.SELECT_OUTLET: (("candidate_index",), ("longitude", "latitude")),
    Intent.SET_RAINFALL_PERIOD: (("rainfall_mode",), ("date",),
                                 ("start_datetime", "end_datetime"), ("station_count",), ("station_count_mode",)),
}

#: Intents with nothing left to collect. They are still gated by the state
#: machine — starting a run needs a plan, showing a result needs a run.
UNCONDITIONAL_INTENTS: frozenset[Intent] = frozenset(Intent) - frozenset(REQUIRED_ENTITIES)


def missing_entities(intent: Intent, provided: dict[str, Any]) -> list[str]:
    """The entity fields this intent still needs, most useful first."""
    alternatives = REQUIRED_ENTITIES.get(intent)
    if alternatives is None:
        return []
    for group in alternatives:
        if all(provided.get(name) not in (None, "") for name in group):
            return []
    ordered: list[str] = []
    for group in alternatives:
        for name in group:
            if provided.get(name) in (None, "") and name not in ordered:
                ordered.append(name)
    return ordered


# ------------------------------------------------------- entity validation

_DATETIME_FIELDS = frozenset({"date", "start_datetime", "end_datetime"})
_INTEGER_FIELDS = frozenset({"lookback_years", "end_year", "candidate_index", "station_count"})
_FLOAT_FIELDS = frozenset({"longitude", "latitude"})


def _normalise_datetime(field: str, value: Any) -> str:
    """A real calendar date or time, or a refusal.

    The rules already refuse ``2024-02-31`` because they build a ``date``
    object rather than matching a pattern; a model's candidate has to clear the
    same bar, so this parses too.
    """
    text = str(value).strip().replace("/", "-")
    try:
        if field == "date":
            return date.fromisoformat(text).isoformat()
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"「{text}」不是有效日期") from exc
    return parsed.strftime("%Y-%m-%d %H:%M")


def _coerce_value(field: str, value: Any) -> Any:
    if field in _FLOAT_FIELDS:
        return float(str(value).strip())
    if field in _INTEGER_FIELDS:
        return int(str(value).strip())
    if field in _DATETIME_FIELDS:
        return _normalise_datetime(field, value)
    if field == "administrative_area":
        # The action is not part of the supplied place name. The spatial
        # catalog still decides whether the normalized place actually exists.
        return RuleParser._extract_administrative_name(str(value)) or str(value).strip()
    return str(value).strip()


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors()
    return str(errors[0].get("msg")) if errors else str(exc)


def validated_payload(candidates: list[EntityCandidate]) -> tuple[dict[str, Any], list[str]]:
    """Turn candidates into a workflow payload, using the workflow's own rules.

    Each field is validated on its own so that one bad value does not discard
    the good ones, and every refusal is reported: a candidate that is dropped
    silently is a candidate the user never learns was not understood.
    """
    payload: dict[str, Any] = {}
    problems: list[str] = []
    for item in candidates:
        try:
            value = _coerce_value(item.field, item.value)
            parsed = ParsedIntent.model_validate({item.field: value})
        except ValidationError as exc:
            problems.append(f"{item.field}「{item.value}」未通过校验：{_first_error(exc)}")
            continue
        except (TypeError, ValueError) as exc:
            problems.append(f"{item.field}「{item.value}」无法解析：{exc}")
            continue
        payload[item.field] = getattr(parsed, item.field)
    return payload, problems


# ------------------------------------------------------------------- router


class DialogueRouter:
    """The model understands the turn; deterministic checks govern execution.

    Every ordinary message reaches the configured model, even when keyword
    rules found a river or an action. Only explicit human decisions bypass it.
    Rules remain available for entity validation and model-unavailable fallback.
    """

    SYSTEM_PROMPT = (
        "本项目允许自然对话开场：礼貌问候、感谢和询问系统能力属于knowledge_chat，不能仅因没有工程术语就判为out_of_scope。"
        "独立的‘你好’‘您好’‘hello’使用knowledge_chat、ask_explanation、topic=greeting；"
        "‘你能做什么’使用knowledge_chat、ask_explanation、topic=capability。这类消息的candidates与missing_fields均为空数组。"
        "问候后若同时提出工程问题或具体任务，应按后面的实质请求路由，不能只回复问候。"
        "out_of_scope用于确实无关的实质请求，不用于礼貌开场；工程知识问答不需要地点、日期或已完成计算。"
        "先核对本轮要求的交付物、当前已知输入和当前步骤，只决定一个已有意图；"
        "缺少当前步骤必需的信息才澄清，保留已知输入，不替用户补年份、地点或授权。"
        "用户询问当前数值、换算、指标口径、旧新方案差别或要求把结果写错时，"
        "使用knowledge_chat、ask_explanation、topic=current_result，由结果证据回答；"
        "请求更改报告文字不等于修改物理模型输入。"
        "询问失败原因、QA具体错误或缺失字段时，使用topic=task_artifacts读取诊断，"
        "不能因为任务失败而回复通用概念，也不能宣称当前无任务。"
        "用户索取已有图件或文件的下载时选择download_artifact，"
        "同一轮若同时索取数值与图件，选择show_visualization并提取具体visualization_kind；"
        "同时索取文件与来源追溯时选择download_artifact，仍保留所指成果类型。程序会补齐其余明确要求，"
        "单个主意图不能代表整轮全部交付已经完成。"
        "它与show_visualization（查看或生成图件）和show_result（解释数值）不同；"
        "只有用户索取下载且文件已登记，程序才能交付，不能编造链接。"
        "用户希望同时了解结果与某项产物状态时，topic=task_artifacts可核对当前结果及成果目录。"
        "context.known中的run_id和输入版本为当前任务依据；用户明确指定其他运行时，"
        "不能忽略该限定后改查当前运行，应澄清任务关联。"
        "候选行政区只包含地名，去掉显示、打开、返回等动作词。"
        "context.resolved_temporal_inputs若非空，是程序根据已知降雨日期核算的相对日期或时段；"
        "仅在用户确实要求修改降雨时采用，原理或假设问答不修改输入。"
        "改变尚未确认边界任务的日期只记录待用输入，不等于准备数据或启动计算。"
        "‘那个区’‘原区域’‘选区’是任务上下文或操作对象，不是新的行政区名；"
        "引用已有位置时沿用context.known，不为指代词编造新的地点候选。"
        '先看当前任务上下文：已有计算结果时，用户说“评测区域稳定性”“评价一下这个区域”'
        '“整体结果怎么样”“解释分布”通常是在要求解读当前成果，使用knowledge_chat、ask_explanation。'
        '应结合现有任务回答，不重复加载地图、不重置边界、不要求重新输入已知条件。'
        '用户明确要求换区域、换降雨或重新计算时，才安排相应工作流。'
        '“为什么这样分区”“参数具体取了多少”“这个文件记录了什么”也是读取当前任务证据后回答。'
        '当前结果解读的topic使用current_result；工作流文件、模型实际参数和产物问题使用task_artifacts。'
        "用户要求布设/更改虚拟雨站数量时，使用workflow_action和set_rainfall_period，"
        "提取station_count（1到100的整数）；‘六个’提取为6，station_count_mode为manual。"
        "‘恢复按面积自动布站’提取station_count_mode=auto，取消旧的手动站数。"
        "只修改站数时保留已有降雨日期或时段；还没有降雨事件时先记录站数再询问事件。"
        "‘6个雨站够吗’或‘为何用了4个雨站’是knowledge_chat，不修改任务。"
        "你是坡面稳定性初步评价智能体的对话路由器。只输出 JSON，不要输出解释。"
        "字段：route、intent、confidence、candidates、missing_fields、topic、ambiguity。\n"
        "route：workflow_action（执行已有工作流）、clarification（还缺信息，先问一个最关键的问题）、"
        "knowledge_chat（水文地质、地质灾害、地理、岩土与泛工程知识，或当前结果的含义）、"
        "out_of_scope（与工程知识及本项目均无关）。\n"
        "先结合本轮语义与会话上下文判断用户是在提问还是布置任务，再决定是否提取工作流输入。"
        "不能因出现河名、地名、日期、坐标、计算、分析、结果等词就认定用户要求执行操作。"
        "提问包括事实查询、概念解释、方法讨论、比较和假设情景；知识提问即使没有问号、"
        "以‘分析’或‘帮我’开头，也可以是 knowledge_chat，candidates 和 missing_fields 应为空。"
        "任务包括具体评价、地图选址、指定降雨、运行、进度查询、图件展示与成果下载；"
        "即使说成‘能不能…’等问句，也可能是任务。不要为了回答知识问题加载地图或索要坐标。"
        "独立知识提问不会延续上一轮计算任务；‘继续’等省略请求必须结合正在等待的步骤理解。\n"
        "按用户希望得到的结果判断，不按某一个关键词或问句形式判断。"
        "‘帮我分析水流怎样影响边坡’是原理问答；‘能否评价永川区的小流域’是安排任务。"
        "‘不对，我想问原理’是在更正你的理解，不代表拒绝流域边界；"
        "‘我说的是永川区’是补充地点，不要要求用户重复完整评价命令。"
        "context.pending_workflow 保存未完成的任务和已提供输入；插入知识问题后该任务仍可继续。"
        "继续任务时可直接使用 context.known 中已有输入，不要编造历史证据或反复索要相同字段。"
        "区分继续当前任务和回滚历史版本：用户从知识讨论回到原评价、沿用现有地点与降雨、"
        "恢复到选区步骤或重新打开选区底图时，按当前要完成的动作选择 start_assessment，"
        "不是 restore_task。restore_task 专指撤销输入修改、还原到之前保存的任务版本；"
        "不能仅因用户使用‘恢复’‘原任务’等词就选择历史版本回滚。"
        "当前停在地图选区步骤且 context.known 或 pending_workflow 已有明确地点时，"
        "用户请求返回选区并显示地图，应返回 workflow_action、start_assessment，"
        "沿用已知输入，无须重复提取地点候选或索要已经给出的日期。"
        "继续任务不等于确认边界或启动计算；仍按当前阶段的前置条件执行。"
        "已确认边界后请求完成后续建模或生成计划，使用start_hydro_run衔接固定准备流程；"
        "程序会停在展示计划，不能提前认定用户批准尚未展示的计划。"
        "context.known.parameters是面板已经保存的参数，不是让你提取的新字段。"
        "用户复述这些参数并要求更新结果、生成匹配计划时，使用start_hydro_run准备计划；"
        "不用索要已有参数、地点或日期，也不能编造参数修改意图字段。"
        "若用户要求的参数与面板记录不同，说明需要在参数面板保存后再生成计划。"
        "已取消或已结束的任务之后，用户明确指定新的雨日并要求展示计划，"
        "使用set_rainfall_period并提取日期；‘暂不启动’只禁止提交计算，不禁止准备新计划。"
        "历史最大日须保留用户给出的截止年份end_year与资料期长度lookback_years；"
        "例如截至某年的过去若干年是样本筛选条件，不是具体降雨日，也不是设计重现期。"
        "只读查询已保存方案的日期、参数和结果使用knowledge_chat，不使用restore_task。"
        "如果用户转而问问题，选择 knowledge_chat，不能因有待处理任务而强行推动工作流。"
        "若无法判断用户想问知识还是实际评价，用一句针对两种可能意图的澄清问题。\n"
        "工作流逐步执行，不要求用户一次提供全部计算参数：start_assessment 在已有河名时"
        "只打开地图供用户手动选位置，河名本身就满足这一步，不缺经纬度或降雨日期；"
        "河名和经纬度都只定位 DEM 地图；用户必须在地图上提交闭合多边形，检查覆盖并确认后才能计算。"
        "用户说‘就看前面那条河’时继承 context.known.river_name，不要反复索要已知河名。"
        "用户要求按区县自行画线选择评价区域时，使用 START_ASSESSMENT 与 administrative_area，"
        "区县名称不是 river_name。行政区只是地图定位，边界由用户在 DEM 上画闭合多边形；"
        "报出区县即可打开绘图，不要提前索要出口、坐标或降雨。知识问题仍走 KNOWLEDGE_CHAT。"
        "‘我想评价永川区的小流域’是 start_assessment，administrative_area 为永川区，"
        "不是查看已有结果，也不是河名永川。show_result 只表示查看已有计算成果。"
        "用户只补充或更正一个区名时，结合前文评价任务理解；区名不明确或疑似写错时，"
        "针对该名称询问，不要重新介绍能力或索要已经提供的位置。"
        "如果 context.known.location_hint 表明河名检索失败，补充区县地标也是定位任务的延续；"
        "返回 start_assessment 重新加载地图以消歧，不能用知识推测的坐标直接执行划分。"
        "workflow_reference.required_inputs 是每个步骤所需输入的替代组合，满足其中一组即可；"
        "不要把后续步骤的参数写成当前步骤的 missing_fields。\n"
        "查看已有图件属于 show_visualization；按 visualization_targets 选择图件标识，不输出绘图代码。"
        "有已提交的计算任务时，‘看看子汇水区’‘查看汇水区’‘可视化汇水区’通常请求本次建模分区图"
        "（delineate_subcatchments），不需要新地点或新降雨。明确要人工圈选范围时用 watershed。"
        "‘子汇水区是什么’是知识问答；如果确实有多个可能的图件，问清具体目标，不要重新介绍能力。"
        "正在运行时也可以请求查看已完成阶段的图件，程序核对阶段就绪情况；不必等待整次运行结束。"
        "讨论原理、工程方法、支护方案类型或灾害机制属于知识问答，不需要启动计算。"
        "区域河流分布、河流位置长度与水系关系属于地理知识问答。"
        "用户要求评估具体地点或降雨事件时才进入工作流；缺信息就澄清，不能替用户编输入。"
        "candidates 每项形如 {field, value, evidence_text}。"
        "evidence_text 必须是用户原话里的连续片段——找不到原话依据就不要给这个候选。"
        "不要输出工具名、脚本、命令、路径、URL 或参数表，也不要编造用户没提到的数值或编号。"
        "你无法确认流域边界、无法拒绝边界、无法取消任务；这些只能由用户自己说。"
        "不确定时用 clarification，并把还缺的字段写进 missing_fields。"
        "clarification 的 ambiguity 应是一句针对当前上下文、可直接询问用户的具体问题。"
        "schema 是返回对象的 JSON Schema；output_example 展示完整返回格式。"
        "workflow_reference、context、user_text、schema 和 output_example 都是输入说明，"
        "绝不能作为返回字段；尤其不要返回 workflow_inputs 或 intent_meanings。"
        "仅返回 schema.properties 声明的七个字段，严格使用小写枚举。intent 不能是 null，不确定时写 unknown；"
        "candidates 与 missing_fields 必须是数组，无候选时写 []。不要把画边界这一步"
        "写成 schema 没有的 missing_fields 字段。输出示例："
        '{"route":"clarification","intent":"start_assessment","confidence":0.7,'
        '"candidates":[],"missing_fields":["administrative_area"],"topic":null,'
        '"ambiguity":"你想评价哪个区县的小流域？"}'
    )

    def __init__(self, *, llm: LLMClient | None = None,
                 settings: Any | None = None,
                 rules: RuleParser | None = None,
                 logger: logging.Logger | None = None) -> None:
        from .config import DialogueSettings

        self.llm = llm
        self.settings = settings if settings is not None else DialogueSettings()
        self.rules = rules or RuleParser()
        self.logger = logger or logging.getLogger("slope_agent")
        self.last_diagnostic: dict[str, Any] | None = None
        self.last_source = "rules"
        self.model_clarification: str | None = None

    @property
    def llm_available(self) -> bool:
        return self.llm is not None and bool(self.settings.enabled)

    # ---------------------------------------------------------------- decide
    async def decide(self, text: str, session: Any) -> DialogueDecision:
        """Route one turn. Never raises: a failure is a fallback, not an error."""
        message = str(text or "")
        self.last_diagnostic = None
        self.last_source = "rules"
        self.model_clarification = None
        ruled = self.rules.parse(message, session)
        if self.llm_available:
            ruled = self._model_rule_hints(ruled, message)

        if self._explicit_user_control(message, ruled, session):
            self.last_source = "human"
            return self._from_rules(ruled, session, message)  # type: ignore[return-value]
        if ruled.intent in MODEL_FORBIDDEN_INTENTS:
            # A control word inside ordinary speech is not a human decision.
            ruled.intent = Intent.UNKNOWN
            ruled.confidence = 0.2

        if not self.llm_available:
            return self._without_model(ruled, session, message)

        try:
            payload = await self._ask(message, session)
            schema_repaired = False
            try:
                proposed = DialogueDecision.from_model(payload)
            except ValidationError as exc:
                # A valid JSON with a contract mistake gets one fresh attempt.
                # Send field/type diagnostics, never raw model output or inputs.
                issues = [{"field": ".".join(map(str, item["loc"])), "type": item["type"]}
                          for item in exc.errors()[:5]]
                payload = await self._ask(message, session, repair=issues)
                proposed = DialogueDecision.from_model(payload)
                schema_repaired = True
                self.last_diagnostic = {"reason": "invalid_route_schema", "issues": issues,
                                        "recovered": True}
            if (not schema_repaired and session.hydro_input_id and session.hydro_input_validated
                    and not session.hydro_run_id
                    and re.search(r'参数|黏聚力|粘聚力|摩擦角|厚度|密度|饱和度|复用|必要更新', message)
                    and (proposed.route is RouteKind.CLARIFICATION or
                         proposed.intent in {Intent.UNKNOWN, Intent.START_ASSESSMENT} or
                         (proposed.route is RouteKind.KNOWLEDGE_CHAT and
                          re.search(r'再.*完成.*更新|生成.*计划', message)))):
                # One bounded semantic check; the model still chooses the
                # intent, and all candidates/thresholds/human gates still apply.
                payload = await self._ask(message, session, semantic_review=True)
                try:
                    proposed = DialogueDecision.from_model(payload)
                except ValidationError as exc:
                    issues = [{"field": ".".join(map(str, item["loc"])), "type": item["type"]}
                              for item in exc.errors()[:5]]
                    payload = await self._ask(message, session, repair=issues, semantic_review=True)
                    proposed = DialogueDecision.from_model(payload)
                self.last_diagnostic = {"reason": "prepared_input_semantic_review", "recovered": True}
        except Exception as exc:  # noqa: BLE001 - the rules' answer still stands
            reason = ("invalid_route_schema" if isinstance(exc, ValidationError)
                      else "model_decision_refused" if isinstance(exc, DecisionRefused)
                      else getattr(exc, "reason", type(exc).__name__))
            self.last_diagnostic = {"reason": reason, "recovered": False}
            if isinstance(exc, ValidationError):
                self.last_diagnostic["issues"] = [
                    {"field": ".".join(map(str, item["loc"])), "type": item["type"]}
                    for item in exc.errors()[:5]]
            self.last_source = "model_fallback"
            self.logger.info("对话路由未成功，回退规则路径：%s", reason)
            if (session.hydro_input_id and session.hydro_input_validated and not session.hydro_run_id
                    and re.search(r'参数|黏聚力|粘聚力|摩擦角|厚度|密度|饱和度|复用', message)):
                return DialogueDecision(route=RouteKind.CLARIFICATION, intent=Intent.UNKNOWN, confidence=0,
                    ambiguity='已保存的参数和已验证输入仍保留。你是要据此生成新计划，还是只查询这些输入？')
            return self._without_model(ruled, session, message, note="模型路由未成功")
        self.last_source = "model"
        proposed = self._resolve_temporal_decision(proposed, message, session)
        if (proposed.intent in {Intent.SET_RAINFALL_PERIOD, Intent.START_ASSESSMENT}
                and proposed.route in {RouteKind.WORKFLOW_ACTION, RouteKind.CLARIFICATION}
                and (ruled.rainfall_mode == "historical_max" or any(
                    item.field == "rainfall_mode" and item.value == "historical_max" for item in proposed.candidates))):
            from .temporal_inputs import historical_period
            values = historical_period(message)
            if values:
                kept = [item for item in proposed.candidates if item.field not in values]
                kept.extend(EntityCandidate(field=key, value=value, evidence_text=message)
                            for key, value in values.items())
                proposed = proposed.model_copy(update={"candidates": kept})
        self.model_clarification = proposed.ambiguity
        return self._merge(ruled, proposed, message, session)

    @staticmethod
    def _resolve_temporal_decision(proposed, message, session):
        """Validate date arithmetic only after the model selected an edit task."""
        if (proposed.intent is not Intent.SET_RAINFALL_PERIOD
                or proposed.route not in {RouteKind.WORKFLOW_ACTION, RouteKind.CLARIFICATION}):
            return proposed
        from .temporal_inputs import resolve_relative_rainfall
        resolved = resolve_relative_rainfall(message, session.rainfall_request)
        if not resolved.values and not resolved.error:
            return proposed
        temporal_fields={"date","start_datetime","end_datetime","rainfall_mode"}
        kept=[item for item in proposed.candidates if item.field not in temporal_fields]
        if resolved.error:
            return proposed.model_copy(update={"route":RouteKind.CLARIFICATION,
                "candidates":kept,"missing_fields":["date"],"ambiguity":resolved.error})
        kept.extend(EntityCandidate(field=key,value=value,evidence_text=resolved.evidence)
                    for key,value in resolved.values.items())
        kept.append(EntityCandidate(field="rainfall_mode",value="explicit_period",evidence_text=resolved.evidence))
        return proposed.model_copy(update={"candidates":kept})

    @staticmethod
    def _explicit_user_control(message: str, ruled: ParsedIntent, session: Any) -> bool:
        text = message.strip().lower().strip("。！! ")
        if text.rstrip(".") in {
            "confirm boundary", "confirm the boundary", "reject boundary",
            "reject the boundary", "cancel run", "cancel the run",
        }:
            return True
        parts = [part.strip() for part in re.split(r"[，,、；;]", text) if part.strip()]
        # Only an imperative main clause grants control. A trailing question
        # must not hide it; a conditional, quote, negation or retraction cannot.
        clause = re.split(r"[，,；;。！!]", text, maxsplit=1)[0].strip()
        cancel = bool(re.fullmatch(
                r"(?:请|帮我|我决定|我要)?(?:先|立即|现在)?(?:" + "|".join(map(re.escape, CANCEL_PHRASES))
                + r")(?:当前|本次|这次|这个)?(?:任务|运行|计算|模型)?(?:吧|了)?", clause))
        retracted = bool(re.search(r"(?:不要|不用|先别|不必|不能)\s*(?:取消|停止|停下|终止)|只是问问|不是指令", text))
        if cancel and not retracted:
            ruled.intent, ruled.confidence = Intent.CANCEL_RUN, .95
            return True
        if ruled.intent not in MODEL_FORBIDDEN_INTENTS or ruled.intent is Intent.CANCEL_RUN:
            return False
        pending = session.has_pending_watershed()
        last_agent = next((turn for turn in reversed(session.recent_conversation())
                           if turn.role == "agent"), None)
        after_knowledge = bool(last_agent and last_agent.answer_source in {
            "general_model_knowledge", "builtin_knowledge"})
        if ruled.intent is Intent.CONFIRM_WATERSHED:
            phrases = set(CONFIRM_PHRASES) | {"确认该边界", "确认边界", "确认流域", "确认流域边界"}
            if parts and all(part in phrases for part in parts):
                if after_knowledge and not any(part.startswith(("确认", "confirm")) for part in parts):
                    return False
                return pending or any(part.startswith(("确认", "confirm")) for part in parts)
            # A user's completed review may precede their explicit decision.
            # Keep both clauses narrowly affirmative: conditions, quotations,
            # questions, delegation and a later retraction do not grant control.
            clauses = [part.strip() for part in re.split(r"[，,；;。！!]", text) if part.strip()]
            reviewed = (r"我(?:已经|已)?(?:核对|检查|复核|查看|看)(?:过|了)?"
                        r"(?:当前|本次|这个|该)?(?:评价范围|选区|流域边界|边界)"
                        r"(?:qa|质量检查)?(?:(?:和|与|及)(?:边界)?(?:qa|质量检查|检查结果))?"
                        r"(?:结果)?(?:无误|通过|没有问题)?")
            confirm = r"(?:我)?确认(?:当前|本次|该|这个)?(?:流域(?:边界)?|边界)"
            undecided_review = pending or (bool(session.review_id) and not session.watershed_confirmed
                                          and session.state == "watershed_qa_failed")
            if (undecided_review and len(clauses) == 2 and re.fullmatch(reviewed, clauses[0])
                    and re.fullmatch(confirm, clauses[1])):
                return True
            return bool(re.match(r"^(?:我|请)?确认(?:该|这个)?(?:流域)?边界(?:[，,。]|并|后|$)", text))
        phrases = set(REJECT_PHRASES) | {"先不确认", "拒绝该边界", "拒绝边界", "换一个位置", "换个位置", "重新绘制", "重画"}
        if after_knowledge and not any(part.startswith(("拒绝", "不确认", "先不确认", "reject")) for part in parts):
            return False
        return bool(parts and all(part in phrases for part in parts)
                    and (pending or "拒绝" in text))

    def _without_model(self, ruled: ParsedIntent, session: Any, message: str,
                       *, note: str = "") -> DialogueDecision:
        """Rules are a fallback, never a veto on a successful model route."""
        district_fact_question = bool(ruled.administrative_area and re.search(
            r"在哪(?:里|儿)?|有哪些|属于哪个|面积.*多大|人口.*多少", message))
        if self.settings.allow_general_chat and (is_engineering_question(message) or district_fact_question):
            return DialogueDecision(route=RouteKind.KNOWLEDGE_CHAT, confidence=0.6,
                                    intent=Intent.ASK_EXPLANATION,
                                    topic=domain_knowledge.match_topic(message) or "engineering_general")
        decided = self._from_rules(ruled, session, message)
        return decided or self._fallback(ruled, session, note=note, message=message)

    # ------------------------------------------------ deterministic fallback
    def _from_rules(self, ruled: ParsedIntent, session: Any,
                    message: str) -> DialogueDecision | None:
        """The rules' verdict, when they reached one that needs no model."""
        intent = ruled.intent
        if ruled.station_count_error and intent is Intent.SET_RAINFALL_PERIOD:
            return DialogueDecision(route=RouteKind.CLARIFICATION, intent=intent, confidence=.9,
                                    missing_fields=["station_count"], ambiguity=ruled.station_count_error)
        if intent in MODEL_FORBIDDEN_INTENTS:
            # A decision the user made: final, and never shown to the model.
            return DialogueDecision(route=RouteKind.WORKFLOW_ACTION, intent=intent,
                                    confidence=max(ruled.confidence, 0.9))
        if intent in CONVERSATIONAL_INTENTS and self.settings.allow_general_chat:
            return DialogueDecision(route=RouteKind.KNOWLEDGE_CHAT, intent=intent,
                                    confidence=ruled.confidence,
                                    topic=deterministic_knowledge_topic(message))
        threshold=(self.settings.clarification_confidence_threshold if intent in INSPECTION_INTENTS else RULE_CERTAINTY)
        if intent is Intent.UNKNOWN or ruled.confidence < threshold:
            return None
        # The rules match on a word; a question about that word is not the same
        # as a request to do it. "这个结果可以直接用于工程设计吗？" is read by the
        # rules as "show me the result", which is a conversation about the
        # subject, not an instruction — and answering it from the glossary is
        # what the user wanted.
        topic = deterministic_knowledge_topic(message) if self.settings.allow_general_chat else None
        if topic:
            return DialogueDecision(route=RouteKind.KNOWLEDGE_CHAT, intent=intent,
                                    confidence=ruled.confidence, topic=topic)
        provided = self.provided_entities(session, ruled)
        missing = missing_entities(intent, provided)
        if missing:
            # The rules know what the user wants but not everything they need.
            return DialogueDecision(route=RouteKind.CLARIFICATION, intent=intent,
                                    confidence=ruled.confidence, missing_fields=missing)
        return DialogueDecision(route=RouteKind.WORKFLOW_ACTION, intent=intent,
                                confidence=ruled.confidence)

    def _fallback(self, ruled: ParsedIntent, session: Any, *,
                  note: str = "", message: str = "") -> DialogueDecision:
        """What to do with no model, or when the model let us down.

        The deterministic workflow is untouched by this: an intent the rules
        placed is still executed. Only a turn nobody could place becomes a
        question — never an action. A greeting or a plain concept question is
        recognised here so that ordinary conversation does not depend on the
        model being reachable.
        """
        topic = deterministic_knowledge_topic(message) if message else None
        if topic and self.settings.allow_general_chat:
            return DialogueDecision(route=RouteKind.KNOWLEDGE_CHAT, confidence=0.6,
                                    topic=topic)
        ambiguity = note or "、".join(ruled.notes) or "没有从这句话里看出要做什么"
        if ruled.intent is not Intent.UNKNOWN:
            provided = self.provided_entities(session, ruled)
            return DialogueDecision(route=RouteKind.CLARIFICATION, intent=ruled.intent,
                                    confidence=ruled.confidence,
                                    missing_fields=missing_entities(ruled.intent, provided),
                                    ambiguity=ambiguity)
        recovery = self._contextual_recovery_question(session)
        if recovery:
            return DialogueDecision(route=RouteKind.CLARIFICATION, intent=Intent.UNKNOWN,
                                    confidence=0.0, ambiguity=recovery)
        return DialogueDecision(route=RouteKind.CLARIFICATION, intent=Intent.UNKNOWN,
                                confidence=0.0, ambiguity=ambiguity)

    def _contextual_recovery_question(self, session: Any) -> str | None:
        """Retain task context without deciding an unclear turn on the user's behalf."""
        if session.has_pending_watershed():
            return ("本轮未能可靠理解你的话，已提交的边界仍保留并等待你核对。"
                    "你想继续检查这份边界，还是询问其他问题？")
        pending = getattr(session, "pending_workflow", {})
        if pending.get("intent") not in {"start_assessment", "select_outlet", "unknown", None}:
            return None
        # Explicitly disputed fields must not become a chosen place in the reply.
        known = self.provided_entities(session)
        location = known.get("administrative_area") or known.get("river_name")
        coordinate_pair = known.get("longitude") is not None and known.get("latitude") is not None
        if not location and not coordinate_pair:
            return None
        label = f"「{location}」" if location else "你之前提供的经纬度位置"
        if session.selection_context.get("map_id"):
            return (f"本轮未能可靠理解你的话，{label}的定位地图仍保留。"
                    "你想继续在当前 DEM 地图上绘制评价边界，还是询问其他问题？")
        return (f"本轮未能可靠理解你的话，你之前提供的{label}仍保留。"
                "你想继续为该位置加载 DEM 地图并手绘评价边界，还是询问其他问题？")

    # ------------------------------------------------------------- the model
    async def _ask(self, message: str, session: Any, *, repair=None, semantic_review=False) -> dict[str, Any]:
        request = {"schema": self.schema(), "output_example": self.output_example(),
                   "workflow_reference": self.workflow_reference(),
                   "context": self.context(session), "user_text": message}
        if semantic_review:
            request['semantic_review'] = (
                '当前输入已导入并验证，parameters是面板已经保存的当前参数。先核对用户给出的值是否已保存；'
                '黏聚力、摩擦角、厚度等参数名称不是河名或行政区。值相同时不用要求再次保存。'
                '用户要求复用已有输入并完成必要更新时，应先用start_hydro_run生成展示计划，'
                '而不是重复定位地图；准备计划不需要运行批准。尚未展示的计划不能被提前批准，'
                '实际提交计算仍需之后的明确批准。只有缺少当前步骤所需信息时才澄清。'
                'start_hydro_run沿用保存输入，candidates必须为空数组[]；参数数值不可映射为river_name等地点字段。'
                '若用户只是问原理则仍为knowledge_chat；若参数值不同且尚未保存，说明需先保存参数。')
        from .temporal_inputs import resolve_relative_rainfall
        temporal = resolve_relative_rainfall(message, session.rainfall_request)
        if temporal.values:
            request["context"]["resolved_temporal_inputs"] = temporal.values
        elif temporal.error:
            request["context"]["temporal_input_question"] = temporal.error
        if repair:
            request["schema_repair"] = {"issues": repair,
                "instruction": ("上一轮 JSON 不符合输出契约。只返回 schema.properties 的七个字段，"
                                "字段格式参照 output_example；删除额外说明字段，"
                                "重新判断语义，不要复制 schema 或 workflow_reference。")}
        payload = await self.llm.complete_json(  # type: ignore[union-attr]
            system=self.SYSTEM_PROMPT,
            user=json.dumps(request, ensure_ascii=False),
        )
        if not isinstance(payload, dict):
            raise ValueError("模型没有返回 JSON 对象")
        return payload

    @staticmethod
    def schema() -> dict[str, Any]:
        """Only the output contract; execution requirements are separate input context."""
        schema = DialogueDecision.model_json_schema()
        schema["required"] = list(DialogueDecision.model_fields)
        schema["$defs"]["Intent"]["enum"] = [
            item.value for item in Intent if item not in MODEL_FORBIDDEN_INTENTS]
        schema["properties"]["missing_fields"]["items"]["enum"] = sorted(ENTITY_FIELDS)
        return schema

    @staticmethod
    def output_example() -> dict[str, Any]:
        return {"route": "clarification", "intent": "start_assessment", "confidence": 0.7,
                "candidates": [], "missing_fields": ["administrative_area"], "topic": None,
                "ambiguity": "你想评价哪个区县的小流域？"}

    @staticmethod
    def workflow_reference() -> dict[str, Any]:
        from .visualization_reference import TARGETS
        return {
            "visualization_targets":dict(TARGETS),
            "required_inputs": {intent.value: [list(group) for group in alternatives]
                                for intent, alternatives in REQUIRED_ENTITIES.items()},
            "intent_meanings": {
                "restore_task": "明确要求撤销修改、回滚或还原到之前保存的任务版本；"
                                "不表示从知识问答返回当前任务，也不表示重开选区地图。",
                "start_assessment": "请求评价具体地点，或沿用当前任务输入继续/返回选区步骤；"
                                    "先加载 DEM 供手绘边界，不确认边界、不提交计算。",
                "select_outlet": "补充位置；只定位 DEM 供手绘，不自动划分",
                "set_rainfall_period": "为待评价范围指定实际降雨事件",
                "start_hydro_run": "衔接已验证输入、面板已保存参数的固定准备流程，展示计划；仅明确批准现有计划才提交运行",
                "show_result": "查看该会话已有真实计算成果",
                "show_visualization":"查看当前任务已有成果图，不能启动、修改或确认计算；由程序校验类型和阶段。",
                "ask_explanation": "问原理、方法或现有结果含义",
            },
        }

    def context(self, session: Any) -> dict[str, Any]:
        """What the router is allowed to see: state and ids, never a path."""
        return {
            "current_state": session.state,
            "pending_action": session.pending_action(),
            "last_clarification": session.last_clarification,
            "last_workflow_intent": session.intent,
            "pending_workflow": getattr(session, "pending_workflow", {}),
            "progress": {"map_loaded": bool(session.selection_context.get("map_id")),
                         "boundary_submitted": bool(session.review_id),
                         "spatial_qa_passed": session.watershed_qa_passed,
                         "run_planned": session.hydro_planned,
                         "input_imported": bool(session.hydro_input_id),
                         "input_validated": session.hydro_input_validated,
                         "has_run": bool(session.hydro_run_id),
                         "run_succeeded": session.hydro_run_status == "succeeded"},
            "has_pending_watershed": session.has_pending_watershed(),
            "watershed_confirmed": session.watershed_confirmed,
            "known": {
                "river_name": session.river_name,
                "administrative_area": getattr(session,"administrative_area",None),
                "selection_mode": getattr(session,"selection_mode",None),
                "location_mode":session.selection_context.get("location_mode"),
                "map_center":session.selection_context.get("map_center"),
                "location_hint": session.location_hint,
                "candidate_count": len(session.outlet_candidates or []),
                "rainfall_request": dict(session.rainfall_request or {}),
                "review_id": session.review_id,
                "hydro_run_id": session.hydro_run_id,
                "hydro_run_status": session.hydro_run_status,
                "hydro_input_id": session.hydro_input_id,
                "parameters": {"preset": session.parameter_preset,
                               "overrides": dict(session.parameter_overrides)},
                "completed_stages":session.run_status_detail.get("completed_stages",[]),
                "current_stage":session.run_status_detail.get("current_stage"),
            },
            "recent_conversation": [
                {"role": turn.role, "text": turn.text}
                for turn in session.recent_conversation(self.settings.recent_turns)
            ],
        }

    # ------------------------------------------------------------- the merge
    @staticmethod
    def _non_specific_model_place(value: str, message: str) -> bool:
        """Do not reinterpret a model-routed context reference as a new place."""
        return (bool(re.fullmatch(r'[+-]?\d+(?:\.\d+)?\s*(?:kPa|m|度|°)?', value, re.IGNORECASE))
                or bool(re.fullmatch(r'(?:饱和|天然|初始)?(?:黏聚力|粘聚力|内?摩擦角|土层厚度|堆积层厚度|密度|饱和度)', value))
                or value.endswith(("地区","小区","山区","灾区","评价区","研究区","工作区","选区","分区","区域"))
                or bool(re.search(r"(?:(?:这|那)(?:一)?(?:个|片)?|该|本|上述|前述|原定|原先|原来)(?:区|县)$",value))
                or bool(re.search(re.escape(value)+r"(?:域|间|划)",message)))

    @classmethod
    def _model_rule_hints(cls, ruled, message):
        # This filter belongs only to successful model routing. The standalone
        # rule parser and its model-unavailable fallback remain unchanged.
        updates = {}
        if ruled.administrative_area and cls._non_specific_model_place(ruled.administrative_area,message):
            updates["administrative_area"] = None
        if ruled.river_name and re.search(re.escape(ruled.river_name) + r"[区县]", message):
            # The river heuristic can truncate an administrative name at 川,
            # e.g. 打开永川区 -> 打开永川. Do not replace a known district with
            # that fragment when the model is continuing the current task.
            updates["river_name"] = None
        return ruled.model_copy(update=updates) if updates else ruled

    def _merge(self, ruled: ParsedIntent, proposed: DialogueDecision,
               message: str, session: Any) -> DialogueDecision:
        """Respect the model's route, then validate inputs only for tasks."""
        ruled = self._model_rule_hints(ruled,message)
        if proposed.route in {RouteKind.KNOWLEDGE_CHAT, RouteKind.OUT_OF_SCOPE}:
            if proposed.route is RouteKind.KNOWLEDGE_CHAT and not self.settings.allow_general_chat:
                return self._fallback(ruled, session, note="对话功能已关闭")
            if proposed.route is RouteKind.OUT_OF_SCOPE and not self.settings.allow_out_of_scope_reply:
                return self._fallback(ruled, session, note="超范围回复已关闭")
            return proposed.model_copy(update={"candidates": [], "missing_fields": []})
        if proposed.intent is Intent.START_HYDRO_RUN:
            # This intent uses the already saved input/parameters. Irrelevant
            # place candidates cannot conflict, invalidate inputs or be saved.
            proposed = proposed.model_copy(update={"candidates": [], "missing_fields": []})
        if (ruled.rainfall_period_error
                and proposed.intent in {Intent.START_ASSESSMENT, Intent.SET_RAINFALL_PERIOD}):
            return proposed.model_copy(update={"route":RouteKind.CLARIFICATION,
                "candidates":[c for c in proposed.candidates if c.field not in
                              {"date", "start_datetime", "end_datetime", "rainfall_mode"}],
                "missing_fields":["start_datetime", "end_datetime"],
                "ambiguity":ruled.rainfall_period_error})
        if ruled.station_count_error:
            return DialogueDecision(route=RouteKind.CLARIFICATION, intent=Intent.SET_RAINFALL_PERIOD,
                                    confidence=.9, missing_fields=["station_count"],
                                    ambiguity=ruled.station_count_error)
        grounded = [item for item in proposed.grounded_candidates(message)
                    if item.field not in {"administrative_area", "river_name"} or not self._non_specific_model_place(str(item.value),message)]
        dropped = proposed.ungrounded_candidates(message)
        if dropped:
            self.logger.info("丢弃 %d 个没有原文依据的实体候选", len(dropped))
        proposed = proposed.model_copy(update={"candidates": grounded})

        conflicts = DialogueDecision.conflicts(grounded)
        if conflicts:
            return DialogueDecision(route=RouteKind.CLARIFICATION,
                                    intent=proposed.intent,
                                    confidence=proposed.confidence,
                                    candidates=grounded,
                                    missing_fields=sorted(conflicts),
                                    ambiguity="同一项给出了多个取值，需要确认")

        payload, problems = validated_payload(grounded)
        if (proposed.intent is Intent.SET_RAINFALL_PERIOD
                and any(item.field in {"station_count", "station_count_mode"} and item.field not in payload
                        for item in grounded)):
            return DialogueDecision(route=RouteKind.CLARIFICATION, intent=proposed.intent, confidence=.9,
                                    missing_fields=["station_count"], ambiguity="请提供1～100个虚拟雨站，或选择按面积自动布站。")
        if problems:
            self.logger.info("实体候选未通过本地校验：%s", problems[:3])

        # Current deterministic coordinates/dates win over model suggestions;
        # session values are defaults, and a supported model name can refine
        # the heuristic river matcher (e.g. "重庆的嘉陵江" -> "嘉陵江").
        from_rules = {name: value for name, value in
                      ruled.model_dump().items()
                      if name in ENTITY_FIELDS and value not in (None, "")}
        if "river_name" in payload or "administrative_area" in payload:
            from_rules.pop("river_name", None)
            from_rules.pop("administrative_area", None)
        merged_values = {**self.provided_entities(session), **payload, **from_rules}
        current_location = {**payload, **from_rules}
        if not any(current_location.get(name) is not None for name in (
                "administrative_area", "river_name", "longitude", "latitude")):
            current_location.update(getattr(session, "pending_workflow", {}).get("entities", {}))
        has_new_name = any(current_location.get(name) for name in ("administrative_area", "river_name"))
        has_new_coordinate = any(current_location.get(name) is not None for name in ("longitude", "latitude"))
        if has_new_coordinate and not has_new_name:
            merged_values.update(river_name=None, administrative_area=None)
        elif has_new_name and not has_new_coordinate:
            merged_values.update(longitude=None, latitude=None)

        # The model has identified an assessment task but may ask for a precise
        # point. A known river enables the manual map, which is how this workflow
        # clarifies that point. Later-stage parameters cannot block map entry.
        if (proposed.route is RouteKind.CLARIFICATION
                and proposed.intent is Intent.START_ASSESSMENT
                and proposed.confidence >= self.settings.clarification_confidence_threshold
                and (merged_values.get("river_name") or merged_values.get("administrative_area"))
                and proposed.missing_fields
                and not proposed.ambiguity
                and set(proposed.missing_fields) <= {"administrative_area","river_name", "longitude", "latitude"}):
            # This only opens a map to clarify the point; it cannot delineate
            # or run a model. Location uncertainty must not block disambiguation.
            return DialogueDecision(route=RouteKind.WORKFLOW_ACTION,
                                    intent=Intent.START_ASSESSMENT,
                                    confidence=proposed.confidence,
                                    candidates=[item for item in grounded if item.field in payload])

        if proposed.route is RouteKind.WORKFLOW_ACTION:
            from .temporal_inputs import resolve_relative_rainfall
            local_date = bool(ruled.date or resolve_relative_rainfall(message, session.rainfall_request).values)
            # Recording a locally verified date before any approved boundary is
            # an input draft, not permission to prepare data or submit a run.
            verified_rainfall_draft = (proposed.intent is Intent.SET_RAINFALL_PERIOD
                and local_date and not session.watershed_confirmed
                and not any((session.rainfall_package_id,session.model_input_package_id,
                             session.hydro_input_id,session.hydro_run_id)))
            verified_plan_preparation = (proposed.intent is Intent.START_HYDRO_RUN
                and session.hydro_input_id and session.hydro_input_validated
                and not session.hydro_planned and not session.hydro_run_id
                and requests_plan_preparation(message))
            threshold = (self.settings.clarification_confidence_threshold
                         if verified_rainfall_draft or verified_plan_preparation or proposed.intent in {Intent.START_ASSESSMENT, Intent.SELECT_OUTLET} | INSPECTION_INTENTS
                         else self.settings.workflow_confidence_threshold)
            if proposed.ambiguity:
                return DialogueDecision(route=RouteKind.CLARIFICATION, intent=proposed.intent,
                                        confidence=proposed.confidence, candidates=grounded,
                                        missing_fields=missing_entities(proposed.intent, merged_values),
                                        ambiguity=proposed.ambiguity)
            # Locating a map clarifies the study domain; it grants no right to
            # submit, confirm or calculate. Other steps retain their threshold.
            if proposed.confidence < threshold:
                return DialogueDecision(route=RouteKind.CLARIFICATION,
                                        intent=proposed.intent,
                                        confidence=proposed.confidence,
                                        candidates=grounded,
                                        missing_fields=missing_entities(
                                            proposed.intent, merged_values),
                                        ambiguity="把握不足，先确认要做什么")
            missing = missing_entities(proposed.intent, merged_values)
            if missing:
                return DialogueDecision(route=RouteKind.CLARIFICATION,
                                        intent=proposed.intent,
                                        confidence=proposed.confidence,
                                        candidates=grounded,
                                        missing_fields=missing)
            # Drop candidates the model produced that validation refused, so
            # nothing unusable is carried forward.
            kept = [item for item in grounded if item.field in payload]
            return DialogueDecision(route=RouteKind.WORKFLOW_ACTION,
                                    intent=proposed.intent,
                                    confidence=proposed.confidence,
                                    candidates=kept)

        if proposed.route is RouteKind.KNOWLEDGE_CHAT and not self.settings.allow_general_chat:
            return self._fallback(ruled, session, note="对话功能已关闭")
        if proposed.route is RouteKind.OUT_OF_SCOPE and not self.settings.allow_out_of_scope_reply:
            return self._fallback(ruled, session, note="超范围回复已关闭")
        if proposed.route is RouteKind.CLARIFICATION and proposed.confidence < \
                self.settings.clarification_confidence_threshold and not proposed.missing_fields:
            return DialogueDecision(route=RouteKind.CLARIFICATION, intent=proposed.intent,
                                    confidence=proposed.confidence,
                                    ambiguity=proposed.ambiguity or "需要更多信息")
        return proposed

    # ------------------------------------------------------------ to workflow
    def provided_entities(self, session: Any, ruled: ParsedIntent | None = None) -> dict[str, Any]:
        """What is known about the user's request: the session, plus this turn."""
        request = session.rainfall_request or {}
        outlet = session.requested_outlet or session.selection_context.get("map_center") or {}
        known: dict[str, Any] = {
            "river_name": session.river_name,
            "administrative_area": getattr(session,"administrative_area",None),
            "longitude": outlet.get("longitude"),
            "latitude": outlet.get("latitude"),
            "candidate_index": None,
            "rainfall_mode": request.get("mode"),
            "station_count": request.get("station_count"),
            "station_count_mode": "manual" if request.get("station_count") is not None else None,
            "date": request.get("date"),
            "start_datetime": request.get("start_datetime"),
            "end_datetime": request.get("end_datetime"),
            "lookback_years": request.get("lookback_years"),
            "end_year": request.get("end_year"),
            "visualization_kind": None,
        }
        pending = getattr(session, "pending_workflow", {}).get("entities", {})
        for name, value in pending.items():
            if name in ENTITY_FIELDS and value not in (None, ""):
                known[name] = value
        for name in getattr(session, "pending_workflow", {}).get("unresolved_fields", []):
            if name in ENTITY_FIELDS:
                known[name] = None
        if ruled is not None:
            for name in ENTITY_FIELDS:
                value = getattr(ruled, name, None)
                if value not in (None, ""):
                    known[name] = value
        return known

    def to_parsed_intent(self, decision: DialogueDecision, message: str,
                         session: Any) -> tuple[ParsedIntent, list[str]]:
        """The workflow's own request object for an accepted decision.

        The rules are re-read (they are deterministic and cheap), then the
        decision's grounded candidates fill in what the rules did not find —
        each one validated by the workflow's existing validators.
        """
        ruled = self.rules.parse(str(message or ""), session)
        if self.last_source == "model":
            ruled = self._model_rule_hints(ruled,message)
        if (self.last_source == "model"
                and decision.intent is Intent.SET_RAINFALL_PERIOD
                and decision.route is RouteKind.CLARIFICATION):
            from .temporal_inputs import resolve_relative_rainfall
            if resolve_relative_rainfall(message, session.rainfall_request).error:
                # An unresolved offset must not smuggle an absolute-date hint
                # into the active inputs while the user is being asked to clarify.
                ruled.date = ruled.start_datetime = ruled.end_datetime = None
                ruled.rainfall_mode = None
        candidates=decision.grounded_candidates(message)
        if self.last_source == "model":
            candidates=[item for item in candidates if item.field!="administrative_area"
                        or not self._non_specific_model_place(str(item.value),message)]
        payload, problems = validated_payload(candidates)
        conflicts = DialogueDecision.conflicts(decision.grounded_candidates(message))
        for name in conflicts:
            payload.pop(name, None)
            setattr(ruled, name, None)
        if "administrative_area" in payload and "river_name" not in payload:
            # A model-grounded district replaces the heuristic that treats names
            # ending in 川 (e.g. 永川) as rivers. This is semantic precedence.
            ruled.river_name=None
        for name, value in payload.items():
            current = getattr(ruled, name, None)
            if current in (None, "") or name in {"river_name", "administrative_area"}:
                setattr(ruled, name, value)
        if ruled.rainfall_period_error:
            ruled.date = ruled.start_datetime = ruled.end_datetime = ruled.rainfall_mode = None
        elif ruled.start_datetime and ruled.end_datetime and not (
                ruled.date and ruled.start_datetime in {f"{ruled.date} 00:00", f"{ruled.date} 00:00:00"}
                and ruled.end_datetime in {f"{ruled.date} 23:00", f"{ruled.date} 23:00:00"}):
            # A complete window must not coexist with a coarse model date hint.
            ruled.date = None
        ruled.intent = decision.intent
        if decision.route is RouteKind.WORKFLOW_ACTION:
            known = self.provided_entities(session)
            if decision.intent in {Intent.START_ASSESSMENT, Intent.SELECT_OUTLET}:
                fields = ("administrative_area", "river_name", "longitude", "latitude")
                if not ruled.river_name and not ruled.administrative_area:
                    if ruled.longitude is not None or ruled.latitude is not None:
                        fields = ("longitude", "latitude")
                    elif ruled.candidate_index is None:
                        pending = getattr(session, "pending_workflow", {}).get("entities", {})
                        mode = session.selection_context.get("location_mode")
                        if pending.get("administrative_area") or pending.get("river_name"):
                            fields = ("administrative_area", "river_name")
                        elif any(pending.get(k) is not None for k in ("longitude", "latitude")):
                            fields = ("longitude", "latitude")
                        elif mode == "district":
                            fields = ("administrative_area",)
                        elif mode == "river":
                            fields = ("river_name",)
                        elif mode == "coordinates":
                            fields = ("longitude", "latitude")
                    else:
                        fields = ()
            elif decision.intent is Intent.SET_RAINFALL_PERIOD:
                fields = ("rainfall_mode", "date", "start_datetime", "end_datetime")
            else:
                fields = ()
            # Inherit a task reference only when this turn supplies no new
            # location/event. A new river must not inherit the old coordinates.
            if fields and (fields == ("longitude", "latitude")
                           or not any(getattr(ruled, name) is not None for name in fields)):
                for name in fields:
                    if getattr(ruled, name) is None and known.get(name) is not None:
                        setattr(ruled, name, known[name])
        if decision.route in {RouteKind.KNOWLEDGE_CHAT, RouteKind.OUT_OF_SCOPE}:
            for name in ENTITY_FIELDS:
                setattr(ruled, name, None)
        elif decision.intent not in {Intent.START_ASSESSMENT,Intent.SELECT_OUTLET,Intent.SET_RAINFALL_PERIOD,Intent.UNKNOWN}:
            # A place/date noun in a viewing request is not an instruction to
            # replace inputs. Rule hints must not invalidate the current run.
            for name in ENTITY_FIELDS-{'visualization_kind'}:
                setattr(ruled,name,None)
        ruled.source = self.last_source if self.last_source != "rules" else ("rules+model" if payload else "rules")
        ruled.confidence = decision.confidence
        return ruled, problems
