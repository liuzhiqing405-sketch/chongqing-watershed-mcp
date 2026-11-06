"""Turn a user's words into a structured intent and entities.

Two parsers, used in this order:

1. **Rules** — identifiers, coordinates, dates and explicit confirm/reject
   wording. Deterministic, always available, and the only parser that can
   decide a confirmation.
2. **Optional language model** — fills in a free-form intent the rules could not
   place, and returns JSON validated against :class:`ParsedIntent`.

The model's output schema has no field for a tool name, a script or a command,
so a prompt that asks for one cannot produce one. Its ``intent`` must be one of
the workflow's own values; anything else is discarded and the rules' verdict
stands.

Both parsers receive the session state and the recent conversation, which is
what makes a bare "确认" land on the right object instead of guessing.
"""

from __future__ import annotations

import json
import logging
import re
from enum import StrEnum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, field_validator

from .models import AgentSession


class Intent(StrEnum):
    """The intents this release understands."""

    START_ASSESSMENT = "start_assessment"
    SELECT_OUTLET = "select_outlet"
    CONFIRM_WATERSHED = "confirm_watershed"
    REJECT_WATERSHED = "reject_watershed"
    SET_RAINFALL_PERIOD = "set_rainfall_period"
    START_HYDRO_RUN = "start_hydro_run"
    CHECK_STATUS = "check_status"
    CANCEL_RUN = "cancel_run"
    RESTORE_TASK = "restore_task"
    SHOW_RESULT = "show_result"
    SHOW_VISUALIZATION = "show_visualization"
    DOWNLOAD_ARTIFACT = "download_artifact"
    ASK_EXPLANATION = "ask_explanation"
    UNKNOWN = "unknown"


#: Phrases that count as an explicit confirmation. Nothing else does — in
#: particular, a bare "继续" is *not* a confirmation, because the workflow may
#: be waiting for something other than a boundary decision.
CONFIRM_PHRASES = (
    "确认", "确定", "就用这个", "就用刚才那个", "用这个", "用刚才那个", "就用它",
    "没有问题", "没问题", "可以，就这个", "可以就这个", "同意", "对的", "是的",
    "approved", "confirm", "yes",
)

REJECT_PHRASES = (
    "拒绝", "不行", "不对", "不是这个", "换一个", "重新选", "重来", "不要这个",
    "不确认", "不用这个", "这个不好", "reject", "no",
)

#: Wording that asks to advance a pending step without naming it. Kept separate
#: from confirmation because it must never decide a watershed boundary.
CONTINUE_PHRASES = ("继续", "下一步", "接着", "然后呢", "go on", "continue")

START_PHRASES = ("开始运行", "开始计算", "启动", "开始任务", "跑起来", "运行吧", "开始吧", "执行")
STATUS_PHRASES = ("状态", "进度", "怎么样了", "到哪了", "好了吗", "完成了吗", "还在跑吗",
                  "跑完了吗", "跑完", "算完了吗", "结束了吗")
CANCEL_PHRASES = ("取消", "停止", "停下", "终止", "不要跑了", "别跑了")


def awaits_panel_approval(text: str) -> bool:
    """An announced future approval is not the approval itself."""
    return bool(re.search(r"(?:我会|我将|待|等待|等|需要先).*?(?:面板|按钮).*?(?:批准|确认)"
                          r"|(?:待|等待|等).*?(?:我|面板).*?(?:批准|确认)后", text))
RESULT_PHRASES = ("结果", "结果怎么样", "看看结果", "结论", "查出什么")
EXPLAIN_PHRASES = ("为什么", "解释", "什么意思", "怎么理解", "是什么", "怎么算", "代表什么",
                   "什么是", "为何")
VISUALIZATION_PHRASES = ("图", "图件", "可视化", "画个", "看图", "plot", "figure")

#: Asking what figures exist, rather than for a particular one.
CATALOG_PHRASES = ("有哪些图", "有什么图", "能看什么图", "可以看什么图", "图件类型",
                   "有哪些图件", "能看哪些图", "有什么图件")
RAINFALL_PHRASES = ("降雨", "雨量", "暴雨", "降雨时段", "历史最大", "最大日", "全天")

#: Asking for a file rather than for a picture. One of these words is what makes
#: a request a download: "安全系数的图" asks for a figure, "下载安全系数的 TIFF"
#: asks for the file behind it.
DOWNLOAD_PHRASES = ("下载", "导出", "另存", "保存为", "存为", "download")

#: Wording that asks *how* one downloads, which is a question about the system
#: rather than a request to download something.
DOWNLOAD_QUESTION_MARKERS = ("怎么下载", "如何下载", "在哪里下载", "哪里下载", "能不能下载",
                             "可以下载吗", "支持下载吗", "怎么导出", "如何导出")

#: The file a user names, mapped onto the format the service publishes. Order
#: matters: a bundle is recognised before the bare archive word inside it.
ARTIFACT_FORMAT_ALIASES: dict[str, tuple[str, ...]] = {
    "model_bundle": ("模型包", "swmm包", "swmm成果包", "swmm模型包", "复算包",
                     "建模成果包", "成果包"),
    "xlsx": ("excel", "xlsx", "xls", "工作簿", "电子表格"),
    "geotiff": ("geotiff", "geo-tiff", "tiff", "tif", "栅格", "原始栅格"),
    "csv": ("csv", "逗号分隔", "原始表", "表格"),
    "inp": ("inp",),
    "rpt": ("rpt",),
    "out": ("out",),
    "zip": ("zip", "压缩包"),
}

#: Figure names a user might ask for, mapped onto the server's catalog names.
#: Only names the server actually offers appear here.
VISUALIZATION_ALIASES: dict[str, str] = {
    "最小fs": "safety_minimum_factor",
    "最小Fs": "safety_minimum_factor",
    "最小FS": "safety_minimum_factor",
    "安全系数": "safety_minimum_factor",
    "最小安全系数": "safety_minimum_factor",
    "失稳持续时间": "safety_unstable_duration",
    "失稳时间": "safety_unstable_duration",
    "孔隙水压力": "safety_max_pore_pressure",
    "孔压": "safety_max_pore_pressure",
    "水深": "safety_max_depth",
    "流速": "safety_max_velocity",
    "速度": "safety_max_velocity",
    "径流": "swmm_runoff_hydrograph",
    "径流过程": "swmm_runoff_hydrograph",
    "峰值径流": "swmm_peak_runoff",
    "累积径流": "swmm_cumulative_runoff",
    "连续性": "swmm_continuity",
    "流域": "watershed",
    "流域边界": "watershed",
    "流域图": "watershed",
    "子汇水区": "delineate_subcatchments",
    "降雨过程": "rainfall_series",
    "累积降雨": "rainfall_cumulative",
    "雨站": "rainfall_station_map",
    "敏感性": "analysis_sensitivity_ranking",
    "参数贡献": "sobol_contribution",
    "qa": "qa_dashboard",
}

#: Values that must never appear in a river name, so a name cannot smuggle a
#: path or a command into a tool argument.
UNSAFE_NAME_PATTERN = re.compile(r"[\\/;|&<>$`\"'\n\r\t]|\b(?:os|sys|subprocess)\.|\bimport\b")

_LONGITUDE = re.compile(
    r"(?:东经|经度|longitude|lon)\s*[:：]?\s*(\d{1,3}(?:\.\d+)?)\s*[°度]?", re.IGNORECASE)
_LATITUDE = re.compile(
    r"(?:北纬|纬度|latitude|lat)\s*[:：]?\s*(\d{1,2}(?:\.\d+)?)\s*[°度]?", re.IGNORECASE)
_COORDINATE_PAIR = re.compile(
    r"(\d{2,3}\.\d+)\s*[,，\s]\s*(\d{1,2}\.\d+)")
_LONLAT_SUFFIX = re.compile(
    r"(\d{1,3}(?:\.\d+)?)\s*°?\s*[EeＥ]\s*[,，]?\s*(\d{1,2}(?:\.\d+)?)\s*°?\s*[NnＮ]")

_DATE_CN = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?")
_DATE_ISO = re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})")
_TIME_RANGE = re.compile(
    r"(\d{1,2})\s*[:：]\s*(\d{2})\s*(?:到|-|~|—|至)\s*(\d{1,2})\s*[:：]\s*(\d{2})")
_HOUR_RANGE = re.compile(r"(\d{1,2})\s*(?:点|时)\s*(?:到|-|~|—|至)\s*(\d{1,2})\s*(?:点|时)")
_CANDIDATE = re.compile(r"第\s*([0-9一二三四五六七八九十]+)\s*(?:个|号|条)")
_CANDIDATE_SHORT = re.compile(r"(?:选择|选|用)\s*([0-9])\s*(?:号|个)?")
_MAX_DAY = re.compile(r"(?:历史最大|最大日|样本最大|最大降雨日)")
_WHOLE_DAY = re.compile(r"(?:全天|整日|当日|一整天)")
_LOOKBACK = re.compile(r"(?:近|最近|过去)\s*(\d{1,3})\s*年")

_NUMBER_WORDS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                 "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

#: A watercourse name: Chinese characters ending in a water word. The match is
#: greedy so the longest plausible name is taken, then a leading verb is
#: stripped — "分析大宁河" must yield 大宁河, not 分析大宁河.
_RIVER_NAME = re.compile(r"([一-龥]{1,32}(?:河|江|溪|川|沟|库))")
_ADMINISTRATIVE_NAME = re.compile(r"([一-龥]{2,12}(?:区|县))")

#: Verbs and particles that commonly precede a river name in a request.
LEADING_VERBS = (
    "能不能", "能否", "可否", "可以", "你能",
    "帮我", "请", "我要", "我想", "给我", "麻烦", "分析一下", "分析", "评估", "评价",
    "计算", "研究", "查询一下", "查询", "查一下", "查看", "看看", "看一下", "看",
    "显示", "打开", "展示", "返回", "回到",
    "测一下", "算一下", "算算", "用", "把", "对", "在", "就", "拿", "选",
    # Changing one's mind is phrased with a verb in front of the new name:
    # "换成嘉陵江" is a river called 嘉陵江, not one called 换成嘉陵江.
    "换成", "换成那个", "改用", "改看", "改成", "改为", "切换到", "定位到", "选择", "换",
    "这次", "本次", "此次", "现在", "接下来", "重新", "先", "仍然", "继续",
)

#: Words that read like a river name but are not one.
NOT_A_RIVER = ("降雨", "降水", "雨水", "水系", "水量", "洪水", "径流", "流域稳定",
               "哪些", "什么", "多少", "几条", "哪里", "哪儿", "哪条", "如何", "怎么", "为什么")

#: Words that point at something already said instead of naming it. A phrase
#: built from these is a reference, not a river name: reading "前面说的那条河"
#: as a name would ask the outlet recommender about a sentence fragment.
REFERENCE_WORDS = ("前面", "刚才", "之前", "先前", "上述", "上面", "刚说",
                   "说的", "这条", "那条", "这个", "那个", "该河", "此河")

#: A phrase that refers back to a place rather than naming one. It has to end
#: in a place word — "那个" alone is far more likely to be about a number than
#: about a river.
REFERENCE_PATTERN = re.compile(
    r"(?:前面|刚才|之前|先前|上述|上面)说?[的]?(?:那|这)?[条个]?"
    r"(?:河|江|溪|川|沟|库|地方|位置|点位)"
    r"|(?:这|那)[条个处](?:河|江|溪|川|沟|库|地方|位置)")

#: Identifiers a user may paste back in.
_ID_PATTERNS = {
    "review_id": re.compile(r"\bbasin-[0-9a-fA-F]{6,}\b"),
    "hydro_run_id": re.compile(r"\bhydro-run-[0-9a-fA-F]{6,}\b"),
    "hydro_input_id": re.compile(r"\bhydro-input-[0-9a-fA-F]{6,}\b"),
    "rainfall_package_id": re.compile(r"\brain-[0-9a-fA-F]{6,}\b"),
    "model_input_package_id": re.compile(r"\bpackage-[0-9a-fA-F]{6,}\b"),
    "visualization_id": re.compile(r"\bviz-[0-9a-fA-F]{6,}\b"),
    "artifact_id": re.compile(r"\bartifact-[0-9a-fA-F]{6,}\b"),
}


#: Intents a language model may produce. It may place an ambiguous request onto
#: this list and nothing more: every one of these names a question the workflow
#: answers, not an action it takes, and none carries an entity. Anything else —
#: a decision, a coordinate, a rainfall window, a run id — is the rules' work.
MODEL_ALLOWED_INTENTS = frozenset({
    Intent.CHECK_STATUS,
    Intent.SHOW_RESULT,
    Intent.SHOW_VISUALIZATION,
    Intent.ASK_EXPLANATION,
    Intent.UNKNOWN,
})

#: Intents a language model may never produce. Confirming or rejecting a
#: watershed is a human decision recorded by deterministic rules; letting a
#: model supply it would make the confirmation gate meaningless.
DECISION_INTENTS = frozenset({Intent.CONFIRM_WATERSHED, Intent.REJECT_WATERSHED})


class ParsedIntent(BaseModel):
    """The parser's verdict.

    Note what is absent: there is no field for a tool name, a script, a command
    or a working directory. A model asked to produce one has nowhere to put it.
    """

    intent: Intent = Intent.UNKNOWN
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    river_name: str | None = None
    administrative_area: str | None = None
    longitude: float | None = None
    latitude: float | None = None
    candidate_index: int | None = Field(default=None, ge=1, le=10)
    date: str | None = None
    start_datetime: str | None = None
    end_datetime: str | None = None
    rainfall_mode: str | None = None
    rainfall_period_error: str | None = None
    station_count: int | None = Field(default=None, ge=1, le=100, strict=True)
    station_count_mode: Literal["auto", "manual"] | None = None
    station_count_error: str | None = None
    lookback_years: int | None = Field(default=None, ge=1, le=100)
    end_year: int | None = Field(default=None, ge=1900, le=2200)
    visualization_kind: str | None = None
    #: Which file the user asked for, by format — an alias resolved against the
    #: service's own format names, never a path or a filename.
    artifact_format: str | None = None
    topic: str | None = None
    identifiers: dict[str, str] = Field(default_factory=dict)
    #: Which parser produced this, for the event log and for debugging.
    source: str = "rules"
    #: Anything the parser noticed but could not use.
    notes: list[str] = Field(default_factory=list)

    @field_validator("river_name", "administrative_area")
    @classmethod
    def _safe_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        name = value.strip()
        if not name:
            return None
        if UNSAFE_NAME_PATTERN.search(name):
            # A name carrying shell or path syntax is not a river name.
            raise ValueError("河流名称包含不允许的字符")
        return name[:60]

    @field_validator("longitude")
    @classmethod
    def _valid_longitude(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not -180.0 <= value <= 180.0:
            raise ValueError(f"经度超出范围：{value}")
        return float(value)

    @field_validator("latitude")
    @classmethod
    def _valid_latitude(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not -90.0 <= value <= 90.0:
            raise ValueError(f"纬度超出范围：{value}")
        return float(value)

    @field_validator("rainfall_mode")
    @classmethod
    def _known_mode(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in {"historical_max", "explicit_period"}:
            raise ValueError(f"未知降雨模式：{value}")
        return value

    @field_validator("topic", "visualization_kind", "artifact_format")
    @classmethod
    def _plain_text(cls, value: str | None) -> str | None:
        return None if value is None else str(value).strip()[:200]

    def entities(self) -> dict[str, Any]:
        """Only the populated entities, for merging and for the event log."""
        return {key: value for key, value in self.model_dump().items()
                if value not in (None, {}, [], 0.0) or key in {"confidence", "intent"}}


class LLMClient(Protocol):
    """What the intent parser needs from a language model."""

    async def complete_json(self, *, system: str, user: str) -> dict[str, Any]:
        """Return a JSON object; raise on any failure."""
        ...


# --------------------------------------------------------------------- helpers


def _chinese_number(text: str) -> int | None:
    if text.isdigit():
        return int(text)
    if text in _NUMBER_WORDS:
        return _NUMBER_WORDS[text]
    if len(text) == 2 and text[0] == "十" and text[1] in _NUMBER_WORDS:
        return 10 + _NUMBER_WORDS[text[1]]
    if len(text) == 2 and text[0] in _NUMBER_WORDS and text[1] == "十":
        return _NUMBER_WORDS[text[0]] * 10
    return None


def _iso_date(year: str, month: str, day: str) -> str | None:
    try:
        from datetime import date

        parsed = date(int(year), int(month), int(day))
    except ValueError:
        return None
    return parsed.isoformat()


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


#: Words that invert the phrase they precede. "不确认" is the opposite of a
#: confirmation, and reading it as one would defeat the confirmation gate — the
#: one decision in this system that must never be taken by accident. Note that
#: "确认" is a substring of "不确认", "没确认" and "不确定", so a plain
#: substring search answers yes to all three.
NEGATIONS = ("不", "没", "别", "勿", "未", "甭")

#: How far back to look for a negation. Wide enough to catch "不用我确认"
#: (three characters between the 不 and the 确认); a false negative here only
#: costs the user one more question, while a false positive takes a decision
#: they did not make.
NEGATION_WINDOW = 4

#: Words that hand the decision to someone else. "你自己确认流域并运行" is an
#: instruction *to the agent*, not the user's own confirmation — and it is
#: exactly what the gate exists to refuse.
SECOND_PERSON = ("你", "您", "系统", "程序", "智能体")

#: Imperatives that name the agent as the one who should decide. Checked across
#: the whole message: these are unmistakable, and a message containing one is
#: not a place where an unqualified "确认" should be read as the user's own.
DECIDE_ON_MY_BEHALF = ("你自己", "你帮我", "你直接", "系统自己", "程序自己",
                       "自动确认", "替你确认", "帮我确认", "代替我确认")


def _negated_before(text: str, start: int) -> bool:
    window = text[max(0, start - NEGATION_WINDOW):start]
    return any(word in window for word in NEGATIONS)


def _addressed_to_the_agent(text: str, start: int) -> bool:
    window = text[max(0, start - 3):start]
    if any(word in window for word in SECOND_PERSON):
        return True
    return any(pattern in text for pattern in DECIDE_ON_MY_BEHALF)


#: Words that ask about a phrase rather than using it. "为什么要我确认边界？"
#: contains 确认 but is a question *about* the gate; reading it as a
#: confirmation would let a question walk through the confirmation gate — the
#: exact thing the gate exists to prevent.
DISCUSSION_MARKERS = ("为什么", "为何", "如何", "怎么", "是什么意思", "什么意思", "什么是",
                      "是否", "能否", "能不能", "可不可以", "可以吗", "何时")


def _asks_about_the_phrase(text: str) -> bool:
    return any(marker in text for marker in DISCUSSION_MARKERS)


def _contains_unnegated(text: str, phrases: tuple[str, ...]) -> bool:
    """True when one of ``phrases`` occurs as the user's own statement.

    An occurrence is skipped when it is negated, when the surrounding words
    attribute the action to the agent rather than to the user, and when the
    message is asking about the phrase instead of using it.
    """
    lowered = str(text).lower()
    if _asks_about_the_phrase(lowered):
        return False
    for phrase in phrases:
        start = lowered.find(phrase)
        while start != -1:
            if not _negated_before(lowered, start) and not _addressed_to_the_agent(lowered, start):
                return True
            start = lowered.find(phrase, start + 1)
    return False


# ------------------------------------------------------------------ rule parser


class RuleParser:
    """Deterministic parsing. Available with no configuration and no network."""

    def parse(self, text: str, session: AgentSession) -> ParsedIntent:
        raw = str(text or "").strip()
        if not raw:
            return ParsedIntent(intent=Intent.UNKNOWN, source="rules",
                                notes=["输入为空"])
        result = ParsedIntent(source="rules")
        result.identifiers = self._identifiers(raw)
        self._coordinates(raw, result)
        self._dates(raw, result)
        self._station_count(raw, result)
        self._candidate(raw, result)
        result.visualization_kind = self._visualization(raw)
        result.topic = raw if len(raw) <= 200 else raw[:200]
        result.administrative_area = self._extract_administrative_name(raw)
        previous_area = session.administrative_area
        if not result.administrative_area and previous_area:
            alias = previous_area.removesuffix("区").removesuffix("县")
            if re.fullmatch(r"(?:对[，,、]?|是的[，,、]?|就是|还是|仍然|就用|用|选|选择)*" + re.escape(alias), raw):
                result.administrative_area = previous_area
        if result.administrative_area and result.administrative_area in raw:
            # "永川区的临江河" names a river with a district hint, whereas
            # "永川区的小流域" has supplied only a district.
            tail = raw.split(result.administrative_area, 1)[1].lstrip("的")
            result.river_name = self._extract_river_name(tail)
        else:
            result.river_name = None if result.administrative_area else self._extract_river_name(raw)
        if result.river_name is None:
            self._resolve_place_reference(raw, session, result)
        self._intent(raw, session, result)
        return result

    @staticmethod
    def _resolve_place_reference(text: str, session: AgentSession,
                                 result: ParsedIntent) -> None:
        """Read "那条河" as the river this session already knows.

        Such a phrase names nothing — it points at something established
        earlier. When the session has that something, using it is the correct
        reading; when it has not, the reference stays unresolved and the agent
        asks which place is meant, which is the only honest answer.
        """
        if not REFERENCE_PATTERN.search(text):
            return
        if session.river_name:
            result.river_name = session.river_name
            result.notes.append(f"按上一轮提到的河流理解：{session.river_name}")
        else:
            result.notes.append("提到了「那条河」之类的说法，但本会话还没有记录过河流")

    # --------------------------------------------------------------- entities
    @staticmethod
    def _identifiers(text: str) -> dict[str, str]:
        found: dict[str, str] = {}
        for key, pattern in _ID_PATTERNS.items():
            match = pattern.search(text)
            if match:
                found[key] = match.group(0)
        return found

    @staticmethod
    def _coordinates(text: str, result: ParsedIntent) -> None:
        lon_match = _LONGITUDE.search(text)
        lat_match = _LATITUDE.search(text)
        if lon_match and lat_match:
            result.longitude = float(lon_match.group(1))
            result.latitude = float(lat_match.group(1))
            return
        suffix = _LONLAT_SUFFIX.search(text)
        if suffix:
            result.longitude = float(suffix.group(1))
            result.latitude = float(suffix.group(2))
            return
        pair = _COORDINATE_PAIR.search(text)
        if pair:
            lon, lat = float(pair.group(1)), float(pair.group(2))
            # Only accept the pair when it plausibly reads as lon/lat.
            if -180 <= lon <= 180 and -90 <= lat <= 90 and lon > lat:
                result.longitude, result.latitude = lon, lat

    @staticmethod
    def _dates(text: str, result: ParsedIntent) -> None:
        from .temporal_inputs import absolute_rainfall_period
        absolute = absolute_rainfall_period(text)
        if absolute.error:
            result.rainfall_period_error = absolute.error
            return
        if absolute.values:
            result.start_datetime = absolute.values['start_datetime']
            result.end_datetime = absolute.values['end_datetime']
            result.rainfall_mode = "explicit_period"
            return
        day: str | None = None
        match = _DATE_CN.search(text) or _DATE_ISO.search(text)
        if match:
            day = _iso_date(*match.groups())
        if day:
            result.date = day
        if _MAX_DAY.search(text):
            result.rainfall_mode = "historical_max"
        elif day or _WHOLE_DAY.search(text):
            result.rainfall_mode = "explicit_period"
        if day:
            window = _TIME_RANGE.search(text)
            hours = _HOUR_RANGE.search(text)
            if window:
                result.start_datetime = f"{day} {int(window.group(1)):02d}:{window.group(2)}"
                result.end_datetime = f"{day} {int(window.group(3)):02d}:{window.group(4)}"
            elif hours:
                result.start_datetime = f"{day} {int(hours.group(1)):02d}:00"
                result.end_datetime = f"{day} {int(hours.group(2)):02d}:00"
            elif _WHOLE_DAY.search(text):
                result.start_datetime = f"{day} 00:00"
                result.end_datetime = f"{day} 23:00"
        lookback = _LOOKBACK.search(text)
        if lookback:
            result.lookback_years = int(lookback.group(1))
        if result.rainfall_mode == "historical_max":
            from .temporal_inputs import historical_period
            period = historical_period(text)
            result.end_year = period.get("end_year")
            result.lookback_years = period.get("lookback_years", result.lookback_years)

    @staticmethod
    def _station_count(text: str, result: ParsedIntent) -> None:
        from .station_count import parse_station_preference
        count, mode, error = parse_station_preference(text)
        result.station_count, result.station_count_mode, result.station_count_error = count, mode, error

    @staticmethod
    def _candidate(text: str, result: ParsedIntent) -> None:
        match = _CANDIDATE.search(text) or _CANDIDATE_SHORT.search(text)
        if not match:
            return
        value = _chinese_number(match.group(1))
        if value and 1 <= value <= 10:
            result.candidate_index = value

    @staticmethod
    def _visualization(text: str) -> str | None:
        lowered = text.lower()
        for alias, kind in VISUALIZATION_ALIASES.items():
            if alias in lowered:
                return kind
        return None

    @staticmethod
    def _artifact_format(text: str) -> str | None:
        """Which file format the user asked for, if they named one.

        Latin aliases are matched on word boundaries so "out" cannot be found
        inside an ordinary English word; Chinese aliases are substrings, which is
        how they are actually written.
        """
        for name, aliases in ARTIFACT_FORMAT_ALIASES.items():
            for alias in aliases:
                if alias.isascii():
                    if re.search(rf"\b{re.escape(alias)}\b", text):
                        return name
                elif alias in text:
                    return name
        return None

    # ----------------------------------------------------------------- intent
    def _intent(self, text: str, session: AgentSession, result: ParsedIntent) -> None:
        lowered = text.lower()
        if lowered.strip("。！! ") in {"恢复上一步","恢复上个版本","撤销上次修改","恢复修改前"}:
            result.intent=Intent.RESTORE_TASK
            result.confidence=.98
            return

        # An explicit decision comes first: it is the one thing the rules must
        # get right, and the one thing a language model may never decide.
        # A *negated* confirmation phrase is not a confirmation, so the search
        # skips those before falling through to the rejection list below.
        location_correction = bool(result.administrative_area and not session.has_pending_watershed()
                                   and not any(word in lowered for word in ("边界", "确认流域")))
        station_preference = bool(result.station_count_mode or result.station_count_error)
        station_only = station_preference and not any(word in lowered for word in ("边界", "流域", "确认"))
        if _contains_unnegated(lowered, CONFIRM_PHRASES) and not location_correction and not station_only:
            result.intent = Intent.CONFIRM_WATERSHED
            result.confidence = 0.95
            return
        if (_contains_any(lowered, REJECT_PHRASES) and not _asks_about_the_phrase(lowered)
                and not location_correction):
            result.intent = Intent.REJECT_WATERSHED
            result.confidence = 0.95
            return
        if _contains_any(lowered, CANCEL_PHRASES) and not _asks_about_the_phrase(lowered):
            result.intent = Intent.CANCEL_RUN
            result.confidence = 0.9
            return
        if _contains_any(lowered, START_PHRASES):
            result.intent = Intent.START_HYDRO_RUN
            result.confidence = 0.85
            return
        if _contains_any(lowered, STATUS_PHRASES):
            result.intent = Intent.CHECK_STATUS
            result.confidence = 0.85
            return
        if _contains_any(lowered, CATALOG_PHRASES):
            # "What figures are there?" — a catalogue question, not a request for
            # one figure. The workflow answers it by listing the server's options.
            result.intent = Intent.SHOW_VISUALIZATION
            result.visualization_kind = None
            result.confidence = 0.85
            return
        if (_contains_any(lowered, DOWNLOAD_PHRASES)
                and not _contains_any(lowered, DOWNLOAD_QUESTION_MARKERS)
                and not _asks_about_the_phrase(lowered)):
            # Asking for a file. Which file, and of what, is resolved by the
            # workflow from this session's own figures — the rule only records
            # what was named, and never a path or a file name.
            result.intent = Intent.DOWNLOAD_ARTIFACT
            result.artifact_format = self._artifact_format(lowered)
            result.confidence = 0.85
            return
        if result.visualization_kind and _contains_any(lowered, VISUALIZATION_PHRASES):
            result.intent = Intent.SHOW_VISUALIZATION
            result.confidence = 0.8
            return
        if (session.hydro_run_id and not any((result.administrative_area, result.river_name,
                result.longitude is not None, result.date, result.rainfall_mode))
                and not any(word in lowered for word in ('重新计算', '重新运行', '换个区域', '新任务'))
                and any(word in lowered for word in ('评测区域', '评价区域稳定性', '评估区域稳定性',
                                                     '评价一下这个区域', '解释分布', '分析这个结果'))):
            result.intent = Intent.SHOW_RESULT
            result.confidence = .9
            return
        if _contains_any(lowered, EXPLAIN_PHRASES):
            result.intent = Intent.ASK_EXPLANATION
            result.confidence = 0.7
            return
        if _contains_any(lowered, RESULT_PHRASES):
            result.intent = Intent.SHOW_RESULT
            result.confidence = 0.8
            return

        # A river name and a coordinate both name a place to analyse. Which
        # intent that is depends on how far the session has already got: before
        # a boundary exists it starts an assessment, afterwards it changes an
        # existing one.
        starting_fresh = session.state in {
            "new", "needs_location", "outlet_candidates_ready", "watershed_rejected",
        }
        if result.administrative_area and starting_fresh:
            result.intent = Intent.START_ASSESSMENT
            result.confidence = 0.85
            return
        if result.river_name and starting_fresh:
            result.intent = Intent.START_ASSESSMENT
            result.confidence = 0.85
            return
        if (result.longitude is not None and result.latitude is not None) and starting_fresh:
            result.intent = Intent.SELECT_OUTLET
            result.confidence = 0.9
            return
        if result.candidate_index is not None and starting_fresh:
            result.intent = Intent.SELECT_OUTLET
            result.confidence = 0.85
            return
        if (result.rainfall_mode or result.date or result.lookback_years
                or result.station_count_mode or result.station_count_error):
            result.intent = Intent.SET_RAINFALL_PERIOD
            result.confidence = 0.8
            return
        if result.administrative_area or result.river_name:
            result.intent = Intent.START_ASSESSMENT
            result.confidence = 0.7
            return
        if result.longitude is not None and result.latitude is not None:
            result.intent = Intent.SELECT_OUTLET
            result.confidence = 0.8
            return
        if result.candidate_index is not None:
            result.intent = Intent.SELECT_OUTLET
            result.confidence = 0.75
            return
        if _contains_any(lowered, CONTINUE_PHRASES):
            # "继续" is not a decision. It only means "do the pending step",
            # and the workflow decides what that is.
            result.intent = Intent.UNKNOWN
            result.confidence = 0.4
            from .state_machine import pending_action

            result.notes.append(f"“继续”未指定动作；{pending_action(session.state)}")
            return
        if _contains_any(lowered, ("我想评价", "我想评估", "我要评价", "我要评估", "请评价", "进行稳定性评价")):
            result.intent = Intent.START_ASSESSMENT
            result.confidence = 0.8
            return
        result.intent = Intent.UNKNOWN
        result.confidence = 0.2

    # --------------------------------------------------------------- extract
    @staticmethod
    def _extract_administrative_name(text: str) -> str | None:
        match = _ADMINISTRATIVE_NAME.search(text)
        if not match:
            return None
        name = match.group(1)
        # This recognises supplied names; the spatial catalog checks existence.
        # Never turn a spelling suggestion into a location selection.
        for separator in ("的", "在", "是"):
            if separator in name:
                name = name.rsplit(separator, 1)[-1]
        changed = True
        while changed:
            changed = False
            for prefix in sorted((*LEADING_VERBS, "选择", "重庆市", "重庆"), key=len, reverse=True):
                if name.startswith(prefix) and len(name) > len(prefix) + 1:
                    name = name[len(prefix):]
                    changed = True
                    break
        if len(name) < 3 or name.endswith(("地区", "小区", "山区", "灾区", "评价区", "研究区", "工作区")):
            return None
        return name

    def _extract_river_name(self, text: str) -> str | None:
        match = _RIVER_NAME.search(text)
        if not match:
            return None
        name = match.group(1)
        if any(word in name for word in NOT_A_RIVER):
            return None
        # Strip a leading verb, possibly several ("给我看看大宁河").
        changed = True
        while changed:
            changed = False
            for verb in sorted(LEADING_VERBS, key=len, reverse=True):
                if name.startswith(verb) and len(name) > len(verb) + 1:
                    name = name[len(verb):]
                    changed = True
                    break
        if len(name) < 2 or any(word in name for word in NOT_A_RIVER):
            return None
        # "前面说的那条河" survives the strip above and is not a name. It is a
        # reference, and it is resolved against the session instead — see
        # ``_resolve_place_reference``.
        if any(word in name for word in REFERENCE_WORDS):
            return None
        return name


class IntentParser:
    """Rules first, then an optional language model for what rules could not place."""

    #: Rules at or above this confidence are final; the model is not consulted.
    RULE_CERTAINTY = 0.7

    SYSTEM_PROMPT = (
        "你是坡面稳定性初步评价智能体的意图解析器。"
        "只输出 JSON，字段见用户消息中的 schema。"
        "intent 必须是给定枚举之一。不要输出工具名、脚本名、命令、路径或代码。"
        "不要编造用户没有提到的数值。无法判断时 intent 用 unknown。"
    )

    def __init__(self, *, llm: LLMClient | None = None,
                 logger: logging.Logger | None = None) -> None:
        self.rules = RuleParser()
        self.llm = llm
        self.logger = logger or logging.getLogger("slope_agent")

    @property
    def llm_available(self) -> bool:
        return self.llm is not None

    async def parse(self, text: str, session: AgentSession) -> ParsedIntent:
        """Parse ``text`` in the context of ``session``."""
        ruled = self.rules.parse(text, session)
        # A decided confirmation or rejection is final.
        if ruled.confidence >= self.RULE_CERTAINTY:
            return ruled
        if self.llm is None:
            return ruled
        try:
            model_result = await self._parse_with_llm(text, session)
        except Exception as exc:  # noqa: BLE001 - the rules' answer still stands
            self.logger.info("语言模型解析失败，使用规则结果：%s", exc)
            ruled.notes.append("语言模型解析失败，已使用规则结果")
            return ruled
        return self._merge(ruled, model_result)

    # -------------------------------------------------------------------- llm
    async def _parse_with_llm(self, text: str, session: AgentSession) -> ParsedIntent:
        schema = {
            "intent": [item.value for item in Intent],
            "river_name": "string|null",
            "longitude": "number|null",
            "latitude": "number|null",
            "date": "YYYY-MM-DD|null",
            "start_datetime": "YYYY-MM-DD HH:MM|null",
            "end_datetime": "YYYY-MM-DD HH:MM|null",
            "lookback_years": "integer|null",
        }
        context = {
            "current_state": session.state,
            "pending_question": session.pending_action(),
            "has_pending_watershed": session.has_pending_watershed(),
            "known": {
                "river_name": session.river_name,
                "review_id": session.review_id,
                "hydro_run_id": session.hydro_run_id,
            },
            "recent_conversation": [
                {"role": turn.role, "text": turn.text}
                for turn in session.recent_conversation(6)
            ],
        }
        payload = await self.llm.complete_json(
            system=self.SYSTEM_PROMPT,
            user=json.dumps({"schema": schema, "context": context, "user_text": text},
                            ensure_ascii=False),
        )
        return self._validate_model_output(payload)

    @staticmethod
    def _validate_model_output(payload: Any) -> ParsedIntent:
        """Validate model JSON, keeping only an intent the model may name.

        Everything else is dropped rather than validated: entities, identifiers
        and parameters are the rules' to establish, so a model that returns one
        cannot influence the workflow through it.
        """
        if not isinstance(payload, dict):
            raise ValueError("模型未返回 JSON 对象")
        raw_intent = payload.get("intent")
        try:
            intent = Intent(raw_intent) if raw_intent is not None else Intent.UNKNOWN
        except ValueError:
            intent = Intent.UNKNOWN
        if intent not in MODEL_ALLOWED_INTENTS:
            return ParsedIntent(
                source="llm", confidence=0.0,
                notes=[f"模型给出的意图「{raw_intent}」不在允许范围内，已忽略"])
        confidence = payload.get("confidence", 0.0)
        try:
            confidence = min(1.0, max(0.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.0
        return ParsedIntent(intent=intent, confidence=confidence, source="llm")

    @staticmethod
    def _merge(ruled: ParsedIntent, model_result: ParsedIntent) -> ParsedIntent:
        """Rules win on facts they established; the model may fill the gaps."""
        merged = model_result.model_copy(deep=True)
        merged.source = "rules+llm"
        for field in ("river_name", "longitude", "latitude", "candidate_index", "date",
                      "start_datetime", "end_datetime", "rainfall_mode", "lookback_years",
                      "station_count", "station_count_mode",
                      "visualization_kind"):
            ruled_value = getattr(ruled, field)
            if ruled_value is not None:
                setattr(merged, field, ruled_value)
        merged.identifiers = {**model_result.identifiers, **ruled.identifiers}
        merged.notes = [*ruled.notes, *model_result.notes]
        if ruled.intent is not Intent.UNKNOWN:
            merged.intent = ruled.intent
        merged.confidence = max(ruled.confidence, model_result.confidence)
        return merged


def looks_like_a_command(text: str) -> bool:
    """True when text is shaped like a command rather than a request.

    Used only to decide whether to say "that is not something I can run" — the
    workflow never acts on the answer.
    """
    lowered = str(text).lower()
    markers = ("os.system", "subprocess", "eval(", "exec(", "import os", "import sys",
               "rm -rf", "del /f", "powershell", "cmd.exe", "bash -c", "&&", "||", ";rm",
               "../", ".." + chr(92))
    return any(marker in lowered for marker in markers)
