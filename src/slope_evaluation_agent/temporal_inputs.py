"""Resolve calendar offsets against a user-supplied rainfall event.

The language model still chooses whether a turn is a task or a question.
This module never changes a session and never authorises an operation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import re
from typing import Any


_NUMBER = r"[0-9零〇一二两三四五六七八九十百]+"
_OFFSETS = [
    (re.compile(rf"(?P<n>{_NUMBER})(?:天|日)(?:之|以)?(?P<direction>前|后)"), None),
    (re.compile(rf"(?:往|向)?(?P<direction>前|后)(?:顺延|移动|调整|挪|移|推)?(?:了)?(?P<n>{_NUMBER})(?:天|日)"), None),
    (re.compile(rf"(?P<direction>前|后)(?:的)?第?(?P<n>{_NUMBER})(?:天|日)"), None),
    (re.compile(rf"(?:提前)(?:了)?(?P<n>{_NUMBER})(?:天|日)"), -1),
    (re.compile(rf"(?:推迟|延后|顺延)(?:了)?(?P<n>{_NUMBER})(?:天|日)"), 1),
    (re.compile(r"(?:次日|翌日|下一天|第二天)"), 1),
    (re.compile(r"(?:前一天|上一天)"), -1),
]
_ABSOLUTE = re.compile(r"\d{4}(?:\s*年\s*\d{1,2}\s*月\s*\d{1,2}|[-/.]\d{1,2}[-/.]\d{1,2})")
_DIGITS = dict(zip("零〇一二两三四五六七八九", (0,0,1,2,2,3,4,5,6,7,8,9)))


def historical_period(message: str) -> dict[str, int]:
    """Extract explicit inclusive archive years; never substitute today's year."""
    end = re.search(r"(?:截至|截止|到)\s*(\d{4})\s*年", message)
    window = re.search(rf"(?:过去|近|前|回溯)\s*({_NUMBER})\s*年", message)
    span = re.search(r"(\d{4})\s*(?:年)?\s*(?:至|到|—|–|-)\s*(\d{4})\s*年", message)
    if span and 1900 <= int(span[1]) <= int(span[2]) <= 2200:
        count = int(span[2]) - int(span[1]) + 1
        if count <= 100:
            return {"end_year": int(span[2]), "lookback_years": count}
    values = {}
    if end and 1900 <= int(end[1]) <= 2200:
        values["end_year"] = int(end[1])
    if window and 1 <= _integer(window[1]) <= 100:
        values["lookback_years"] = _integer(window[1])
    return values


def _integer(text: str) -> int:
    if text.isascii() and text.isdigit():
        return int(text)
    total = pending = 0
    for char in text:
        if char in _DIGITS:
            pending = _DIGITS[char]
        elif char in "十百":
            total += (pending or 1) * (10 if char == "十" else 100)
            pending = 0
        else:
            raise ValueError("unsupported day count")
    return total + pending


@dataclass(frozen=True)
class TemporalResolution:
    values: dict[str, str] = field(default_factory=dict)
    evidence: str = ""
    error: str | None = None


_FULL_TIMESTAMP = re.compile(
    r"(?P<year>\d{4})(?:年|[-/.])(?P<month>\d{1,2})(?:月|[-/.])"
    r"(?P<day>\d{1,2})日?(?:\s*|T)(?P<hour>\d{1,2})"
    r"(?:[:：](?P<minute>\d{2})(?::00)?|时(?:(?P<cn_minute>\d{1,2})分)?)")


def absolute_rainfall_period(message: str) -> TemporalResolution:
    """Read an explicitly connected pair of full timestamps, including dates.

    This extracts input values only; the model still decides whether the user
    is requesting an action. Unconnected dates are not an event selection.
    """
    stamps = list(_FULL_TIMESTAMP.finditer(message))
    pairs = [(a, b) for a, b in zip(stamps, stamps[1:])
             if re.fullmatch(r"\s*(?:至|到|—|–|－|-|~|～)\s*", message[a.end():b.start()])]
    if not pairs:
        return TemporalResolution()
    if len(pairs) != 1:
        return TemporalResolution(error="给出了多个完整降雨时段，请明确本次使用哪一个。")
    a, b = pairs[0]
    evidence = message[a.start():b.end()]
    try:
        def value(match):
            return datetime(int(match['year']), int(match['month']), int(match['day']),
                            int(match['hour']), int(match['minute'] or match['cn_minute'] or 0))
        start, end = value(a), value(b)
        if end < start:
            raise ValueError('reversed window')
    except ValueError:
        return TemporalResolution(evidence=evidence, error="降雨起止时刻无效或结束早于开始，请核对完整时段。")
    return TemporalResolution(values={'start_datetime':start.isoformat(sep=' ', timespec='minutes'),
                                      'end_datetime':end.isoformat(sep=' ', timespec='minutes')},
                              evidence=evidence)


def resolve_relative_rainfall(message: str, request: dict[str, Any] | None) -> TemporalResolution:
    matches = []
    negated = []
    seen = []
    for pattern, sign in _OFFSETS:
        for match in pattern.finditer(message):
            # Several patterns can recognise the same phrase; count it once.
            if any(match.start() < end and match.end() > start for start, end in seen):
                continue
            seen.append(match.span())
            groups = match.groupdict()
            amount = _integer(groups["n"]) if groups.get("n") else 1
            direction = sign if sign is not None else (-1 if groups["direction"] == "前" else 1)
            prefix = re.split(r"[，,。；;！？!?]",message[:match.start()])[-1]
            if re.search(r"(?:不要|别|不必|无需|不用|不能|不再|并非)(?:再|去|直接|继续|进行|将|把|降雨|日期|事件|时段|往|向|改用|改成|换成|设为|设置为|调整到)*$",prefix):
                negated.append(match.group(0))
                continue
            if message[match.end():match.end()+1] == "半":
                return TemporalResolution(evidence=match.group(0),error="请明确新的降雨日期或完整起止时刻，避免把半天偏移当成整天。")
            matches.append((match.start(), match.end(), direction * amount, match.group(0)))
    if not matches:
        if negated:
            return TemporalResolution(evidence=negated[0],error="本轮否定了日期偏移，原降雨设置保持不变；如需修改，请明确新的日期。")
        return TemporalResolution()
    evidence = matches[0][3]
    if len({item[2] for item in matches}) != 1:
        return TemporalResolution(evidence=evidence, error="给出了不同的日期偏移，请明确要提前或顺延几天。")
    if _ABSOLUTE.search(message):
        return TemporalResolution(evidence=evidence, error="同时提到了具体日期和相对日期，请直接确认修改后的降雨日期。")
    request = request or {}
    if request.get("mode") == "historical_max":
        return TemporalResolution(evidence=evidence, error="当前是历史最大日筛选，请先指定要作为偏移基准的具体日期。")
    try:
        delta = timedelta(days=matches[0][2])
        start, end = request.get("start_datetime"), request.get("end_datetime")
        if start and end:
            a,b = datetime.fromisoformat(start),datetime.fromisoformat(end)
            if b < a:
                raise ValueError("reversed rainfall window")
            values={"start_datetime":(a+delta).isoformat(sep=" ",timespec="minutes"),
                    "end_datetime":(b+delta).isoformat(sep=" ",timespec="minutes")}
        elif request.get("date"):
            anchor = date.fromisoformat(request["date"])
            values={"date":(anchor+delta).isoformat()}
        else:
            return TemporalResolution(evidence=evidence, error="还没有可用的基准日期，请先给出降雨日期或完整时段。")
    except (TypeError, ValueError, OverflowError):
        return TemporalResolution(evidence=evidence, error="原降雨日期或时段无法用于日期偏移，请给出有效的具体日期。")
    return TemporalResolution(values=values,evidence=evidence)
