"""What may be claimed, and what may not.

This module is the gate between a model result and a statement about slope
stability. It decides three things:

* whether an assessment may be called a *valid preliminary* result, may only be
  used *diagnostically*, or is *unavailable* at all;
* which interpretations the evidence supports, phrased so a local minimum is
  never stretched into a statement about a whole watershed;
* whether a finished reply is faithful — every number in it traceable to the
  ledger, and no forbidden claim in it.

The audit works from an explicit :class:`EvidenceLedger` rather than from a
loose dictionary walk. Each entry declares its own display precision and whether
a percentage form is meaningful for it, so `0.2752` may be written `0.2752` or
`0.275`, an unstable-area fraction of `0.42` may also be written `42%`, and a
safety factor may **not** be written `27.52%` — it has no percent form.

No value is exempt for being small. `0.9`, `0.42` and `1.0` are audited exactly
like any other number, because those are the values a reader acts on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable

from .response_writer import FORBIDDEN_PHRASES, NEGATIONS, find_forbidden, sanitize


class AssessmentStatus(StrEnum):
    """How far a result may be taken."""

    VALID_PRELIMINARY = "valid_preliminary"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PolicyDecision:
    status: AssessmentStatus
    reasons: list[str]

    @property
    def is_formal(self) -> bool:
        return self.status is AssessmentStatus.VALID_PRELIMINARY


def decide_status(*, qa_passed: bool | None, run_state: str | None,
                  safety_available: bool) -> PolicyDecision:
    """Decide whether a result may be presented as a preliminary assessment.

    A run that failed a quality gate, crashed or was cancelled never yields a
    formal assessment, no matter how complete its numbers look.
    """
    reasons: list[str] = []
    if run_state in {"failed", "cancelled"}:
        reasons.append(f"任务状态为 {run_state}，没有可用于评价的结果")
        return PolicyDecision(AssessmentStatus.UNAVAILABLE, reasons)
    if run_state == "qa_failed":
        reasons.append("质量门槛未通过：只能给出诊断信息，不能给出稳定性结论")
        return PolicyDecision(AssessmentStatus.DIAGNOSTIC_ONLY, reasons)
    if run_state != "succeeded":
        reasons.append(f"任务尚未结束（{run_state or '未知'}）")
        return PolicyDecision(AssessmentStatus.UNAVAILABLE, reasons)
    if not safety_available:
        reasons.append("缺少安全系数摘要，无法给出稳定性评价")
        return PolicyDecision(AssessmentStatus.UNAVAILABLE, reasons)
    if qa_passed is not True:
        reasons.append("质量门槛的通过状态未知，按未通过处理")
        return PolicyDecision(AssessmentStatus.DIAGNOSTIC_ONLY, reasons)
    reasons.append("计算完成且全部质量门槛通过")
    return PolicyDecision(AssessmentStatus.VALID_PRELIMINARY, reasons)


# ------------------------------------------------------------------- ledger


@dataclass(frozen=True)
class LedgerEntry:
    """One number the reply is allowed to state, and how it may be written."""

    name: str
    value: float | None
    unit: str = ""
    #: Display precision. The rounding tolerance is half a unit in the last
    #: place, so a value declared at three decimals accepts 0.275 for 0.2752 but
    #: not 0.27.
    decimals: int = 4
    #: Minimum precision a reply must retain.  The previous implementation used
    #: the reply token's own precision, which made ``0`` a "valid rounding" of
    #: 0.269.  Scientific quantities may be rounded, but not until their meaning
    #: changes.
    min_decimals: int = 0
    #: Whether a percentage form of this value is meaningful. A fraction may be
    #: written ``42%``; a safety factor may not.
    percent_equivalent: bool = False
    source: str = "get_hydro_run_result"
    resource_id: str = ""
    visible: bool = True

    def accepts(self, token: str, *, as_percent: bool,
                observed_unit: str = "") -> bool:
        """Whether ``token`` is a faithful way of writing this value.

        The rule is rounding, not a global tolerance: the token must equal the
        value rounded to the token's own precision. So 0.28 and 0.275 are both
        faithful writings of 0.2752, while 0.29 and 0.2 are not — a coarser
        rounding is fine as long as it rounds correctly. A token may not claim
        more precision than the field declares.
        """
        if self.value is None or not self.visible:
            return False
        if as_percent and not self.percent_equivalent:
            return False
        target = self.value * 100.0 if as_percent else self.value
        try:
            written = float(token)
        except ValueError:
            return False
        decimals = _token_decimals(token)
        if decimals is None:
            # Scientific notation states a value, not a precision: compare it
            # exactly rather than rounding it.
            return abs(written - target) <= 1e-9 * max(1.0, abs(target))
        required_decimals = (max(0, self.min_decimals - 2)
                             if as_percent else self.min_decimals)
        if decimals < required_decimals:
            return False
        # The service's original precision is evidence too. A field's normal
        # display precision must not reject an unrounded source value.
        source_decimals = _token_decimals(str(target))
        if decimals > max(self.decimals + 2, source_decimals or 0):
            return False
        if observed_unit:
            expected_unit = "%" if as_percent else _normalise_unit(self.unit)
            if not expected_unit or _normalise_unit(observed_unit) != expected_unit:
                return False
        expected = round(target, decimals)
        return abs(written - expected) <= 1e-9 * max(1.0, abs(expected))


#: How each factor is declared to the ledger. Percent equivalence is granted
#: only to quantities that are genuinely ratios.
#: ``decimals`` is the display precision the renderers actually use, because the
#: rounding tolerance follows from it: a safety factor printed to three decimals
#: accepts 0.275 for 0.2752, and accepts nothing coarser.
FACTOR_SPECS: tuple[dict[str, Any], ...] = (
    {"name": "rainfall.total_rain_mm", "path": ("rainfall", "total_rain_mm"),
     "unit": "mm", "decimals": 2, "min_decimals": 1},
    {"name": "rainfall.peak_hourly_rain_mm", "path": ("rainfall", "peak_hourly_rain_mm"),
     "unit": "mm/h", "decimals": 2, "min_decimals": 1},
    {"name": "safety.minimum_safety_factor", "path": ("safety_factor", "minimum"),
     "decimals": 3, "min_decimals": 2},
    {"name": "safety.mean_safety_factor", "path": ("safety_factor", "mean"),
     "decimals": 3, "min_decimals": 2},
    {"name": "safety.p95_safety_factor", "path": ("safety_factor", "p95"),
     "decimals": 3, "min_decimals": 2},
    {"name": "safety.maximum_safety_factor", "path": ("safety_factor", "maximum"),
     "decimals": 3, "min_decimals": 2},
    {"name": "safety.max_unstable_fraction",
     "path": ("instability_extent_duration", "max_unstable_fraction"),
     "decimals": 4, "min_decimals": 2, "percent_equivalent": True},
    {"name": "safety.max_unstable_duration_h",
     "path": ("instability_extent_duration", "max_unstable_duration_h"),
     "unit": "h", "decimals": 2, "min_decimals": 1},
    {"name": "safety.max_flow_depth_m",
     "path": ("hydraulic_pore_pressure", "max_flow_depth_m"), "unit": "m",
     "decimals": 4, "min_decimals": 2},
    {"name": "safety.max_velocity_ms",
     "path": ("hydraulic_pore_pressure", "max_velocity_ms"), "unit": "m/s",
     "decimals": 4, "min_decimals": 2},
    {"name": "safety.max_pore_water_pressure_kpa",
     "path": ("hydraulic_pore_pressure", "max_pore_water_pressure_kpa"),
     "unit": "kPa", "decimals": 2, "min_decimals": 1},
    {"name": "model.subcatchment_count", "path": ("model_context", "subcatchment_count"),
     "root": "", "decimals": 0},
)


#: Per-field rules, keyed by the leaf name. A value's own name says whether a
#: percentage form means the same thing: an area fraction may be written 42%,
#: a safety factor may not, and a continuity error is itself a percentage.
LEAF_RULES: dict[str, dict[str, Any]] = {
    "minimum_safety_factor": {"decimals": 3, "min_decimals": 2, "percent_equivalent": False},
    "mean_safety_factor": {"decimals": 3, "min_decimals": 2, "percent_equivalent": False},
    "p95_safety_factor": {"decimals": 3, "min_decimals": 2, "percent_equivalent": False},
    "maximum_safety_factor": {"decimals": 3, "min_decimals": 2, "percent_equivalent": False},
    "max_unstable_fraction": {"decimals": 4, "min_decimals": 2, "percent_equivalent": True},
    "max_unstable_duration_h": {"decimals": 2, "min_decimals": 1, "unit": "h", "percent_equivalent": False},
    "max_flow_depth_m": {"decimals": 4, "min_decimals": 2, "unit": "m", "percent_equivalent": False},
    "max_velocity_ms": {"decimals": 4, "min_decimals": 2, "unit": "m/s", "percent_equivalent": False},
    "max_pore_water_pressure_kpa": {"decimals": 2, "min_decimals": 1, "unit": "kPa", "percent_equivalent": False},
    "total_rain_mm": {"decimals": 2, "min_decimals": 1, "unit": "mm", "percent_equivalent": False},
    "peak_hourly_rain_mm": {"decimals": 2, "min_decimals": 1, "unit": "mm/h", "percent_equivalent": False},
    "runoff_continuity_error_percent": {"decimals": 3, "percent_equivalent": True},
    "flow_routing_continuity_error_percent": {"decimals": 3, "percent_equivalent": True},
    "error": {"decimals": 3, "percent_equivalent": True},
    "unstable_duration_h_max": {"decimals": 2, "min_decimals": 1, "unit": "h", "percent_equivalent": False},
    # Coordinates and elevations are stated at the precision the layers carry.
    "longitude": {"decimals": 7, "min_decimals": 1, "percent_equivalent": False},
    "latitude": {"decimals": 7, "min_decimals": 1, "percent_equivalent": False},
    "requested_longitude": {"decimals": 7, "min_decimals": 1, "percent_equivalent": False},
    "requested_latitude": {"decimals": 7, "min_decimals": 1, "percent_equivalent": False},
    "snapped_longitude": {"decimals": 7, "min_decimals": 1, "percent_equivalent": False},
    "snapped_latitude": {"decimals": 7, "min_decimals": 1, "percent_equivalent": False},
    "elevation_m": {"decimals": 2, "min_decimals": 1, "unit": "m", "percent_equivalent": False},
    "shift_m": {"decimals": 2, "min_decimals": 1, "unit": "m", "percent_equivalent": False},
    "river_distance_m": {"decimals": 2, "min_decimals": 1, "unit": "m", "percent_equivalent": False},
    "area_km2": {"decimals": 3, "min_decimals": 2, "unit": "km2", "percent_equivalent": False},
    "selected_cells": {"decimals": 0, "percent_equivalent": False},
    "dem_edge_distance_km": {"decimals": 2, "min_decimals": 1, "unit": "km", "percent_equivalent": False},
    "timestamp_count": {"decimals": 0, "percent_equivalent": False},
    "station_count": {"decimals": 0, "percent_equivalent": False},
    "active_rows": {"decimals": 0, "percent_equivalent": False},
    "size_bytes": {"decimals": 0, "percent_equivalent": False},
    "size_mb": {"decimals": 3, "percent_equivalent": False},
    "artifact_count": {"decimals": 0, "percent_equivalent": False},
    "total_mm": {"decimals": 3, "min_decimals": 1, "unit": "mm", "percent_equivalent": False},
    "peak_mm_h": {"decimals": 3, "min_decimals": 1, "unit": "mm/h", "percent_equivalent": False},
    "read": {"decimals": 0, "percent_equivalent": False},
    "kept": {"decimals": 0, "percent_equivalent": False},
    "lon": {"decimals": 7, "percent_equivalent": False},
    "lat": {"decimals": 7, "percent_equivalent": False},
}

#: Leaf names that identify a ratio even when no rule is declared for them.
RATIO_HINTS = ("fraction", "ratio", "percent")


def _leaf_rule(name: str, value: float | None) -> dict[str, Any]:
    leaf = name.rsplit(".", 1)[-1].split("[")[0]
    if leaf in LEAF_RULES:
        return LEAF_RULES[leaf]
    if any(hint in leaf.lower() for hint in RATIO_HINTS):
        return {"decimals": 2, "min_decimals": 2, "percent_equivalent": True}
    # An unnamed value in [0, 1] is a fraction unless its name says otherwise;
    # a safety factor is the case that must not be read as a percentage.
    anonymous_ratio = value is not None and 0.0 <= value <= 1.0 and "factor" not in leaf.lower()
    integral = value is not None and float(value).is_integer()
    lowered = leaf.lower()
    unit = ""
    if lowered.endswith("_mm_h") or lowered.endswith("_mm_per_h"):
        unit = "mm/h"
    elif lowered.endswith("_mm"):
        unit = "mm"
    elif lowered.endswith("_kpa"):
        unit = "kPa"
    elif lowered.endswith("_km2"):
        unit = "km2"
    elif lowered.endswith("_m2"):
        unit = "m2"
    elif lowered.endswith("_km"):
        unit = "km"
    elif lowered.endswith("_ms"):
        unit = "m/s"
    elif lowered.endswith("_m"):
        unit = "m"
    elif lowered.endswith("_h"):
        unit = "h"
    return {"decimals": 2, "min_decimals": 0 if integral else 1,
            "unit": unit, "percent_equivalent": anonymous_ratio}


def _token_decimals(token: str) -> int | None:
    """Decimal places the token states, or ``None`` when it states none."""
    text = str(token)
    if "e" in text.lower():
        return None
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


class EvidenceLedger:
    """Every value a reply may quote, with the precision it may be quoted at."""

    def __init__(self, entries: Iterable[LedgerEntry], *, identifiers: Iterable[str] = ()) -> None:
        self.entries = [entry for entry in entries if entry.visible]
        self.identifiers = set(identifiers)

    def extend(self, other: "EvidenceLedger") -> None:
        self.entries.extend(other.entries)
        self.identifiers.update(other.identifiers)

    @classmethod
    def from_assessment(cls, assessment: dict[str, Any] | None) -> "EvidenceLedger":
        if not isinstance(assessment, dict):
            return cls([])
        entries: list[LedgerEntry] = []
        for spec in FACTOR_SPECS:
            path = spec.get("path")
            if path is None:
                continue
            node: Any = (assessment.get("factors") or {}) if spec.get("root", "factors") == "factors"                 else assessment
            for key in path:
                node = node.get(key) if isinstance(node, dict) else None
            value = _number(node)
            entries.append(LedgerEntry(
                name=spec["name"], value=value, unit=spec.get("unit", ""),
                decimals=int(spec.get("decimals", 4)),
                min_decimals=int(spec.get("min_decimals", 0)),
                percent_equivalent=bool(spec.get("percent_equivalent", False)),
                resource_id=str(assessment.get("hydro_run_id") or "")))
        distribution = (assessment.get('factors') or {}).get('spatial_distribution') or {}
        entries.extend(cls.from_evidence({'spatial_distribution': distribution}).entries)
        # These are the unabridged facts of this same run, not model output or
        # historical conversation. Keep their names for field/unit binding.
        entries.extend(cls.from_evidence({'result_details': assessment.get('result_details') or {}}).entries)
        for item in assessment.get('derived_evidence') or []:
            entries.append(LedgerEntry(
                name=str(item['name']), value=_number(item['value']),
                unit='' if item.get('unit') == 'fraction' else str(item.get('unit') or ''),
                decimals=6, min_decimals=0,
                percent_equivalent=item.get('unit') == 'fraction',
                source='report_conversion:' + str(item.get('operation')),
                resource_id=str(assessment.get('hydro_run_id') or '')))
        return cls(entries, identifiers=_hash_evidence(assessment))

    @classmethod
    def from_evidence(cls, evidence: Any) -> "EvidenceLedger":
        """A ledger for a plain payload, used by the unit-level audit tests."""
        if isinstance(evidence, dict) and "factors" in evidence:
            return cls.from_assessment(evidence)
        entries: list[LedgerEntry] = []
        for name, value in _flatten(evidence):
            rule = _leaf_rule(name, _number(value))
            entries.append(LedgerEntry(
                name=name, value=_number(value),
                unit=str(rule.get("unit", "")),
                decimals=int(rule["decimals"]),
                min_decimals=int(rule.get("min_decimals", 0)),
                percent_equivalent=bool(rule["percent_equivalent"])))
        return cls(entries, identifiers=_hash_evidence(evidence))

    def values(self) -> list[float]:
        return [entry.value for entry in self.entries if entry.value is not None]

    def audit(self, text: str) -> list[str]:
        """Numbers in ``text`` that no visible entry accounts for."""
        problems: list[str] = []
        def identifier(match):
            token = match.group()
            parts = re.split(r'…|\.\.\.', token, maxsplit=1)
            valid = (token in self.identifiers if len(parts) == 1 else any(
                value.startswith(parts[0]) and value.endswith(parts[1]) for value in self.identifiers))
            if not valid:
                problems.append('输入指纹或摘要未与登记证据匹配')
            return ' ' * len(token)
        cleaned = _strip_non_values(_HASH_TOKEN.sub(identifier, str(text)))
        for match in _NUMBER.finditer(cleaned):
            token = match.group("number")
            as_percent = bool(match.group("percent"))
            observed_unit = "%" if as_percent else _unit_after(cleaned, match.end())
            context = _claim_context(cleaned, match.start(), match.end())
            permitted = [entry for entry in self.entries
                         if _entry_matches_context(entry, context, observed_unit)]
            if not permitted and self.entries and all(
                    _anonymous_entry(entry) for entry in self.entries):
                # Unit helpers and low-level policy tests may provide a bare
                # ``{"x": value}``.  There is no field semantics to bind in
                # that shape; production assessment ledgers use named fields.
                permitted = self.entries
            # A token written with a percent sign is a percentage and nothing
            # else, so only percent-capable entries may account for it.
            if any(entry.accepts(token, as_percent=as_percent,
                                 observed_unit=observed_unit)
                   for entry in permitted):
                continue
            if observed_unit:
                reason = "数值、字段或单位与证据不一致"
            elif context != cleaned:
                reason = "数值与所述字段不一致"
            else:
                reason = "证据中没有这个数值"
            problems.append(f"{token}{'%' if as_percent else ''}（{reason}）")
        return problems


#: A number, optionally followed by a percent sign. A digit inside a word is not
#: a value: "ERA5" and identifier suffixes are skipped.
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])(?P<number>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
                     r"(?P<percent>\s*%)?")
_HASH_TOKEN = re.compile(r'(?<![A-Za-z0-9_])(?:[0-9a-fA-F]{32,64}|'
                         r'[0-9a-fA-F]{8,}(?:…|\.\.\.)[0-9a-fA-F]*)(?![A-Za-z0-9_])')
_EXPLICIT_UNIT = re.compile(
    r"^\s*(mm/h|m/s|kPa|km²|km2|m²|m2|平方千米|平方米|mm|km|m|h|小时)", re.IGNORECASE)
_UNIT_ALIASES = {
    "km²": "km2", "km2": "km2", "平方千米": "km2", "m²": "m2", "m2": "m2",
    "平方米": "m2", "kpa": "kpa", "小时": "h",
    "mm/h": "mm/h", "m/s": "m/s", "mm": "mm", "km": "km",
    "m": "m", "h": "h", "%": "%",
}
#: Thousands separators, so "2,389" is one number.
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
#: Dates and clock times, removed before scanning. The shape has to be a real
#: date or clock — a bare "digit.digit" is a decimal number, not a date.
_TIME_LIKE = re.compile(
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?"
    r"|\d{4}[-/.]\d{1,2}"
    r"|\d{1,2}:\d{2}(?::\d{2})?")
#: A numbered list marker at the start of a line.
_LIST_MARKER = re.compile(r"^\s*[-*]?\s*\d+[.、)]\s+", re.MULTILINE)
#: Identifiers, whose digits are not scientific values.
_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_-])(?:basin|hydro-run|hydro-input|run|input|review|revision|rain|package|viz|artifact)-[0-9a-zA-Z_-]+")
_CALENDAR_ZH = re.compile(r"(?<!\d)(?:(?P<year>\d{4})年)?(?P<month>\d{1,2})月(?P<day>\d{1,2})[日号]")
_CALENDAR_YEAR = re.compile(r"(?<!\d)(?:1\d{3}|2\d{3})年(?!\d)")
#: Version strings and option-style tokens.
_VERSION_LIKE = re.compile(r"\bv?\d+(?:\.\d+){2,}\b")
_YEAR_RANGE = re.compile(r"(?<!\d)\d{4}\s*[–—~-]\s*\d{4}(?!\d)")
# Only the unitless Fs decision boundary at one is a policy constant. Keep
# reported values (including Fs=1), noncanonical thresholds, and attached units
# in the audit. In particular, never consume just the "1" prefix of "1.5".
_FS_SYMBOL = r"(?<![A-Za-z0-9_])(?:F\s*_?\s*s|Fₛ)(?![A-Za-z0-9_])"
_FS_NAME = rf"(?:安全系数(?:\s*[（(]\s*{_FS_SYMBOL}\s*[)）])?|{_FS_SYMBOL})"
_FS_COMPARISON = (
    r"(?:[<>＜＞≤≥≦≧][=＝]?|小于或等于|大于或等于|小于等于|大于等于|"
    r"不大于|不小于|不高于|不低于|小于|大于|低于|高于)"
)
_FS_UNIT_SUFFIX = (
    r"(?:[%％]|(?:mm/h|m/s|kPa|km²|km2|m²|m2|mm|km|kN|kg|Pa|N|m|h|s)(?![A-Za-z])|"
    r"小时|平方千米|平方米|千米|毫米|米|秒)"
)
_FS_ONE = (r"(?P<threshold>1(?:\.0+)?)"
           rf"(?![\dA-Za-z_]|\.\d|,\d{{3}})(?![ \t]*{_FS_UNIT_SUFFIX})")
_FS_BOUNDARY_LABEL = r"(?:判别|判断)?(?:阈值|临界值|界限|分界线|界)"
_FS_THRESHOLDS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    rf"{_FS_NAME}\s*{_FS_COMPARISON}\s*{_FS_ONE}",
    rf"{_FS_NAME}\s*(?:的\s*)?{_FS_BOUNDARY_LABEL}\s*(?:为|取(?:值)?(?:为)?|[=＝])\s*{_FS_ONE}",
    # Equality alone may report a measured value. Require an explicit boundary
    # definition, such as "以Fs=1为界", before exempting that operand.
    rf"{_FS_NAME}\s*(?:[=＝]|等于)\s*{_FS_ONE}"
    rf"(?=\s*(?:[)）]\s*)?(?:作为|是|为)\s*{_FS_BOUNDARY_LABEL})",
))


def _mask_fs_thresholds(text: str) -> str:
    """Remove a decision condition from result-field matching as well.

    In "事件最小安全系数小于1的区域面积为...", the quantity reported is
    an area, not a minimum safety factor. Leave reported field labels outside
    the matched condition intact, while excluding the condition's own label.
    """
    for pattern in _FS_THRESHOLDS:
        text = pattern.sub(lambda match: " " * len(match.group()), text)
    return text

# A number must belong to the field named in its own clause.  Without this,
# rainfall 150.72 could be relabelled as a safety factor merely because the
# number occurred somewhere in the evidence ledger.
_CONTEXT_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"峰值小时雨量|小时雨量|[Pp]eak hourly rainfall"), ("peak_hourly_rain", "peak_mm_h")),
    (re.compile(r"累计降雨|累计量|累计|总雨量|日均值|[Tt]otal rainfall"), ("total_rain", "total_mm")),
    (re.compile(r"平均安全系数|安全系数均值|(?<!日)均值|[Mm]ean factor of safety"), ("mean_safety_factor", ".mean")),
    (re.compile(r"最[小低]安全系数|安全系数最小|最[小低]\s*F\s*_?\s*s|最小值|minimum factor of safety", re.IGNORECASE), ("minimum_safety_factor", ".minimum")),
    (re.compile(r"P95", re.IGNORECASE), ("p95_safety_factor", ".p95")),
    (re.compile(r"失稳面积比例|失稳比例|unstable fraction"), ("max_unstable_fraction",)),
    (re.compile(r"失稳持续时间|最长持续时间"), ("unstable_duration",)),
    (re.compile(r"水深|flow depth"), ("flow_depth",)),
    (re.compile(r"流速|flow velocity"), ("velocity",)),
    (re.compile(r"孔隙水压力|孔压|pore water pressure"), ("pore_water_pressure",)),
    (re.compile(r"面积"), ("area_km2", "area_m2", "area_fraction", "unstable_fraction")),
    (re.compile(r"像元(?:数量|个数|数)|选中|个(?:像元|栅格)"), ("selected_cells", "cell_count", "pixel_count")),
    (re.compile(r"高程"), ("elevation_m",)),
    (re.compile(r"偏移"), ("shift_m",)),
    (re.compile(r"经纬度|经度|纬度"), ("longitude", "latitude", "lon", "lat")),
    (re.compile(r"峰值"), ("peak_hourly_rain", "peak_mm_h", "max_flow_depth", "max_velocity", "max_pore_water_pressure")),
    (re.compile(r"占"), ("fraction", "share_of")),
    (re.compile(r"无数据区接触|NoData接触", re.IGNORECASE), ("nodata_contact_cells", "nodata_contact_fraction")),
    (re.compile(r"虚拟站|雨站|站点"), ("station_count",)),
    (re.compile(r"项产物|成果数量|文件数量"), ("artifact_count", "artifact_catalog")),
)


def _normalise_unit(unit: str) -> str:
    return _UNIT_ALIASES.get(str(unit).strip().lower(), str(unit).strip().lower())


def _unit_after(text: str, end: int) -> str:
    matched = _EXPLICIT_UNIT.match(text[end:])
    return matched.group(1) if matched else ""


def _claim_context(text: str, start: int, end: int) -> str:
    left = max(text.rfind(mark, 0, start) for mark in ("\n", "。", "；", ";", "，", "、")) + 1
    right_candidates = [position for mark in ("\n", "。", "；", ";", "，", "、")
                        if (position := text.find(mark, end)) != -1]
    right = min(right_candidates) if right_candidates else len(text)
    # A later label in the same clause must not rename an earlier quantity:
    # "面积3.92 km²，占计算域20.12 km²的19.5%" contains three fields.
    prefix = text[left:start]
    count_label = re.match(r'\s*(?:个|项|套)\s*(?:[A-Za-z][A-Za-z0-9_-]*\s*)?'
                           r'(?:虚拟雨?站|雨站|站点|产物|成果|文件|栅格|像元)', text[end:right])
    if count_label and not any(pattern.search(prefix) for pattern, _ in _CONTEXT_RULES):
        return prefix + count_label.group()
    if any(pattern.search(prefix) for pattern, _ in _CONTEXT_RULES):
        return prefix
    return text[left:right]


def _entry_matches_context(entry: LedgerEntry, context: str, observed_unit: str = '') -> bool:
    lowered = entry.name.lower()
    # Bind to the closest field label. "逐像元事件最小值的面积" is an
    # area, whereas "最小安全系数" is a factor even if given a wrong unit.
    matches = [(match.end(), -index, allowed, pattern.pattern) for index, (pattern, allowed)
               in enumerate(_CONTEXT_RULES) for match in pattern.finditer(context)]
    allowed = max(matches)[2] if matches else None
    if matches and max(matches)[3] == '峰值':
        specific = [item for item in matches if item[3] != '峰值']
        if specific and not any(key in max(specific)[2] for key in ('total_rain', 'total_mm')):
            allowed = max(specific)[2]
    if (_normalise_unit(observed_unit) in {'km2', 'm2'} and allowed
            and all(key in {'fraction', 'share_of'} for key in allowed)):
        # A parenthesised area after a percentage retains the same footprint,
        # e.g. "占19.5%（约3.92 km²）". A factor label never gets this override.
        allowed = ('area_km2', 'area_m2')
    if 'spatial_distribution' in lowered:
        if allowed and any(key in allowed for key in ('minimum_safety_factor', 'mean_safety_factor', 'p95_safety_factor')):
            return False
        regions = ('西北部', '北部', '东北部', '西部', '中部', '东部', '西南部', '南部', '东南部')
        mentioned = [r for r in regions if r in context]
        if mentioned and '.sectors.' in lowered and not any(r in entry.name for r in mentioned):
            return False
        if re.search(r'同一时刻|瞬时|最大失稳(?:面积)?比例', context):
            return False
        if re.search(r'Fs|安全系数|区域|面积|比例|占|阈值', context, re.IGNORECASE):
            if observed_unit == '%' or (not observed_unit and re.search(r'占|比例|百分', context)):
                return any(key in lowered for key in ('fraction', 'share_of'))
            if allowed:
                return any(fragment.lower() in lowered for fragment in allowed)
            return True
    if allowed:
        return any(fragment.lower() in lowered for fragment in allowed)
    return True


def _anonymous_entry(entry: LedgerEntry) -> bool:
    leaf = entry.name.rsplit(".", 1)[-1].split("[")[0].lower()
    return leaf in {"x", "v", "value"}


def _strip_non_values(text: str) -> str:
    """Remove everything that is not a reported scientific value."""
    cleaned = _IDENTIFIER.sub(" ", text)
    cleaned = _VERSION_LIKE.sub(" ", cleaned)
    cleaned = _YEAR_RANGE.sub(" ", cleaned)
    cleaned = _mask_fs_thresholds(cleaned)
    cleaned = _TIME_LIKE.sub(" ", cleaned)
    def calendar(match):
        from datetime import date
        try:
            date(int(match.group('year') or 2000), int(match.group('month')), int(match.group('day')))
        except ValueError:
            return match.group(0)
        return ' ' * len(match.group(0))
    cleaned = _CALENDAR_ZH.sub(calendar, cleaned)
    cleaned = _CALENDAR_YEAR.sub(" ", cleaned)
    cleaned = _LIST_MARKER.sub("", cleaned)
    return _THOUSANDS.sub("", cleaned)


def numeric_audit(text: str, evidence: Any) -> list[str]:
    """Numbers in ``text`` that cannot be found in ``evidence``."""
    ledger = (evidence if isinstance(evidence, EvidenceLedger)
              else EvidenceLedger.from_evidence(evidence))
    return ledger.audit(text)


def _flatten(node: Any, trail: str = "", depth: int = 0) -> list[tuple[str, Any]]:
    if depth > 12:
        return []
    found: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.extend(_flatten(value, f"{trail}.{key}".strip("."), depth + 1))
    elif isinstance(node, (list, tuple)):
        found.append((trail, float(len(node))))
        for index, value in enumerate(node):
            found.extend(_flatten(value, f"{trail}[{index}]", depth + 1))
    elif not isinstance(node, bool):
        found.append((trail, node))
    return found


def _hash_evidence(evidence):
    return {value for name, value in _flatten(evidence)
            if isinstance(value, str) and re.fullmatch(r'[0-9a-fA-F]{32,64}', value)
            and any(key in name.lower() for key in ('fingerprint', 'sha256', 'hash'))}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- forbidden


#: Claims that may not appear while the evidence only supports a screening
#: result. Ordered so the most consequential are reported first.
FORBIDDEN_PHRASES_EXTENDED: tuple[str, ...] = FORBIDDEN_PHRASES + (
    "已证明", "证明了", "必然", "一定会", "导致", "造成失稳",
    "低风险", "中风险", "高风险", "风险等级", "安全等级", "稳定等级",
    "泥石流易发性", "易发性等级", "工程安全", "可以放心",
)


def forbidden_reasons(text: str) -> list[str]:
    """Forbidden claims present in a reply, with the phrase that triggered each.

    The original text is searched, not the sanitized one: ``sanitize`` rewrites
    these very phrases, so cleaning first would hide every violation.
    """
    cleaned = str(text)
    found: list[str] = []
    for phrase in FORBIDDEN_PHRASES_EXTENDED:
        position = cleaned.find(phrase)
        if position != -1 and not _negated(cleaned, position):
            if phrase in {'导致', '造成失稳'} and any(modal in cleaned[max(0,position-14):position]
                    for modal in ('可能', '通常', '一般', '机理上')):
                continue
            found.append(phrase)
    return [f"出现不允许的表述：{phrase}" for phrase in dict.fromkeys(found)]


_NEGATIONS = NEGATIONS


def _negated(text: str, start: int, window: int = 6) -> bool:
    return any(negation in text[max(0, start - window):start] for negation in _NEGATIONS)


#: Phrases that assert stability or safety rather than describing a simulation.
STABILITY_CLAIMS = ("坡体安全", "边坡安全", "已经稳定", "不会失稳", "不会滑坡",
                    "整体稳定", "保证安全")


def stability_claim_reasons(text: str) -> list[str]:
    cleaned = str(text)
    found = [phrase for phrase in STABILITY_CLAIMS
             if phrase in cleaned and not _negated(cleaned, cleaned.find(phrase))]
    return [f"出现不允许的稳定性结论：{phrase}" for phrase in found]


def audit_reply(text: str, evidence: Any) -> list[str]:
    """Everything wrong with a reply: untraceable numbers and forbidden claims."""
    ledger = (evidence if isinstance(evidence, EvidenceLedger)
              else EvidenceLedger.from_evidence(evidence))
    problems = [f"回答中的数字在证据中找不到：{item}" for item in ledger.audit(text)]
    problems.extend(forbidden_reasons(text))
    problems.extend(stability_claim_reasons(text))
    return problems


# ---------------------------------------------------------------- statements


def interpretations(safety: dict[str, Any], rainfall: dict[str, Any],
                    extent: dict[str, Any]) -> list[dict[str, Any]]:
    """The statements the evidence supports, each pointing at its own numbers."""
    statements: list[dict[str, Any]] = []

    minimum = _number(safety.get("minimum_safety_factor"))
    mean = _number(safety.get("mean_safety_factor"))
    if minimum is not None and minimum < 1.0:
        statements.append({
            "text": f"事件期内存在局部像元进入 FS<1 状态，最小值为 {minimum:.3f}",
            "basis": "safety.minimum_safety_factor",
            "scope": "local",
        })
    elif minimum is not None:
        statements.append({
            "text": f"事件期内每个像元的最小安全系数均不低于 {minimum:.3f}",
            "basis": "safety.minimum_safety_factor",
            "scope": "cell-wise minimum",
        })
    if mean is not None:
        statements.append({
            "text": f"流域像元最小安全系数的均值为 {mean:.3f}，"
                    "可与最小值一并说明「局部」与「整体」的差别",
            "basis": "safety.mean_safety_factor",
            "scope": "domain mean",
        })

    fraction = _number(extent.get("max_unstable_fraction"))
    if fraction is not None:
        statements.append({
            "text": f"模型判定的最大失稳面积比例为 {fraction * 100:.2f}%"
                    "（某一时刻的瞬时值，不是全时段合计）",
            "basis": "safety.max_unstable_fraction",
            "scope": "instantaneous extent",
        })
    duration = _number(extent.get("max_unstable_duration_h"))
    if duration is not None:
        statements.append({
            "text": f"局部像元处于失稳状态的最长持续时间为 {duration:.2f} 小时",
            "basis": "safety.max_unstable_duration_h",
            "scope": "local duration",
        })

    depth = _number(extent.get("max_flow_depth_m"))
    velocity = _number(extent.get("max_velocity_ms"))
    pore = _number(extent.get("max_pore_water_pressure_kpa"))
    hydraulic = [f"最大水深 {depth:.4f} m" if depth is not None else None,
                 f"最大流速 {velocity:.4f} m/s" if velocity is not None else None,
                 f"最大孔隙水压力 {pore:.2f} kPa" if pore is not None else None]
    hydraulic = [item for item in hydraulic if item]
    if hydraulic:
        statements.append({
            "text": "水力与孔压极值：" + "；".join(hydraulic),
            "basis": "safety hydraulics",
            "scope": "domain maximum",
        })

    total = _number(rainfall.get("total_rain_mm"))
    peak = _number(rainfall.get("peak_hourly_rain_mm"))
    if total is not None:
        detail = f"事件期间虚拟雨站面平均累计降雨 {total:.2f} mm"
        if peak is not None:
            detail += f"，峰值小时雨量 {peak:.2f} mm/h"
        statements.append({"text": detail, "basis": "rainfall",
                           "scope": "virtual stations"})
    return statements


def limitations(*, rainfall_source: str | None = None,
                preset: str | None = None,
                extra: Iterable[str] = ()) -> list[str]:
    """The caveats that apply to every result of this kind."""
    items = [
        "本结果为区域尺度、降雨事件驱动的坡面稳定性初步筛查与诊断，不是工程级边坡安全鉴定。",
        "降雨来自 ERA5 网格虚拟雨站，不是地面实测雨量站。",
        "当前模型为独立子汇水区出水口配置：各子汇水区径流之和不是选定流域出口的路由流量过程。",
        "安全系数由简化力学模型计算，未做场地勘察标定。",
    ]
    if preset:
        items.append(f"结果为参数预设「{preset}」下的取值；更换预设或分辨率后不可直接排名。")
    if rainfall_source == "historical_max":
        items.append("降雨事件为资料期内样本最大日，不是重现期设计暴雨。")
    items.extend(extra)
    return [sanitize(item) for item in items]


FORBIDDEN_CLAIMS = FORBIDDEN_PHRASES_EXTENDED
