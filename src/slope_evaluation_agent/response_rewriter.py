"""Constrained rewriting of an already-decided reply.

The order is fixed and not negotiable:

1. deterministic code builds a :class:`CanonicalReply` — the facts, the caveats,
   the next actions, and the template text;
2. the model is asked to *rephrase* that text, receiving nothing else;
3. the result is audited against the ledger, and on any failure the canonical
   text is what the user gets.

The model is never asked for a number, a status, an identifier or a conclusion,
and it has nowhere to put one: the facts are already strings, and the audit
compares what comes back against the ledger.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .scientific_policy import EvidenceLedger, audit_reply
from .response_writer import sanitize


SYSTEM_PROMPT = (
    "你是坡面稳定性初步评价智能体的语言改写器。"
    "你会收到一段已经定稿的中文回复，以及它的结构化事实。"
    "只允许改写措辞，使表达更自然、更贴合用户的问题。"
    "不得新增、删除或修改任何数字、单位、状态、编号或结论；"
    "不得把局部结论扩大为整体结论；不得使用因果性措辞；"
    "不得给出风险等级、工程处置建议或泥石流易发性判断。"
    "输入中的 [[LITERAL_X]] 和 [[BOUNDARY_X]] 是不可修改的占位符，"
    "必须各保留一次且顺序不变。"
    "只输出一个 JSON 对象，唯一字段为 text；text 的值是改写后的正文。"
)


_BOUNDARY_MARKERS = ("限制：", "不是", "不等于", "不构成", "不能", "不得", "不要据此")
_PLACEHOLDER = re.compile(r"\[\[(?:LITERAL|BOUNDARY)_[A-Z]+\]\]")
_LITERAL = re.compile(
    r"\b(?:basin|hydro-run|hydro-input|rain|package|viz|artifact)-[0-9A-Za-z]+"
    r"|(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?"
    r"(?:\s*(?:mm/h|m/s|kPa|km²|km2|mm|km|m|h|小时|个|年|时次|%))?",
    re.IGNORECASE,
)


def _letters(index: int) -> str:
    """A, B, ..., Z, AA ... without digits inside a placeholder."""
    result = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result


def protect_reply(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Protect scientific boundaries, numbers, units and opaque identifiers.

    A model is useful for connective prose, not for editing evidence.  Boundary
    lines are frozen verbatim and every literal is replaced by an ordered token.
    The response is accepted only when the exact token sequence comes back.
    """
    replacements: list[tuple[str, str]] = []
    lines: list[str] = []
    for line in str(text).splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body):]
        if body and any(marker in body for marker in _BOUNDARY_MARKERS):
            token = f"[[BOUNDARY_{_letters(len(replacements))}]]"
            replacements.append((token, body))
            lines.append(token + ending)
        else:
            lines.append(line)
    protected = "".join(lines)

    def replace_literal(match: re.Match[str]) -> str:
        token = f"[[LITERAL_{_letters(len(replacements))}]]"
        replacements.append((token, match.group(0)))
        return token

    protected = _LITERAL.sub(replace_literal, protected)
    by_token = dict(replacements)
    ordered = [(token, by_token[token]) for token in _PLACEHOLDER.findall(protected)]
    return protected, ordered


def restore_reply(text: str, replacements: list[tuple[str, str]]) -> tuple[str | None, str | None]:
    """Restore protected material, rejecting deletion, duplication or invention."""
    expected = [token for token, _ in replacements]
    found = _PLACEHOLDER.findall(str(text))
    if found != expected:
        return None, "模型删除、复制或调换了受保护的事实"
    residue = _PLACEHOLDER.sub("", str(text))
    if re.search(r"\d", residue):
        return None, "模型新增了不受保护的数字"
    restored = str(text)
    for token, original in replacements:
        restored = restored.replace(token, original)
    return restored, None


@dataclass(frozen=True)
class Fact:
    """One value the canonical reply states."""

    name: str
    value: str
    unit: str = ""
    source: str = ""


@dataclass
class CanonicalReply:
    """The deterministic reply, with everything it is allowed to say."""

    intent: str
    status: str
    canonical_text: str
    facts: list[Fact] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)

    def model_payload(self, *, text_override: str | None = None,
                      include_fact_values: bool = True) -> str:
        """The only thing the model is shown: the text and its facts."""
        lines = [f"状态：{self.status}", "", "事实："]
        for fact in self.facts:
            if include_fact_values:
                unit = f" {fact.unit}" if fact.unit else ""
                lines.append(f"- {fact.name} = {fact.value}{unit}")
            else:
                lines.append(f"- {fact.name}（值已在正文中锁定）")
        if self.caveats:
            lines.append("")
            lines.append("必须保留的科学边界：")
            lines.extend(f"- {item}" for item in self.caveats)
        lines.append("")
        lines.append("待改写的正文：")
        lines.append(self.canonical_text if text_override is None else text_override)
        return "\n".join(lines)


@dataclass
class RewriteOutcome:
    """What was returned, and why it was not the model's text."""

    text: str
    used_model: bool = False
    fallback_reason: str | None = None
    duration_ms: int = 0


class ResponseRewriter:
    """Rephrase a canonical reply, within the audit."""

    def __init__(self, llm: Any | None = None,
                 logger: logging.Logger | None = None) -> None:
        self.llm = llm
        self.logger = logger or logging.getLogger("slope_agent")

    @property
    def available(self) -> bool:
        return self.llm is not None

    async def rewrite(self, canonical: CanonicalReply,
                      ledger: EvidenceLedger) -> RewriteOutcome:
        """Return an audited reply: the model's if it passes, else the canonical."""
        if self.llm is None:
            return RewriteOutcome(text=canonical.canonical_text,
                                  fallback_reason="未配置语言模型")
        import time

        started = time.monotonic()
        protected, replacements = protect_reply(canonical.canonical_text)
        try:
            payload = await self.llm.complete_json(
                system=SYSTEM_PROMPT,
                user=canonical.model_payload(text_override=protected,
                                             include_fact_values=False))
        except Exception as exc:  # noqa: BLE001 - any failure falls back
            return RewriteOutcome(
                text=canonical.canonical_text, fallback_reason=f"改写失败：{type(exc).__name__}",
                duration_ms=int((time.monotonic() - started) * 1000))
        text = self._extract(payload)
        if not text:
            return RewriteOutcome(text=canonical.canonical_text,
                                  fallback_reason="改写结果为空",
                                  duration_ms=int((time.monotonic() - started) * 1000))
        text, integrity_problem = restore_reply(text, replacements)
        if integrity_problem or text is None:
            return RewriteOutcome(
                text=canonical.canonical_text,
                fallback_reason=integrity_problem or "无法恢复受保护事实",
                duration_ms=int((time.monotonic() - started) * 1000))
        problems = audit_reply(text, ledger)
        if problems:
            self.logger.info("改写未通过审计，回退确定性文本：%s", problems[:3])
            return RewriteOutcome(
                text=canonical.canonical_text,
                fallback_reason=f"审计未通过：{'; '.join(problems[:2])}",
                duration_ms=int((time.monotonic() - started) * 1000))
        return RewriteOutcome(text=sanitize(text), used_model=True,
                              duration_ms=int((time.monotonic() - started) * 1000))

    @staticmethod
    def _extract(payload: Any) -> str:
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, dict):
            for key in ("text", "reply", "content", "canonical_text"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""


def build_rewriter(llm_settings: Any, logger: logging.Logger | None = None) -> ResponseRewriter:
    """A rewriter carrying a model when one is configured, else a pass-through."""
    from .llm import build_llm_client

    return ResponseRewriter(build_llm_client(llm_settings, logger), logger)
