"""Slope-stability preliminary evaluation agent.

An orchestration agent that sits *above* the two existing MCP servers. It owns
natural-language understanding, session state, a deterministic workflow, human
confirmation gates, task tracking and evidence-bound explanations. It never
performs project science itself: every reported computation metric comes from
an MCP response. General engineering knowledge uses a separately labelled
language-model channel, which may explain formulas and hypothetical examples.

Computation scope, fixed for this release:

    Regional-scale, rainfall-event-driven slope-stability preliminary screening
    and diagnosis.

It is not an engineering slope-safety determination, a hazard forecast or a
debris-flow susceptibility assessment, and its output must never be described
as any of those.
"""

from __future__ import annotations

__all__ = ["__version__", "SCOPE_STATEMENT"]

__version__ = "1.0"

#: The only accurate description of what this system does. Used verbatim by the
#: response writer so the scope cannot drift between releases.
SCOPE_STATEMENT = "区域尺度、降雨事件驱动的坡面稳定性初步筛查与诊断智能体"

DISCLAIMER_ZH = (
    "本系统给出的是区域尺度、降雨事件驱动的坡面稳定性初步筛查与诊断结果，"
    "不是工程级边坡安全鉴定，不是灾害预报，也不是泥石流易发性评价。"
)
