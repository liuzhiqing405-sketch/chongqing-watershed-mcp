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
SCOPE_STATEMENT = "Preliminary regional screening and diagnosis of rainfall-driven slope stability"

DISCLAIMER_ZH = (
    "The results describe preliminary regional screening under the modelled rainfall event. "
    "Their scope excludes engineering safety certification, hazard forecasting and debris-flow susceptibility."
)
