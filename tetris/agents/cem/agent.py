"""Cross-entropy method linear heuristic agent."""

from __future__ import annotations

from tetris.agents.linear import LinearAgent


class CEMAgent(LinearAgent):
    key = "cem"
    name = "Cross-entropy method"
    description = "Linear feature weights optimized with the cross-entropy method."
