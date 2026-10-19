"""GA-tuned linear heuristic agent."""

from __future__ import annotations

from tetris.agents.linear import LinearAgent


class GAAgent(LinearAgent):
    key = "ga"
    name = "GA heuristic"
    description = "Linear feature weights evolved with a genetic algorithm."
