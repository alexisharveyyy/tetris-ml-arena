"""LLM agent: a novelty opponent that asks a language model where to place each piece.

Each decision renders the game as text (:mod:`tetris.agents.llm.prompt`),
sends it to the configured backend and plays the numbered placement the model
picks. Anything that goes wrong falls back to the GA heuristic's move:

* ``timeout``: no reply within ``timeout_seconds``;
* ``invalid``: the reply is not JSON with a listed number;
* ``error``: the backend failed (connection, HTTP, missing key);
* ``budget``: the API spending cap would be crossed (the call is not sent).

Counters for every outcome, seconds per decision and token use are kept in
:attr:`LLMAgent.stats`. With ``stop_on_budget`` (evaluation runs) a spent
budget raises instead of falling back, so an evaluation stops cleanly.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tetris.agents.base import Agent, ScoredPlacement
from tetris.agents.llm.backends import Backend, BackendError, BackendTimeout, make_backend
from tetris.agents.llm.prompt import (
    SYSTEM_PROMPT,
    InvalidAnswer,
    options,
    parse_choice,
    user_prompt,
)
from tetris.agents.llm.spending import BudgetExhausted
from tetris.engine.game import Game

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
OUTCOMES = ("llm", "timeout", "invalid", "error", "budget")


def load_llm_config(path: Path | None = None) -> dict[str, Any]:
    return yaml.safe_load((path or DEFAULT_CONFIG).read_text())


@dataclass
class LLMStats:
    outcomes: Counter = field(default_factory=Counter)
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    last_error: str = ""

    @property
    def decisions(self) -> int:
        return sum(self.outcomes.values())

    @property
    def fallbacks(self) -> int:
        return self.decisions - self.outcomes["llm"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decisions": self.decisions,
            "outcomes": {k: self.outcomes[k] for k in OUTCOMES},
            "fallback_rate": self.fallbacks / max(self.decisions, 1),
            "seconds_per_decision": self.seconds / max(self.decisions, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }


class LLMAgent(Agent):
    key = "llm"
    name = "LLM (novelty)"
    description = "A language model picks each placement from a text description of the board."

    def __init__(self, seed: int | None = None, config: dict[str, Any] | None = None) -> None:
        super().__init__(seed)
        self.cfg = config or load_llm_config()
        self.variant: str = self.cfg["prompt"]
        self.timeout = float(self.cfg["timeout_seconds"])
        self.stop_on_budget = False
        self._backend: Backend | None = None
        self._fallback: Agent | None = None
        self.stats = LLMStats()

    def configure(
        self,
        backend: Backend | None = None,
        variant: str | None = None,
        timeout: float | None = None,
        stop_on_budget: bool | None = None,
    ) -> None:
        if backend is not None:
            self._backend = backend
        if variant is not None:
            self.variant = variant
        if timeout is not None:
            self.timeout = timeout
        if stop_on_budget is not None:
            self.stop_on_budget = stop_on_budget

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = make_backend(self.cfg)
        return self._backend

    @property
    def fallback(self) -> Agent:
        if self._fallback is None:
            from tetris.agents.registry import create_agent

            self._fallback = create_agent("ga")
        return self._fallback

    def rank_placements(self, game: Game) -> list[ScoredPlacement]:
        """The model's choice first, then the GA heuristic's ranking."""
        heuristic = self.fallback.rank_placements(game)
        if not heuristic:
            return []
        opts = options(game)
        t0 = time.perf_counter()
        outcome, chosen = "llm", None
        try:
            reply = self.backend.complete(
                SYSTEM_PROMPT, user_prompt(game, opts, self.variant), self.timeout
            )
            self.stats.input_tokens += reply.input_tokens
            self.stats.output_tokens += reply.output_tokens
            self.stats.cost_usd += reply.cost_usd
            chosen = opts[parse_choice(reply.text, len(opts)) - 1].placement
        except BudgetExhausted as exc:
            outcome, self.stats.last_error = "budget", str(exc)
            if self.stop_on_budget:
                self.stats.outcomes[outcome] += 1
                raise
        except BackendTimeout as exc:
            outcome, self.stats.last_error = "timeout", str(exc)
        except InvalidAnswer as exc:
            outcome, self.stats.last_error = "invalid", str(exc)
        except BackendError as exc:
            outcome, self.stats.last_error = "error", str(exc)
        finally:
            self.stats.seconds += time.perf_counter() - t0
        self.stats.outcomes[outcome] += 1
        if chosen is None:
            return heuristic
        first = [s for s in heuristic if s.placement == chosen]
        if not first:  # an equivalent placement the heuristic ranked under another index
            first = [ScoredPlacement(chosen, float("inf"))]
        return first + [s for s in heuristic if s.placement != chosen]
