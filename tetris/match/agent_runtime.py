"""Agent runners: where agent inference happens.

The controller talks to agents only through :class:`AgentRunner`, using an
asynchronous submit/poll protocol with serialized game states. The inline
runner below answers immediately in-process; a process-pool runner can
implement the same interface without any change to the controller.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from tetris.agents.difficulty import DifficultyParams
from tetris.engine.placement import Placement


@dataclass
class Decision:
    """An agent's answer to one decision request."""

    request_id: int
    placement: Placement | None
    candidates: list[tuple[Placement, float]] = field(default_factory=list)
    think_time: float = 0.0
    error: str | None = None


class AgentRunner(ABC):
    @abstractmethod
    def add_agent(
        self,
        player_id: str,
        agent_key: str,
        difficulty: DifficultyParams,
        checkpoint: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Create (and load) the agent that plays for ``player_id``."""

    @abstractmethod
    def set_difficulty(self, player_id: str, difficulty: DifficultyParams) -> None: ...

    @abstractmethod
    def submit(self, player_id: str, request_id: int, state: dict[str, Any], top_k: int) -> None:
        """Ask for a decision on a serialized game state."""

    @abstractmethod
    def poll(self, player_id: str) -> Decision | None:
        """Return the latest finished decision for a player, if any."""

    def close(self) -> None:  # noqa: B027 - optional hook, default does nothing
        """Release resources."""


def decide(agent: Any, request_id: int, state: dict[str, Any], top_k: int) -> Decision:
    """Run one decision with error capture (shared by all runners)."""
    t0 = time.perf_counter()
    try:
        placement, ranked = agent.decide(state, top_k)
        return Decision(
            request_id,
            placement,
            [(c.placement, c.score) for c in ranked],
            time.perf_counter() - t0,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the driver, which falls back
        return Decision(request_id, None, [], time.perf_counter() - t0, error=repr(exc))


class InlineRunner(AgentRunner):
    """Runs agents synchronously in the calling thread. For tests and light agents."""

    def __init__(self) -> None:
        self._agents: dict[str, Any] = {}
        self._results: dict[str, Decision] = {}

    def add_agent(
        self,
        player_id: str,
        agent_key: str,
        difficulty: DifficultyParams,
        checkpoint: str | None = None,
        seed: int | None = None,
    ) -> None:
        from tetris.agents.registry import create_agent

        self._agents[player_id] = create_agent(agent_key, checkpoint, difficulty, seed)

    def set_difficulty(self, player_id: str, difficulty: DifficultyParams) -> None:
        self._agents[player_id].set_difficulty(difficulty)

    def submit(self, player_id: str, request_id: int, state: dict[str, Any], top_k: int) -> None:
        self._results[player_id] = decide(self._agents[player_id], request_id, state, top_k)

    def poll(self, player_id: str) -> Decision | None:
        return self._results.pop(player_id, None)
