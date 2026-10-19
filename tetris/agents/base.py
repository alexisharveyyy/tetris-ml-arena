"""Common agent interface.

Every agent chooses a :class:`~tetris.engine.placement.Placement` for the
current piece. The shared path planner turns that placement into moves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tetris.agents.difficulty import DifficultyParams
from tetris.engine.game import Game
from tetris.engine.placement import Placement
from tetris.env.afterstates import legal_placements

#: Noise mistakes pick among the next this-many ranked placements...
MISTAKE_POOL = 5
#: ...with probability decaying by this factor per rank (rank 2 most likely).
MISTAKE_DECAY = 0.5


class NoLegalPlacementError(RuntimeError):
    """Raised when an agent is asked to move with no legal placement."""


@dataclass(frozen=True)
class ScoredPlacement:
    placement: Placement
    score: float


def as_game(game_state: Game | dict[str, Any]) -> Game:
    """Accept a live game or its serialized state dict."""
    return game_state if isinstance(game_state, Game) else Game.from_dict(game_state)


class Agent(ABC):
    """Base class for all agents.

    Subclasses implement :meth:`rank_placements` (best first) and usually
    :meth:`load`. Difficulty noise (``epsilon``) is applied here.
    """

    #: Registry key, also the checkpoint directory name.
    key: str = "agent"
    name: str = "Agent"
    description: str = ""

    def __init__(self, seed: int | None = None) -> None:
        self.difficulty = DifficultyParams()
        self.rng = np.random.default_rng(seed)

    def load(self, checkpoint: str | Path) -> None:
        """Load model parameters from a checkpoint."""
        raise NotImplementedError(f"{self.name} does not use checkpoints")

    def seed(self, seed: int | None) -> None:
        self.rng = np.random.default_rng(seed)

    def set_difficulty(self, difficulty: DifficultyParams) -> None:
        self.difficulty = difficulty

    @abstractmethod
    def rank_placements(self, game: Game) -> list[ScoredPlacement]:
        """Legal placements with scores, best first. May be truncated."""

    def _choose(self, game: Game, ranked: list[ScoredPlacement]) -> Placement:
        """Apply difficulty noise to a ranking and return the chosen placement."""
        d = self.difficulty
        if d.epsilon > 0 and self.rng.random() < d.epsilon:
            options = legal_placements(game)
            if options:
                return options[int(self.rng.integers(len(options)))][0]
        if not ranked:
            raise NoLegalPlacementError("no legal placement")
        if d.noise > 0 and len(ranked) > 1 and self.rng.random() < d.noise:
            pool = ranked[1 : 1 + MISTAKE_POOL]
            weights = MISTAKE_DECAY ** np.arange(len(pool))
            return pool[int(self.rng.choice(len(pool), p=weights / weights.sum()))].placement
        return ranked[0].placement

    def select_placement(self, game_state: Game | dict[str, Any]) -> Placement:
        """Choose a placement for the current piece (difficulty noise applied)."""
        game = as_game(game_state)
        return self._choose(game, self.rank_placements(game))

    def decide(
        self, game_state: Game | dict[str, Any], top_k: int = 0
    ) -> tuple[Placement, list[ScoredPlacement]]:
        """Placement plus the top ``top_k`` ranked candidates (for overlays)."""
        game = as_game(game_state)
        ranked = self.rank_placements(game)
        return self._choose(game, ranked), ranked[:top_k]
