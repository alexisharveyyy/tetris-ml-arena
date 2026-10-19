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

    def select_placement(self, game_state: Game | dict[str, Any]) -> Placement:
        """Choose a placement for the current piece."""
        game = as_game(game_state)
        if self.difficulty.epsilon > 0 and self.rng.random() < self.difficulty.epsilon:
            options = legal_placements(game)
            if not options:
                raise NoLegalPlacementError("no legal placement")
            return options[int(self.rng.integers(len(options)))][0]
        ranked = self.rank_placements(game)
        if not ranked:
            raise NoLegalPlacementError("no legal placement")
        return ranked[0].placement

    def decide(
        self, game_state: Game | dict[str, Any], top_k: int = 0
    ) -> tuple[Placement, list[ScoredPlacement]]:
        """Placement plus the top ``top_k`` ranked candidates (for overlays)."""
        game = as_game(game_state)
        placement = self.select_placement(game)
        candidates = self.rank_placements(game)[:top_k] if top_k > 0 else []
        return placement, candidates
