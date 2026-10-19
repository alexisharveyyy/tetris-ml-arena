"""PPO agent: an actor-critic policy over the 80 placement actions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tetris.agents.base import Agent, ScoredPlacement
from tetris.engine.game import Game
from tetris.engine.placement import Placement
from tetris.env.afterstates import action_mask
from tetris.rl.encoding import observation_arrays
from tetris.rl.nets import load_checkpoint


class PPOAgent(Agent):
    """Plays the most probable legal placement (deterministic at play time).

    Difficulty mistakes pick among the next five legal actions by policy probability.
    """

    key = "ppo"
    name = "PPO"
    description = "Actor-critic policy over the masked 80-action placement space."

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed)
        self.model: torch.nn.Module | None = None

    def load(self, checkpoint: str | Path) -> None:
        self.model, _ = load_checkpoint(checkpoint, "ppo", "cpu")

    def probabilities(self, game: Game) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("PPO agent has no checkpoint loaded")
        board, extras = observation_arrays(game)
        mask = action_mask(game)
        with torch.inference_mode():
            logits, _ = self.model(
                torch.from_numpy(board[None].astype(np.float32)),
                torch.from_numpy(extras[None]),
                torch.from_numpy(mask[None]),
            )
            probs = torch.softmax(logits, -1)[0].numpy()
        return probs, mask

    def rank_placements(self, game: Game) -> list[ScoredPlacement]:
        probs, mask = self.probabilities(game)
        legal = np.flatnonzero(mask)
        order = legal[np.argsort(-probs[legal], kind="stable")]
        return [ScoredPlacement(Placement.from_index(int(a)), float(probs[a])) for a in order]
