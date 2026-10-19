"""DQN agent: values every afterstate with a CNN and picks the best placement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tetris.agents.base import Agent, ScoredPlacement
from tetris.engine.game import Game
from tetris.env.afterstates import enumerate_afterstates
from tetris.rl.encoding import afterstate_arrays, unpack_boards
from tetris.rl.nets import load_checkpoint


class DQNAgent(Agent):
    """Ranks placements by ``r(s, a) + gamma * V(afterstate)`` (V = 0 after a top-out).

    Difficulty mistakes pick among the next five placements by this value.
    """

    key = "dqn"
    name = "DQN (afterstates)"
    description = "Double DQN with prioritized replay; a CNN values each afterstate."

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed)
        self.model: torch.nn.Module | None = None
        self.gamma = 0.99

    def load(self, checkpoint: str | Path) -> None:
        self.model, data = load_checkpoint(checkpoint, "dqn", "cpu")
        self.gamma = float(data.get("gamma", 0.99))

    def values(self, game: Game):
        """Afterstates and their action values."""
        if self.model is None:
            raise RuntimeError("DQN agent has no checkpoint loaded")
        afs = enumerate_afterstates(game, dedupe=True)
        if not afs:
            return afs, np.zeros(0, dtype=np.float32)
        boards, extras, rewards, tops, _ = afterstate_arrays(game, afs)
        with torch.inference_mode():
            v = self.model(
                torch.from_numpy(unpack_boards(boards)), torch.from_numpy(extras)
            ).numpy()
        return afs, rewards + self.gamma * v * (~tops)

    def rank_placements(self, game: Game) -> list[ScoredPlacement]:
        afs, q = self.values(game)
        order = np.argsort(-q, kind="stable")
        return [ScoredPlacement(afs[i].placement, float(q[i])) for i in order]
