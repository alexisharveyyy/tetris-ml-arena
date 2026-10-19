"""PPO agents: afterstate-scored (``ppo``) and raw-observation (``ppo_raw``) policies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tetris.agents.base import Agent, ScoredPlacement
from tetris.engine.game import Game
from tetris.engine.placement import Placement
from tetris.env.afterstates import action_mask
from tetris.rl.encoding import action_afterstate_arrays, observation_arrays
from tetris.rl.nets import load_checkpoint


class RawPPOAgent(Agent):
    """PPO with logits from the board observation alone (the originally specified design).

    Plays the most probable legal placement (deterministic at play time).
    Difficulty mistakes pick among the next five legal actions by policy probability.
    """

    key = "ppo_raw"
    name = "PPO (raw actions)"
    description = "PPO policy over the 80 actions from the board alone; weak, kept for comparison."
    kind = "ppo_raw"

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed)
        self.model: torch.nn.Module | None = None

    def load(self, checkpoint: str | Path) -> None:
        self.model, _ = load_checkpoint(checkpoint, self.kind, "cpu")

    def _inputs(self, game: Game) -> tuple[list[torch.Tensor], np.ndarray]:
        board, extras = observation_arrays(game)
        mask = action_mask(game)
        tensors = [
            torch.from_numpy(board[None].astype(np.float32)),
            torch.from_numpy(extras[None]),
            torch.from_numpy(mask[None]),
        ]
        return tensors, mask

    def probabilities(self, game: Game) -> tuple[np.ndarray, np.ndarray]:
        """Policy probabilities over the 80 actions and the legal mask."""
        if self.model is None:
            raise RuntimeError("PPO agent has no checkpoint loaded")
        tensors, mask = self._inputs(game)
        with torch.inference_mode():
            logits, _ = self.model(*tensors)
            probs = torch.softmax(logits, -1)[0].numpy()
        return probs, mask

    def rank_placements(self, game: Game) -> list[ScoredPlacement]:
        probs, mask = self.probabilities(game)
        legal = np.flatnonzero(mask)
        order = legal[np.argsort(-probs[legal], kind="stable")]
        return [ScoredPlacement(Placement.from_index(int(a)), float(probs[a])) for a in order]


class PPOAgent(RawPPOAgent):
    """PPO whose logit for each action scores that action's afterstate.

    The output is still a distribution over the same 80-action space (illegal
    actions masked), so the shared action-space rule holds.
    """

    key = "ppo"
    name = "PPO"
    description = "PPO policy that scores each placement's afterstate (masked 80-action space)."
    kind = "ppo"

    def _inputs(self, game: Game) -> tuple[list[torch.Tensor], np.ndarray]:
        board, extras = observation_arrays(game)
        cand_boards, cand_extras, mask = action_afterstate_arrays(game)
        tensors = [
            torch.from_numpy(board[None].astype(np.float32)),
            torch.from_numpy(extras[None]),
            torch.from_numpy(mask[None]),
            torch.from_numpy(cand_boards[None]),
            torch.from_numpy(cand_extras[None]),
        ]
        return tensors, mask
