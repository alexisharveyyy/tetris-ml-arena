"""Decision Transformer agent: plays toward a target return.

The agent keeps the history of the game it is playing (the last ``context``
pieces: return-to-go, state and the chosen afterstate) and lowers the
return-to-go by each move's reward, which it reads from the score change
between decisions.

* **Target return:** ``DifficultyParams.target_return`` (fitness units),
  else the checkpoint's default (the calibrated best). Changing it mid-game
  takes effect at the next piece: the return-to-go becomes the new target
  minus the return already earned.
* **Segments:** the model was trained on 500-piece games. Live games have no
  cap, so every 500 pieces the agent starts a new segment: history cleared,
  piece counter and return-to-go reset.
* **New games** are detected when the piece count does not advance. A gap of
  several pieces (another policy moved in between, such as the worker
  fallback) is credited as one step with the combined reward.
* The return-to-go is floored at one piece's survival reward, so playing
  above the target never feeds the model a negative return it never saw.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tetris.agents.base import Agent, ScoredPlacement
from tetris.agents.decision_transformer.model import DecisionTransformer
from tetris.engine.game import Game
from tetris.engine.placement import Placement
from tetris.rl.encoding import (
    REWARD_SCORE_SCALE,
    REWARD_SURVIVAL,
    action_afterstate_arrays,
    observation_arrays,
    pack_boards,
)
from tetris.rl.nets import load_checkpoint


@dataclass
class _Step:
    rtg: float
    obs_board: np.ndarray  # packed (30,)
    obs_extras: np.ndarray  # (62,)
    timestep: int
    actions: np.ndarray  # legal 80-space actions of this state
    emb: torch.Tensor  # (n_legal, d) afterstate embeddings
    chosen: torch.Tensor | None = None  # (d,) embedding of the chosen afterstate


class DTAgent(Agent):
    key = "dt"
    name = "Decision Transformer"
    description = "Return-conditioned transformer trained on the other agents' games."

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed)
        self.model: DecisionTransformer | None = None
        self.default_target = 1.0
        self.cap = 500
        self.reset()

    def load(self, checkpoint: str | Path) -> None:
        self.model, meta = load_checkpoint(checkpoint, "dt", "cpu")
        self.default_target = float(meta.get("default_target", 1.0))
        self.cap = int(meta.get("return_cap", 500))
        self.reset()

    # ------------------------------------------------------------ history

    def reset(self) -> None:
        context = self.model.context if self.model is not None else 1
        self.history: deque[_Step] = deque(maxlen=max(1, context - 1))
        self.current: _Step | None = None
        self.last_pieces = -1
        self.last_score = 0
        self.segment_start = 0
        self.earned = 0.0  # return earned in the current segment

    @property
    def target(self) -> float:
        t = self.difficulty.target_return
        return float(t) if t is not None else self.default_target

    def _sync(self, game: Game) -> None:
        """Account for the moves since the last decision (or start a new game)."""
        pieces = game.pieces_placed
        if self.current is None or pieces <= self.last_pieces:
            self.reset()
            self.segment_start = pieces
        else:
            moved = pieces - self.last_pieces
            reward = (
                REWARD_SURVIVAL * moved + REWARD_SCORE_SCALE * (game.score - self.last_score)
            ) / self.cap
            done = self.current
            if done.chosen is None:  # ranked but never chosen: assume the top choice
                done.chosen = done.emb[0]
            self.history.append(done)
            self.earned += reward
            if pieces - self.segment_start >= self.cap:
                self.history.clear()
                self.segment_start = pieces
                self.earned = 0.0
        self.last_pieces = pieces
        self.last_score = game.score

    # ----------------------------------------------------------- decisions

    @torch.inference_mode()
    def rank_placements(self, game: Game) -> list[ScoredPlacement]:
        if self.model is None:
            raise RuntimeError("Decision Transformer has no checkpoint loaded")
        self._sync(game)
        model = self.model
        cand_boards, cand_extras, mask = action_afterstate_arrays(game)
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            self.current = None
            return []
        emb = model.embed_afterstates(
            torch.from_numpy(cand_boards[legal]), torch.from_numpy(cand_extras[legal])
        )
        board, extras = observation_arrays(game)
        rtg = max(self.target - self.earned, REWARD_SURVIVAL / self.cap)
        step = _Step(
            rtg,
            pack_boards(board.astype(bool)),
            extras,
            game.pieces_placed - self.segment_start,
            legal,
            emb,
        )
        steps = [*self.history, step]
        k = model.context
        pad = k - len(steps)
        d = model.d_model
        rtgs = torch.zeros(1, k)
        obs_b = torch.zeros(1, k, step.obs_board.shape[0], dtype=torch.uint8)
        obs_e = torch.zeros(1, k, step.obs_extras.shape[0])
        acts = torch.zeros(1, k, d)
        ts = torch.zeros(1, k, dtype=torch.long)
        valid = torch.zeros(1, k, dtype=torch.bool)
        for i, s in enumerate(steps):
            j = pad + i
            rtgs[0, j] = s.rtg
            obs_b[0, j] = torch.from_numpy(s.obs_board)
            obs_e[0, j] = torch.from_numpy(s.obs_extras)
            ts[0, j] = s.timestep
            valid[0, j] = True
            if s.chosen is not None:
                acts[0, j] = s.chosen
        query = model(rtgs, obs_b, obs_e, acts, ts, valid)[0, -1]
        scores = model.action_logits(query.expand_as(emb), emb).numpy()
        order = np.argsort(-scores, kind="stable")
        step.actions, step.emb = legal[order], emb[torch.from_numpy(order.copy())]
        self.current = step
        return [
            ScoredPlacement(Placement.from_index(int(legal[i])), float(scores[i])) for i in order
        ]

    @torch.inference_mode()
    def _choose(self, game: Game, ranked: list[ScoredPlacement]) -> Placement:
        placement = super()._choose(game, ranked)
        cur = self.current
        if cur is not None:
            hit = np.flatnonzero(cur.actions == placement.index)
            if len(hit):
                cur.chosen = cur.emb[int(hit[0])]
            else:  # a random (epsilon) placement outside the ranked list
                af, ex, _ = action_afterstate_arrays(game)
                cur.chosen = self.model.embed_afterstates(  # type: ignore[union-attr]
                    torch.from_numpy(af[placement.index : placement.index + 1]),
                    torch.from_numpy(ex[placement.index : placement.index + 1]),
                )[0]
        return placement
