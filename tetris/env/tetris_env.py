"""Gymnasium environment with placement-level actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tetris.engine.board import HEIGHT, WIDTH, board_to_strings
from tetris.engine.game import Game, GameConfig, IllegalPlacementError
from tetris.engine.placement import NUM_ACTIONS, Placement
from tetris.env.afterstates import action_mask, enumerate_afterstates
from tetris.env.features import BOARD_FEATURES, board_features
from tetris.env.rewards import RewardConfig

QUEUE_OBS = 5


def inject_scheduled_garbage(game: Game, every: int | None, rows: int) -> None:
    """Evaluation pressure: add ``rows`` garbage rows after every ``every`` pieces."""
    if every and not game.game_over and game.pieces_placed % every == 0:
        game.receive_garbage(rows)


def encode_observation(game: Game) -> dict[str, np.ndarray]:
    """Observation dict for a game (also used by agents outside the env)."""
    queue = np.array([int(p) - 1 for p in game.queue[:QUEUE_OBS]], dtype=np.int64)
    return {
        "board": (game.board != 0).astype(np.int8),
        "piece": np.int64(int(game.active.piece) - 1 if game.active else 0),
        "hold": np.int64(int(game.hold) if game.hold else 0),
        "can_hold": np.int64(int(game.can_hold)),
        "queue": queue,
        "features": board_features(game.board != 0),
    }


class TetrisEnv(gym.Env):
    """Tetris with an 80-way placement action space.

    Observations: ``board`` (HxW occupancy), ``piece`` (0-6), ``hold`` (0 none,
    1-7 piece), ``can_hold``, ``queue`` (next 5 pieces, 0-6) and ``features``
    (board feature vector, see :data:`BOARD_FEATURES`).

    ``info["action_mask"]`` holds the legal-action mask after every reset and
    step; :meth:`action_masks` returns the same for masking libraries.

    Optional evaluation pressure: with ``garbage_every=N``, ``garbage_rows``
    rows of garbage are inserted after every N placed pieces.

    Illegal actions end the episode with the game-over penalty.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        reward: RewardConfig | dict[str, Any] | str | Path | None = None,
        max_pieces: int | None = None,
        include_hold: bool = True,
        garbage_every: int | None = None,
        garbage_rows: int = 1,
        game_config: GameConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        if isinstance(reward, (str, Path)):
            reward = RewardConfig.from_yaml(reward)
        elif isinstance(reward, dict):
            reward = RewardConfig.from_dict(reward)
        self.reward_config = reward or RewardConfig()
        self.max_pieces = max_pieces
        self.include_hold = include_hold
        self.garbage_every = garbage_every
        self.garbage_rows = garbage_rows
        self.game_config = game_config
        self.render_mode = render_mode
        self.game: Game | None = None

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(0, 1, (HEIGHT, WIDTH), dtype=np.int8),
                "piece": spaces.Discrete(7),
                "hold": spaces.Discrete(8),
                "can_hold": spaces.Discrete(2),
                "queue": spaces.MultiDiscrete([7] * QUEUE_OBS),
                "features": spaces.Box(0.0, np.inf, (len(BOARD_FEATURES),), dtype=np.float32),
            }
        )

    # ------------------------------------------------------------------ API

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        game_seed = seed if seed is not None else int(self.np_random.integers(0, 2**63 - 1))
        self.game = Game(game_seed, self.game_config)
        self._obs = encode_observation(self.game)
        return self._obs, self._info()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        game = self._require_game()
        placement = Placement.from_index(int(action))
        feats_before = self._obs["features"]
        score_before = game.score
        if placement.use_hold and not self.include_hold:
            return self._illegal(placement)
        try:
            result = game.place(placement)
        except IllegalPlacementError:
            return self._illegal(placement)

        inject_scheduled_garbage(game, self.garbage_every, self.garbage_rows)

        terminated = game.game_over or not self._mask().any()
        if terminated and not game.game_over:
            game.game_over = True
        truncated = bool(
            not terminated and self.max_pieces is not None and game.pieces_placed >= self.max_pieces
        )
        self._obs = encode_observation(game)
        reward = self.reward_config.compute(
            result.lines,
            game.score - score_before,
            result.attack,
            feats_before,
            self._obs["features"],
            terminated,
        )
        info = self._info()
        info["lines_cleared"] = result.lines
        info["attack"] = result.attack
        return self._obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Legal-action mask (for maskable policy libraries)."""
        return self._mask()

    def afterstates(self, dedupe: bool = False):
        """Evaluated afterstates for the current decision point."""
        return enumerate_afterstates(self._require_game(), self.include_hold, dedupe)

    def render(self) -> str | None:
        if self.render_mode != "ansi" or self.game is None:
            return None
        rows = board_to_strings(self.game.board)
        a = self.game.active
        header = (
            f"piece={a.piece.letter if a else '-'} lines={self.game.lines} score={self.game.score}"
        )
        return "\n".join([header, *rows])

    # -------------------------------------------------------------- helpers

    def _require_game(self) -> Game:
        if self.game is None:
            raise RuntimeError("call reset() first")
        return self.game

    def _mask(self) -> np.ndarray:
        return action_mask(self._require_game(), self.include_hold)

    def _info(self) -> dict[str, Any]:
        g = self._require_game()
        return {
            "action_mask": self._mask(),
            "score": g.score,
            "lines": g.lines,
            "pieces": g.pieces_placed,
            "attack_sent": g.attack_sent,
        }

    def _illegal(self, placement: Placement):
        game = self._require_game()
        game.game_over = True
        info = self._info()
        info["illegal_action"] = placement.index
        return self._obs, float(self.reward_config.game_over), True, False, info
