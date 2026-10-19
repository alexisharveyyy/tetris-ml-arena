"""Tensor encodings and the shared reward for the learned agents.

NumPy only, so environment worker processes never import torch.

Reward (DQN and PPO): each placed piece earns ``1 + 0.25 * score_delta / 1000``.
This is the per-step form of the evaluation fitness
``mean(pieces / cap) + 0.25 * mean(score_per_piece) / 1000`` multiplied by the
cap: survival counts 1 per piece, and score adds the same 0.25-per-1000 bonus
the fitness gives. Topping out ends the stream of rewards; there is no extra
penalty, so the learned objective matches the one the leaderboard measures.
"""

from __future__ import annotations

import numpy as np

from tetris.engine.board import HEIGHT, WIDTH
from tetris.engine.game import Game
from tetris.engine.pieces import Piece
from tetris.env.afterstates import Afterstate
from tetris.env.features import BOARD_FEATURES, FEATURE_NAMES, board_features

REWARD_SURVIVAL = 1.0
REWARD_SCORE_SCALE = 0.25 / 1000.0

BOARD_SHAPE = (HEIGHT, WIDTH)
PACKED_BOARD_BYTES = HEIGHT * WIDTH // 8  # 240 cells -> 30 bytes

#: Fixed scales that bring every feature to roughly [0, 1].
_FEATURE_SCALES = {
    "aggregate_height": 200.0,
    "max_height": 24.0,
    "holes": 40.0,
    "rows_with_holes": 20.0,
    "hole_depth": 100.0,
    "bumpiness": 60.0,
    "wells": 60.0,
    "max_well_depth": 20.0,
    "row_transitions": 120.0,
    "col_transitions": 120.0,
    "lines_cleared": 4.0,
    "landing_height": 24.0,
    "eroded_cells": 16.0,
}
FEATURE_SCALE = np.array([_FEATURE_SCALES[n] for n in FEATURE_NAMES], dtype=np.float32)
BOARD_FEATURE_SCALE = np.array([_FEATURE_SCALES[n] for n in BOARD_FEATURES], dtype=np.float32)

#: Afterstate extras: 13 features, next piece (7), hold after placement (8: none + 7), level.
AFTERSTATE_EXTRA_DIM = len(FEATURE_NAMES) + 7 + 8 + 1
#: PPO extras: current piece (7), hold (8), can_hold (1), next 5 pieces (35),
#: 10 board features, level.
OBS_EXTRA_DIM = 7 + 8 + 1 + 35 + len(BOARD_FEATURES) + 1
QUEUE_LEN = 5


def reward_for_score(score_delta: np.ndarray | float) -> np.ndarray | float:
    """Per-placement reward from the score gained by that placement."""
    return REWARD_SURVIVAL + REWARD_SCORE_SCALE * score_delta


def pack_boards(filled: np.ndarray) -> np.ndarray:
    """Bool ``(..., 24, 10)`` boards to ``(..., 30)`` uint8."""
    flat = np.asarray(filled, dtype=bool).reshape(*filled.shape[:-2], HEIGHT * WIDTH)
    return np.packbits(flat, axis=-1)


def unpack_boards(packed: np.ndarray) -> np.ndarray:
    """``(..., 30)`` uint8 to float32 ``(..., 24, 10)`` occupancy."""
    bits = np.unpackbits(packed, axis=-1, count=HEIGHT * WIDTH)
    return bits.reshape(*packed.shape[:-1], HEIGHT, WIDTH).astype(np.float32)


def _level_after(game: Game, lines: np.ndarray) -> np.ndarray:
    cfg = game.config
    lvl = cfg.start_level + (game.lines + lines) // cfg.lines_per_level
    return np.minimum(cfg.max_level, np.maximum(game.level, lvl))


def afterstate_arrays(
    game: Game, afterstates: list[Afterstate]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Network inputs for a list of afterstates of ``game``.

    Returns packed boards ``(n, 30)``, extras ``(n, 29)``, rewards ``(n,)``,
    top-out flags ``(n,)`` and 80-space action indices ``(n,)``.

    The extras encode the state *after* the placement: the piece that spawns
    next and the hold slot, so a hold placement is represented correctly.
    """
    n = len(afterstates)
    boards = np.stack([a.board != 0 for a in afterstates])
    extras = np.zeros((n, AFTERSTATE_EXTRA_DIM), dtype=np.float32)
    feats = np.stack([a.features for a in afterstates]) / FEATURE_SCALE
    nf = len(FEATURE_NAMES)
    extras[:, :nf] = feats
    current = game.active.piece
    lines = np.array([a.lines_cleared for a in afterstates])
    for i, a in enumerate(afterstates):
        if a.placement.use_hold:
            next_piece = game.queue[1] if game.hold is None else game.queue[0]
            hold_after: Piece | None = current
        else:
            next_piece = game.queue[0]
            hold_after = game.hold
        extras[i, nf + int(next_piece) - 1] = 1.0
        extras[i, nf + 7 + (int(hold_after) if hold_after else 0)] = 1.0
    extras[:, -1] = _level_after(game, lines) / 20.0
    rewards = reward_for_score(np.array([a.score for a in afterstates], dtype=np.float32))
    tops = np.array([a.top_out for a in afterstates], dtype=bool)
    actions = np.array([a.placement.index for a in afterstates], dtype=np.int16)
    return pack_boards(boards), extras, rewards.astype(np.float32), tops, actions


def observation_arrays(game: Game) -> tuple[np.ndarray, np.ndarray]:
    """PPO observation: board occupancy ``(24, 10)`` uint8 and extras ``(62,)``."""
    board = (game.board != 0).astype(np.uint8)
    extras = np.zeros(OBS_EXTRA_DIM, dtype=np.float32)
    o = 0
    if game.active is not None:
        extras[o + int(game.active.piece) - 1] = 1.0
    o += 7
    extras[o + (int(game.hold) if game.hold else 0)] = 1.0
    o += 8
    extras[o] = float(game.can_hold)
    o += 1
    for k, piece in enumerate(game.queue[:QUEUE_LEN]):
        extras[o + 7 * k + int(piece) - 1] = 1.0
    o += 7 * QUEUE_LEN
    extras[o : o + len(BOARD_FEATURES)] = board_features(game.board != 0) / BOARD_FEATURE_SCALE
    o += len(BOARD_FEATURES)
    extras[o] = game.level / 20.0
    return board, extras


def action_afterstate_arrays(game: Game) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Afterstate inputs laid out over the full 80-action space.

    Returns packed boards ``(80, 30)``, extras ``(80, 29)`` and the legal mask
    ``(80,)``. Row ``a`` describes the afterstate of action ``a``; illegal rows
    are zero. No deduplication, so every legal action has its own row.
    """
    from tetris.engine.placement import NUM_ACTIONS
    from tetris.env.afterstates import enumerate_afterstates

    boards = np.zeros((NUM_ACTIONS, PACKED_BOARD_BYTES), np.uint8)
    extras = np.zeros((NUM_ACTIONS, AFTERSTATE_EXTRA_DIM), np.float32)
    mask = np.zeros(NUM_ACTIONS, bool)
    afs = enumerate_afterstates(game, dedupe=False)
    if afs:
        b, e, _, _, actions = afterstate_arrays(game, afs)
        idx = actions.astype(np.int64)
        boards[idx], extras[idx], mask[idx] = b, e, True
    return boards, extras, mask
