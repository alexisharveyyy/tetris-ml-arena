"""Enumeration of legal placements and their afterstates.

Every agent sees the same 80-action placement space (see
:mod:`tetris.engine.placement`). This module turns a :class:`Game` into the
list of legal placements, an action mask, and fully evaluated afterstates
(board after line clears plus features).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tetris.engine import scoring
from tetris.engine.board import HEIGHT, HIDDEN_ROWS, clear_full_rows
from tetris.engine.game import Game
from tetris.engine.pieces import CELLS, Piece
from tetris.engine.placement import NUM_ACTIONS, Placement, Target, legal_targets, spawn_position
from tetris.env.features import full_features


@dataclass
class Afterstate:
    """Result of putting the current (or hold) piece at a placement."""

    placement: Placement
    target: Target
    board: np.ndarray  # int8 (HEIGHT, WIDTH), after line clears
    lines_cleared: int
    landing_height: float
    eroded_cells: int
    attack: int
    top_out: bool
    features: np.ndarray  # float32 (NUM_FEATURES,)

    @property
    def piece(self) -> Piece:
        return self.target.piece


def _hold_piece_and_next(game: Game) -> tuple[Piece, Piece]:
    """Piece placed when holding, and the piece that spawns after it."""
    if game.hold is not None:
        return game.hold, game.queue[0]
    return game.queue[0], game.queue[1]


def legal_placements(game: Game, include_hold: bool = True) -> list[tuple[Placement, Target]]:
    """All legal placements for the game's current decision point."""
    if game.game_over or game.active is None:
        return []
    out = [
        (Placement(t.rotation, t.column, False), t)
        for t in legal_targets(game.board, game.active.piece)
    ]
    if include_hold and game.can_hold:
        piece, _ = _hold_piece_and_next(game)
        out.extend(
            (Placement(t.rotation, t.column, True), t) for t in legal_targets(game.board, piece)
        )
    return out


def action_mask(game: Game, include_hold: bool = True) -> np.ndarray:
    """Boolean mask over the 80-action space."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for p, _ in legal_placements(game, include_hold):
        mask[p.index] = True
    return mask


def enumerate_afterstates(
    game: Game, include_hold: bool = True, dedupe: bool = False
) -> list[Afterstate]:
    """Evaluate every legal placement.

    With ``dedupe=True``, placements that yield an identical board (for example
    the two horizontal orientations of an I piece) are reduced to one, keeping
    the non-hold variant first.
    """
    candidates = legal_placements(game, include_hold)
    if not candidates:
        return []
    boards = np.empty((len(candidates), HEIGHT, game.board.shape[1]), dtype=np.int8)
    lines = np.zeros(len(candidates), dtype=np.int32)
    landing = np.zeros(len(candidates), dtype=np.float32)
    eroded = np.zeros(len(candidates), dtype=np.int32)
    top_out = np.zeros(len(candidates), dtype=bool)
    attacks = np.zeros(len(candidates), dtype=np.int32)

    next_plain = game.queue[0]
    _, next_hold = _hold_piece_and_next(game)

    for i, (placement, t) in enumerate(candidates):
        board = game.board.copy()
        rows = []
        for dr, dc in CELLS[t.piece][t.rotation]:
            r = t.row + dr
            board[r, t.box_col + dc] = int(t.piece)
            rows.append(r)
        min_r, max_r = min(rows), max(rows)
        landing[i] = HEIGHT - (min_r + max_r) / 2.0
        lock_out = max_r < HIDDEN_ROWS
        board, cleared = clear_full_rows(board)
        n = len(cleared)
        lines[i] = n
        if n:
            eroded[i] = n * sum(1 for r in rows if r in cleared)
            combo = game.combo + 1
            b2b = n == 4 and game.b2b
            attacks[i] = scoring.attack_for(n, combo, b2b)
        boards[i] = board
        nxt = next_hold if placement.use_hold else next_plain
        top_out[i] = lock_out or spawn_position(board, nxt) is None

    feats = full_features(boards != 0, lines, landing, eroded)
    result = [
        Afterstate(
            placement=p,
            target=t,
            board=boards[i],
            lines_cleared=int(lines[i]),
            landing_height=float(landing[i]),
            eroded_cells=int(eroded[i]),
            attack=int(attacks[i]),
            top_out=bool(top_out[i]),
            features=feats[i],
        )
        for i, (p, t) in enumerate(candidates)
    ]
    if dedupe:
        seen: set[tuple[bool, bytes]] = set()
        unique = []
        for a in result:
            key = (a.placement.use_hold, a.board.tobytes())
            if key not in seen:
                seen.add(key)
                unique.append(a)
        result = unique
    return result
