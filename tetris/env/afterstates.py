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
from tetris.engine.board import HEIGHT, HIDDEN_ROWS
from tetris.engine.game import Game
from tetris.engine.pieces import CELLS, Piece
from tetris.engine.placement import (
    FULL_ROW,
    NUM_ACTIONS,
    BoardIndex,
    Placement,
    Target,
    board_rows,
    piece_row_masks,
)
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
    #: Exact score gained by this placement (hard-drop points plus line clear).
    score: int
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


def _cache_key(game: Game, include_hold: bool) -> tuple:
    a = game.active
    return (
        game.board.tobytes(),
        (a.piece, a.rot, a.row, a.col) if a else None,
        game.hold,
        game.can_hold,
        game.queue[0] if game.queue else None,
        include_hold,
    )


def legal_placements(game: Game, include_hold: bool = True) -> list[tuple[Placement, Target]]:
    """All legal placements for the game's current decision point.

    Results are cached on the game object, keyed by everything that affects
    legality, so repeated calls (mask, afterstates, step) cost one lookup.
    """
    if game.game_over or game.active is None:
        return []
    key = _cache_key(game, include_hold)
    cached = getattr(game, "_placement_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    index = BoardIndex(game.board)
    a = game.active
    out = [
        (Placement(t.rotation, t.column, False), t)
        for t in index.targets(a.piece, (a.rot, a.row, a.col))
    ]
    if include_hold and game.can_hold:
        piece, _ = _hold_piece_and_next(game)
        out.extend((Placement(t.rotation, t.column, True), t) for t in index.targets(piece))
    game._placement_cache = (key, out)
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
    n_cand = len(candidates)
    boards = np.empty((n_cand, HEIGHT, game.board.shape[1]), dtype=np.int8)
    lines = np.zeros(n_cand, dtype=np.int32)
    landing = np.zeros(n_cand, dtype=np.float32)
    eroded = np.zeros(n_cand, dtype=np.int32)
    top_out = np.zeros(n_cand, dtype=bool)
    attacks = np.zeros(n_cand, dtype=np.int32)
    scores = np.zeros(n_cand, dtype=np.int64)
    index = BoardIndex(game.board)
    start_rows: dict[tuple[bool, Piece], int] = {}

    rows0 = board_rows(game.board)
    next_plain = game.queue[0]
    _, next_hold = _hold_piece_and_next(game)

    for i, (placement, t) in enumerate(candidates):
        rows = rows0[:]
        touched = []
        for dr, mask in piece_row_masks(t.piece, t.rotation):
            r = t.row + dr
            rows[r] |= mask << t.box_col if t.box_col >= 0 else mask >> -t.box_col
            touched.append(r)
        min_r, max_r = touched[0], touched[-1]
        landing[i] = HEIGHT - (min_r + max_r) / 2.0
        lock_out = max_r < HIDDEN_ROWS

        board = boards[i]
        board[:] = game.board
        for dr, dc in CELLS[t.piece][t.rotation]:
            board[t.row + dr, t.box_col + dc] = int(t.piece)
        full = [r for r in touched if rows[r] == FULL_ROW]
        if full:
            n = len(full)
            kept = [r for r in range(HEIGHT) if r not in full]
            board[n:] = board[kept]
            board[:n] = 0
            rows = [0] * n + [rows[r] for r in kept]
            lines[i] = n
            eroded[i] = n * sum(1 for dr, dc in CELLS[t.piece][t.rotation] if t.row + dr in full)
            attacks[i] = scoring.attack_for(n, game.combo + 1, n == 4 and game.b2b)
        # Hard-drop points: 2 per row from where the piece starts to where it lands.
        key = (placement.use_hold, t.piece)
        if key not in start_rows:
            if placement.use_hold:
                spawn = index.spawn(t.piece)
                start_rows[key] = spawn[0] if spawn else t.row
            else:
                start_rows[key] = game.active.row
        n_lines = int(lines[i])
        scores[i] = 2 * max(0, t.row - start_rows[key]) + scoring.score_for(
            n_lines, game.level, game.combo + 1 if n_lines else 0, n_lines == 4 and game.b2b
        )
        nxt = next_hold if placement.use_hold else next_plain
        top_out[i] = lock_out or BoardIndex(rows=rows).spawn(nxt) is None

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
            score=int(scores[i]),
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
