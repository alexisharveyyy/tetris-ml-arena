"""Placement-level actions shared by the engine, environment and agents.

A :class:`Placement` is ``(rotation, column, use_hold)`` where ``column`` is
the board column of the piece's leftmost occupied cell. The action space has
``4 rotations x 10 columns x 2 hold options = 80`` actions.

Legality (v1): a placement is legal when the piece, created at its spawn
position, can be rotated in place to ``rotation`` and shifted horizontally at
spawn height to ``column`` without collision, then hard dropped. Tucks and
spins are out of scope. Live play uses :mod:`tetris.agents.planner` to find an
actual SRS move sequence to the same final position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tetris.engine.board import SPAWN_ROW, WIDTH, drop_row, fits, spawn_col
from tetris.engine.pieces import MIN_DC, Piece

NUM_ROTATIONS = 4
NUM_COLUMNS = WIDTH
NUM_ACTIONS = NUM_ROTATIONS * NUM_COLUMNS * 2


@dataclass(frozen=True, slots=True)
class Placement:
    """Target placement for the current piece (or the hold piece)."""

    rotation: int
    column: int
    use_hold: bool = False

    @property
    def index(self) -> int:
        """Index into the 80-action space."""
        return (NUM_ROTATIONS * NUM_COLUMNS if self.use_hold else 0) + (
            self.rotation * NUM_COLUMNS + self.column
        )

    @classmethod
    def from_index(cls, index: int) -> Placement:
        if not 0 <= index < NUM_ACTIONS:
            raise ValueError(f"action index out of range: {index}")
        use_hold, rest = divmod(index, NUM_ROTATIONS * NUM_COLUMNS)
        rotation, column = divmod(rest, NUM_COLUMNS)
        return cls(rotation, column, bool(use_hold))

    def to_dict(self) -> dict:
        return {"rotation": self.rotation, "column": self.column, "use_hold": self.use_hold}

    @classmethod
    def from_dict(cls, d: dict) -> Placement:
        return cls(int(d["rotation"]), int(d["column"]), bool(d.get("use_hold", False)))


@dataclass(frozen=True, slots=True)
class Target:
    """A resolved landing position for a piece."""

    piece: Piece
    rotation: int
    column: int  # leftmost occupied column
    box_col: int  # bounding-box column
    row: int  # bounding-box row after hard drop


def spawn_position(board: np.ndarray, piece: Piece) -> tuple[int, int] | None:
    """Spawn ``(row, box_col)`` for a piece, or None if blocked (block out).

    Pieces are created at :data:`SPAWN_ROW` and immediately drop one row if
    possible so the piece is visible.
    """
    col = spawn_col(piece)
    if not fits(board, piece, 0, SPAWN_ROW, col):
        return None
    if fits(board, piece, 0, SPAWN_ROW + 1, col):
        return SPAWN_ROW + 1, col
    return SPAWN_ROW, col


def legal_targets(board: np.ndarray, piece: Piece) -> list[Target]:
    """All legal hard-drop targets for ``piece`` on ``board`` (see module doc)."""
    spawn = spawn_position(board, piece)
    if spawn is None:
        return []
    row, start = spawn
    targets: list[Target] = []
    for rot in range(NUM_ROTATIONS):
        if not fits(board, piece, rot, row, start):
            continue
        lo = start
        while fits(board, piece, rot, row, lo - 1):
            lo -= 1
        hi = start
        while fits(board, piece, rot, row, hi + 1):
            hi += 1
        min_dc = MIN_DC[piece][rot]
        for box_col in range(lo, hi + 1):
            targets.append(
                Target(
                    piece, rot, box_col + min_dc, box_col, drop_row(board, piece, rot, row, box_col)
                )
            )
    return targets


def resolve_target(board: np.ndarray, piece: Piece, rotation: int, column: int) -> Target | None:
    """Resolve a rotation/column pair to a landing target, or None if illegal."""
    for t in legal_targets(board, piece):
        if t.rotation == rotation and t.column == column:
            return t
    return None
