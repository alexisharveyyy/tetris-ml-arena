"""Placement-level actions shared by the engine, environment and agents.

A :class:`Placement` is ``(rotation, column, use_hold)`` where ``column`` is
the board column of the piece's leftmost occupied cell. The action space has
``4 rotations x 10 columns x 2 hold options = 80`` actions.

Legality (v1): a placement is legal when the piece, from its current position
(the spawn position for a fresh or held piece), can reach ``rotation`` by
in-place rotations (every intermediate orientation must
fit without a kick, so the engine's first SRS test succeeds), then shift
horizontally at spawn height to ``column`` and hard drop. This is exactly the
move sequence :func:`tetris.agents.planner.plan_path` falls back on, so every
legal placement is reachable through real engine moves (verified by tests).
Tucks and spins are out of scope.

Collision checks here use per-row bitmasks and a precomputed "next filled
cell below" table, which is much faster than cell-by-cell NumPy indexing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tetris.engine.board import HEIGHT, SPAWN_ROW, WIDTH, spawn_col
from tetris.engine.pieces import CELLS, MAX_DC, MIN_DC, Piece

NUM_ROTATIONS = 4
NUM_COLUMNS = WIDTH
NUM_ACTIONS = NUM_ROTATIONS * NUM_COLUMNS * 2

_COL_BITS = np.array([1 << c for c in range(WIDTH)], dtype=np.int64)

# Per piece/rotation: list of (row_offset, row_bitmask at box column 0).
_ROW_MASKS: dict[Piece, tuple[tuple[tuple[int, int], ...], ...]] = {}
# Per piece/rotation: list of (col_offset, lowest row offset in that column).
_BOTTOMS: dict[Piece, tuple[tuple[tuple[int, int], ...], ...]] = {}
for _p in Piece:
    masks, bottoms = [], []
    for _r in range(4):
        rows: dict[int, int] = {}
        low: dict[int, int] = {}
        for dr, dc in CELLS[_p][_r]:
            rows[dr] = rows.get(dr, 0) | (1 << dc)
            low[dc] = max(low.get(dc, -1), dr)
        masks.append(tuple(sorted(rows.items())))
        bottoms.append(tuple(sorted(low.items())))
    _ROW_MASKS[_p] = tuple(masks)
    _BOTTOMS[_p] = tuple(bottoms)

#: Starting state for enumeration: (rotation, box row, box column).
Start = tuple[int, int, int]


def _rotation_paths(start_rot: int, rot: int) -> tuple[tuple[int, ...], ...]:
    """In-place rotation sequences from ``start_rot`` to ``rot`` (each state must fit)."""
    d = (rot - start_rot) % 4
    cw1, cw2 = (start_rot + 1) % 4, (start_rot + 2) % 4
    ccw1 = (start_rot - 1) % 4
    if d == 0:
        return ((),)
    if d == 1:
        return ((cw1,),)
    if d == 3:
        return ((ccw1,),)
    return ((cw1, cw2), (ccw1, cw2))


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


FULL_ROW = (1 << WIDTH) - 1


def board_rows(board: np.ndarray) -> list[int]:
    """Occupancy of each row as a bitmask (bit c set when column c is filled)."""
    return ((board != 0).astype(np.int64) @ _COL_BITS).tolist()


def piece_row_masks(piece: Piece, rot: int) -> tuple[tuple[int, int], ...]:
    """``(row_offset, bitmask)`` pairs for a piece at box column 0."""
    return _ROW_MASKS[piece][rot]


class BoardIndex:
    """Fast collision and landing queries for one board."""

    __slots__ = ("rows", "_next_filled")

    def __init__(self, board: np.ndarray | None = None, rows: list[int] | None = None) -> None:
        if rows is None:
            assert board is not None
            rows = board_rows(board)
        self.rows = rows
        self._next_filled: list[list[int]] | None = None

    @property
    def next_filled(self) -> list[list[int]]:
        """``next_filled[r][c]``: first filled row >= r in column c (HEIGHT if none)."""
        if self._next_filled is None:
            nf = [[HEIGHT] * WIDTH for _ in range(HEIGHT + 1)]
            below = [HEIGHT] * WIDTH
            for r in range(HEIGHT - 1, -1, -1):
                bits = self.rows[r]
                if bits:
                    below = [r if bits >> c & 1 else below[c] for c in range(WIDTH)]
                nf[r] = below
            self._next_filled = nf
        return self._next_filled

    def fits(self, piece: Piece, rot: int, row: int, col: int) -> bool:
        if col + MIN_DC[piece][rot] < 0 or col + MAX_DC[piece][rot] >= WIDTH:
            return False
        rows = self.rows
        for dr, mask in _ROW_MASKS[piece][rot]:
            r = row + dr
            if r < 0 or r >= HEIGHT:
                return False
            if rows[r] & (mask << col if col >= 0 else mask >> -col):
                return False
        return True

    def landing_row(self, piece: Piece, rot: int, row: int, col: int) -> int:
        """Box row after a hard drop from a fitting position."""
        nf = self.next_filled
        best = HEIGHT
        for dc, low in _BOTTOMS[piece][rot]:
            land = nf[row + low + 1][col + dc] - 1 - low
            if land < best:
                best = land
        return best

    def spawn(self, piece: Piece) -> tuple[int, int] | None:
        col = spawn_col(piece)
        if not self.fits(piece, 0, SPAWN_ROW, col):
            return None
        if self.fits(piece, 0, SPAWN_ROW + 1, col):
            return SPAWN_ROW + 1, col
        return SPAWN_ROW, col

    def rotation_reachable(
        self, piece: Piece, start_rot: int, rot: int, row: int, col: int
    ) -> bool:
        """Can ``rot`` be reached from ``start_rot`` by kick-free in-place turns?"""
        for path in _rotation_paths(start_rot, rot):
            if all(self.fits(piece, r, row, col) for r in path):
                return True
        return False

    def _start(self, piece: Piece, start: Start | None) -> Start | None:
        if start is not None:
            return start
        spawn = self.spawn(piece)
        return None if spawn is None else (0, spawn[0], spawn[1])

    def targets(self, piece: Piece, start: Start | None = None) -> list[Target]:
        """Legal targets from ``start`` = (rot, row, box_col); defaults to the spawn position."""
        origin = self._start(piece, start)
        if origin is None:
            return []
        start_rot, row, start_col = origin
        out: list[Target] = []
        for rot in range(NUM_ROTATIONS):
            if not self.rotation_reachable(piece, start_rot, rot, row, start_col):
                continue
            lo = start_col
            while self.fits(piece, rot, row, lo - 1):
                lo -= 1
            hi = start_col
            while self.fits(piece, rot, row, hi + 1):
                hi += 1
            min_dc = MIN_DC[piece][rot]
            for box_col in range(lo, hi + 1):
                out.append(
                    Target(
                        piece,
                        rot,
                        box_col + min_dc,
                        box_col,
                        self.landing_row(piece, rot, row, box_col),
                    )
                )
        return out

    def resolve(
        self, piece: Piece, rotation: int, column: int, start: Start | None = None
    ) -> Target | None:
        """Landing target for one rotation/column, or None if illegal (see :meth:`targets`)."""
        origin = self._start(piece, start)
        if origin is None or not 0 <= rotation < NUM_ROTATIONS:
            return None
        start_rot, row, start_col = origin
        if not self.rotation_reachable(piece, start_rot, rotation, row, start_col):
            return None
        box_col = column - MIN_DC[piece][rotation]
        step = 1 if box_col >= start_col else -1
        for c in range(start_col, box_col + step, step):
            if not self.fits(piece, rotation, row, c):
                return None
        return Target(
            piece, rotation, column, box_col, self.landing_row(piece, rotation, row, box_col)
        )


def spawn_position(board: np.ndarray, piece: Piece) -> tuple[int, int] | None:
    """Spawn ``(row, box_col)`` for a piece, or None if blocked (block out).

    Pieces are created at :data:`SPAWN_ROW` and immediately drop one row if
    possible so the piece is visible.
    """
    return BoardIndex(board).spawn(piece)


def legal_targets(board: np.ndarray, piece: Piece) -> list[Target]:
    """All legal hard-drop targets for ``piece`` on ``board`` (see module doc)."""
    return BoardIndex(board).targets(piece)


def resolve_target(board: np.ndarray, piece: Piece, rotation: int, column: int) -> Target | None:
    """Resolve a rotation/column pair to a landing target, or None if illegal."""
    return BoardIndex(board).resolve(piece, rotation, column)
