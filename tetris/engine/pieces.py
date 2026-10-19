"""Tetromino definitions, SRS rotation states and wall kick tables.

Coordinates use ``(row, col)`` with rows increasing downward. Rotation states
are 0 (spawn), 1 (R, clockwise), 2 (180) and 3 (L, counterclockwise). SRS
rotation is exactly a rotation of the piece's bounding box (3x3 for JLSTZ, 4x4
for I, 2x2 for O), so the states are generated from the spawn matrices.
"""

from __future__ import annotations

from enum import IntEnum


class Piece(IntEnum):
    """Piece kinds. Values double as board cell codes (0 is empty, 8 is garbage)."""

    I = 1  # noqa: E741
    O = 2  # noqa: E741
    T = 3
    S = 4
    Z = 5
    J = 6
    L = 7

    @property
    def letter(self) -> str:
        return self.name

    @classmethod
    def from_letter(cls, letter: str) -> Piece:
        return cls[letter]


GARBAGE = 8
CELL_CHARS = ".IOTSZJLG"
ALL_PIECES: tuple[Piece, ...] = tuple(Piece)

_SPAWN_MATRICES: dict[Piece, list[str]] = {
    Piece.I: ["....", "####", "....", "...."],
    Piece.O: ["##", "##"],
    Piece.T: [".#.", "###", "..."],
    Piece.S: [".##", "##.", "..."],
    Piece.Z: ["##.", ".##", "..."],
    Piece.J: ["#..", "###", "..."],
    Piece.L: ["..#", "###", "..."],
}

Cells = tuple[tuple[int, int], ...]


def _rotate_cw(matrix: list[str]) -> list[str]:
    n = len(matrix)
    return ["".join(matrix[n - 1 - c][r] for c in range(n)) for r in range(n)]


def _cells(matrix: list[str]) -> Cells:
    return tuple((r, c) for r, row in enumerate(matrix) for c, ch in enumerate(row) if ch == "#")


def _build_states() -> dict[Piece, tuple[Cells, Cells, Cells, Cells]]:
    states = {}
    for piece, matrix in _SPAWN_MATRICES.items():
        rots = []
        m = matrix
        for _ in range(4):
            rots.append(_cells(m))
            m = _rotate_cw(m)
        states[piece] = tuple(rots)
    return states


#: CELLS[piece][rotation] -> tuple of (row, col) offsets inside the bounding box.
CELLS: dict[Piece, tuple[Cells, Cells, Cells, Cells]] = _build_states()

#: Bounding box size per piece.
BOX_SIZE: dict[Piece, int] = {p: len(m) for p, m in _SPAWN_MATRICES.items()}

#: Leftmost occupied column offset per piece/rotation.
MIN_DC: dict[Piece, tuple[int, ...]] = {
    p: tuple(min(c for _, c in CELLS[p][r]) for r in range(4)) for p in Piece
}
#: Rightmost occupied column offset per piece/rotation.
MAX_DC: dict[Piece, tuple[int, ...]] = {
    p: tuple(max(c for _, c in CELLS[p][r]) for r in range(4)) for p in Piece
}

# SRS kick data in guideline notation: (dx, dy) with y pointing up.
_KICKS_JLSTZ_UP: dict[tuple[int, int], list[tuple[int, int]]] = {
    (0, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (1, 0): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (1, 2): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    (2, 1): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
    (2, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
    (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
    (0, 3): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
}
_KICKS_I_UP: dict[tuple[int, int], list[tuple[int, int]]] = {
    (0, 1): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (1, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (1, 2): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
    (2, 1): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (2, 3): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
    (3, 2): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
    (3, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
    (0, 3): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
}


def _to_row_col(table: dict[tuple[int, int], list[tuple[int, int]]]):
    # Convert (dx, dy_up) to (dcol, drow) with rows increasing downward.
    return {k: tuple((-dy, dx) for dx, dy in v) for k, v in table.items()}


KICKS_JLSTZ: dict[tuple[int, int], tuple[tuple[int, int], ...]] = _to_row_col(_KICKS_JLSTZ_UP)
KICKS_I: dict[tuple[int, int], tuple[tuple[int, int], ...]] = _to_row_col(_KICKS_I_UP)
_NO_KICK: tuple[tuple[int, int], ...] = ((0, 0),)


def kicks(piece: Piece, from_rot: int, to_rot: int) -> tuple[tuple[int, int], ...]:
    """Return the ordered list of ``(drow, dcol)`` kick offsets to try."""
    if piece is Piece.O:
        return _NO_KICK
    table = KICKS_I if piece is Piece.I else KICKS_JLSTZ
    return table[(from_rot, to_rot)]
