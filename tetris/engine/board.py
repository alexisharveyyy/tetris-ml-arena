"""Board geometry and low-level board operations.

The board is a ``numpy.int8`` array of shape ``(HEIGHT, WIDTH)``. Row 0 is the
top. The first ``HIDDEN_ROWS`` rows are the spawn buffer above the visible
10x20 playfield.
"""

from __future__ import annotations

import numpy as np

from tetris.engine.pieces import BOX_SIZE, CELL_CHARS, CELLS, GARBAGE, Piece

WIDTH = 10
VISIBLE_HEIGHT = 20
HIDDEN_ROWS = 4
HEIGHT = VISIBLE_HEIGHT + HIDDEN_ROWS

#: Bounding-box row at which pieces are created (before the one-row spawn drop).
SPAWN_ROW = 2


def spawn_col(piece: Piece) -> int:
    """Bounding-box column at which a piece spawns (centered, left-biased)."""
    return 4 if BOX_SIZE[piece] == 2 else 3


def empty_board() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH), dtype=np.int8)


def fits(board: np.ndarray, piece: Piece, rot: int, row: int, col: int) -> bool:
    """Return True if the piece at the given box position overlaps nothing."""
    for dr, dc in CELLS[piece][rot]:
        r = row + dr
        c = col + dc
        if c < 0 or c >= WIDTH or r < 0 or r >= HEIGHT or board[r, c]:
            return False
    return True


def piece_cells(piece: Piece, rot: int, row: int, col: int) -> list[tuple[int, int]]:
    """Absolute board cells occupied by a piece."""
    return [(row + dr, col + dc) for dr, dc in CELLS[piece][rot]]


def drop_row(board: np.ndarray, piece: Piece, rot: int, row: int, col: int) -> int:
    """Lowest row reachable by dropping straight down from ``row``."""
    while fits(board, piece, rot, row + 1, col):
        row += 1
    return row


def clear_full_rows(board: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Remove full rows. Returns the new board and the cleared row indices."""
    full = np.flatnonzero(board.all(axis=1))
    if full.size == 0:
        return board, []
    kept = np.delete(board, full, axis=0)
    new = np.zeros_like(board)
    new[full.size :] = kept
    return new, full.tolist()


def insert_garbage_rows(board: np.ndarray, rows: int, hole: int) -> tuple[np.ndarray, bool]:
    """Push the stack up and add ``rows`` garbage rows with a hole at ``hole``.

    Returns the new board and whether filled cells were pushed off the top
    (a top out).
    """
    if rows <= 0:
        return board, False
    rows = min(rows, HEIGHT)
    overflow = bool(board[:rows].any())
    new = np.empty_like(board)
    new[: HEIGHT - rows] = board[rows:]
    new[HEIGHT - rows :] = GARBAGE
    new[HEIGHT - rows :, hole] = 0
    return new, overflow


def board_to_strings(board: np.ndarray) -> list[str]:
    return ["".join(CELL_CHARS[v] for v in row) for row in board.tolist()]


def board_from_strings(rows: list[str]) -> np.ndarray:
    board = np.array([[CELL_CHARS.index(ch) for ch in row] for row in rows], dtype=np.int8)
    if board.shape != (HEIGHT, WIDTH):
        raise ValueError(f"board must be {HEIGHT}x{WIDTH}, got {board.shape}")
    return board
