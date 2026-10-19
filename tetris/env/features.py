"""Hand-crafted board features, vectorized over batches of boards.

All functions accept a boolean array of shape ``(N, H, W)`` (or a single
``(H, W)`` board) where True means filled. Batching lets agents score every
candidate afterstate with a handful of NumPy calls.
"""

from __future__ import annotations

import numpy as np

#: Board-level features in vector order.
BOARD_FEATURES: tuple[str, ...] = (
    "aggregate_height",
    "max_height",
    "holes",
    "rows_with_holes",
    "hole_depth",
    "bumpiness",
    "wells",
    "max_well_depth",
    "row_transitions",
    "col_transitions",
)
#: Placement-level features appended for afterstates.
PLACEMENT_FEATURES: tuple[str, ...] = ("lines_cleared", "landing_height", "eroded_cells")
FEATURE_NAMES: tuple[str, ...] = BOARD_FEATURES + PLACEMENT_FEATURES
NUM_FEATURES = len(FEATURE_NAMES)


def _as_batch(filled: np.ndarray) -> tuple[np.ndarray, bool]:
    filled = np.asarray(filled, dtype=bool)
    if filled.ndim == 2:
        return filled[None], True
    return filled, False


def column_heights(filled: np.ndarray) -> np.ndarray:
    """Column heights measured from the floor, shape ``(N, W)``."""
    filled, single = _as_batch(filled)
    h = filled.shape[1]
    any_filled = filled.any(axis=1)
    top = filled.argmax(axis=1)
    heights = np.where(any_filled, h - top, 0)
    return heights[0] if single else heights


def board_features(filled: np.ndarray) -> np.ndarray:
    """Board features (see :data:`BOARD_FEATURES`), shape ``(N, 10)`` float32."""
    filled, single = _as_batch(filled)
    n, h, w = filled.shape
    heights = column_heights(filled)

    covered = np.maximum.accumulate(filled, axis=1)
    hole_mask = covered & ~filled
    holes = hole_mask.sum(axis=(1, 2))
    rows_with_holes = hole_mask.any(axis=2).sum(axis=1)
    # Hole depth: for each hole, number of filled cells above it in its column.
    filled_above = np.cumsum(filled, axis=1) - filled
    hole_depth = (filled_above * hole_mask).sum(axis=(1, 2))

    bumpiness = np.abs(np.diff(heights, axis=1)).sum(axis=1)

    # Wells: empty cells whose left and right neighbours (walls count) are filled.
    padded = np.pad(filled, ((0, 0), (0, 0), (1, 1)), constant_values=True)
    well_cells = ~filled & padded[:, :, :-2] & padded[:, :, 2:]
    run = np.zeros((n, w), dtype=np.int32)
    wells = np.zeros(n, dtype=np.int64)
    max_well = np.zeros((n, w), dtype=np.int32)
    for r in range(h):
        run = (run + 1) * well_cells[:, r]
        wells += run.sum(axis=1)
        max_well = np.maximum(max_well, run)
    max_well_depth = max_well.max(axis=1)

    row_transitions = (padded[:, :, 1:] != padded[:, :, :-1]).sum(axis=(1, 2))
    vpad = np.concatenate(
        [np.zeros((n, 1, w), dtype=bool), filled, np.ones((n, 1, w), dtype=bool)], axis=1
    )
    col_transitions = (vpad[:, 1:] != vpad[:, :-1]).sum(axis=(1, 2))

    feats = np.stack(
        [
            heights.sum(axis=1),
            heights.max(axis=1),
            holes,
            rows_with_holes,
            hole_depth,
            bumpiness,
            wells,
            max_well_depth,
            row_transitions,
            col_transitions,
        ],
        axis=1,
    ).astype(np.float32)
    return feats[0] if single else feats


def full_features(
    filled: np.ndarray,
    lines_cleared: np.ndarray,
    landing_height: np.ndarray,
    eroded_cells: np.ndarray,
) -> np.ndarray:
    """Board features plus placement features, shape ``(N, NUM_FEATURES)``."""
    filled, single = _as_batch(filled)
    board = board_features(filled)
    extra = np.stack(
        [
            np.atleast_1d(lines_cleared),
            np.atleast_1d(landing_height),
            np.atleast_1d(eroded_cells),
        ],
        axis=1,
    ).astype(np.float32)
    out = np.concatenate([board, extra], axis=1)
    return out[0] if single else out
