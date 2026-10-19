"""Headless Tetris engine. Pure Python and NumPy, no rendering dependencies."""

from tetris.engine.board import HEIGHT, HIDDEN_ROWS, VISIBLE_HEIGHT, WIDTH
from tetris.engine.game import Event, Game, GameConfig, IllegalPlacementError, LockResult, Move
from tetris.engine.pieces import GARBAGE, Piece
from tetris.engine.placement import NUM_ACTIONS, Placement, Target, legal_targets

__all__ = [
    "GARBAGE",
    "HEIGHT",
    "HIDDEN_ROWS",
    "NUM_ACTIONS",
    "VISIBLE_HEIGHT",
    "WIDTH",
    "Event",
    "Game",
    "GameConfig",
    "IllegalPlacementError",
    "LockResult",
    "Move",
    "Piece",
    "Placement",
    "Target",
    "legal_targets",
]
