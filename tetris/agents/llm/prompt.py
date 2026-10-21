"""Text rendering of a decision for the LLM agent, and parsing of its answer.

The prompt shows the board as a text grid, the current piece, the hold slot
and the next pieces, and a numbered list of every distinct legal placement
(equivalent placements that give the same board are listed once). Two
variants:

* ``features``: each placement also gives the lines it clears, the holes and
  the stack height after it;
* ``bare``: only the piece, rotation and leftmost column (an ablation: how
  well the model plays from the board picture alone).

The model must answer with JSON ``{"choice": <number>}``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from tetris.engine.board import HIDDEN_ROWS, piece_cells
from tetris.engine.game import Game
from tetris.engine.placement import Placement
from tetris.env.afterstates import enumerate_afterstates
from tetris.env.features import FEATURE_NAMES

PROMPT_VARIANTS = ("features", "bare")
HOLES = FEATURE_NAMES.index("holes")
MAX_HEIGHT = FEATURE_NAMES.index("max_height")
QUEUE_SHOWN = 5

SYSTEM_PROMPT = """You are playing Tetris. Each turn you choose where the current piece lands.
The board is 10 columns wide (0 is the left) and 20 rows tall. '#' is a filled cell, '.' is empty.
Every placement is a hard drop: the piece falls straight down from above.
A garbage row with one gap rises from the bottom every 3 pieces, so keep the stack low.
Good placements clear lines, create no holes (empty cells covered from above)
and keep the stack low and flat.
You may hold: a placement marked "hold" swaps the current piece with the held one.
Answer with JSON only, exactly {"choice": N}, where N is the number of one listed placement."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"choice": {"type": "integer"}},
    "required": ["choice"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Option:
    number: int
    placement: Placement
    piece: str
    leftmost_column: int
    lines: int
    holes: int
    height: int


def board_text(game: Game) -> str:
    """The visible 20 rows (plus any filled buffer row) with a column header."""
    board = game.board
    first = HIDDEN_ROWS
    for r in range(HIDDEN_ROWS):
        if (board[r] != 0).any():
            first = r
            break
    lines = ["   0123456789"]
    for r in range(first, board.shape[0]):
        row = "".join("#" if v else "." for v in board[r])
        lines.append(f"{board.shape[0] - r:2d} {row}")
    return "\n".join(lines)


def options(game: Game) -> list[Option]:
    """Distinct legal placements, numbered from 1 (non-hold placements first)."""
    out = []
    afs = enumerate_afterstates(game, dedupe=True)
    afs.sort(key=lambda a: (a.placement.use_hold, a.placement.rotation, a.placement.column))
    for n, a in enumerate(afs, 1):
        t = a.target
        cols = [c for _, c in piece_cells(t.piece, t.rotation, t.row, t.box_col)]
        out.append(
            Option(
                n,
                a.placement,
                t.piece.letter,
                min(cols),
                int(a.lines_cleared),
                int(a.features[HOLES]),
                int(a.features[MAX_HEIGHT]),
            )
        )
    return out


def user_prompt(game: Game, opts: list[Option], variant: str) -> str:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}")
    current = game.active.piece.letter if game.active is not None else "?"
    hold = game.hold.letter if game.hold else "empty"
    queue = " ".join(p.letter for p in game.queue[:QUEUE_SHOWN])
    lines = [
        "Board (row numbers count from the bottom):",
        board_text(game),
        "",
        f"Current piece: {current}    Hold: {hold}    Next: {queue}",
        "",
        "Placements:",
    ]
    for o in opts:
        where = f"{o.piece}{' (hold)' if o.placement.use_hold else ''}, rotation "
        where += f"{o.placement.rotation}, leftmost column {o.leftmost_column}"
        if variant == "features":
            where += f": clears {o.lines}, holes {o.holes}, height {o.height}"
        lines.append(f"{o.number}. {where}")
    lines += ["", 'Reply with {"choice": N}.']
    return "\n".join(lines)


class InvalidAnswer(ValueError):
    """The model's reply is not JSON with a listed choice."""


def parse_choice(text: str, n_options: int) -> int:
    """The chosen option number from a reply, or :class:`InvalidAnswer`."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match is None:
        raise InvalidAnswer(f"no JSON object in reply: {text[:80]!r}")
    try:
        value = json.loads(match.group(0))["choice"]
    except (ValueError, KeyError, TypeError) as exc:
        raise InvalidAnswer(f"bad JSON: {match.group(0)[:80]!r}") from exc
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise InvalidAnswer(f"choice is not a number: {value!r}")
    try:
        number = int(value)
    except ValueError as exc:
        raise InvalidAnswer(f"choice is not a number: {value!r}") from exc
    if not 1 <= number <= n_options:
        raise InvalidAnswer(f"choice {number} is not listed (1 to {n_options})")
    return number
