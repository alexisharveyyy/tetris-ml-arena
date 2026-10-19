"""Headless, deterministic Tetris game.

The :class:`Game` supports two styles of use:

* Real-time play: :meth:`Game.apply` for player moves and :meth:`Game.tick` to
  advance gravity and lock delay.
* Placement-level play (training, evaluation): :meth:`Game.place` puts the
  current piece at a :class:`~tetris.engine.placement.Placement` instantly.

All randomness comes from seeded :class:`~tetris.engine.rng.Rng` streams, and
the full state round-trips through :meth:`Game.to_dict` / :meth:`Game.from_dict`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from tetris.engine import scoring
from tetris.engine.board import (
    HEIGHT,
    HIDDEN_ROWS,
    WIDTH,
    board_from_strings,
    board_to_strings,
    clear_full_rows,
    drop_row,
    empty_board,
    fits,
    insert_garbage_rows,
    piece_cells,
)
from tetris.engine.pieces import ALL_PIECES, CELLS, Piece, kicks
from tetris.engine.placement import BoardIndex, Placement, spawn_position
from tetris.engine.rng import Rng, derive_seed

_BAG_STREAM = 1
_GARBAGE_STREAM = 2


class Move(StrEnum):
    """Player inputs. String values keep them JSON friendly."""

    LEFT = "left"
    RIGHT = "right"
    CW = "cw"
    CCW = "ccw"
    SOFT = "soft"
    HARD = "hard"
    HOLD = "hold"


class IllegalPlacementError(ValueError):
    """Raised by :meth:`Game.place` for a placement that is not legal."""


@dataclass
class GameConfig:
    """Tunable game rules."""

    start_level: int = 1
    max_level: int = 20
    lines_per_level: int = 10
    queue_size: int = 5
    lock_delay: float = 0.5
    max_lock_resets: int = 15
    #: Fixed gravity in rows per second; None uses the guideline level curve.
    gravity: float | None = None
    #: Maximum garbage rows inserted per lock; None means no cap.
    garbage_cap: int | None = None
    #: Chance that the garbage hole column moves between batches (rows within a
    #: batch always share one hole).
    hole_change_prob: float = 0.3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GameConfig:
        return cls(**d)


@dataclass
class ActivePiece:
    piece: Piece
    rot: int
    row: int
    col: int

    def to_dict(self) -> dict[str, Any]:
        return {"piece": self.piece.letter, "rot": self.rot, "row": self.row, "col": self.col}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ActivePiece:
        return cls(Piece.from_letter(d["piece"]), d["rot"], d["row"], d["col"])


@dataclass
class LockResult:
    """Summary of a piece lock, also emitted as a ``lock`` event."""

    piece: Piece
    lines: int
    combo: int
    b2b: bool
    attack: int  # raw attack before cancelling incoming garbage
    sent: int  # attack left after cancelling incoming garbage
    score: int
    cells: list[tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["piece"] = self.piece.letter
        d["cells"] = [list(c) for c in self.cells]
        return d


@dataclass
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, **self.data}


class Game:
    """A single Tetris board with its own piece sequence."""

    def __init__(
        self, seed: int = 0, config: GameConfig | None = None, garbage_seed: int | None = None
    ) -> None:
        """``seed`` drives the piece sequence. ``garbage_seed`` (default: derived
        from ``seed``) drives garbage hole columns, so boards in a match can share
        pieces but have independent holes."""
        self.seed = seed
        self.config = config or GameConfig()
        self.board = empty_board()
        self.bag_rng = Rng(derive_seed(seed, _BAG_STREAM))
        self.garbage_rng = Rng(
            derive_seed(seed if garbage_seed is None else garbage_seed, _GARBAGE_STREAM)
        )
        self.queue: list[Piece] = []
        self.active: ActivePiece | None = None
        self.hold: Piece | None = None
        self.can_hold = True
        self.score = 0
        self.lines = 0
        self.level = self.config.start_level
        self.combo = -1
        self.b2b = False
        self.pieces_placed = 0
        self.attack_sent = 0
        self.game_over = False
        self.pending_garbage: list[int] = []
        self.garbage_hole: int | None = None
        self.time = 0.0
        self._gravity_acc = 0.0
        self._lock_timer = 0.0
        self._lock_resets = 0
        self._lowest_row = 0
        self._events: list[Event] = []
        self._fill_queue()
        self._spawn(self._next_piece())

    # ------------------------------------------------------------------ queue

    def _fill_queue(self) -> None:
        while len(self.queue) < max(self.config.queue_size + 1, 7):
            bag = list(ALL_PIECES)
            self.bag_rng.shuffle(bag)
            self.queue.extend(bag)

    def _next_piece(self) -> Piece:
        piece = self.queue.pop(0)
        self._fill_queue()
        return piece

    @property
    def next_queue(self) -> list[Piece]:
        """The visible next pieces."""
        return self.queue[: self.config.queue_size]

    # ------------------------------------------------------------------ spawn

    def _spawn(self, piece: Piece) -> bool:
        pos = spawn_position(self.board, piece)
        if pos is None:
            self.active = None
            self._top_out("block_out")
            return False
        row, col = pos
        self.active = ActivePiece(piece, 0, row, col)
        self._gravity_acc = 0.0
        self._lock_timer = 0.0
        self._lock_resets = 0
        self._lowest_row = row
        self._emit("spawn", piece=piece.letter)
        return True

    def _top_out(self, reason: str) -> None:
        if not self.game_over:
            self.game_over = True
            self._emit("game_over", reason=reason)

    # ----------------------------------------------------------------- events

    def _emit(self, type_: str, **data: Any) -> None:
        self._events.append(Event(type_, data))

    def pop_events(self) -> list[Event]:
        """Return and clear events emitted since the last call."""
        events, self._events = self._events, []
        return events

    # ----------------------------------------------------------- live control

    def gravity(self) -> float:
        """Current gravity in rows per second."""
        if self.config.gravity is not None:
            return self.config.gravity
        return scoring.gravity_rows_per_second(self.level)

    def _grounded(self) -> bool:
        a = self.active
        return a is not None and not fits(self.board, a.piece, a.rot, a.row + 1, a.col)

    def _on_successful_move(self, was_grounded: bool) -> None:
        """Lock-delay bookkeeping after a shift, rotation or soft drop.

        A move counts toward the reset limit if the piece was on the ground
        before or after it, so floor kicks that lift the piece still count.
        """
        a = self.active
        assert a is not None
        if a.row > self._lowest_row:
            self._lowest_row = a.row
            self._lock_resets = 0
            self._lock_timer = 0.0
        elif (was_grounded or self._grounded()) and self._lock_resets < self.config.max_lock_resets:
            self._lock_resets += 1
            self._lock_timer = 0.0

    def _shift(self, dcol: int) -> bool:
        a = self.active
        if a is None or not fits(self.board, a.piece, a.rot, a.row, a.col + dcol):
            return False
        was_grounded = self._grounded()
        a.col += dcol
        self._on_successful_move(was_grounded)
        return True

    def _rotate(self, direction: int) -> bool:
        a = self.active
        if a is None:
            return False
        new_rot = (a.rot + direction) % 4
        for drow, dcol in kicks(a.piece, a.rot, new_rot):
            if fits(self.board, a.piece, new_rot, a.row + drow, a.col + dcol):
                was_grounded = self._grounded()
                a.rot = new_rot
                a.row += drow
                a.col += dcol
                self._on_successful_move(was_grounded)
                return True
        return False

    def _soft_drop(self) -> bool:
        a = self.active
        if a is None or not fits(self.board, a.piece, a.rot, a.row + 1, a.col):
            return False
        a.row += 1
        self.score += 1
        self._on_successful_move(False)
        return True

    def _hard_drop(self) -> LockResult | None:
        a = self.active
        if a is None:
            return None
        target = drop_row(self.board, a.piece, a.rot, a.row, a.col)
        self.score += 2 * (target - a.row)
        a.row = target
        return self._lock()

    def _hold(self) -> bool:
        a = self.active
        if a is None or not self.can_hold:
            return False
        current = a.piece
        incoming = self.hold if self.hold is not None else self._next_piece()
        self.hold = current
        self.can_hold = False
        self._emit("hold", piece=current.letter)
        self._spawn(incoming)
        return True

    def apply(self, move: Move | str) -> bool:
        """Apply a player move. Returns True if the move changed the state."""
        if self.game_over or self.active is None:
            return False
        move = Move(move)
        if move is Move.LEFT:
            return self._shift(-1)
        if move is Move.RIGHT:
            return self._shift(1)
        if move is Move.CW:
            return self._rotate(1)
        if move is Move.CCW:
            return self._rotate(-1)
        if move is Move.SOFT:
            return self._soft_drop()
        if move is Move.HARD:
            return self._hard_drop() is not None
        if move is Move.HOLD:
            return self._hold()
        raise ValueError(move)

    def tick(self, dt: float) -> None:
        """Advance gravity and lock delay by ``dt`` seconds."""
        if self.game_over:
            return
        self.time += dt
        a = self.active
        if a is None:
            return
        self._gravity_acc += dt * self.gravity()
        while self._gravity_acc >= 1.0:
            if not fits(self.board, a.piece, a.rot, a.row + 1, a.col):
                self._gravity_acc = 0.0
                break
            self._gravity_acc -= 1.0
            a.row += 1
            if a.row > self._lowest_row:
                self._lowest_row = a.row
                self._lock_resets = 0
                self._lock_timer = 0.0
        if self._grounded():
            self._lock_timer += dt
            # Extended placement: once the move resets are used up, the piece locks
            # as soon as it touches down (prevents infinite spinning with floor kicks).
            if (
                self._lock_timer >= self.config.lock_delay
                or self._lock_resets >= self.config.max_lock_resets
            ):
                self._lock()
        else:
            self._lock_timer = 0.0

    def ghost_row(self) -> int | None:
        """Row the active piece would land on if hard dropped."""
        a = self.active
        if a is None:
            return None
        return drop_row(self.board, a.piece, a.rot, a.row, a.col)

    # ------------------------------------------------------- placement level

    def place(self, placement: Placement) -> LockResult:
        """Instantly place the current piece (or hold piece) and lock it."""
        if self.game_over or self.active is None:
            raise IllegalPlacementError("game is over")
        index = BoardIndex(self.board)
        if placement.use_hold:
            if not self.can_hold:
                raise IllegalPlacementError("hold already used for this piece")
            piece = self.hold if self.hold is not None else self.queue[0]
            target = index.resolve(piece, placement.rotation, placement.column)
        else:
            a = self.active
            piece = a.piece
            target = index.resolve(
                piece, placement.rotation, placement.column, (a.rot, a.row, a.col)
            )
        if target is None:
            raise IllegalPlacementError(f"illegal placement {placement} for {piece.letter}")
        if placement.use_hold:
            self._hold()
            if self.active is None:  # spawn failed; cannot happen if target resolved
                raise IllegalPlacementError("hold piece could not spawn")
        a = self.active
        start_row = a.row
        a.rot, a.col, a.row = target.rotation, target.box_col, target.row
        self.score += 2 * max(0, target.row - start_row)
        result = self._lock()
        assert result is not None
        return result

    # ------------------------------------------------------------------ lock

    def _lock(self) -> LockResult | None:
        a = self.active
        if a is None:
            return None
        cells = piece_cells(a.piece, a.rot, a.row, a.col)
        for r, c in cells:
            self.board[r, c] = int(a.piece)
        self.active = None
        self.pieces_placed += 1
        lock_out = all(r < HIDDEN_ROWS for r, _ in cells)

        self.board, cleared = clear_full_rows(self.board)
        lines = len(cleared)
        b2b = False
        if lines > 0:
            self.combo += 1
            if lines == 4:
                b2b = self.b2b
                self.b2b = True
            else:
                self.b2b = False
        else:
            self.combo = -1

        combo = max(self.combo, 0)
        score = scoring.score_for(lines, self.level, combo, b2b)
        attack = scoring.attack_for(lines, combo, b2b)
        sent = attack
        while sent > 0 and self.pending_garbage:
            cancel = min(sent, self.pending_garbage[0])
            sent -= cancel
            self.pending_garbage[0] -= cancel
            if self.pending_garbage[0] == 0:
                self.pending_garbage.pop(0)

        self.score += score
        self.lines += lines
        self.attack_sent += sent
        self.level = min(
            self.config.max_level,
            max(self.level, self.config.start_level + self.lines // self.config.lines_per_level),
        )
        self.can_hold = True
        result = LockResult(a.piece, lines, combo if lines else -1, b2b, attack, sent, score, cells)
        self._emit("lock", **result.to_dict())

        if lock_out:
            self._top_out("lock_out")
            return result
        if lines == 0 and self.pending_garbage:
            self._insert_pending_garbage()
            if self.game_over:
                return result
        self._spawn(self._next_piece())
        return result

    # --------------------------------------------------------------- garbage

    def queue_garbage(self, rows: int) -> None:
        """Add incoming garbage to the pending meter (inserted on a non-clearing lock)."""
        if rows > 0:
            self.pending_garbage.append(rows)
            self._emit("garbage_queued", rows=rows, pending=self.pending_total)

    @property
    def pending_total(self) -> int:
        return sum(self.pending_garbage)

    def _insert_pending_garbage(self) -> None:
        budget = self.config.garbage_cap
        while self.pending_garbage and not self.game_over:
            rows = self.pending_garbage[0]
            if budget is not None:
                if budget <= 0:
                    break
                rows = min(rows, budget)
                budget -= rows
            self.pending_garbage[0] -= rows
            if self.pending_garbage[0] == 0:
                self.pending_garbage.pop(0)
            self.receive_garbage(rows)

    def receive_garbage(self, rows: int) -> None:
        """Immediately insert ``rows`` garbage rows sharing one hole column.

        The hole column carries over from the previous batch and moves to a
        different random column with probability ``config.hole_change_prob``.
        """
        if rows <= 0 or self.game_over:
            return
        hole = self.garbage_hole
        if hole is None:
            hole = self.garbage_rng.randbelow(WIDTH)
        elif self.garbage_rng.random() < self.config.hole_change_prob:
            hole = (hole + 1 + self.garbage_rng.randbelow(WIDTH - 1)) % WIDTH
        self.garbage_hole = hole
        self.board, overflow = insert_garbage_rows(self.board, rows, hole)
        self._emit("garbage", rows=rows, hole=hole)
        a = self.active
        if a is not None:
            while not fits(self.board, a.piece, a.rot, a.row, a.col):
                a.row -= 1
                if a.row + min(dr for dr, _ in CELLS[a.piece][a.rot]) < 0:
                    overflow = True
                    break
            self._lowest_row = min(self._lowest_row, a.row)
        if overflow:
            self._top_out("garbage_out")

    # --------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        """Full, JSON-serializable state."""
        return {
            "seed": self.seed,
            "config": self.config.to_dict(),
            "board": board_to_strings(self.board),
            "bag_rng": self.bag_rng.state,
            "garbage_rng": self.garbage_rng.state,
            "queue": "".join(p.letter for p in self.queue),
            "active": self.active.to_dict() if self.active else None,
            "hold": self.hold.letter if self.hold else None,
            "can_hold": self.can_hold,
            "score": self.score,
            "lines": self.lines,
            "level": self.level,
            "combo": self.combo,
            "b2b": self.b2b,
            "pieces_placed": self.pieces_placed,
            "attack_sent": self.attack_sent,
            "game_over": self.game_over,
            "pending_garbage": list(self.pending_garbage),
            "garbage_hole": self.garbage_hole,
            "time": self.time,
            "gravity_acc": self._gravity_acc,
            "lock_timer": self._lock_timer,
            "lock_resets": self._lock_resets,
            "lowest_row": self._lowest_row,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Game:
        game = cls.__new__(cls)
        game.seed = d["seed"]
        game.config = GameConfig.from_dict(d["config"])
        game.board = board_from_strings(d["board"])
        game.bag_rng = Rng(0)
        game.bag_rng.state = d["bag_rng"]
        game.garbage_rng = Rng(0)
        game.garbage_rng.state = d["garbage_rng"]
        game.queue = [Piece.from_letter(ch) for ch in d["queue"]]
        game.active = ActivePiece.from_dict(d["active"]) if d["active"] else None
        game.hold = Piece.from_letter(d["hold"]) if d["hold"] else None
        game.can_hold = d["can_hold"]
        game.score = d["score"]
        game.lines = d["lines"]
        game.level = d["level"]
        game.combo = d["combo"]
        game.b2b = d["b2b"]
        game.pieces_placed = d["pieces_placed"]
        game.attack_sent = d["attack_sent"]
        game.game_over = d["game_over"]
        game.pending_garbage = list(d["pending_garbage"])
        game.garbage_hole = d.get("garbage_hole")
        game.time = d["time"]
        game._gravity_acc = d["gravity_acc"]
        game._lock_timer = d["lock_timer"]
        game._lock_resets = d["lock_resets"]
        game._lowest_row = d["lowest_row"]
        game._events = []
        return game

    def copy(self) -> Game:
        """Deep copy without going through JSON."""
        new = Game.__new__(Game)
        new.__dict__.update(self.__dict__)
        new.config = GameConfig(**asdict(self.config))
        new.board = self.board.copy()
        new.bag_rng = Rng(0)
        new.bag_rng.state = self.bag_rng.state
        new.garbage_rng = Rng(0)
        new.garbage_rng.state = self.garbage_rng.state
        new.queue = list(self.queue)
        new.active = (
            ActivePiece(self.active.piece, self.active.rot, self.active.row, self.active.col)
            if self.active
            else None
        )
        new.pending_garbage = list(self.pending_garbage)
        new._events = []
        return new

    def visible_board(self) -> np.ndarray:
        """The 20x10 visible playfield (a view, do not mutate)."""
        return self.board[HIDDEN_ROWS:HEIGHT]
