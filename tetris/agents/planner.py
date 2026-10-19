"""Path planner: turns a placement into player moves.

Agents choose placements; the planner finds a sequence of real engine moves
(shift, rotate with SRS kicks, soft drop, hold, hard drop) that reaches it, so
agents visibly move pieces like a player. The search mirrors the engine's
rotation and kick logic exactly, so executing the plan through
:meth:`Game.apply` produces the planned board.
"""

from __future__ import annotations

from collections import deque

from tetris.engine.game import Game, Move
from tetris.engine.pieces import MIN_DC, Piece, kicks
from tetris.engine.placement import BoardIndex, Placement, Target

State = tuple[int, int, int]  # (rot, row, col)

#: Re-plans allowed per executor before the target is treated as unreachable.
MAX_REPLANS = 8

_LATERAL = (Move.CW, Move.CCW, Move.LEFT, Move.RIGHT)


def _successor(index: BoardIndex, piece: Piece, state: State, move: Move) -> State | None:
    rot, row, col = state
    if move is Move.LEFT or move is Move.RIGHT:
        c = col - 1 if move is Move.LEFT else col + 1
        return (rot, row, c) if index.fits(piece, rot, row, c) else None
    if move is Move.SOFT:
        return (rot, row + 1, col) if index.fits(piece, rot, row + 1, col) else None
    new_rot = (rot + (1 if move is Move.CW else -1)) % 4
    for drow, dcol in kicks(piece, rot, new_rot):
        if index.fits(piece, new_rot, row + drow, col + dcol):
            return (new_rot, row + drow, col + dcol)
    return None


def _bfs(
    index: BoardIndex, piece: Piece, start: State, goal: Target, moves: tuple[Move, ...]
) -> list[Move] | None:
    def is_goal(s: State) -> bool:
        return (
            s[0] == goal.rotation
            and s[2] == goal.box_col
            and index.landing_row(piece, s[0], s[1], s[2]) == goal.row
        )

    if is_goal(start):
        return []
    parents: dict[State, tuple[State, Move]] = {}
    seen = {start}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        for move in moves:
            nxt = _successor(index, piece, state, move)
            if nxt is None or nxt in seen:
                continue
            seen.add(nxt)
            parents[nxt] = (state, move)
            if is_goal(nxt):
                path = []
                s = nxt
                while s != start:
                    s, m = parents[s]
                    path.append(m)
                path.reverse()
                return path
            queue.append(nxt)
    return None


def find_path(index: BoardIndex, piece: Piece, start: State, goal: Target) -> list[Move] | None:
    """Shortest move list from ``start`` to a state that hard drops onto ``goal``.

    Tries shifts and rotations first; falls back to also allowing soft drops.
    The returned list does not include the final hard drop.
    """
    path = _bfs(index, piece, start, goal, _LATERAL)
    if path is None:
        path = _bfs(index, piece, start, goal, (*_LATERAL, Move.SOFT))
    return path


def plan_path(game: Game, placement: Placement) -> list[Move] | None:
    """Moves from the game's current state to ``placement``, ending with a hard drop.

    Returns None when the placement cannot be reached. The game is not modified.
    """
    if game.game_over or game.active is None:
        return None
    prefix: list[Move] = []
    sim = game
    if placement.use_hold:
        if not game.can_hold:
            return None
        sim = game.copy()
        if not sim.apply(Move.HOLD) or sim.active is None:
            return None
        prefix.append(Move.HOLD)
    a = sim.active
    index = BoardIndex(sim.board)
    goal = index.resolve(a.piece, placement.rotation, placement.column, (a.rot, a.row, a.col))
    if goal is None:
        # Not in the canonical action set; aim for any landing with this rotation/column.
        box_col = placement.column - MIN_DC[a.piece][placement.rotation % 4]
        if not index.fits(a.piece, placement.rotation % 4, a.row, box_col):
            return None
        row = index.landing_row(a.piece, placement.rotation % 4, a.row, box_col)
        goal = Target(a.piece, placement.rotation % 4, placement.column, box_col, row)
    path = find_path(index, a.piece, (a.rot, a.row, a.col), goal)
    if path is None:
        return None
    return [*prefix, *path, Move.HARD]


class PlanExecutor:
    """Feeds a placement's moves to a live game one at a time.

    Gravity can move the piece between moves; when the piece is not where the
    plan expects, the executor re-plans from the current state. If the target
    becomes unreachable, :attr:`unreachable` is set and the next move is a hard
    drop in place.
    """

    def __init__(self, game: Game, placement: Placement) -> None:
        self.placement = placement
        self.done = False
        self.replans = 0
        self._piece_count = game.pieces_placed
        self._plan(game)

    def _plan(self, game: Game) -> None:
        moves = plan_path(game, self.placement)
        #: True when the target can no longer be reached (for example gravity
        #: pulled the piece below a ledge). Callers may choose a new placement;
        #: otherwise :meth:`next_move` hard drops in place.
        self.unreachable = moves is None
        self.moves: deque[Move] = deque(moves if moves is not None else [Move.HARD])
        self._expected: State | None = None

    def next_move(self, game: Game) -> Move | None:
        """Apply the next move. Returns it, or None when finished or when the
        target just became unreachable (see :attr:`unreachable`)."""
        if self.done or game.game_over or game.active is None:
            self.done = True
            return None
        if game.pieces_placed != self._piece_count:
            # The piece locked on its own (lock delay); nothing left to do.
            self.done = True
            return None
        a = game.active
        if self._expected is not None and (a.rot, a.row, a.col) != self._expected:
            self.replans += 1
            self.placement = Placement(self.placement.rotation, self.placement.column, False)
            self._plan(game)
            if self.replans > MAX_REPLANS:
                # Gravity keeps undoing the plan; treat the target as unreachable.
                self.unreachable = True
                self.moves = deque([Move.HARD])
            if self.unreachable:
                return None  # let the caller re-decide before dropping
        if not self.moves:
            self.moves.append(Move.HARD)
        move = self.moves.popleft()
        ok = game.apply(move)
        if move is Move.HARD or game.pieces_placed != self._piece_count:
            self.done = True
        elif not ok:
            self.replans += 1
            self._plan(game)
        else:
            if move is Move.HOLD:
                self.placement = Placement(self.placement.rotation, self.placement.column, False)
            a = game.active
            self._expected = (a.rot, a.row, a.col) if a else None
        return move
