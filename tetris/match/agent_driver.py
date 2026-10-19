"""Drives one agent-controlled board in real time.

For each new piece the driver requests a decision from the runner, waits out
the reaction delay, then feeds the planned moves to the game at the
difficulty's move rate, holding the hard drop back to respect the
pieces-per-second cap.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from tetris.agents.difficulty import DifficultyParams
from tetris.agents.planner import PlanExecutor
from tetris.engine.board import piece_cells
from tetris.engine.game import Game, Move
from tetris.engine.placement import BoardIndex, Placement
from tetris.match.agent_runtime import AgentRunner, Decision


class Phase(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    WAITING = "waiting"
    MOVING = "moving"


class AgentDriver:
    def __init__(
        self,
        player_id: str,
        runner: AgentRunner,
        difficulty: DifficultyParams,
        top_k: int = 0,
    ) -> None:
        self.player_id = player_id
        self.runner = runner
        self.difficulty = difficulty
        self.top_k = top_k
        self.clock = 0.0
        self.phase = Phase.IDLE
        self.request_id = 0
        self.last_error: str | None = None
        self.decisions = 0
        self.total_think_time = 0.0
        self._token = -1
        self._spawn_clock = 0.0
        self._last_lock_clock = -1e9
        self._decision: Decision | None = None
        self._executor: PlanExecutor | None = None
        self._move_acc = 0.0
        self.candidate_cells: list[dict[str, Any]] = []

    @property
    def thinking(self) -> bool:
        return self.phase is Phase.THINKING

    def set_difficulty(self, difficulty: DifficultyParams) -> None:
        self.difficulty = difficulty
        self.runner.set_difficulty(self.player_id, difficulty)

    def reset(self) -> None:
        self.__init__(self.player_id, self.runner, self.difficulty, self.top_k)  # type: ignore[misc]

    # ------------------------------------------------------------------ loop

    def update(self, game: Game, dt: float) -> None:
        self.clock += dt
        if game.game_over or game.active is None:
            return
        if self.phase is not Phase.IDLE and game.pieces_placed != self._token:
            # The piece locked without us (for example lock delay while thinking).
            self._finish_piece(locked_by_driver=False)

        if self.phase is Phase.IDLE:
            self._request(game)
        if self.phase is Phase.THINKING:
            decision = self.runner.poll(self.player_id)
            if decision is not None and decision.request_id == self.request_id:
                self._accept(game, decision)
        if (
            self.phase is Phase.WAITING
            and self.clock - self._spawn_clock >= self.difficulty.reaction_delay
        ):
            placement = self._decision.placement if self._decision else None
            self._executor = PlanExecutor(game, placement or Placement(0, 0))
            if placement is None:
                self._executor.moves.clear()
                self._executor.moves.append(Move.HARD)
            self.phase = Phase.MOVING
            self._move_acc = 1.0  # first move happens immediately
        if self.phase is Phase.MOVING:
            self._run_moves(game, dt)

    def _request(self, game: Game) -> None:
        self.request_id += 1
        self._token = game.pieces_placed
        self._spawn_clock = self.clock
        self.phase = Phase.THINKING
        self.runner.submit(self.player_id, self.request_id, game.to_dict(), self.top_k)

    def _accept(self, game: Game, decision: Decision) -> None:
        self._decision = decision
        self.decisions += 1
        self.total_think_time += decision.think_time
        self.last_error = decision.error
        self.candidate_cells = self._cells_for(game, decision.candidates)
        self.phase = Phase.WAITING

    def _drop_allowed(self) -> bool:
        pps = self.difficulty.pps
        return pps is None or self.clock - self._last_lock_clock >= 1.0 / pps

    def _run_moves(self, game: Game, dt: float) -> None:
        ex = self._executor
        assert ex is not None
        rate = self.difficulty.move_rate
        if rate is None:
            self._move_acc = float("inf")
        else:
            self._move_acc += dt * rate
        while self._move_acc >= 1.0 and not ex.done:
            upcoming = ex.moves[0] if ex.moves else Move.HARD
            if upcoming is Move.HARD and not self._drop_allowed():
                self._move_acc = min(self._move_acc, 1.0)
                return
            ex.next_move(game)
            self._move_acc -= 1.0
        if ex.done:
            self._finish_piece(locked_by_driver=True)

    def _finish_piece(self, locked_by_driver: bool) -> None:
        if locked_by_driver:
            self._last_lock_clock = self.clock
        self.phase = Phase.IDLE
        self._decision = None
        self._executor = None
        self.candidate_cells = []

    @staticmethod
    def _cells_for(game: Game, candidates: list[tuple[Placement, float]]) -> list[dict[str, Any]]:
        """Landing cells of ranked candidates, for ghost overlays."""
        out = []
        index = BoardIndex(game.board)
        for rank, (placement, score) in enumerate(candidates, start=1):
            a = game.active
            if placement.use_hold:
                piece = game.hold if game.hold is not None else game.queue[0]
                target = index.resolve(piece, placement.rotation, placement.column)
            else:
                piece = a.piece
                target = index.resolve(
                    piece, placement.rotation, placement.column, (a.rot, a.row, a.col)
                )
            if target is None:
                continue
            cells = piece_cells(piece, target.rotation, target.row, target.box_col)
            out.append(
                {
                    "rank": rank,
                    "score": score,
                    "piece": piece.letter,
                    "hold": placement.use_hold,
                    "cells": [list(c) for c in cells],
                }
            )
        return out
