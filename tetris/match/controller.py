"""Match controller: owns every board and the shared clock.

Frontends (Pygame now, a WebSocket server later) only call
:meth:`MatchController.step`, :meth:`MatchController.apply_input` and
:meth:`MatchController.get_snapshot`. The snapshot is plain JSON-serializable
data, so the same controller can drive any renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tetris.engine.board import HIDDEN_ROWS, board_to_strings, piece_cells
from tetris.engine.game import Game, GameConfig, Move


@dataclass
class PlayerSpec:
    """Who sits at a board."""

    player_id: str
    name: str
    kind: str = "human"  # "human" or "agent"


@dataclass
class MatchConfig:
    mode: str = "solo"
    seed: int = 0
    game: GameConfig = field(default_factory=GameConfig)


@dataclass
class PlayerState:
    spec: PlayerSpec
    game: Game
    thinking: bool = False
    finish_time: float | None = None


class MatchController:
    """Runs one match. Phase 1 supports solo play; later phases add race and battle."""

    def __init__(self, config: MatchConfig, players: list[PlayerSpec]) -> None:
        if not players:
            raise ValueError("a match needs at least one player")
        ids = [p.player_id for p in players]
        if len(set(ids)) != len(ids):
            raise ValueError("player ids must be unique")
        self.config = config
        self.specs = list(players)
        self.reset()

    def reset(self) -> None:
        """Restart the match with the same configuration and seed."""
        self.time = 0.0
        self.paused = False
        self.finished = False
        self.winner: str | None = None
        self.players: dict[str, PlayerState] = {
            spec.player_id: PlayerState(spec, Game(self.config.seed, self.config.game))
            for spec in self.specs
        }
        self.events: list[dict[str, Any]] = []

    # ------------------------------------------------------------ control

    def set_paused(self, paused: bool) -> None:
        if not self.finished:
            self.paused = paused

    def apply_input(self, player_id: str, action: Move | str) -> bool:
        """Apply a move for a player. Ignored while paused or after game over."""
        if self.paused or self.finished:
            return False
        player = self.players[player_id]
        changed = player.game.apply(action)
        self._collect(player)
        return changed

    def step(self, dt: float) -> None:
        """Advance the shared clock by ``dt`` seconds."""
        if self.paused or self.finished:
            return
        self.time += dt
        for player in self.players.values():
            if not player.game.game_over:
                player.game.tick(dt)
            self._collect(player)
        self._check_finished()

    def pop_events(self) -> list[dict[str, Any]]:
        events, self.events = self.events, []
        return events

    def _collect(self, player: PlayerState) -> None:
        for ev in player.game.pop_events():
            d = ev.to_dict()
            d["player_id"] = player.spec.player_id
            self.events.append(d)
            if ev.type == "game_over" and player.finish_time is None:
                player.finish_time = self.time

    def _check_finished(self) -> None:
        alive = [p for p in self.players.values() if not p.game.game_over]
        if not alive:
            self.finished = True
            best = max(self.players.values(), key=lambda p: (p.game.score, p.game.lines))
            self.winner = best.spec.player_id

    # ----------------------------------------------------------- snapshot

    def get_snapshot(self) -> dict[str, Any]:
        """JSON-serializable view of the match for renderers."""
        return {
            "mode": self.config.mode,
            "time": self.time,
            "paused": self.paused,
            "finished": self.finished,
            "winner": self.winner,
            "hidden_rows": HIDDEN_ROWS,
            "players": [self._player_snapshot(p) for p in self.players.values()],
        }

    def _player_snapshot(self, player: PlayerState) -> dict[str, Any]:
        g = player.game
        a = g.active
        active = None
        if a is not None:
            ghost = g.ghost_row()
            active = {
                "piece": a.piece.letter,
                "cells": [list(c) for c in piece_cells(a.piece, a.rot, a.row, a.col)],
                "ghost": [list(c) for c in piece_cells(a.piece, a.rot, ghost, a.col)],
            }
        return {
            "id": player.spec.player_id,
            "name": player.spec.name,
            "kind": player.spec.kind,
            "board": board_to_strings(g.board),
            "active": active,
            "hold": g.hold.letter if g.hold else None,
            "can_hold": g.can_hold,
            "queue": [p.letter for p in g.next_queue],
            "score": g.score,
            "lines": g.lines,
            "level": g.level,
            "pieces": g.pieces_placed,
            "combo": g.combo,
            "b2b": g.b2b,
            "attack_sent": g.attack_sent,
            "pending_garbage": g.pending_total,
            "game_over": g.game_over,
            "thinking": player.thinking,
        }
