"""Match controller: owns every board and the shared clock.

Frontends (Pygame now, a WebSocket server later) only call
:meth:`MatchController.step`, :meth:`MatchController.apply_input` and
:meth:`MatchController.get_snapshot`. The snapshot is plain JSON-serializable
data, so the same controller can drive any renderer.

Two clock modes:

* Real time (default): agent decisions arrive asynchronously from the runner
  and are applied on the first frame after they arrive. ``step`` never blocks,
  so the human's input is never delayed by agents.
* Lockstep (``MatchConfig.lockstep``): ``step`` waits for each requested agent
  decision before advancing. Results depend only on the seed and the inputs,
  so headless matches and tests are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tetris.agents.difficulty import DifficultyParams
from tetris.engine.board import HIDDEN_ROWS, board_to_strings, piece_cells
from tetris.engine.game import Game, GameConfig, Move
from tetris.engine.rng import derive_seed
from tetris.env.tetris_env import inject_scheduled_garbage
from tetris.match.agent_driver import AgentDriver
from tetris.match.agent_runtime import AgentRunner, InlineRunner
from tetris.match.rules import (
    FIRST_TO_LINES,
    LINES_IN_TIME,
    SCORE_IN_TIME,
    TARGET_LEADER,
    AttackRouter,
    BattleRules,
    RaceRules,
)

_GARBAGE_SEED_STREAM = 1000
_ROUTER_SEED_STREAM = 2000

MODES = ("solo", "watch", "race", "battle", "eval")


@dataclass
class PlayerSpec:
    """Who sits at a board. Agent players name a registry key and a difficulty."""

    player_id: str
    name: str
    kind: str = "human"  # "human" or "agent"
    agent_key: str | None = None
    difficulty: DifficultyParams = field(default_factory=DifficultyParams)
    checkpoint: str | None = None


@dataclass
class MatchConfig:
    mode: str = "solo"
    seed: int = 0
    game: GameConfig = field(default_factory=GameConfig)
    race: RaceRules | None = None
    battle: BattleRules | None = None
    #: Wait for every agent decision before advancing (deterministic).
    lockstep: bool = False
    #: Seconds of countdown before the clock starts.
    countdown: float = 0.0
    #: Evaluation pressure: direct (non-cancellable) garbage every N pieces per board.
    eval_garbage_every: int | None = None
    eval_garbage_rows: int = 1
    #: Evaluation piece cap per board (a board that reaches it is done, not topped out).
    eval_max_pieces: int | None = None


@dataclass
class PlayerState:
    spec: PlayerSpec
    game: Game
    driver: AgentDriver | None = None
    finish_time: float | None = None  # topped out or reached a goal
    done: bool = False  # finished without topping out (eval cap, race goal)
    goal_time: float | None = None  # first-to-N-lines reach time
    attack_received: int = 0

    @property
    def alive(self) -> bool:
        return not self.game.game_over and not self.done


class MatchController:
    """Runs one match: solo, watch, race, battle or a headless evaluation."""

    def __init__(
        self,
        config: MatchConfig,
        players: list[PlayerSpec],
        runner: AgentRunner | None = None,
        top_k: int = 0,
    ) -> None:
        if not players:
            raise ValueError("a match needs at least one player")
        if config.mode not in MODES:
            raise ValueError(f"unknown mode {config.mode}")
        ids = [p.player_id for p in players]
        if len(set(ids)) != len(ids):
            raise ValueError("player ids must be unique")
        if config.mode == "race" and config.race is None:
            config.race = RaceRules()
        if config.mode == "battle":
            if config.battle is None:
                config.battle = BattleRules()
            if len(players) < 2:
                raise ValueError("battle needs at least two players")
        self.config = config
        self.specs = list(players)
        self.top_k = top_k
        self.runner = runner
        agents = [p for p in players if p.kind == "agent"]
        if agents and self.runner is None:
            self.runner = InlineRunner()
        for i, spec in enumerate(agents):
            if spec.agent_key is None:
                raise ValueError(f"agent player {spec.player_id} needs an agent_key")
            self.runner.add_agent(
                spec.player_id,
                spec.agent_key,
                spec.difficulty,
                spec.checkpoint,
                seed=derive_seed(config.seed, 3000 + i),
            )
        self.reset()

    # ------------------------------------------------------------------ setup

    def reset(self, seed: int | None = None) -> None:
        """Restart the match (optionally with a new seed). Agents stay loaded."""
        if seed is not None:
            self.config.seed = seed
        seed = self.config.seed
        self.time = 0.0
        self.countdown = self.config.countdown
        self.paused = False
        self.finished = False
        self.winner: str | None = None
        self.standings: list[dict[str, Any]] = []
        self.players: dict[str, PlayerState] = {}
        for i, spec in enumerate(self.specs):
            driver = None
            if spec.kind == "agent":
                assert self.runner is not None
                driver = AgentDriver(
                    spec.player_id, self.runner, spec.difficulty, self.top_k, self.config.lockstep
                )
            # Every board shares the piece seed; garbage holes are per board.
            game = Game(
                seed, self.config.game, garbage_seed=derive_seed(seed, _GARBAGE_SEED_STREAM + i)
            )
            self.players[spec.player_id] = PlayerState(spec, game, driver)
        default_mode = self.config.battle.targeting if self.config.battle else TARGET_LEADER
        self.router = AttackRouter(
            [s.player_id for s in self.specs], derive_seed(seed, _ROUTER_SEED_STREAM), default_mode
        )
        self.events: list[dict[str, Any]] = []

    def close(self) -> None:
        """Shut down agent runners."""
        if self.runner is not None:
            self.runner.close()

    @property
    def ready(self) -> bool:
        """False while agent workers are still loading."""
        return self.runner is None or self.runner.ready()

    # ---------------------------------------------------------------- control

    def set_paused(self, paused: bool) -> None:
        if not self.finished:
            self.paused = paused

    def set_difficulty(self, player_id: str, difficulty: DifficultyParams) -> None:
        player = self.players[player_id]
        player.spec.difficulty = difficulty
        if player.driver is not None:
            player.driver.set_difficulty(difficulty)

    def set_top_k(self, top_k: int) -> None:
        self.top_k = top_k
        for player in self.players.values():
            if player.driver is not None:
                player.driver.top_k = top_k

    def set_targeting(self, player_id: str, mode: str) -> None:
        """Change who a player's attacks go to (battle)."""
        self.router.set_mode(player_id, mode)

    def end_match(self) -> None:
        """Stop now and rank players as they stand."""
        if not self.finished:
            self._finish()

    def _running(self) -> bool:
        return not (self.paused or self.finished or self.countdown > 0)

    def apply_input(self, player_id: str, action: Move | str) -> bool:
        """Apply a human move. Ignored before the start, while paused, after game
        over, and for agent boards."""
        player = self.players[player_id]
        if not self._running() or player.driver is not None or not player.alive:
            return False
        changed = player.game.apply(action)
        self._collect(player)
        self._check_end()
        return changed

    def step(self, dt: float) -> None:
        """Advance the shared clock by ``dt`` seconds."""
        if self.paused or self.finished:
            return
        if not self.config.lockstep and not self.ready:
            return  # workers still loading; the snapshot reports it
        if self.countdown > 0:
            self.countdown = max(0.0, self.countdown - dt)
            return
        self.time += dt
        for player in self.players.values():
            if player.driver is not None and player.alive:
                before = player.game.pieces_placed
                player.driver.update(player.game, dt)
                self._collect(player)
                # A piece locked this frame: lock events (garbage, attacks) are
                # applied above, so the next decision sees the updated board.
                if player.driver.idle and player.alive and player.game.pieces_placed != before:
                    player.driver.update(player.game, 0.0)
                    self._collect(player)
            if player.alive:
                player.game.tick(dt)
            self._collect(player)
        self._check_end()

    def pop_events(self) -> list[dict[str, Any]]:
        events, self.events = self.events, []
        return events

    # ------------------------------------------------------------- events

    def _alive_map(self) -> dict[str, bool]:
        return {pid: p.alive for pid, p in self.players.items()}

    def _stats_map(self) -> dict[str, tuple[int, int]]:
        return {pid: (p.game.lines, p.game.score) for pid, p in self.players.items()}

    def _collect(self, player: PlayerState) -> None:
        pid = player.spec.player_id
        for ev in player.game.pop_events():
            d = ev.to_dict()
            d["player_id"] = pid
            self.events.append(d)
            if ev.type == "lock":
                self._on_lock(player, ev.data)
            elif ev.type == "game_over" and player.finish_time is None:
                player.finish_time = self.time

    def _on_lock(self, player: PlayerState, data: dict[str, Any]) -> None:
        game = player.game
        cfg = self.config
        if cfg.mode == "battle" and data["sent"] > 0 and not game.game_over:
            deliveries = self.router.route(
                player.spec.player_id, data["sent"], self._alive_map(), self._stats_map()
            )
            for target_id, rows in deliveries:
                target = self.players[target_id]
                target.game.queue_garbage(rows)
                target.attack_received += rows
                self.events.append(
                    {"type": "attack", "from": player.spec.player_id, "to": target_id, "rows": rows}
                )
        if cfg.eval_garbage_every:
            inject_scheduled_garbage(game, cfg.eval_garbage_every, cfg.eval_garbage_rows)
        if cfg.eval_max_pieces and game.pieces_placed >= cfg.eval_max_pieces and not game.game_over:
            player.done = True
            player.finish_time = self.time
        race = cfg.race
        if (
            cfg.mode == "race"
            and race is not None
            and race.condition == FIRST_TO_LINES
            and game.lines >= race.target_lines
            and player.goal_time is None
        ):
            player.goal_time = self.time
            player.done = True
            player.finish_time = self.time

    # ------------------------------------------------------------- ending

    def _check_end(self) -> None:
        if self.finished:
            return
        alive = [p for p in self.players.values() if p.alive]
        cfg = self.config
        if cfg.mode == "battle":
            if len(alive) <= 1:
                self._finish()
        elif cfg.mode == "race":
            race = cfg.race
            assert race is not None
            if not alive:
                self._finish()
            elif race.condition == FIRST_TO_LINES:
                if any(p.goal_time is not None for p in self.players.values()):
                    self._finish()
            elif self.time >= race.time_limit:
                self._finish()
        elif not alive:
            self._finish()

    def _finish(self) -> None:
        self.finished = True
        self.paused = False
        self.standings = self._rank()
        self.winner = self.standings[0]["id"] if self.standings else None
        self.events.append({"type": "match_over", "winner": self.winner})

    def _rank(self) -> list[dict[str, Any]]:
        players = list(self.players.values())
        mode = self.config.mode
        seat = {s.player_id: i for i, s in enumerate(self.specs)}

        def key(p: PlayerState) -> tuple:
            g = p.game
            if mode == "battle":
                # Survivors first, then later eliminations ahead of earlier ones.
                return (
                    0 if not g.game_over else 1,
                    -(p.finish_time or 0.0),
                    seat[p.spec.player_id],
                )
            if mode == "race":
                cond = self.config.race.condition
                if cond == FIRST_TO_LINES:
                    reached = p.goal_time is not None
                    return (0 if reached else 1, p.goal_time or 0.0, -g.lines, -g.score)
                if cond == SCORE_IN_TIME:
                    return (-g.score, -g.lines)
                assert cond == LINES_IN_TIME
                return (-g.lines, -g.score)
            return (-g.score, -g.lines)

        ranked = sorted(players, key=key)
        return [{"place": i + 1, **self._result_row(p)} for i, p in enumerate(ranked)]

    def _result_row(self, p: PlayerState) -> dict[str, Any]:
        g = p.game
        active = p.finish_time if p.finish_time is not None else self.time
        return {
            "id": p.spec.player_id,
            "name": p.spec.name,
            "kind": p.spec.kind,
            "lines": g.lines,
            "score": g.score,
            "pieces": g.pieces_placed,
            "attack_sent": g.attack_sent,
            "attack_received": p.attack_received,
            "pps": g.pieces_placed / active if active > 0 else 0.0,
            "topped_out": g.game_over,
            "finish_time": p.finish_time,
        }

    # ----------------------------------------------------------- snapshot

    def get_snapshot(self) -> dict[str, Any]:
        """JSON-serializable view of the match for renderers."""
        snap: dict[str, Any] = {
            "mode": self.config.mode,
            "seed": self.config.seed,
            "time": self.time,
            "countdown": self.countdown,
            "loading": not self.config.lockstep and not self.ready,
            "paused": self.paused,
            "finished": self.finished,
            "winner": self.winner,
            "standings": self.standings,
            "hidden_rows": HIDDEN_ROWS,
            "players": [],
        }
        race = self.config.race
        if self.config.mode == "race" and race is not None:
            snap["race"] = {
                **race.to_dict(),
                "description": race.describe(),
                "time_remaining": (
                    max(0.0, race.time_limit - self.time)
                    if race.condition != FIRST_TO_LINES
                    else None
                ),
            }
        alive, stats = self._alive_map(), self._stats_map()
        targets: dict[str, list[str]] = {}
        if self.config.mode == "battle":
            for pid in self.players:
                targets[pid] = self.router.targets(pid, alive, stats) if alive[pid] else []
        for player in self.players.values():
            snap["players"].append(self._player_snapshot(player, targets))
        return snap

    def _player_snapshot(
        self, player: PlayerState, targets: dict[str, list[str]]
    ) -> dict[str, Any]:
        g = player.game
        pid = player.spec.player_id
        a = g.active
        active = None
        if a is not None and player.alive:
            ghost = g.ghost_row()
            active = {
                "piece": a.piece.letter,
                "cells": [list(c) for c in piece_cells(a.piece, a.rot, a.row, a.col)],
                "ghost": [list(c) for c in piece_cells(a.piece, a.rot, ghost, a.col)],
            }
        driver = player.driver
        row = self._result_row(player)
        return {
            **row,
            "board": board_to_strings(g.board),
            "active": active,
            "hold": g.hold.letter if g.hold else None,
            "can_hold": g.can_hold,
            "queue": [p.letter for p in g.next_queue],
            "level": g.level,
            "combo": g.combo,
            "b2b": g.b2b,
            "pending_garbage": g.pending_total,
            "game_over": g.game_over,
            "done": player.done,
            "thinking": driver.thinking if driver else False,
            "candidates": driver.candidate_cells if driver else [],
            "warning": self.runner.status(pid) if driver and self.runner else None,
            "targeting": self.router.modes[pid] if self.config.mode == "battle" else None,
            "targets": targets.get(pid, []),
            "targeted_by": [o for o, ts in targets.items() if pid in ts],
            "difficulty": player.spec.difficulty.to_dict() if driver else None,
        }
