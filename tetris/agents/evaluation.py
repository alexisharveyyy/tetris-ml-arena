"""Headless rollouts, evaluation settings, fitness and parallel evaluation.

Shared by the GA and CEM trainers, game logging and the evaluation script.
Rollouts use the engine directly (no observation encoding), applying the same
garbage schedule as :class:`~tetris.env.TetrisEnv`.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tetris.engine.game import Game
from tetris.engine.placement import Placement
from tetris.env.afterstates import action_mask
from tetris.env.tetris_env import inject_scheduled_garbage


@dataclass(frozen=True)
class EvalSetting:
    """A rollout configuration. ``garbage_every=None`` means no garbage."""

    name: str
    max_pieces: int
    garbage_every: int | None = None
    garbage_rows: int = 1
    include_hold: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalSetting:
        return cls(**d)


#: Primary leaderboard and training setting. See the GA README for calibration.
#: Evaluation garbage is inserted directly after the scheduled piece (it does
#: not go through the pending meter and cannot be cancelled by attacks), so the
#: setting measures survival under a fixed incoming load.
GARBAGE_SETTING = EvalSetting("garbage", max_pieces=500, garbage_every=3, garbage_rows=1)
#: Secondary sanity check: no garbage, capped.
PLAIN_SETTING = EvalSetting("plain", max_pieces=1000)

SETTINGS: dict[str, EvalSetting] = {s.name: s for s in (GARBAGE_SETTING, PLAIN_SETTING)}

#: Fixed seeds for final reported numbers. Never used for training.
HELD_OUT_SEEDS: tuple[int, ...] = tuple(range(900_000, 900_500))


@dataclass
class GameStats:
    seed: int
    pieces: int
    lines: int
    score: int
    attack: int
    topped_out: bool
    max_pieces: int

    @property
    def score_per_piece(self) -> float:
        return self.score / max(self.pieces, 1)

    @property
    def lines_per_piece(self) -> float:
        return self.lines / max(self.pieces, 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Policy signature: given a game, return an 80-space action index (None = no move).
Policy = Callable[[Game], int | None]
#: Optional per-step hook: (game before the move, action index) -> None.
StepHook = Callable[[Game, int], None]


def play_game(
    policy: Policy, seed: int, setting: EvalSetting, on_step: StepHook | None = None
) -> GameStats:
    """Play one capped game with a placement policy."""
    game = Game(seed)
    while not game.game_over and game.pieces_placed < setting.max_pieces:
        action = policy(game)
        if action is None:
            game.game_over = True
            break
        if on_step is not None:
            on_step(game, action)
        game.place(Placement.from_index(action))
        inject_scheduled_garbage(game, setting.garbage_every, setting.garbage_rows)
    return GameStats(
        seed=seed,
        pieces=game.pieces_placed,
        lines=game.lines,
        score=game.score,
        attack=game.attack_sent,
        topped_out=game.game_over,
        max_pieces=setting.max_pieces,
    )


# ------------------------------------------------------------------ fitness

#: Score-per-piece normalizer for the fitness efficiency term.
SCORE_PER_PIECE_SCALE = 1000.0
#: Weight of the efficiency term relative to survival.
EFFICIENCY_WEIGHT = 0.25


def fitness(stats: Sequence[GameStats]) -> float:
    """Fitness = mean survival fraction + 0.25 * mean score-per-piece / 1000.

    Survival (pieces / cap) dominates: a game that tops out early loses up to
    1.0. The efficiency term separates agents that all survive by rewarding
    score per piece, which favors multi-line clears (and Tetrises at higher
    levels) over singles. See the GA README for the calibration runs.
    """
    return float(np.mean([game_fitness(s) for s in stats]))


#: Assumed play speed for attack-per-minute in headless runs (placements are instant).
ASSUMED_PPS = 2.0


def game_fitness(s: GameStats) -> float:
    """Per-game fitness contribution (``fitness`` is the mean of these)."""
    return s.pieces / s.max_pieces + EFFICIENCY_WEIGHT * s.score_per_piece / SCORE_PER_PIECE_SCALE


def paired_bootstrap(
    a: Sequence[GameStats], b: Sequence[GameStats], n_boot: int = 10_000, seed: int = 0
) -> dict[str, tuple[float, float, float]]:
    """Mean difference (a - b) and 95% interval for fitness and top-out rate.

    ``a`` and ``b`` must be played on the same seeds in the same order.
    """
    if [s.seed for s in a] != [s.seed for s in b]:
        raise ValueError("paired bootstrap needs the same seeds in the same order")
    rng = np.random.default_rng(seed)
    out = {}
    for name, f in (("fitness", game_fitness), ("top_out_rate", lambda s: float(s.topped_out))):
        d = np.array([f(x) - f(y) for x, y in zip(a, b, strict=True)])
        idx = rng.integers(0, len(d), size=(n_boot, len(d)))
        boots = d[idx].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[name] = (float(d.mean()), float(lo), float(hi))
    return out


def summarize(stats: Sequence[GameStats]) -> dict[str, float]:
    """Aggregate metrics for reports.

    ``attack_per_minute`` assumes the agent plays at :data:`ASSUMED_PPS`.
    """
    minutes = [s.pieces / ASSUMED_PPS / 60.0 for s in stats]
    return {
        "games": len(stats),
        "fitness": fitness(stats),
        "mean_pieces": float(np.mean([s.pieces for s in stats])),
        "mean_lines": float(np.mean([s.lines for s in stats])),
        "mean_score": float(np.mean([s.score for s in stats])),
        "score_per_piece": float(np.mean([s.score_per_piece for s in stats])),
        "lines_per_piece": float(np.mean([s.lines_per_piece for s in stats])),
        "top_out_rate": float(np.mean([s.topped_out for s in stats])),
        "attack_per_minute": float(
            np.mean([s.attack / m if m > 0 else 0.0 for s, m in zip(stats, minutes, strict=True)])
        ),
    }


# ------------------------------------------------------------- parallelism


def _linear_task(args: tuple[tuple[float, ...], int, dict[str, Any]]) -> dict[str, Any]:
    from tetris.agents.linear import best_placement_index

    weights, seed, setting_d = args
    setting = EvalSetting.from_dict(setting_d)
    w = np.asarray(weights)
    stats = play_game(lambda g: best_placement_index(g, w, setting.include_hold), seed, setting)
    return stats.to_dict()


def _agent_task(args: tuple) -> dict[str, Any]:
    """``(key, checkpoint, seed, setting dict[, difficulty dict])``."""
    from tetris.agents.difficulty import DifficultyParams
    from tetris.agents.registry import create_agent

    key, checkpoint, seed, setting_d, *rest = args
    setting = EvalSetting.from_dict(setting_d)
    difficulty = DifficultyParams.from_dict(rest[0]) if rest and rest[0] else None
    agent = create_agent(key, checkpoint, difficulty, seed=seed)

    def policy(game: Game) -> int | None:
        if not action_mask(game, setting.include_hold).any():
            return None
        return agent.select_placement(game).index

    return play_game(policy, seed, setting).to_dict()


def evaluate_agent(
    key: str,
    seeds: Sequence[int],
    setting: EvalSetting,
    workers: int | None = None,
    checkpoint: str | None = None,
    difficulty: Any = None,
) -> list[GameStats]:
    """Play a registry agent on every seed (in worker processes when ``workers`` > 0).

    ``difficulty`` (a :class:`~tetris.agents.difficulty.DifficultyParams`)
    applies mistakes or a target return; speed settings do not affect headless games.
    """
    diff = difficulty.to_dict() if difficulty is not None else None
    tasks = [(key, checkpoint, int(s), setting.to_dict(), diff) for s in seeds]
    workers = default_workers() if workers is None else workers
    if workers > 0:
        with mp.get_context("spawn").Pool(min(workers, len(tasks))) as pool:
            results = pool.map(_agent_task, tasks, chunksize=1)
    else:
        results = [_agent_task(t) for t in tasks]
    return [GameStats(**r) for r in results]


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


class LinearEvaluator:
    """Evaluates linear weight vectors on seed sets across worker processes.

    Uses the ``spawn`` start method (the macOS default) with a persistent pool.
    With ``workers=0`` everything runs in-process (tests, debugging).
    """

    def __init__(self, workers: int | None = None) -> None:
        self.workers = default_workers() if workers is None else workers
        self._pool = mp.get_context("spawn").Pool(self.workers) if self.workers > 0 else None

    def evaluate(
        self, population: Iterable[np.ndarray], seeds: Sequence[int], setting: EvalSetting
    ) -> list[list[GameStats]]:
        """Stats per individual, each played on every seed (common random numbers)."""
        pop = [tuple(float(x) for x in w) for w in population]
        tasks = [(w, int(s), setting.to_dict()) for w in pop for s in seeds]
        if self._pool is None:
            results = [_linear_task(t) for t in tasks]
        else:
            results = self._pool.map(_linear_task, tasks, chunksize=1)
        out: list[list[GameStats]] = []
        n = len(seeds)
        for i in range(len(pop)):
            out.append([GameStats(**r) for r in results[i * n : (i + 1) * n]])
        return out

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> LinearEvaluator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
