"""Headless Marathon evaluation: how beatable is each agent at each preset?

Each game is a lockstep Marathon match with one agent and no human: normal
level gravity, no garbage, real speed caps, reaction delays and move rates on
a simulated 60 fps clock, capped at 2,000 pieces.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from tetris.agents.difficulty import PRESETS
from tetris.match.controller import MatchConfig, MatchController, PlayerSpec

MARATHON_MAX_PIECES = 2000
DT = 1 / 60
#: Play times (seconds, after the countdown) at which score and lines are recorded.
#: A Marathon match ends when the human tops out, so these are what a human
#: has to beat, not the agent's final score.
CHECKPOINT_SECONDS = (60, 180, 300, 600)


@dataclass
class MarathonResult:
    agent: str
    preset: str
    seed: int
    score: int
    lines: int
    level: int
    pieces: int
    tetrises: int
    survival_time: float
    hit_cap: bool
    #: Score and lines at each of CHECKPOINT_SECONDS (final values if the
    #: game ended earlier).
    checkpoint_scores: list[int] = field(default_factory=list)
    checkpoint_lines: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def play_marathon(
    agent_key: str, preset: str, seed: int, max_pieces: int = MARATHON_MAX_PIECES
) -> MarathonResult:
    match = MatchController(
        MatchConfig(mode="marathon", seed=seed, lockstep=True, eval_max_pieces=max_pieces),
        [PlayerSpec("a", agent_key, "agent", agent_key, PRESETS[preset], preset=preset)],
    )
    game = match.players["a"].game
    scores: list[int] = []
    lines: list[int] = []
    while not match.finished:
        match.step(DT)
        while (
            len(scores) < len(CHECKPOINT_SECONDS) and match.time >= CHECKPOINT_SECONDS[len(scores)]
        ):
            scores.append(game.score)
            lines.append(game.lines)
    row = match.standings[0]
    scores += [row["score"]] * (len(CHECKPOINT_SECONDS) - len(scores))
    lines += [row["lines"]] * (len(CHECKPOINT_SECONDS) - len(lines))
    match.close()
    return MarathonResult(
        agent_key,
        preset,
        seed,
        row["score"],
        row["lines"],
        row["level"],
        row["pieces"],
        row["tetrises"],
        row["survival_time"],
        not row["topped_out"] and row["pieces"] >= max_pieces,
        scores,
        lines,
    )


def _task(args: tuple[str, str, int, int]) -> dict[str, Any]:
    return play_marathon(*args).to_dict()


def evaluate_marathon(
    agents: Sequence[str],
    presets: Sequence[str],
    seeds: Sequence[int],
    workers: int,
    max_pieces: int = MARATHON_MAX_PIECES,
) -> list[MarathonResult]:
    """All (agent, preset, seed) games, longest-expected first for load balance."""
    order = {"Unlimited": 0, "Hard": 1, "Medium": 2, "Easy": 3}
    tasks = [
        (a, p, int(s), max_pieces)
        for p in sorted(presets, key=lambda x: order.get(x, 9), reverse=True)
        for a in agents
        for s in seeds
    ]
    if workers > 0:
        with mp.get_context("spawn").Pool(workers) as pool:
            rows = pool.map(_task, tasks, chunksize=1)
    else:
        rows = [_task(t) for t in tasks]
    return [MarathonResult(**r) for r in rows]


def summarize_marathon(results: Sequence[MarathonResult]) -> dict[str, float]:
    return {
        "games": len(results),
        "mean_score": float(np.mean([r.score for r in results])),
        "mean_lines": float(np.mean([r.lines for r in results])),
        "mean_level": float(np.mean([r.level for r in results])),
        "mean_pieces": float(np.mean([r.pieces for r in results])),
        "cap_hit_rate": float(np.mean([r.hit_cap for r in results])),
        "mean_survival_s": float(np.mean([r.survival_time for r in results])),
        **{
            f"score_{t // 60}min": float(np.mean([r.checkpoint_scores[k] for r in results]))
            for k, t in enumerate(CHECKPOINT_SECONDS)
        },
        **{
            f"lines_{t // 60}min": float(np.mean([r.checkpoint_lines[k] for r in results]))
            for k, t in enumerate(CHECKPOINT_SECONDS)
        },
        **{
            f"alive_{t // 60}min": float(np.mean([r.survival_time >= t for r in results]))
            for k, t in enumerate(CHECKPOINT_SECONDS)
        },
    }
