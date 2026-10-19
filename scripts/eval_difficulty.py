"""Measure difficulty presets in real-time play under the garbage evaluation.

Each game is a headless lockstep match (real gravity, lock delay, reaction
delay, move rate and speed cap) with the primary garbage setting: one
non-cancellable garbage row every 3 pieces, 500-piece cap.

Usage:
    python scripts/eval_difficulty.py [--agents dellacherie ga cem] [--games 30]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from tetris.agents.difficulty import PRESET_NAMES, PRESETS
from tetris.agents.evaluation import GARBAGE_SETTING, default_workers
from tetris.agents.registry import AGENTS, ROOT
from tetris.match import MatchConfig, MatchController, PlayerSpec
from tetris.match.rules import LINES_IN_TIME, RaceRules

SEED_BASE = 950_000
DT = 1 / 60
MAX_SECONDS = 3600.0


def race_lines(args: tuple[str, str, int]) -> dict:
    """Lines cleared in a 2-minute race without garbage (human-scale reference)."""
    key, preset, seed = args
    match = MatchController(
        MatchConfig(mode="race", seed=seed, lockstep=True, race=RaceRules(LINES_IN_TIME, 120.0)),
        [PlayerSpec("a", key, "agent", key, PRESETS[preset])],
    )
    while not match.finished:
        match.step(DT)
    row = match.get_snapshot()["players"][0]
    match.close()
    return {"agent": key, "preset": preset, "seed": seed, "race_lines": row["lines"]}


def play(args: tuple[str, str, int]) -> dict:
    key, preset, seed = args
    match = MatchController(
        MatchConfig(
            mode="eval",
            seed=seed,
            lockstep=True,
            eval_garbage_every=GARBAGE_SETTING.garbage_every,
            eval_garbage_rows=GARBAGE_SETTING.garbage_rows,
            eval_max_pieces=GARBAGE_SETTING.max_pieces,
        ),
        [PlayerSpec("a", key, "agent", key, PRESETS[preset])],
    )
    while not match.finished and match.time < MAX_SECONDS:
        match.step(DT)
    row = match.get_snapshot()["players"][0]
    match.close()
    return {
        "agent": key,
        "preset": preset,
        "seed": seed,
        **{k: row[k] for k in ("pieces", "lines", "pps", "topped_out")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--agents", nargs="+", default=["dellacherie", "ga", "cem"])
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--out", default=str(ROOT / "reports" / "difficulty.md"))
    args = parser.parse_args()
    tasks = [
        (k, p, SEED_BASE + i) for p in PRESET_NAMES for k in args.agents for i in range(args.games)
    ]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(play, tasks, chunksize=1)
        races = pool.map(race_lines, tasks, chunksize=1)
    lines = [
        "# Difficulty presets",
        "",
        f"Real-time headless matches, {args.games} games per cell, seeds {SEED_BASE}+. Garbage "
        f"evaluation: 1 direct (non-cancellable) garbage row every {GARBAGE_SETTING.garbage_every} "
        f"pieces, {GARBAGE_SETTING.max_pieces}-piece cap, normal gravity curve.",
        "",
        "| Preset | pps cap | Move rate | Reaction | Noise | Epsilon |",
        "|---|---|---|---|---|---|",
    ]
    for name in PRESET_NAMES:
        d = PRESETS[name]
        cap = "none" if d.pps is None else f"{d.pps:g}"
        rate = "instant" if d.move_rate is None else f"{d.move_rate:g}/s"
        reaction = f"{d.reaction_delay * 1000:.0f} ms"
        lines.append(f"| {name} | {cap} | {rate} | {reaction} | {d.noise:g} | {d.epsilon:g} |")
    lines += [
        "",
        "Top-out rate, pieces, lines and measured pieces/s come from the garbage evaluation. "
        "The last column is lines per minute in a separate 2-minute race with no garbage, "
        "for comparison with human play.",
        "",
        "| Preset | Agent | Top-out rate | Pieces | Lines | Measured pps | Race lines/min |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in PRESET_NAMES:
        for key in args.agents:
            sel = [r for r in rows if r["preset"] == name and r["agent"] == key]
            top = np.mean([r["topped_out"] for r in sel])
            pieces = np.mean([r["pieces"] for r in sel])
            lines_cleared = np.mean([r["lines"] for r in sel])
            pps = np.mean([r["pps"] for r in sel])
            race = np.mean(
                [r["race_lines"] for r in races if r["preset"] == name and r["agent"] == key]
            )
            lines.append(
                f"| {name} | {AGENTS[key].name} | {top:.2f} | {pieces:.0f} | "
                f"{lines_cleared:.0f} | {pps:.2f} | {race / 2:.1f} |"
            )
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n{len(rows)} games in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
