"""Rerun afterstate PPO after the MPS gradient fix (Phase 4, option B), unattended.

Order: three 1-hour seed runs, the 3-hour main run (promoted to best.pt), the
leaderboard with Marathon, then the seed spread. Every training run starts
with the MPS pre-flight and runs under the progress watchdog
(tetris/rl/safeguards.py). Meant to run from a git worktree pinned to a commit
(see reports/ppo_mps_bug.md), so the working copy can change meanwhile.

Progress goes to ``logs/phase4_queue.log`` (of the checkout it runs from) and each step's own log in
``logs/``. Steps whose output already exists are skipped on rerun.

Usage: python scripts/run_ppo_rerun.py
"""

from __future__ import annotations

import sys

from run_phase4 import CKPT, LOGS, log, run

TRAIN = ["-m", "tetris.agents.ppo.train"]


def main() -> None:
    LOGS.mkdir(exist_ok=True)
    log("ppo rerun queue: start")
    for seed in (0, 1, 2):
        name = f"short{seed}"
        run(
            f"ppo_{name}",
            [
                *TRAIN,
                "--budget-minutes",
                "60",
                "--seed",
                str(seed),
                "--run-name",
                name,
                "--no-promote",
            ],
            CKPT / "ppo" / name / "selection.json",
        )
    run("ppo_seed0", TRAIN, CKPT / "ppo" / "seed0" / "selection.json")
    run("leaderboard", ["scripts/evaluate.py", "--games", "500", "--marathon-games", "100"], None)
    run("seed_spread", ["scripts/seed_spread.py"], None)
    log("ppo rerun queue finished")


if __name__ == "__main__":
    sys.exit(main())
