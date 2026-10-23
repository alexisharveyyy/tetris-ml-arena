"""Run the Phase 5 training and evaluation steps in order, unattended.

Order: pure BC (30 min, MPS), BC + DAgger (2 rounds), the Imitation (BC)
version choice, BC (you) (skipped below 800 on-space placements or 3 games),
NEAT (3 hours, CPU), then the full leaderboard with Marathon. Each step is a
separate process; nothing runs in parallel with GPU training. Meant to run
from a git worktree pinned to a commit, like scripts/queue/run_ppo_rerun.py.

Progress goes to ``logs/phase4_queue.log`` (shared helper; of the checkout it
runs from) and each step's own log in ``logs/``. Finished steps are skipped
on rerun.

Usage: python scripts/queue/run_phase5.py
"""

from __future__ import annotations

import sys

from run_phase4 import CKPT, LOGS, log, run

BC = ["-m", "tetris.agents.imitation.train"]


def main() -> None:
    LOGS.mkdir(exist_ok=True)
    log("phase 5 queue: start")
    run("bc_pure", [*BC, "pure"], CKPT / "bc_pure" / "seed0" / "selection.json")
    run("bc_dagger", [*BC, "dagger"], CKPT / "bc_dagger" / "seed0" / "selection.json")
    run("bc_choose", [*BC, "choose"], None)
    run("bc_you", [*BC, "you"], None)
    run(
        "neat",
        ["-m", "tetris.agents.neuroevolution.train"],
        CKPT / "neat" / "seed0" / "selection.json",
    )
    run(
        "leaderboard",
        ["scripts/eval/evaluate.py", "--games", "500", "--marathon-games", "100"],
        None,
    )
    log("phase 5 queue finished")


if __name__ == "__main__":
    sys.exit(main())
