"""Run the Phase 6 Decision Transformer steps in order, unattended.

Order: trajectory collection (CPU), Decision Transformer training (2 hours,
MPS), target-return calibration, then the full leaderboard with Marathon.
Each step is a separate process; nothing runs in parallel with GPU training.
Meant to run from a git worktree pinned to a commit, like run_phase5.py.

The LLM evaluation is not part of this queue: it runs GPU inference (Ollama)
and API calls, and starts separately with ``scripts/eval/eval_llm.py`` once the
Decision Transformer is done and GPU use is approved.

Progress goes to ``logs/phase4_queue.log`` (shared helper) and each step's
own log in ``logs/``. Finished steps are skipped on rerun.

Usage: python scripts/queue/run_phase6.py
"""

from __future__ import annotations

import sys

from run_phase4 import CKPT, LOGS, ROOT, log, run

DT = "tetris.agents.decision_transformer"


def main() -> None:
    LOGS.mkdir(exist_ok=True)
    log("phase 6 queue: start")
    run("dt_collect", ["-m", f"{DT}.collect"], ROOT / "data" / "logs" / "dt" / "manifest.json")
    run("dt_train", ["-m", f"{DT}.train", "train"], CKPT / "dt" / "seed0" / "selection.json")
    run("dt_calibrate", ["-m", f"{DT}.train", "calibrate"], CKPT / "dt" / "calibration.json")
    run(
        "leaderboard",
        ["scripts/eval/evaluate.py", "--games", "500", "--marathon-games", "100"],
        None,
    )
    log("phase 6 queue finished")


if __name__ == "__main__":
    sys.exit(main())
