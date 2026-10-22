"""Phase 6, second queue: Decision Transformer v2, then the local LLM reruns.

Order: DT v2 trajectory collection (CPU), DT v2 training (2 hours, MPS),
calibration of the model selected in that run, the v1/v2 choice, the DT-only
held-out and Marathon evaluation, then the LLM evaluation of qwen3:8b and
llama3.1:8b with features at the 15 s evaluation timeout (Ollama, GPU). The
API runs are recorded as pending (no valid API key) and not run. Each step is
a separate process, run pinned to this checkout's commit
(``scripts/pinned_step.py``); nothing runs in parallel with GPU work.

Progress goes to ``logs/phase4_queue.log`` (shared helper) and each step's
own log in ``logs/``. Finished steps are skipped on rerun.

Usage: python scripts/run_phase6b.py
"""

from __future__ import annotations

import sys

from run_phase4 import CKPT, LOGS, ROOT, log, pinned_commit, run

DT = "tetris.agents.decision_transformer"
V2 = ["--config", "tetris/agents/decision_transformer/config_v2.yaml", "--run-name", "seed0_v2"]


def main() -> None:
    LOGS.mkdir(exist_ok=True)
    log("phase 6b queue: start")
    pinned_commit()
    run(
        "dt_v2_collect",
        ["-m", f"{DT}.collect", "--config", V2[1]],
        ROOT / "data" / "logs" / "dt_v2" / "manifest.json",
    )
    run(
        "dt_v2_train",
        ["-m", f"{DT}.train", "train", *V2, "--no-promote"],
        CKPT / "dt" / "seed0_v2" / "selection.json",
    )
    run(
        "dt_v2_calibrate",
        [
            "-m",
            f"{DT}.train",
            "calibrate",
            *V2,
            "--report",
            "dt_calibration_v2",
            "--title",
            "Decision Transformer v2",
        ],
        CKPT / "dt" / "seed0_v2" / "calibration.json",
    )
    run("dt_choose", ["-m", f"{DT}.train", "choose", "--run-name", "seed0_v2"], None)
    run("dt_versions_eval", ["scripts/eval_dt_versions.py"], ROOT / "reports" / "dt_v2.md")
    run(
        "llm_mark_api_pending",
        [
            "scripts/eval_llm.py",
            "--mark-pending",
            "api-features,api-bare",
            "--reason",
            "no valid API key",
        ],
        None,
    )
    run(
        "llm_eval_local", ["scripts/eval_llm.py", "--runs", "qwen3-features,llama31-features"], None
    )
    log("phase 6b queue finished")


if __name__ == "__main__":
    sys.exit(main())
