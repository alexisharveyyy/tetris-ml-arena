"""Run the remaining Phase 4 training and evaluation steps in order, unattended.

Each step is a separate process; nothing runs in parallel with GPU training.
Progress goes to ``logs/phase4_queue.log`` and each step's own log in ``logs/``.
Steps whose output already exists are skipped, so the script can be rerun
after an interruption.

Usage: python scripts/run_phase4.py [--wait-for-pid PID]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
CKPT = ROOT / "checkpoints"
PY = sys.executable
RAW_CONFIG = "tetris/agents/ppo/config_raw.yaml"
#: Raw PPO tuning candidates (the sanity run used ent 0.01, lr 3e-4 and collapsed).
RAW_TUNING = {"tuneA": {"ent_coef": 0.03, "lr": 3e-4}, "tuneB": {"ent_coef": 0.05, "lr": 1e-4}}
SEED_RUNS = [0, 1, 2]


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with (LOGS / "phase4_queue.log").open("a") as f:
        f.write(line + "\n")


def run(name: str, args: list[str], done_marker: Path | None = None) -> None:
    if done_marker is not None and done_marker.exists():
        log(f"skip {name} (already done: {done_marker.relative_to(ROOT)})")
        return
    log(f"start {name}: {' '.join(args)}")
    t0 = time.time()
    with (LOGS / f"{name}.log").open("w") as out:
        proc = subprocess.run([PY, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        log(f"FAILED {name} (exit {proc.returncode}); see logs/{name}.log")
        raise SystemExit(1)
    log(f"done {name} in {(time.time() - t0) / 60:.1f} min")


def selection_fitness(agent: str, run_name: str) -> float:
    return json.loads((CKPT / agent / run_name / "selection.json").read_text())["selection_fitness"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--wait-for-pid", type=int)
    args = parser.parse_args()
    LOGS.mkdir(exist_ok=True)
    if args.wait_for_pid:
        log(f"waiting for pid {args.wait_for_pid} (DQN main run)")
        while True:
            try:
                os.kill(args.wait_for_pid, 0)
            except OSError:
                break
            time.sleep(30)
    if not (CKPT / "dqn" / "best.pt").exists():
        log("DQN main run did not produce checkpoints/dqn/best.pt; stopping")
        raise SystemExit(1)

    train_ppo = ["-m", "tetris.agents.ppo.train"]
    run(
        "ppo_smoke",
        [*train_ppo, "--budget-minutes", "3", "--run-name", "smoke", "--no-promote"],
        CKPT / "ppo" / "smoke" / "selection.json",
    )

    for name, hp in RAW_TUNING.items():
        run(
            f"ppo_raw_{name}",
            [
                *train_ppo,
                "--config",
                RAW_CONFIG,
                "--budget-minutes",
                "15",
                "--run-name",
                name,
                "--no-promote",
                "--ent-coef",
                str(hp["ent_coef"]),
                "--lr",
                str(hp["lr"]),
            ],
            CKPT / "ppo_raw" / name / "selection.json",
        )
    scores = {name: selection_fitness("ppo_raw", name) for name in RAW_TUNING}
    best = max(scores, key=scores.get)
    log(f"raw PPO tuning: {scores} -> {best} {RAW_TUNING[best]}")
    (CKPT / "ppo_raw").mkdir(parents=True, exist_ok=True)
    (CKPT / "ppo_raw" / "tuning.json").write_text(
        json.dumps(
            {"candidates": RAW_TUNING, "selection_fitness": scores, "chosen": best}, indent=2
        )
    )
    hp = RAW_TUNING[best]
    run(
        "ppo_raw_seed0",
        [
            *train_ppo,
            "--config",
            RAW_CONFIG,
            "--ent-coef",
            str(hp["ent_coef"]),
            "--lr",
            str(hp["lr"]),
        ],
        CKPT / "ppo_raw" / "seed0" / "selection.json",
    )

    run("ppo_seed0", train_ppo, CKPT / "ppo" / "seed0" / "selection.json")

    run("leaderboard", ["scripts/evaluate.py", "--games", "500", "--marathon-games", "100"], None)

    for agent, module in (("dqn", "tetris.agents.dqn.train"), ("ppo", "tetris.agents.ppo.train")):
        for seed in SEED_RUNS:
            name = f"short{seed}"
            run(
                f"{agent}_{name}",
                [
                    "-m",
                    module,
                    "--budget-minutes",
                    "60",
                    "--seed",
                    str(seed),
                    "--run-name",
                    name,
                    "--no-promote",
                ],
                CKPT / agent / name / "selection.json",
            )

    run("seed_spread", ["scripts/seed_spread.py"], None)
    log("phase 4 queue finished")


if __name__ == "__main__":
    main()
