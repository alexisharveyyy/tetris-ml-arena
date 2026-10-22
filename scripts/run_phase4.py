"""Run the remaining Phase 4 training and evaluation steps in order, unattended.

Order: DQN v2 (3 h) and the v1/v2 choice, PPO smoke, raw PPO tuning, raw PPO
(1 h), afterstate PPO (3 h), leaderboard, DQN and PPO 1-hour seed runs, seed
spread. Each step is a separate process; nothing runs in parallel with GPU
training.
Progress goes to ``logs/phase4_queue.log`` and each step's own log in ``logs/``.
Steps whose output already exists are skipped, so the script can be rerun
after an interruption.

Every step runs through ``scripts/pinned_step.py`` with this checkout first on
``PYTHONPATH``: it imports this checkout's code (never the one an editable
install points at), and it aborts unless this checkout is still at the commit
the queue started from, with ``tetris/`` and ``scripts/`` unmodified.

Usage: python scripts/run_phase4.py [--wait-for-pid PID]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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


_pinned_commit: str | None = None


def pinned_commit() -> str:
    """The commit this queue runs (HEAD of this checkout when first asked)."""
    global _pinned_commit
    if _pinned_commit is None:
        _pinned_commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        log(f"queue pinned to {ROOT} at {_pinned_commit}")
    return _pinned_commit


def step_env() -> dict[str, str]:
    """Environment for a step: this checkout first on the import path, and the pin."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), *filter(None, [env.get("PYTHONPATH")])])
    env["TETRIS_PINNED_ROOT"] = str(ROOT)
    env["TETRIS_PINNED_COMMIT"] = pinned_commit()
    return env


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with (LOGS / "phase4_queue.log").open("a") as f:
        f.write(line + "\n")


def run(name: str, args: list[str], done_marker: Path | None = None) -> None:
    if done_marker is not None and done_marker.exists():
        log(f"skip {name} (already done: {done_marker.relative_to(ROOT)})")
        return
    env = step_env()
    log(f"start {name}: {' '.join(args)}")
    t0 = time.time()
    with (LOGS / f"{name}.log").open("w") as out:
        proc = subprocess.run(
            [PY, str(ROOT / "scripts" / "pinned_step.py"), *args],
            cwd=ROOT,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode == 3:
        log(f"ABORTED {name}: not running the pinned code; see logs/{name}.log")
        raise SystemExit(1)
    if proc.returncode != 0:
        log(f"FAILED {name} (exit {proc.returncode}); see logs/{name}.log")
        raise SystemExit(1)
    log(f"done {name} in {(time.time() - t0) / 60:.1f} min")


def selection_fitness(agent: str, run_name: str) -> float:
    return json.loads((CKPT / agent / run_name / "selection.json").read_text())["selection_fitness"]


def choose_dqn() -> None:
    """Leaderboard DQN: v1 (seed0) unless v2 (seed0_v2) has a strictly higher selection fitness.

    Both checkpoints were chosen on the selection seeds only, so the held-out
    seeds play no part in this choice.
    """
    fits = {run: selection_fitness("dqn", run) for run in ("seed0", "seed0_v2")}
    chosen = "seed0_v2" if fits["seed0_v2"] > fits["seed0"] else "seed0"
    sel = json.loads((CKPT / "dqn" / chosen / "selection.json").read_text())
    shutil.copy(sel["selected_checkpoint"], CKPT / "dqn" / "best.pt")
    (CKPT / "dqn" / "selection.json").write_text(json.dumps(sel, indent=2))
    (CKPT / "dqn" / "version_choice.json").write_text(
        json.dumps({"selection_fitness": fits, "chosen": chosen}, indent=2)
    )
    log(f"leaderboard DQN: {fits} -> {chosen} ({Path(sel['selected_checkpoint']).name})")


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

    # DQN v2: the stability fix (reports/dqn_overestimation.md), fresh 3-hour run.
    run(
        "dqn_v2",
        ["-m", "tetris.agents.dqn.train", "--run-name", "seed0_v2", "--no-promote"],
        CKPT / "dqn" / "seed0_v2" / "selection.json",
    )
    choose_dqn()

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
