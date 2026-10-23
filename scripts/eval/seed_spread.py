"""Seed variance of the 1-hour runs: held-out results of each run's selected checkpoint.

Usage: python scripts/eval/seed_spread.py [--agents dqn ppo] [--games 500]
Writes reports/training/seed_spread.md and .json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from tetris.agents.evaluation import GARBAGE_SETTING, HELD_OUT_SEEDS, evaluate_agent, summarize
from tetris.agents.registry import AGENTS, CHECKPOINT_DIR, ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--agents", nargs="+", default=["dqn", "ppo"])
    parser.add_argument("--runs", nargs="+", default=["short0", "short1", "short2"])
    parser.add_argument("--games", type=int, default=500)
    args = parser.parse_args()
    seeds = list(HELD_OUT_SEEDS[: args.games])
    out: dict[str, list[dict]] = {}
    md = [
        "# Seed variance (1-hour runs)",
        "",
        f"Each run's checkpoint chosen on the 30 selection seeds, evaluated on {len(seeds)} "
        "held-out seeds in the garbage setting (1 direct garbage row every 3 pieces, "
        "500-piece cap).",
        "",
        "The short runs have a 1-hour budget, not the main runs' 3 hours. Every schedule is "
        "defined as a fraction of the budget, so it is scaled to 1 hour: epsilon decays over "
        "the first 30% (18 minutes), the DQN learning rate follows the same cosine decay "
        "(1e-4 to 1e-5) over the hour, and the PPO learning rate decays linearly over the "
        "hour (to 5% of its start value). DQN short runs use the v2 config (soft target "
        "update, LR schedule, 2M replay; see reports/training/dqn_overestimation.md).",
        "",
        "| Agent | Run | Selected at | Selection fitness | Held-out fitness | Top-out rate "
        "| Pieces |",
        "|---|---|---|---|---|---|---|",
    ]
    for agent in args.agents:
        out[agent] = []
        for run in args.runs:
            sel_path = CHECKPOINT_DIR / agent / run / "selection.json"
            if not sel_path.exists():
                continue
            sel = json.loads(sel_path.read_text())
            stats = evaluate_agent(
                agent, seeds, GARBAGE_SETTING, checkpoint=sel["selected_checkpoint"]
            )
            m = summarize(stats)
            row = {
                "run": run,
                "selected_minutes": sel["selected_minutes"],
                "selection_fitness": sel["selection_fitness"],
                **m,
            }
            out[agent].append(row)
            md.append(
                f"| {AGENTS[agent].name} | {run} | {row['selected_minutes']:.0f} min | "
                f"{row['selection_fitness']:.3f} | {m['fitness']:.3f} | {m['top_out_rate']:.2f} | "
                f"{m['mean_pieces']:.0f} |"
            )
        fits = [r["fitness"] for r in out[agent]]
        if fits:
            md.append(
                f"| {AGENTS[agent].name} | mean +/- std | | | {np.mean(fits):.3f} +/- "
                f"{np.std(fits, ddof=1) if len(fits) > 1 else 0:.3f} | | |"
            )
    md.append("")
    (ROOT / "reports" / "training" / "seed_spread.json").write_text(json.dumps(out, indent=2))
    (ROOT / "reports" / "training" / "seed_spread.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
