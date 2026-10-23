"""Compare Decision Transformer v1 and v2: data, selection, calibration, held-out play.

Usage: python scripts/eval/eval_dt_versions.py [--games 500] [--marathon-games 100]

Run after ``train choose`` (``checkpoints/dt/versions/v1`` and ``v2``). Plays
both versions, each at its calibrated default target, on the held-out seeds
in the garbage and plain settings (the same seeds and settings as
``scripts/eval/evaluate.py``), with the paired bootstrap of v2 minus v1, and v2 in
the Marathon section. Other agents are not replayed: v2 is placed against their
results in ``reports/leaderboard/leaderboard_*.csv``. v1's replay should reproduce its
leaderboard row exactly (play is deterministic per seed); the report checks it.

Writes ``reports/analysis/dt_v2.md``, ``reports/analysis/dt_v1_vs_v2_calibration.png`` and
``reports/analysis/dt_versions.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from tetris.agents.decision_transformer.collect import sources
from tetris.agents.decision_transformer.train import version_dir
from tetris.agents.evaluation import (
    GARBAGE_SETTING,
    HELD_OUT_SEEDS,
    PLAIN_SETTING,
    default_workers,
    evaluate_agent,
    fitness,
    paired_bootstrap,
    summarize,
)
from tetris.agents.registry import CHECKPOINT_DIR, CHECKPOINT_ENV, ROOT
from tetris.match.marathon_eval import evaluate_marathon, summarize_marathon
from tetris.rl.training import load_config

DT_DIR = ROOT / "tetris" / "agents" / "decision_transformer"
CONFIGS = {"v1": DT_DIR / "config.yaml", "v2": DT_DIR / "config_v2.yaml"}
VERSIONS = ("v1", "v2")
PRESETS = ("Easy", "Medium", "Hard")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def leaderboard_rows(setting: str) -> list[dict[str, Any]]:
    with (ROOT / "reports" / "leaderboard" / f"leaderboard_{setting}.csv").open() as f:
        return list(csv.DictReader(f))


def marathon_v2(games: int, workers: int) -> list[dict[str, Any]]:
    """v2 in the Marathon section, read through the registry from its version folder."""
    old = os.environ.get(CHECKPOINT_ENV)
    os.environ[CHECKPOINT_ENV] = str(version_dir("v2").parent)
    try:
        results = evaluate_marathon(["dt"], list(PRESETS), list(HELD_OUT_SEEDS[:games]), workers)
    finally:
        if old is None:
            os.environ.pop(CHECKPOINT_ENV, None)
        else:
            os.environ[CHECKPOINT_ENV] = old
    return [
        {"preset": p, **summarize_marathon([r for r in results if r.preset == p])} for p in PRESETS
    ]


def plot_calibration(cal: dict[str, dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for v, c in cal.items():
        t = [r["target"] for r in c["rows"]]
        ax.plot(
            t,
            [r["achieved_fitness"] for r in c["rows"]],
            marker="o",
            label=f"{v} (rank correlation {c['spearman']:+.2f})",
        )
    t = [r["target"] for r in cal["v1"]["rows"]]
    ax.plot(t, t, linestyle="--", color="gray", label="achieved = target")
    ax.set_xlabel("target return (fitness units)")
    ax.set_ylabel(f"achieved fitness ({cal['v1']['games_per_target']} selection seeds)")
    ax.set_title("Decision Transformer v1 and v2: target return vs achieved")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def data_table(version: str) -> list[str]:
    cfg = load_config(str(CONFIGS[version]))
    manifest = read_json(ROOT / cfg["collect"]["out_dir"] / "manifest.json")
    srcs = {s.name: s for s in sources(cfg)}
    lines = [
        f"{version}: {manifest['total_games']} games, {manifest['total_states']:,} states.",
        "",
        "| Source | Checkpoint | Mistake rate | Games | States | Mean return |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in manifest["sources"].items():
        src = srcs[name]
        ckpt = Path(src.checkpoint).name if src.checkpoint else "promoted"
        lines.append(
            f"| {name} | {ckpt} | {src.noise:g} | {s['games']} | {s['states']:,} "
            f"| {s['mean_fitness']:.3f} |"
        )
    return [*lines, ""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--marathon-games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=default_workers())
    args = parser.parse_args()
    t0 = time.time()
    seeds = list(HELD_OUT_SEEDS[: args.games])
    cal = {v: read_json(version_dir(v) / "calibration.json") for v in VERSIONS}
    sel = {v: read_json(version_dir(v) / "selection.json") for v in VERSIONS}
    choice = read_json(CHECKPOINT_DIR / "dt" / "version_choice.json")

    held: dict[str, dict[str, Any]] = {}
    for setting in (GARBAGE_SETTING, PLAIN_SETTING):
        stats = {}
        for v in VERSIONS:
            stats[v] = evaluate_agent(
                "dt", seeds, setting, args.workers, str(version_dir(v) / "best.pt")
            )
            print(f"{setting.name:8s} {v}: fitness {fitness(stats[v]):.3f}", flush=True)
        board = leaderboard_rows(setting.name)
        recorded = next(float(r["fitness"]) for r in board if r["key"] == "dt")
        others = sorted(
            [(r["agent"], float(r["fitness"])) for r in board if r["key"] != "dt"],
            key=lambda x: -x[1],
        )
        v2_fit = fitness(stats["v2"])
        held[setting.name] = {
            v: {**summarize(stats[v]), "fitness": fitness(stats[v])} for v in VERSIONS
        }
        held[setting.name]["v1_leaderboard_fitness"] = recorded
        held[setting.name]["v2_minus_v1"] = paired_bootstrap(stats["v2"], stats["v1"])
        held[setting.name]["v2_rank"] = 1 + sum(f > v2_fit for _, f in others)
        held[setting.name]["agents"] = len(others) + 1
    marathon = marathon_v2(args.marathon_games, args.workers) if args.marathon_games else []
    v1_marathon = [r for r in leaderboard_rows("marathon") if r["key"] == "dt"]

    reports = ROOT / "reports" / "analysis"
    reports.mkdir(parents=True, exist_ok=True)
    plot_calibration(cal, reports / "dt_v1_vs_v2_calibration.png")
    (reports / "dt_versions.json").write_text(
        json.dumps(
            {
                "choice": choice,
                "calibration": cal,
                "held_out": held,
                "marathon_v2": marathon,
                "seeds": [seeds[0], seeds[-1]],
            },
            indent=2,
        )
    )

    md = [
        "# Decision Transformer v2: does skill the board shows make the target return work?",
        "",
        "DT v1's target return barely changed its play (rank correlation +0.43 between target "
        "and achieved fitness; asking for 0.2 still played at 1.32). Hypothesis: v1's weak "
        "games came from injected mistakes, which the board does not predict, so the model "
        "learned to imitate the strong placements and ignore the return-to-go. v2 keeps the "
        "model, training budget and selection protocol and changes only the data: no injected "
        "mistakes; instead each trained agent plays at three stages of its training (first "
        "evaluated checkpoint, the checkpoint with selection fitness closest to halfway, and "
        "the selected one), plus BC (you) and raw-action PPO.",
        "",
        "## Data",
        "",
        *data_table("v1"),
        *data_table("v2"),
        "## Selection (30 selection seeds, at each version's selection target)",
        "",
        "| Version | Selection target | Checkpoints evaluated | Selection fitness range "
        "| Selected | Selection fitness |",
        "|---|---|---|---|---|---|",
    ]
    for v in VERSIONS:
        fits = [e["fitness"] for e in sel[v]["evaluations"]]
        md.append(
            f"| {v} | {sel[v]['selection_target']:.3f} | {len(fits)} "
            f"| {min(fits):.3f} to {max(fits):.3f} | {Path(sel[v]['selected_checkpoint']).name} "
            f"| {sel[v]['selection_fitness']:.3f} |"
        )
    md += [
        "",
        f"Playable DT: **{choice['chosen']}** (v2 only if its selection fitness is strictly "
        "higher; both were chosen on the selection seeds, so the held-out results below play "
        "no part in the choice).",
        "",
        "## Calibration",
        "",
        f"Each version at each target on {cal['v1']['games_per_target']} selection seeds.",
        "",
        "| Target | " + " | ".join(f"{r['target']:.2f}" for r in cal["v1"]["rows"]) + " |",
        "|---" * (len(cal["v1"]["rows"]) + 1) + "|",
    ]
    for v in VERSIONS:
        md.append(
            f"| {v} achieved | "
            + " | ".join(f"{r['achieved_fitness']:.3f}" for r in cal[v]["rows"])
            + " |"
        )
    md += [
        "",
        "| Version | Rank correlation (target, achieved) | Lowest achieved | Highest achieved "
        "| Default target |",
        "|---|---|---|---|---|",
    ]
    for v in VERSIONS:
        ach = [r["achieved_fitness"] for r in cal[v]["rows"]]
        md.append(
            f"| {v} | {cal[v]['spearman']:+.2f} | {min(ach):.3f} | {max(ach):.3f} "
            f"| {cal[v]['leaderboard_target']:.2f} |"
        )
    md += ["", "![calibration](dt_v1_vs_v2_calibration.png)", ""]
    md += [
        "## Held-out play (DT only)",
        "",
        f"Held-out seeds {seeds[0]}..{seeds[-1]}, each version at its default target. The "
        "other agents were not replayed; v2's rank is against their leaderboard results "
        "(`reports/leaderboard/leaderboard.md`).",
        "",
        "| Setting | Version | Fitness | Top-out rate [95% Wilson] | Pieces survived "
        "| Lines/piece | Rank |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, h in held.items():
        for v in VERSIONS:
            s = h[v]
            rank = f"{h['v2_rank']} of {h['agents']}" if v == "v2" else "(leaderboard row)"
            md.append(
                f"| {name} | {v} | {s['fitness']:.3f} | {s['top_out_rate']:.2f} "
                f"[{s['top_out_lo']:.2f}, {s['top_out_hi']:.2f}] | {s['mean_pieces']:.1f} "
                f"| {s['lines_per_piece']:.3f} | {rank} |"
            )
    md += [
        "",
        "| Setting | v2 minus v1: fitness | v2 minus v1: top-out rate | v1 replay matches "
        "the leaderboard |",
        "|---|---|---|---|",
    ]
    for name, h in held.items():
        cells = []
        for metric in ("fitness", "top_out_rate"):
            m, lo, hi = h["v2_minus_v1"][metric]
            mark = "*" if lo > 0 or hi < 0 else ""
            cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]{mark}")
        same = abs(h["v1"]["fitness"] - h["v1_leaderboard_fitness"]) < 1e-9
        check = "yes" if same else f"no ({h['v1_leaderboard_fitness']:.3f} on the leaderboard)"
        md.append(f"| {name} | {cells[0]} | {cells[1]} | {check} |")
    md += ["", "Paired bootstrap (10,000 resamples); intervals that exclude 0 are marked *.", ""]
    if marathon:
        md += [
            f"## Marathon ({args.marathon_games} games per preset, held-out seeds)",
            "",
            "| Version | Preset | Score 1 min | Score 3 min | Score 5 min | Score 10 min "
            "| Cap-hit rate |",
            "|---|---|---|---|---|---|---|",
        ]
        rows = [("v1", r) for r in v1_marathon] + [("v2", r) for r in marathon]
        for v, r in rows:
            scores = " | ".join(f"{float(r[f'score_{m}min']):,.0f}" for m in (1, 3, 5, 10))
            md.append(f"| {v} | {r['preset']} | {scores} | {float(r['cap_hit_rate']):.2f} |")
        md.append("")
    md.append(f"Runtime: {(time.time() - t0) / 60:.0f} minutes.")
    md.append("")
    (reports / "dt_v2.md").write_text("\n".join(md))
    print("wrote reports/analysis/dt_v2.md", flush=True)


if __name__ == "__main__":
    main()
