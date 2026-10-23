"""Evaluate agents on fixed held-out seeds and write a leaderboard.

The garbage setting is the primary leaderboard; the plain capped setting is a
secondary sanity check.

Usage:
    python scripts/evaluate.py [--agents ga cem dellacherie] [--games 100] [--workers N]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import time
from pathlib import Path

from tetris.agents.evaluation import (
    ASSUMED_PPS,
    GARBAGE_SETTING,
    HELD_OUT_SEEDS,
    PLAIN_SETTING,
    GameStats,
    default_workers,
    evaluate_agent,
    paired_bootstrap,
    summarize,
)
from tetris.agents.provenance import checkpoint_source, write_sources
from tetris.agents.registry import AGENTS, ROOT, leaderboard_agents
from tetris.match.marathon_eval import MARATHON_MAX_PIECES, evaluate_marathon, summarize_marathon

COLUMNS = (
    ("mean_lines", "Mean lines", "{:.1f}"),
    ("mean_score", "Mean score", "{:,.0f}"),
    ("mean_pieces", "Pieces survived", "{:.1f}"),
    ("attack_per_minute", "Attack/min", "{:.1f}"),
    ("lines_per_piece", "Lines/piece", "{:.3f}"),
    ("score_per_piece", "Score/piece", "{:.0f}"),
    ("top_out_rate", "Top-out rate", "{:.2f}"),
    ("top_out_ci", "Top-out 95% interval (Wilson)", "{}"),
    ("fitness", "Fitness", "{:.3f}"),
)


def record_sources(out: Path, section: str, keys: list[str], before: dict) -> None:
    """Write which checkpoint each result came from; flag any that changed mid-run."""
    sources = {}
    for key in keys:
        src = dict(before[key])
        if checkpoint_source(key) != before[key]:
            src["invalid"] = "the promoted checkpoint changed during the evaluation"
        sources[key] = src
    write_sources(out, section, sources)


def write_games(path: Path, keys: list[str], all_stats: dict, setting: str) -> None:
    """Per-game results, so later comparisons can bootstrap without replaying games."""
    fields = [f.name for f in dataclasses.fields(GameStats)]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", *fields])
        for key in keys:
            for g in all_stats[(setting, key)]:
                writer.writerow([key, *(getattr(g, name) for name in fields)])


def marathon_section(
    keys: list[str], games: int, workers: int, out: Path, presets: list[str]
) -> list[str]:
    """Headless Marathon games per agent and preset (how beatable is each agent?)."""
    t0 = time.time()
    seeds = list(HELD_OUT_SEEDS[:games])
    before = {k: checkpoint_source(k) for k in keys}
    results = evaluate_marathon(keys, presets, seeds, workers)
    record_sources(out, "marathon", keys, before)
    minutes = (time.time() - t0) / 60
    rows = []
    for key in keys:
        for preset in presets:
            sel = [r for r in results if r.agent == key and r.preset == preset]
            rows.append(
                {"agent": AGENTS[key].name, "key": key, "preset": preset, **summarize_marathon(sel)}
            )
    with (out / "leaderboard_marathon.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md = [
        f"## Marathon (no garbage, level gravity, {MARATHON_MAX_PIECES}-piece cap)",
        "",
        f"{games} headless games per agent and preset on held-out seeds {seeds[0]}..{seeds[-1]}, "
        "lockstep with a simulated 60 fps clock, so speed caps, reaction delays, move rates and "
        "mistakes apply exactly as in real-time play. Runtime: "
        f"{minutes:.0f} minutes.",
        "",
        "A Marathon match ends when the human tops out, so the numbers that matter are the "
        "agent's score and lines at the moment you top out. The table gives the agent's mean "
        "score and lines after 1, 3, 5 and 10 minutes of play (after the countdown). An agent "
        "that has already topped out keeps its final score. If you top out at 3 minutes with "
        "more points than the 3-minute score, you beat the average game of that agent.",
        "",
        "At Easy, the heuristic agents (Dellacherie, GA, CEM) and DQN are all limited by the "
        "preset's speed cap and reaction delay rather than by their placement choices, so they "
        "score almost the same.",
        "",
        "Time to top-out is the mean time until the agent tops out. Games that reach the "
        f"{MARATHON_MAX_PIECES}-piece cap count at the time they reached it, so where the "
        "cap-hit rate is above 0 the mean is a lower bound (marked >=).",
        "",
        "| Agent | Preset | Score 1 min | Score 3 min | Score 5 min | Score 10 min "
        "| Lines 1 / 3 / 5 / 10 min | Time to top-out | Cap-hit rate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        survival = r["mean_survival_s"]
        bound = ">= " if r["cap_hit_rate"] > 0 else ""
        lines = " / ".join(f"{r[f'lines_{m}min']:.0f}" for m in (1, 3, 5, 10))
        md.append(
            f"| {r['agent']} | {r['preset']} | "
            + " | ".join(f"{r[f'score_{m}min']:,.0f}" for m in (1, 3, 5, 10))
            + f" | {lines} | {bound}{int(survival // 60)}:{int(survival % 60):02d} "
            f"| {r['cap_hit_rate']:.2f} |"
        )
    md.append("")
    print(f"marathon: {len(results)} games in {minutes:.1f} min", flush=True)
    return md


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--agents", nargs="+", default=None, help="registry keys (default: all available)"
    )
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--out", default=str(ROOT / "reports" / "leaderboard"))
    parser.add_argument(
        "--marathon-games", type=int, default=100, help="0 skips the Marathon section"
    )
    parser.add_argument(
        "--marathon-presets",
        nargs="+",
        default=["Easy", "Medium", "Hard"],
        help="difficulty presets for the Marathon section (Unlimited only measures the cap)",
    )
    parser.add_argument("--skip-garbage", action="store_true", help="only run the Marathon section")
    args = parser.parse_args()

    keys = args.agents or [a.key for a in leaderboard_agents()]
    seeds = list(HELD_OUT_SEEDS[: args.games])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = [
        "# Leaderboard",
        "",
        f"Held-out seeds {seeds[0]}..{seeds[-1]} ({len(seeds)} games per agent and setting). "
        f"Attack/min assumes {ASSUMED_PPS:g} pieces per second.",
        "",
        "Evaluation garbage is inserted directly after every scheduled piece. It does not go "
        "through the pending meter and cannot be cancelled by the agent's attacks, so the "
        "garbage setting measures survival under a fixed load. Placements are instant "
        "(no gravity or speed limits).",
        "",
        "Top-out intervals are 95% Wilson score intervals, which stay inside 0 to 100% and "
        "have nonzero width when every game or no game topped out.",
        "",
    ]
    all_stats: dict[tuple[str, str], list] = {}
    settings = ((GARBAGE_SETTING, "Primary: garbage"), (PLAIN_SETTING, "Secondary: plain"))
    if args.skip_garbage:
        # Keep the existing garbage and plain sections; only the Marathon section is redone.
        settings = ()
        old = out / "leaderboard.md"
        if old.exists():
            md = old.read_text().split("## Marathon")[0].rstrip("\n").split("\n") + [""]
    for setting, title in settings:
        before = {k: checkpoint_source(k) for k in keys}
        rows = []
        for key in keys:
            t0 = time.time()
            stats = evaluate_agent(key, seeds, setting, args.workers)
            all_stats[(setting.name, key)] = stats
            rows.append({"agent": AGENTS[key].name, "key": key, **summarize(stats)})
            print(f"{setting.name:8s} {key:12s} done in {time.time() - t0:.0f}s", flush=True)
        record_sources(out, setting.name, keys, before)
        write_games(out / f"leaderboard_games_{setting.name}.csv", keys, all_stats, setting.name)
        rows.sort(key=lambda r: (-r["fitness"], -r["mean_score"]))
        with (out / f"leaderboard_{setting.name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        desc = (
            f"{setting.garbage_rows} garbage row every {setting.garbage_every} pieces, "
            if setting.garbage_every
            else "no garbage, "
        ) + f"{setting.max_pieces}-piece cap"
        md += [f"## {title} ({desc})", ""]
        md.append("| Rank | Agent | " + " | ".join(c[1] for c in COLUMNS) + " |")
        md.append("|---" * (len(COLUMNS) + 2) + "|")
        for i, r in enumerate(rows, 1):
            r = {**r, "top_out_ci": f"[{r['top_out_lo']:.2f}, {r['top_out_hi']:.2f}]"}
            cells = " | ".join(fmt.format(r[k]) for k, _, fmt in COLUMNS)
            md.append(f"| {i} | {r['agent']} | {cells} |")
        md.append("")
        if setting is GARBAGE_SETTING and len(rows) > 1:
            md += [
                "Paired bootstrap on the same seeds (10,000 resamples), mean difference and 95% "
                "interval. Intervals that exclude 0 are marked *.",
                "",
                "| Comparison | Fitness difference | Top-out rate difference |",
                "|---|---|---|",
            ]
            ordered = [r["key"] for r in rows]
            ties = []
            boot_rows = []
            for i, a in enumerate(ordered):
                for b in ordered[i + 1 :]:
                    res = paired_bootstrap(
                        all_stats[(setting.name, a)], all_stats[(setting.name, b)]
                    )
                    _, lo, hi = res["fitness"]
                    if b == ordered[i + 1] and lo <= 0 <= hi:
                        ties.append(
                            f"{AGENTS[a].name} and {AGENTS[b].name} "
                            f"(interval {lo:+.3f} to {hi:+.3f})"
                        )
                    cells = []
                    for metric in ("fitness", "top_out_rate"):
                        mean, lo, hi = res[metric]
                        boot_rows.append(
                            {"a": a, "b": b, "metric": metric, "mean": mean, "lo": lo, "hi": hi}
                        )
                        mark = "*" if lo > 0 or hi < 0 else ""
                        cells.append(f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]{mark}")
                    md.append(
                        f"| {AGENTS[a].name} minus {AGENTS[b].name} | {cells[0]} | {cells[1]} |"
                    )
            md.append("")
            with (out / "leaderboard_bootstrap.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(boot_rows[0]))
                writer.writeheader()
                writer.writerows(boot_rows)
            if ties:
                md += [
                    "Adjacent ranks whose fitness interval includes 0 are statistically tied; "
                    "their order in the table is not a result: " + "; ".join(ties) + ".",
                    "",
                ]
    if args.marathon_games > 0:
        md += marathon_section(keys, args.marathon_games, args.workers, out, args.marathon_presets)
    (out / "leaderboard.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
