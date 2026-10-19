"""Evaluate agents on fixed held-out seeds and write a leaderboard.

The garbage setting is the primary leaderboard; the plain capped setting is a
secondary sanity check.

Usage:
    python scripts/evaluate.py [--agents ga cem dellacherie] [--games 100] [--workers N]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from tetris.agents.evaluation import (
    ASSUMED_PPS,
    GARBAGE_SETTING,
    HELD_OUT_SEEDS,
    PLAIN_SETTING,
    default_workers,
    evaluate_agent,
    paired_bootstrap,
    summarize,
)
from tetris.agents.registry import AGENTS, ROOT, available_agents

COLUMNS = (
    ("mean_lines", "Mean lines", "{:.1f}"),
    ("mean_score", "Mean score", "{:,.0f}"),
    ("mean_pieces", "Pieces survived", "{:.1f}"),
    ("attack_per_minute", "Attack/min", "{:.1f}"),
    ("lines_per_piece", "Lines/piece", "{:.3f}"),
    ("score_per_piece", "Score/piece", "{:.0f}"),
    ("top_out_rate", "Top-out rate", "{:.2f}"),
    ("fitness", "Fitness", "{:.3f}"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--agents", nargs="+", default=None, help="registry keys (default: all available)"
    )
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--out", default=str(ROOT / "reports"))
    args = parser.parse_args()

    keys = args.agents or [a.key for a in available_agents()]
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
    ]
    all_stats: dict[tuple[str, str], list] = {}
    for setting, title in (
        (GARBAGE_SETTING, "Primary: garbage"),
        (PLAIN_SETTING, "Secondary: plain"),
    ):
        rows = []
        for key in keys:
            t0 = time.time()
            stats = evaluate_agent(key, seeds, setting, args.workers)
            all_stats[(setting.name, key)] = stats
            rows.append({"agent": AGENTS[key].name, "key": key, **summarize(stats)})
            print(f"{setting.name:8s} {key:12s} done in {time.time() - t0:.0f}s", flush=True)
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
            for i, a in enumerate(ordered):
                for b in ordered[i + 1 :]:
                    res = paired_bootstrap(
                        all_stats[(setting.name, a)], all_stats[(setting.name, b)]
                    )
                    cells = []
                    for metric in ("fitness", "top_out_rate"):
                        mean, lo, hi = res[metric]
                        mark = "*" if lo > 0 or hi < 0 else ""
                        cells.append(f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]{mark}")
                    md.append(
                        f"| {AGENTS[a].name} minus {AGENTS[b].name} | {cells[0]} | {cells[1]} |"
                    )
            md.append("")
    (out / "leaderboard.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
