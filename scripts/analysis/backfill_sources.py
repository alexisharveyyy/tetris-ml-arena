"""Infer leaderboard_sources.json for reports written before evaluate.py recorded sources.

For each section (garbage, plain, marathon) and each agent in that section's
CSV, the source is the agent's current promoted checkpoint, if it passes two
checks:

* it was last modified before the report was written (``--written``), so the
  evaluation could only have read this file; and
* it has not changed since (its sha256 is recorded now, and the UI compares it
  with the promoted file every time it shows the number).

An agent that fails either check gets no entry, so the UI shows "Pending
rerun". Sections replace the ones already in the file; other sections are
kept. Checkpoints are only read.

Usage:
    TETRIS_CHECKPOINT_DIR=checkpoints python scripts/analysis/backfill_sources.py \\
        --written garbage=2026-10-21T03:19:42 plain=2026-10-21T03:59:48 \\
        marathon=2026-10-21T04:36:02 --note "Phase 5 queue, ..."
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from tetris.agents.provenance import SOURCES_FILE, file_sha256, load_sources
from tetris.agents.registry import AGENTS, ROOT, read_checkpoint_dirs


def _relative(path: Path) -> str:
    for d in read_checkpoint_dirs():
        try:
            return str(path.relative_to(d))
        except ValueError:
            continue
    return str(path)


def infer_section(csv_path: Path, written: datetime) -> tuple[dict[str, dict], list[str]]:
    """Source records for the agents in one CSV, and the problems found."""
    with csv_path.open(newline="") as f:
        keys = list(dict.fromkeys(row["key"] for row in csv.DictReader(f)))
    agents: dict[str, dict] = {}
    problems: list[str] = []
    for key in keys:
        spec = AGENTS.get(key)
        if spec is None:
            problems.append(f"{key}: not in the registry")
            continue
        path = spec.checkpoint_path
        if path is None:
            agents[key] = {"checkpoint": None}
            continue
        if not path.exists():
            problems.append(f"{key}: no promoted checkpoint at {path}")
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime)
        if modified >= written:
            problems.append(
                f"{key}: {path} modified {modified:%Y-%m-%d %H:%M:%S}, "
                f"after the report ({written:%Y-%m-%d %H:%M:%S})"
            )
            continue
        agents[key] = {
            "checkpoint": _relative(path),
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
            "modified": f"{modified:%Y-%m-%dT%H:%M:%S}",
        }
    return agents, problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--reports", type=Path, default=ROOT / "reports" / "leaderboard")
    parser.add_argument(
        "--written",
        nargs="+",
        required=True,
        metavar="SECTION=ISO_TIME",
        help="when each section's CSV was written (local time)",
    )
    parser.add_argument("--note", default="", help="where the reports came from")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data = load_sources(args.reports)
    failed = False
    for item in args.written:
        section, _, when = item.partition("=")
        written = datetime.fromisoformat(when)
        agents, problems = infer_section(args.reports / f"leaderboard_{section}.csv", written)
        for p in problems:
            print(f"{section}: NOT RECORDED {p}")
        failed |= bool(problems)
        data.setdefault("sections", {})[section] = {
            "written_at": f"{written:%Y-%m-%dT%H:%M:%S}",
            "basis": (
                f"inferred on {datetime.now():%Y-%m-%d} by scripts/analysis/backfill_sources.py: "
                "every checkpoint listed was last modified before the report was written "
                f"({written:%Y-%m-%d %H:%M:%S}) and has not changed since. {args.note}"
            ).strip(),
            "agents": agents,
        }
        print(f"{section}: {len(agents)} agents recorded ({', '.join(agents)})")
    if args.dry_run:
        print(json.dumps(data, indent=2))
    else:
        (args.reports / SOURCES_FILE).write_text(json.dumps(data, indent=2) + "\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
