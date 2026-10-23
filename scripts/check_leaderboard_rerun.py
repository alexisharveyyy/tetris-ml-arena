"""Check a leaderboard rerun against the previous leaderboard.

Evaluation is deterministic (fixed seeds, fixed checkpoints), so every row of
an agent whose checkpoint did not change must come out identical. This
compares the garbage, plain and Marathon rows and the bootstrap rows of every
agent except the changed ones, writes ``reports/leaderboard/leaderboard_rerun_check.md``
and prints any mismatch. It never fails the queue: a mismatch is reported,
not acted on.

Usage: python scripts/check_leaderboard_rerun.py OLD_DIR [--changed dt] [--new reports]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def keyed(rows: list[dict[str, str]], fields: tuple[str, ...]) -> dict[tuple, dict[str, str]]:
    return {tuple(r[f] for f in fields): r for r in rows}


def compare(old_dir: Path, new_dir: Path, changed: set[str]) -> tuple[list[str], list[str]]:
    """(summary lines, mismatch lines)."""
    summary, mismatches = [], []
    tables = (
        ("garbage", ("key",), lambda k: k[0] not in changed),
        ("plain", ("key",), lambda k: k[0] not in changed),
        ("marathon", ("key", "preset"), lambda k: k[0] not in changed),
        ("bootstrap", ("a", "b", "metric"), lambda k: not {k[0], k[1]} & changed),
    )
    for name, fields, keep in tables:
        old = keyed(read_rows(old_dir / f"leaderboard_{name}.csv"), fields)
        new = keyed(read_rows(new_dir / f"leaderboard_{name}.csv"), fields)
        same = 0
        for k in sorted(k for k in old if keep(k)):
            if k not in new:
                mismatches.append(f"{name} {k}: missing from the rerun")
            elif new[k] != old[k]:
                diff = {f: (old[k][f], new[k].get(f)) for f in old[k] if old[k][f] != new[k].get(f)}
                mismatches.append(f"{name} {k}: {diff}")
            else:
                same += 1
        summary.append(f"| {name} | {sum(keep(k) for k in old)} | {same} |")
    return summary, mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_dir")
    parser.add_argument("--new", default=str(ROOT / "reports" / "leaderboard"))
    parser.add_argument("--changed", nargs="+", default=["dt"])
    args = parser.parse_args()
    old_dir, new_dir = Path(args.old_dir), Path(args.new)
    summary, mismatches = compare(old_dir, new_dir, set(args.changed))
    md = [
        "# Leaderboard rerun check",
        "",
        f"Rows of every agent except {', '.join(args.changed)} in the rerun "
        f"(`{new_dir.name}/`) compared with the previous leaderboard "
        f"(`{old_dir.relative_to(ROOT) if old_dir.is_relative_to(ROOT) else old_dir}/`). "
        "Evaluation is deterministic, so they should be identical.",
        "",
        "| Table | Rows compared | Identical |",
        "|---|---|---|",
        *summary,
        "",
        "Mismatches: " + ("none." if not mismatches else ""),
        *(f"- {m}" for m in mismatches),
        "",
    ]
    (new_dir / "leaderboard_rerun_check.md").write_text("\n".join(md))
    print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
