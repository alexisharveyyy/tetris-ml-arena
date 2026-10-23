"""The leaderboard saves per-game results; a rerun is checked row by row."""

from __future__ import annotations

import csv
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "eval"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def test_rerun_is_identical_and_a_changed_row_is_reported(tmp_path) -> None:
    def leaderboard(out: Path) -> None:
        subprocess.run(
            [sys.executable, str(SCRIPTS / "evaluate.py"), "--agents", "dellacherie", "ga",
             "--games", "3", "--workers", "1", "--marathon-games", "1",
             "--marathon-presets", "Easy", "--out", str(out)],
            check=True, capture_output=True,
        )  # fmt: skip

    old, new = tmp_path / "old", tmp_path / "new"
    leaderboard(old)
    games = _rows(old / "leaderboard_games_garbage.csv")
    assert [g["key"] for g in games] == ["dellacherie"] * 3 + ["ga"] * 3
    assert {"seed", "pieces", "lines", "score", "attack", "topped_out"} <= set(games[0])
    leaderboard(new)

    check = _load("check_leaderboard_rerun")
    summary, mismatches = check.compare(old, new, {"ga"})
    assert mismatches == []
    assert "| garbage | 1 | 1 |" in summary and "| bootstrap | 0 | 0 |" in summary

    # A changed row of an unchanged agent is reported.
    path = new / "leaderboard_plain.csv"
    rows = _rows(path)
    for r in rows:
        if r["key"] == "dellacherie":
            r["mean_lines"] = "0.0"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _, mismatches = check.compare(old, new, {"ga"})
    assert len(mismatches) == 1 and "plain ('dellacherie',)" in mismatches[0]
    shutil.rmtree(new)
