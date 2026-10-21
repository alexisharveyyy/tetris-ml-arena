"""Leaderboard data for the web UI, read from ``reports/``.

Nothing here computes agent results: rates, piece counts and Marathon
checkpoints come from the evaluation reports, and agents missing from them are
returned as not evaluated. Every result is checked against the checkpoint that
produced it (:mod:`tetris.agents.provenance`); a result from a buggy, changed
or unrecorded checkpoint is returned as ``pending_rerun`` with no numbers.
Top-out intervals are 95% Wilson score intervals computed from each report's
rate and game count. The significance note is generated from the paired
bootstrap table (``leaderboard_bootstrap.csv``, or the table in
``leaderboard.md`` for reports written before that file existed).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from tetris.agents.evaluation import wilson_interval
from tetris.agents.identity import identity
from tetris.agents.provenance import PENDING, REASON_TEXT, load_sources, result_status

PRESETS = ("Easy", "Medium", "Hard", "Unlimited")
MINUTES = (1, 3, 5, 10)
SCALE_MAX = 0.5
BASELINE = "dellacherie"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(v: str | None) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def _setting(md: str) -> dict[str, int]:
    m = re.search(r"garbage \((\d+) garbage rows? every (\d+) pieces, (\d+)-piece cap\)", md)
    if not m:
        return {}
    return {"garbage_rows": int(m[1]), "garbage_every": int(m[2]), "max_pieces": int(m[3])}


def bootstrap_pairs(reports: Path, name_to_key: dict[str, str]) -> dict[tuple[str, str], tuple]:
    """Top-out rate differences (a minus b): ``{(a, b): (mean, lo, hi)}``."""
    pairs: dict[tuple[str, str], tuple] = {}
    for r in _read_csv(reports / "leaderboard_bootstrap.csv"):
        if r["metric"] == "top_out_rate":
            pairs[(r["a"], r["b"])] = (float(r["mean"]), float(r["lo"]), float(r["hi"]))
    if pairs:
        return pairs
    md = reports / "leaderboard.md"
    if not md.exists():
        return pairs
    row = re.compile(
        r"^\| (.+?) minus (.+?) \| [^|]+ \| ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]\*? \|$"
    )
    for line in md.read_text().splitlines():
        m = row.match(line)
        if m and m[1] in name_to_key and m[2] in name_to_key:
            pairs[(name_to_key[m[1]], name_to_key[m[2]])] = (float(m[3]), float(m[4]), float(m[5]))
    return pairs


def _diff(pairs: dict, a: str, b: str) -> tuple[float, float, float] | None:
    if (a, b) in pairs:
        return pairs[(a, b)]
    if (b, a) in pairs:
        mean, lo, hi = pairs[(b, a)]
        return (-mean, -hi, -lo)
    return None


def _join(names: list[str]) -> str:
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def significance_note(order: list[str], pairs: dict) -> str:
    """Plain sentences from the bootstrap: each agent against the baseline, then
    neighbors in the ranking that are not significantly different."""
    name = {k: identity(k).name for k in order}
    better, worse, same = [], [], []
    if BASELINE in order:
        for k in order:
            if k == BASELINE:
                continue
            d = _diff(pairs, k, BASELINE)
            if d is None:
                continue
            _, lo, hi = d
            (better if hi < 0 else worse if lo > 0 else same).append(name[k])
    parts = []
    if better or worse:
        clauses = []
        if better:
            clauses.append(f"{_join(better)} better than the baseline")
        if worse:
            clauses.append(f"{_join(worse)} worse")
        parts.append("A paired bootstrap on the same seeds finds " + ", and ".join(clauses) + ".")
    if same:
        verb = "is" if len(same) == 1 else "are"
        parts.append(f"{_join(same)} {verb} not significantly different from the baseline.")
    ties = []
    for a, b in zip(order, order[1:], strict=False):
        if BASELINE in (a, b):
            continue
        d = _diff(pairs, a, b)
        if d is not None and d[1] <= 0 <= d[2]:
            ties.append(f"{name[a]} and {name[b]}")
    if ties:
        text = f"{ties[0]} are not significantly different from each other"
        text += "".join(f", nor are {t}" for t in ties[1:])
        parts.append(text + ".")
    return " ".join(parts)


def survival(reports: Path, keys: list[str], name_to_key: dict[str, str]) -> dict[str, Any]:
    """Survival tab: one row per key, evaluated ones first by top-out rate."""
    by_key = {r["key"]: r for r in _read_csv(reports / "leaderboard_garbage.csv")}
    md_path = reports / "leaderboard.md"
    setting = _setting(md_path.read_text()) if md_path.exists() else {}
    sources = load_sources(reports)
    rows = []
    pending = []
    for key in keys:
        r = by_key.get(key)
        if r is None:
            rows.append({"key": key, "evaluated": False, "status": "not_evaluated"})
            continue
        status, code, detail = result_status(sources, "garbage", key)
        if status == PENDING:
            pending.append(
                {
                    "key": key,
                    "evaluated": False,
                    "status": PENDING,
                    "reason": REASON_TEXT[code],
                    "detail": detail,
                }
            )
            continue
        games = int(float(r["games"]))
        rate = float(r["top_out_rate"])
        lo, hi = wilson_interval(round(rate * games), games)
        rows.append(
            {
                "key": key,
                "evaluated": True,
                "status": status,
                "source": detail,
                "games": games,
                "top_out_rate": rate,
                "lo": lo,
                "hi": hi,
                "off_scale": hi > SCALE_MAX,
                "pieces_survived": float(r["mean_pieces"]),
                "score_per_piece": float(r["score_per_piece"]),
            }
        )
    evaluated = sorted((r for r in rows if r["evaluated"]), key=lambda r: r["top_out_rate"])
    rows = evaluated + pending + [r for r in rows if not r["evaluated"]]
    order = [r["key"] for r in evaluated]
    games = sorted({r["games"] for r in evaluated})
    if setting:
        rows_text = (
            "A garbage row arrives"
            if setting["garbage_rows"] == 1
            else f"{setting['garbage_rows']} garbage rows arrive"
        )
        setting_text = (
            f" {rows_text} every {setting['garbage_every']} pieces and cannot be cancelled, "
            f"and games stop at {setting['max_pieces']:,} pieces."
        )
    else:
        setting_text = ""
    games_text = f"{games[0]:,} games each" if len(games) == 1 else "the games listed per agent"
    method = f"Intervals are 95% Wilson score intervals from {games_text}.{setting_text}"
    note = significance_note(order, bootstrap_pairs(reports, name_to_key)) if order else ""
    return {
        "scale_max": SCALE_MAX,
        "setting": setting,
        "method": method,
        "significance": note,
        "rows": rows,
    }


def marathon(reports: Path, keys: list[str]) -> dict[str, Any]:
    """Against people tab: score and lines at 1, 3, 5 and 10 minutes, and time to
    top-out, per agent and preset. Presets not in the report are pending (None)."""
    data: dict[tuple[str, str], dict[str, str]] = {
        (r["key"], r["preset"]): r for r in _read_csv(reports / "leaderboard_marathon.csv")
    }
    sources = load_sources(reports)
    rows = []
    for key in keys:
        if not any((key, p) in data for p in PRESETS):
            rows.append(
                {
                    "key": key,
                    "presets": dict.fromkeys(PRESETS),
                    "evaluated": False,
                    "status": "not_evaluated",
                }
            )
            continue
        status, code, detail = result_status(sources, "marathon", key)
        if status == PENDING:
            rows.append(
                {
                    "key": key,
                    "presets": dict.fromkeys(PRESETS),
                    "evaluated": False,
                    "status": PENDING,
                    "reason": REASON_TEXT[code],
                    "detail": detail,
                }
            )
            continue
        presets = {}
        for preset in PRESETS:
            r = data.get((key, preset))
            if r is None:
                presets[preset] = None
                continue
            cap = _float(r.get("cap_hit_rate")) or 0.0
            presets[preset] = {
                "games": int(float(r["games"])),
                "score": {m: _float(r.get(f"score_{m}min")) for m in MINUTES},
                "lines": {m: _float(r.get(f"lines_{m}min")) for m in MINUTES},
                "alive": {m: _float(r.get(f"alive_{m}min")) for m in MINUTES},
                "time_to_top_out": _float(r.get("mean_survival_s")),
                # Games that hit the piece cap count at the cap, so the mean is a lower bound.
                "time_is_lower_bound": cap > 0,
                "cap_hit_rate": cap,
            }
        rows.append(
            {"key": key, "presets": presets, "evaluated": True, "status": status, "source": detail}
        )
    return {"minutes": list(MINUTES), "presets": list(PRESETS), "rows": rows}
