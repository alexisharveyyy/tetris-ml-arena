"""Leaderboard data for the web UI, read from ``reports/``.

Nothing here computes agent results: rates, piece counts and Marathon
checkpoints come from the evaluation reports, and agents missing from them are
returned as not evaluated. Every result is checked against the checkpoint that
produced it (:mod:`tetris.agents.provenance`); a result from a buggy, changed
or unrecorded checkpoint is returned as ``pending_rerun`` with no numbers.
Top-out intervals are 95% Wilson score intervals computed from each report's
rate and game count. The significance note (tied groups, significant gaps,
comparisons with the baseline) and the full comparison table come from the
paired bootstrap (``leaderboard_bootstrap.csv``, or the table in
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


METRICS = ("fitness", "top_out_rate")
_INTERVAL = r"([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]\*?"


def bootstrap_table(reports: Path, name_to_key: dict[str, str]) -> dict[tuple[str, str], dict]:
    """Paired bootstrap differences (a minus b): ``{(a, b): {metric: (mean, lo, hi)}}``."""
    table: dict[tuple[str, str], dict] = {}
    for r in _read_csv(reports / "leaderboard_bootstrap.csv"):
        if r["metric"] in METRICS:
            table.setdefault((r["a"], r["b"]), {})[r["metric"]] = (
                float(r["mean"]),
                float(r["lo"]),
                float(r["hi"]),
            )
    if table:
        return table
    md = reports / "leaderboard.md"
    if not md.exists():
        return table
    row = re.compile(rf"^\| (.+?) minus (.+?) \| {_INTERVAL} \| {_INTERVAL} \|$")
    for line in md.read_text().splitlines():
        m = row.match(line)
        if m and m[1] in name_to_key and m[2] in name_to_key:
            values = [float(v) for v in m.groups()[2:]]
            table[(name_to_key[m[1]], name_to_key[m[2]])] = {
                "fitness": tuple(values[:3]),
                "top_out_rate": tuple(values[3:]),
            }
    return table


def _diff(pairs: dict, a: str, b: str) -> tuple[float, float, float] | None:
    if (a, b) in pairs:
        return pairs[(a, b)]
    if (b, a) in pairs:
        mean, lo, hi = pairs[(b, a)]
        return (-mean, -hi, -lo)
    return None


def _tied(pairs: dict, a: str, b: str) -> bool:
    d = _diff(pairs, a, b)
    return d is not None and d[1] <= 0 <= d[2]


def _join(names: list[str]) -> str:
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + ", and " + names[-1]


def tie_groups(order: list[str], pairs: dict) -> list[list[str]]:
    """Runs of neighbors in the ranking in which every pair is tied (the top-out
    rate interval includes 0). A pair missing from the bootstrap ends a run."""
    groups: list[list[str]] = []
    for k in order:
        if groups and all(_tied(pairs, k, g) for g in groups[-1]):
            groups[-1].append(k)
        else:
            groups.append([k])
    return groups


def significance_note(order: list[str], pairs: dict) -> list[dict[str, str]]:
    """The bootstrap in three labelled lines: groups that are statistically tied,
    each significant gap between neighbors in the ranking, and each agent
    against the baseline. Lines with nothing to say are left out."""
    name = {k: identity(k).name for k in order}
    lines = []
    groups = tie_groups(order, pairs)
    tied = [g for g in groups if len(g) > 1]
    if tied:
        text = "; ".join(_join([name[k] for k in g]) for g in tied) + "."
        # A run ends when the next agent differs from someone in it, not
        # necessarily from its neighbor, so say where groups still overlap.
        for left, right in zip(groups, groups[1:], strict=False):
            if _tied(pairs, left[-1], right[0]):
                text += f" {name[left[-1]]} is also tied with {name[right[0]]}."
        lines.append({"label": "Statistically tied", "text": text})
    gaps = []
    for a, b in zip(order, order[1:], strict=False):
        d = _diff(pairs, a, b)
        if d is None or _tied(pairs, a, b):
            continue
        how = "less" if d[0] < 0 else "more"
        gaps.append(f"{name[a]} tops out {how} often than {name[b]}")
    if gaps:
        lines.append({"label": "Significant gaps", "text": "; ".join(gaps) + "."})
    if BASELINE in order:
        better, worse, same = [], [], []
        for k in order:
            d = None if k == BASELINE else _diff(pairs, k, BASELINE)
            if d is None:
                continue
            (better if d[2] < 0 else worse if d[1] > 0 else same).append(name[k])
        clauses = []
        if better:
            clauses.append(f"{_join(better)} better")
        if worse:
            clauses.append(f"{_join(worse)} worse")
        if same:
            clauses.append(f"{_join(same)} not significantly different")
        if clauses:
            lines.append(
                {
                    "label": f"Against the baseline ({name[BASELINE]})",
                    "text": "; ".join(clauses) + ".",
                }
            )
    return lines


def comparisons(order: list[str], table: dict) -> list[dict[str, Any]]:
    """Every pair of ranked agents, in ranking order, for the full table."""
    by_metric = {m: {pair: v[m] for pair, v in table.items() if m in v} for m in METRICS}
    rows = []
    for i, a in enumerate(order):
        for b in order[i + 1 :]:
            entry: dict[str, Any] = {"a": a, "b": b}
            for metric, pairs in by_metric.items():
                d = _diff(pairs, a, b)
                entry[metric] = (
                    None
                    if d is None
                    else {
                        "mean": d[0],
                        "lo": d[1],
                        "hi": d[2],
                        "significant": not _tied(pairs, a, b),
                    }
                )
            if any(entry[m] for m in METRICS):
                rows.append(entry)
    return rows


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
    table = bootstrap_table(reports, name_to_key) if order else {}
    pairs = {k: v["top_out_rate"] for k, v in table.items() if "top_out_rate" in v}
    if pairs:
        method += " Ties and gaps come from a paired bootstrap on the same seeds."
    return {
        "scale_max": SCALE_MAX,
        "setting": setting,
        "method": method,
        "significance": significance_note(order, pairs),
        "comparisons": comparisons(order, table),
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
