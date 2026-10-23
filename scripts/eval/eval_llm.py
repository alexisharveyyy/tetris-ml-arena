"""Evaluate the LLM agent: backends x prompt variants on the same garbage-setting seeds.

Usage:
    python scripts/eval/eval_llm.py [--runs NAME,...] [--games 20] [--max-pieces 100]
                               [--timeout SECONDS] [--dry-run]
    python scripts/eval/eval_llm.py --mark-pending NAME,... --reason TEXT

Runs (defaults to all):
    qwen3-features     Ollama qwen3:8b, placements listed with lines, holes, height (default agent)
    qwen3-bare         Ollama qwen3:8b, rotation and column only
    llama31-features   Ollama llama3.1:8b, with features
    api-features       API model from the LLM config (OpenAI by default), with features
    api-bare           API model, rotation and column only

Every run plays the first ``--games`` held-out seeds, one game at a time,
and the GA heuristic plays the same seeds with the same cap as the paired
reference. The per-decision timeout is ``eval_timeout_seconds`` from the LLM
config (15 s; live play keeps ``timeout_seconds``, 8 s), so the evaluation
measures the model rather than the fallback. Each finished game is appended
to ``data/llm/eval/<run>_cap<pieces>_t<timeout>.jsonl``, so an interrupted
evaluation resumes where it stopped.

A run stops after 5 backend errors in a row (the game in progress is
discarded). API runs need prices checked within the last 7 days
(``prices_checked`` in ``tetris/agents/llm/config.yaml``), make one test call
first and are skipped if it fails, and stop cleanly when the spending cap
would be crossed. Runs without results appear in the report as pending, with
the reason from ``data/llm/eval/status.json`` (``--mark-pending`` sets one).

Writes ``reports/llm/llm_agent.md`` and ``reports/llm/llm_agent.json``.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tetris.agents.evaluation import (
    GARBAGE_SETTING,
    HELD_OUT_SEEDS,
    EvalSetting,
    GameStats,
    evaluate_agent,
    play_game,
)
from tetris.agents.llm.agent import LLMAgent, load_llm_config
from tetris.agents.llm.backends import BackendError, make_backend
from tetris.agents.llm.spending import BudgetExhausted, SpendingLedger
from tetris.agents.registry import ROOT
from tetris.env.afterstates import action_mask

OUT_DIR = ROOT / "data" / "llm" / "eval"
STATUS_PATH = OUT_DIR / "status.json"
PRICE_MAX_AGE_DAYS = 7
#: A run stops after this many backend errors in a row.
ERROR_STREAK_LIMIT = 5
#: Timeout of the one test call made before an API run.
TEST_CALL_TIMEOUT = 30.0
#: name -> (backend, model override or None, prompt variant)
RUNS: dict[str, tuple[str, str | None, str]] = {
    "qwen3-features": ("ollama", "qwen3:8b", "features"),
    "qwen3-bare": ("ollama", "qwen3:8b", "bare"),
    "llama31-features": ("ollama", "llama3.1:8b", "features"),
    "api-features": ("api", None, "features"),
    "api-bare": ("api", None, "bare"),
}
#: Paired comparisons in the report: (a, b) means a minus b.
COMPARISONS = (
    ("qwen3-features", "llama31-features"),
    ("qwen3-features", "qwen3-bare"),
    ("api-features", "api-bare"),
    ("api-features", "qwen3-features"),
)


def run_config(cfg: dict[str, Any], name: str) -> tuple[dict[str, Any], str]:
    """The LLM config for a run, and its label."""
    backend, model, variant = RUNS[name]
    run_cfg = copy.deepcopy(cfg)
    run_cfg["prompt"] = variant
    if backend == "ollama":
        run_cfg["backend"] = "ollama"
        run_cfg["ollama"]["model"] = model
        label = f"Ollama {model}"
    else:
        provider = cfg["backend"] if cfg["backend"] != "ollama" else "openai"
        run_cfg["backend"] = provider
        label = f"{run_cfg[provider]['model']} ({run_cfg[provider]['description']})"
    return run_cfg, f"{label}, {variant}"


def api_note(cfg: dict[str, Any]) -> str:
    """One sentence naming the API model and its class, for the report."""
    provider = cfg["backend"] if cfg["backend"] != "ollama" else "openai"
    section = cfg[provider]
    return f"The API model is {section['model']}, a {section['description']}. "


def check_prices(cfg: dict[str, Any], today: dt.date | None = None) -> None:
    checked = cfg["prices_checked"]
    if isinstance(checked, str):
        checked = dt.date.fromisoformat(checked)
    age = ((today or dt.date.today()) - checked).days
    if age > PRICE_MAX_AGE_DAYS:
        sys.exit(
            f"API prices were checked {age} days ago ({checked}); recheck them and update "
            "prices and prices_checked in tetris/agents/llm/config.yaml first"
        )


def load_done(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class RunStopped(RuntimeError):
    """The run cannot continue (repeated backend errors, failed API test call)."""


def results_path(name: str, max_pieces: int, timeout: float) -> Path:
    return OUT_DIR / f"{name}_cap{max_pieces}_t{timeout:g}.jsonl"


def load_status() -> dict[str, str]:
    """Run name -> why it has no results yet."""
    return json.loads(STATUS_PATH.read_text()) if STATUS_PATH.exists() else {}


def set_status(name: str, reason: str | None) -> None:
    status = load_status()
    if reason is None:
        status.pop(name, None)
    else:
        status[name] = reason
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n")


def play_llm_game(agent: LLMAgent, seed: int, setting: EvalSetting) -> dict[str, Any]:
    before = agent.stats.to_dict()
    first_move = len(agent.stats.moves)

    def policy(game):
        if not action_mask(game, setting.include_hold).any():
            return None
        index = agent.select_placement(game).index
        if agent.stats.error_streak >= ERROR_STREAK_LIMIT:
            raise RunStopped(
                f"stopped after {ERROR_STREAK_LIMIT} backend errors in a row "
                f"(last: {agent.stats.last_error[:200]})"
            )
        return index

    t0 = time.perf_counter()
    stats = play_game(policy, seed, setting)
    after = agent.stats.to_dict()
    diff = {k: after["outcomes"][k] - before["outcomes"][k] for k in after["outcomes"]}
    return {
        **stats.to_dict(),
        "timeout_seconds": agent.timeout,
        "seconds": time.perf_counter() - t0,
        "outcomes": diff,
        "llm_seconds": after["seconds_per_decision"] * after["decisions"]
        - before["seconds_per_decision"] * before["decisions"],
        "input_tokens": after["input_tokens"] - before["input_tokens"],
        "output_tokens": after["output_tokens"] - before["output_tokens"],
        "cost_usd": after["cost_usd"] - before["cost_usd"],
        "move_seconds": [round(t, 3) for _, t in agent.stats.moves[first_move:]],
    }


def warm_up(agent: LLMAgent) -> None:
    """Load a local model before timing (the first request can take many seconds)."""
    if agent.backend.name == "ollama":
        t0 = time.perf_counter()
        agent.backend.complete("Reply with {}.", "{}", timeout=300.0)
        print(f"  loaded {agent.backend.model} in {time.perf_counter() - t0:.1f}s", flush=True)


def api_test_call(agent: LLMAgent) -> None:
    """One minimal request before an API run; raises RunStopped if it fails."""
    try:
        agent.backend.complete(
            'Reply with the JSON {"choice": 1}.', "Choose placement 1.", TEST_CALL_TIMEOUT
        )
    except BudgetExhausted as exc:
        raise RunStopped(f"spending cap reached ({exc})") from exc
    except BackendError as exc:
        if exc.status in (401, 403) or "is not set" in str(exc):
            raise RunStopped("no valid API key") from exc
        raise RunStopped(f"API test call failed: {str(exc)[:200]}") from exc


def evaluate_run(
    cfg: dict[str, Any], name: str, seeds: list[int], setting: EvalSetting, timeout: float
) -> dict[str, Any]:
    run_cfg, label = run_config(cfg, name)
    path = results_path(name, setting.max_pieces, timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(path)
    agent = LLMAgent(seed=0, config=run_cfg)
    agent.configure(backend=make_backend(run_cfg), timeout=timeout, stop_on_budget=True)
    stopped = None
    todo = [s for s in seeds if s not in {g["seed"] for g in done}]
    print(
        f"{name} ({label}, timeout {timeout:g} s): {len(done)} games done, {len(todo)} to play",
        flush=True,
    )
    if todo and RUNS[name][0] == "api":
        try:
            api_test_call(agent)
        except RunStopped as exc:
            print(f"  skipped: {exc}", flush=True)
            if not done:
                set_status(name, str(exc))
            return {"name": name, "label": label, "games": done, "stopped": str(exc)}
    if todo:
        warm_up(agent)
    for seed in todo:
        try:
            game = play_llm_game(agent, seed, setting)
        except (BudgetExhausted, RunStopped) as exc:
            stopped = str(exc)
            print(f"  stopped: {exc}", flush=True)
            break
        done.append(game)
        with path.open("a") as f:
            f.write(json.dumps(game) + "\n")
        o = game["outcomes"]
        print(
            f"  seed {seed}: {game['pieces']} pieces, {game['lines']} lines, "
            f"fallbacks {sum(o.values()) - o['llm']}/{sum(o.values())}, "
            f"{game['seconds'] / max(game['pieces'], 1):.2f} s/piece, ${game['cost_usd']:.4f}",
            flush=True,
        )
    if done:
        set_status(name, None)
    elif stopped:
        set_status(name, stopped)
    return {"name": name, "label": label, "games": done, "stopped": stopped}


def stored_run(cfg: dict[str, Any], name: str, max_pieces: int, timeout: float) -> dict[str, Any]:
    """Results saved by an earlier invocation: the file for ``timeout`` if it has
    games, else the file with the most games."""
    _, label = run_config(cfg, name)
    preferred = load_done(results_path(name, max_pieces, timeout))
    others = [load_done(f) for f in sorted(OUT_DIR.glob(f"{name}_cap{max_pieces}_t*.jsonl"))]
    games = preferred or max(others, key=len, default=[])
    return {"name": name, "label": label, "games": games, "stopped": None}


def _stats(game: dict[str, Any]) -> GameStats:
    return GameStats(**{k: game[k] for k in GameStats.__dataclass_fields__})


#: Main metrics. Fitness is not used: with a 100-piece cap the GA reaches the
#: cap in almost every game, so survival and line efficiency say more.
METRICS: dict[str, tuple[str, Any]] = {
    "pieces": ("Pieces survived", lambda s: float(s.pieces)),
    "top_out": ("Top-out rate", lambda s: float(s.topped_out)),
    "lines_per_piece": ("Lines per piece", lambda s: s.lines_per_piece),
}


def bootstrap_mean(values: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    boots = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def metric_summary(stats: list[GameStats]) -> dict[str, dict[str, float]]:
    """Mean and 95% bootstrap interval (over games) of each main metric."""
    out = {}
    for key, (_, f) in METRICS.items():
        values = [f(s) for s in stats]
        lo, hi = bootstrap_mean(values)
        out[key] = {"mean": float(np.mean(values)), "ci95": [lo, hi]}
    return out


def paired_metrics(
    a: list[GameStats], b: list[GameStats], n: int = 10_000, seed: int = 0
) -> dict[str, list[float]]:
    """Mean difference a - b and 95% interval of each main metric, on the same seeds."""
    if [s.seed for s in a] != [s.seed for s in b]:
        raise ValueError("paired comparison needs the same seeds in the same order")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    out = {}
    for key, (_, f) in METRICS.items():
        d = np.array([f(x) - f(y) for x, y in zip(a, b, strict=True)])
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        out[key] = [float(d.mean()), float(lo), float(hi)]
    return out


def summarize_run(
    run: dict[str, Any], ga: dict[int, GameStats], pending: str | None = None
) -> dict[str, Any]:
    games = run["games"]
    if not games:
        return {
            "name": run["name"],
            "label": run["label"],
            "games": 0,
            "stopped": run["stopped"],
            "pending": pending or run["stopped"] or "not run",
        }
    stats = [_stats(g) for g in games]
    timeouts = sorted({float(g["timeout_seconds"]) for g in games})
    outcomes = {k: sum(g["outcomes"][k] for g in games) for k in games[0]["outcomes"]}
    decisions = sum(outcomes.values())
    pieces = sum(g["pieces"] for g in games)
    return {
        "name": run["name"],
        "label": run["label"],
        "games": len(games),
        "stopped": run["stopped"],
        "timeout_seconds": timeouts[0] if len(timeouts) == 1 else timeouts,
        "metrics": metric_summary(stats),
        "fallback_rate": (decisions - outcomes["llm"]) / max(decisions, 1),
        "fallbacks": {k: v for k, v in outcomes.items() if k != "llm"},
        "seconds_per_piece": sum(g["seconds"] for g in games) / max(pieces, 1),
        "llm_seconds_per_decision": sum(g["llm_seconds"] for g in games) / max(decisions, 1),
        "tokens_per_game": sum(g["input_tokens"] + g["output_tokens"] for g in games) / len(games),
        "input_tokens": sum(g["input_tokens"] for g in games),
        "output_tokens": sum(g["output_tokens"] for g in games),
        "cost_usd": sum(g["cost_usd"] for g in games),
        "minus_ga": paired_metrics(stats, [ga[s.seed] for s in stats]),
        "move_times": move_times(games),
    }


#: Per-decision limits to report the share of slower moves against (live play
#: and the earlier live default).
MOVE_LIMITS = (3.0, 8.0)


def move_times(games: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Per-move wait quantiles and the share over each limit (None if not recorded)."""
    times = [t for g in games for t in g.get("move_seconds", [])]
    if not times or len(times) < sum(g["pieces"] for g in games) // 2:
        return None  # runs from before per-move times were recorded
    arr = np.array(times)
    return {
        "moves": len(times),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "over": {f"{lim:g}": float((arr > lim).mean()) for lim in MOVE_LIMITS},
    }


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any] | None:
    """Paired differences a - b on the seeds both runs finished."""
    sa = {g["seed"]: _stats(g) for g in a["games"]}
    sb = {g["seed"]: _stats(g) for g in b["games"]}
    common = sorted(set(sa) & set(sb))
    if len(common) < 2:
        return None
    diff = paired_metrics([sa[s] for s in common], [sb[s] for s in common])
    return {"a": a["name"], "b": b["name"], "games": len(common), "difference": diff}


def _timeout_label(value: Any) -> str:
    if isinstance(value, list):
        return " / ".join(f"{v:g} s" for v in value)
    return f"{value:g} s"


def _fmt(key: str, value: float, signed: bool = False) -> str:
    sign = "+" if signed else ""
    if key == "pieces":
        return f"{value:{sign}.1f}"
    if key == "top_out":
        return f"{value:{sign}.2f}"
    return f"{value:{sign}.3f}"


def _diff_cells(diff: dict[str, list[float]]) -> str:
    cells = []
    for key in METRICS:
        m, lo, hi = diff[key]
        cells.append(f"{_fmt(key, m, True)} [{_fmt(key, lo, True)}, {_fmt(key, hi, True)}]")
    return " | ".join(cells)


def write_report(
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    ga_stats: list[GameStats],
    setting: EvalSetting,
    spent: dict[str, Any],
    api_note: str = "",
    live_timeout: float = 3.0,
) -> None:
    ga = metric_summary(ga_stats)
    out = {
        "setting": setting.to_dict(),
        "main_metrics": list(METRICS),
        "ga_reference": ga,
        "runs": summaries,
        "comparisons": comparisons,
        "api_ledger": spent,
    }
    reports = ROOT / "reports" / "llm"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "llm_agent.json").write_text(json.dumps(out, indent=2))
    n = len(ga_stats)
    ga_capped = sum(not s.topped_out for s in ga_stats)
    names = [label for label, _ in METRICS.values()]
    lines = [
        "# LLM agent (novelty opponent)",
        "",
        f"Garbage setting (1 row every 3 pieces), {setting.max_pieces}-piece cap, the first "
        f"{n} held-out seeds. The GA heuristic played the same seeds with the same cap as the "
        f"reference; it reached the cap in {ga_capped} of {n} games.",
        "",
        "**Main metrics: pieces survived, top-out rate and lines per piece.** Leaderboard "
        f"fitness is not used here: with a {setting.max_pieces}-piece cap the GA reaches the "
        "cap in almost "
        "every game, so fitness would mostly measure score per piece. Pieces survived and "
        "top-out rate measure whether the model keeps the stack alive under garbage; lines "
        "per piece measures how efficiently it clears.",
        "",
        f"**{n} games per run gives wide intervals.** The 95% intervals below (bootstrap over "
        "games) are often wider than the differences between runs; treat small differences "
        "as ties. The LLM agent is not on the main leaderboard.",
        "",
        "Every decision falls back to the GA heuristic's move on a timeout, an unparseable "
        "or unlisted answer, or a backend error, so a run with many fallbacks partly "
        f"measures the GA. {api_note}The local models are 8-billion-parameter models "
        "quantized to 4 bits. The timeout column is the per-decision limit the run used; "
        f"live play in the app uses {live_timeout:g} s.",
        "",
        f"| Run | Games | Timeout | {' | '.join(f'{x} [95% interval]' for x in names)} "
        "| Fallback rate | Timeouts / invalid / errors | Seconds per piece | Tokens per game "
        "| Cost |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def metric_cells(m: dict[str, dict[str, Any]]) -> str:
        return " | ".join(
            f"{_fmt(k, m[k]['mean'])} [{_fmt(k, m[k]['ci95'][0])}, {_fmt(k, m[k]['ci95'][1])}]"
            for k in METRICS
        )

    lines.append(f"| GA heuristic (reference) | {n} | | {metric_cells(ga)} | | | | | |")
    for s in summaries:
        if not s["games"]:
            pending = f"Pending ({s['pending']})"
            lines.append(f"| {s['label']} | 0 | | {pending} | | | | | | | |")
            continue
        f = s["fallbacks"]
        lines.append(
            f"| {s['label']} | {s['games']} | {_timeout_label(s['timeout_seconds'])} "
            f"| {metric_cells(s['metrics'])} "
            f"| {s['fallback_rate']:.1%} "
            f"| {f.get('timeout', 0)} / {f.get('invalid', 0)} / {f.get('error', 0)} "
            f"| {s['seconds_per_piece']:.2f} | {s['tokens_per_game']:,.0f} "
            f"| ${s['cost_usd']:.2f} |"
        )
    lines += [
        "",
        "Paired differences on the same seeds (mean, 95% bootstrap interval over games):",
        "",
        f"| Comparison | Games | {' | '.join(names)} |",
        "|---|---|---|---|---|",
    ]
    by_name = {s["name"]: s for s in summaries}
    for s in summaries:
        if s["games"]:
            lines.append(f"| {s['label']} minus GA | {s['games']} | {_diff_cells(s['minus_ga'])} |")
    mixed = False
    for c in comparisons:
        ta, tb = by_name[c["a"]]["timeout_seconds"], by_name[c["b"]]["timeout_seconds"]
        mark = ""
        if ta != tb:
            mixed = True
            mark = f" (timeouts {_timeout_label(ta)} vs {_timeout_label(tb)})"
        lines.append(
            f"| {by_name[c['a']]['label']} minus {by_name[c['b']]['label']}{mark} | {c['games']} "
            f"| {_diff_cells(c['difference'])} |"
        )
    if mixed:
        lines += [
            "",
            "Comparisons marked with two timeouts pair runs made with different per-decision "
            "limits; part of the difference can come from the timeout (a shorter limit means "
            "more fallbacks to the GA).",
        ]
    timed = [s for s in summaries if s["games"] and s.get("move_times")]
    if timed:
        limits = list(timed[0]["move_times"]["over"])
        lines += [
            "",
            "Time waited for the model per move (runs that recorded per-move times):",
            "",
            "| Run | Timeout | Moves | Median | 90th percentile | Longest | "
            + " | ".join(f"Share over {lim} s" for lim in limits)
            + " |",
            "|---|---|---|---|---|---|" + "---|" * len(limits),
        ]
        for s in timed:
            t = s["move_times"]
            lines.append(
                f"| {s['label']} | {_timeout_label(s['timeout_seconds'])} | {t['moves']} "
                f"| {t['median']:.2f} s | {t['p90']:.2f} s | {t['max']:.2f} s | "
                + " | ".join(f"{t['over'][lim]:.1%}" for lim in limits)
                + " |"
            )
    stopped = [s for s in summaries if s.get("stopped") and s["games"]]
    if stopped:
        lines += ["", "Runs stopped early (results cover the games finished before the stop):"]
        lines += [f"- {s['label']}: {s['stopped']}" for s in stopped]
    lines += [
        "",
        f"API spending to date: ${spent.get('spent_usd', 0.0):.2f} of the "
        f"${spent.get('cap_usd', 0.0):.2f} cap.",
        "",
    ]
    (reports / "llm_agent.md").write_text("\n".join(lines))


def _run_names(text: str) -> list[str]:
    names = [n.strip() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in RUNS]
    if unknown:
        sys.exit(f"unknown runs: {unknown}; choose from {list(RUNS)}")
    return names


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--runs", default=",".join(RUNS))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--max-pieces", type=int, default=100)
    parser.add_argument("--timeout", type=float, help="per decision (default eval_timeout_seconds)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mark-pending", help="record runs as pending, with --reason, and exit")
    parser.add_argument("--reason", help="why the --mark-pending runs have no results")
    args = parser.parse_args(argv)
    cfg = load_llm_config()
    if args.mark_pending:
        if not args.reason:
            sys.exit("--mark-pending needs --reason")
        for name in _run_names(args.mark_pending):
            set_status(name, args.reason)
        print(f"marked pending: {args.mark_pending} ({args.reason})")
        return
    names = _run_names(args.runs)
    timeout = float(args.timeout or cfg.get("eval_timeout_seconds", cfg["timeout_seconds"]))
    if any(RUNS[n][0] == "api" for n in names):
        check_prices(cfg)
    setting = EvalSetting(**{**GARBAGE_SETTING.to_dict(), "max_pieces": args.max_pieces})
    seeds = list(HELD_OUT_SEEDS[: args.games])
    ledger = SpendingLedger(ROOT / cfg["spending"]["ledger"], float(cfg["spending"]["cap_usd"]))
    print(
        f"runs {names}: {args.games} games each, cap {args.max_pieces} pieces, "
        f"timeout {timeout:g} s; "
        f"API spent so far ${ledger.spent:.2f} of ${ledger.cap_usd:.2f}",
        flush=True,
    )
    if args.dry_run:
        return
    ga_stats = evaluate_agent("ga", seeds, setting, workers=0)
    ga = {s.seed: s for s in ga_stats}
    played = {n: evaluate_run(cfg, n, seeds, setting, timeout) for n in names}
    # The report covers every run: earlier results for runs not played now.
    runs = [played.get(n) or stored_run(cfg, n, args.max_pieces, timeout) for n in RUNS]
    for r in runs:
        r["games"] = [g for g in r["games"] if g["seed"] in ga]
    status = load_status()
    summaries = [summarize_run(r, ga, status.get(r["name"])) for r in runs]
    by_name = {s["name"]: s for s in summaries}
    runs_by_name = {r["name"]: r for r in runs}
    comparisons = [
        c
        for a, b in COMPARISONS
        if by_name[a]["games"] and by_name[b]["games"]
        for c in [compare(runs_by_name[a], runs_by_name[b])]
        if c is not None
    ]
    write_report(
        summaries,
        comparisons,
        ga_stats,
        setting,
        ledger.snapshot(),
        api_note(cfg),
        float(cfg["timeout_seconds"]),
    )
    print("wrote reports/llm/llm_agent.md", flush=True)


if __name__ == "__main__":
    main()
