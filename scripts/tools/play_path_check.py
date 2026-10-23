"""Play-path check for the learned agents: worker startup, decision latency, modes, fallback.

Runs real-time matches (not lockstep) with persistent worker processes, as the
desktop app does, without opening a window:

1. Startup: time from starting the workers until every worker has loaded its
   model and answered "ready".
2. Latency: per-decision think time inside the worker (CPU, one thread) and
   the round trip from submit to the decision being picked up by the match.
3. Modes: Watch AI, Race and Marathon, each for ``--seconds`` of real time at
   60 fps, checking that every agent keeps placing pieces with no fallback.
4. Fallback: a DQN worker that hangs (paused with SIGSTOP) must hit its 2 s
   decision timeout, and one that crashes (SIGKILL) must be detected; both
   switch to the in-process Dellacherie agent, show a warning and keep playing.

Usage: python scripts/tools/play_path_check.py [--agents dqn ppo ppo_raw] [--seconds 20]
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from collections import defaultdict

import numpy as np

from tetris.agents.difficulty import PRESETS
from tetris.match.agent_runtime import ProcessRunner
from tetris.match.controller import MatchConfig, MatchController, PlayerSpec
from tetris.match.rules import RaceRules

DT = 1 / 60


class TimedRunner(ProcessRunner):
    """ProcessRunner that records think time and round-trip latency per decision."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.submitted: dict[str, float] = {}
        self.think: dict[str, list[float]] = defaultdict(list)
        self.round_trip: dict[str, list[float]] = defaultdict(list)

    def submit(self, player_id, request_id, state, top_k) -> None:
        self.submitted[player_id] = time.perf_counter()
        super().submit(player_id, request_id, state, top_k)

    def poll(self, player_id):
        decision = super().poll(player_id)
        if decision is not None and player_id in self.submitted:
            self.think[player_id].append(decision.think_time)
            self.round_trip[player_id].append(time.perf_counter() - self.submitted.pop(player_id))
        return decision


def agent_spec(key: str, preset: str = "Unlimited", player_id: str | None = None) -> PlayerSpec:
    return PlayerSpec(player_id or key, key, "agent", key, PRESETS[preset], preset=preset)


def run_match(
    mode: str, specs: list[PlayerSpec], seconds: float, after_ready=None, **runner_kwargs
) -> dict:
    runner = TimedRunner(**runner_kwargs)
    config = MatchConfig(mode=mode, seed=123, lockstep=False)
    if mode == "race":
        config.race = RaceRules(time_limit=10_000.0)
    t0 = time.perf_counter()
    match = MatchController(config, specs, runner=runner)
    while not match.ready:
        match.step(DT)
        time.sleep(0.001)
    startup = time.perf_counter() - t0
    match.countdown = 0.0
    if after_ready is not None:
        after_ready(runner)
    start = time.perf_counter()
    frame = 0
    while time.perf_counter() - start < seconds:
        match.step(DT)
        if match.finished and not match.spectating:
            match.keep_watching()  # Marathon: the idle human topped out; keep the agents going
        frame += 1
        delay = start + frame * DT - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
    agents = [s.player_id for s in specs if s.kind == "agent"]
    out = {
        "startup": startup,
        "per_worker_startup": runner.startup_times(),
        "pieces": {p: match.players[p].game.pieces_placed for p in agents},
        "warnings": {p: runner.status(p) for p in agents},
        "think": dict(runner.think),
        "round_trip": dict(runner.round_trip),
    }
    match.close()
    return out


def pct(values: list[float]) -> str:
    if not values:
        return "no decisions"
    v = np.array(values) * 1000
    return f"p50 {np.percentile(v, 50):.2f} ms  p99 {np.percentile(v, 99):.2f} ms  (n={len(v)})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--agents", nargs="+", default=["dqn", "ppo", "ppo_raw"])
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()
    human = PlayerSpec("you", "You", "human")
    ok = True

    print("== Watch AI (one agent at a time, Unlimited)")
    for key in args.agents:
        r = run_match("watch", [agent_spec(key)], args.seconds)
        print(
            f"{key:8s} startup {r['startup']:.2f}s  pieces {r['pieces'][key]:4d}"
            f"  warning {r['warnings'][key]}"
        )
        print(f"         think      {pct(r['think'].get(key, []))}")
        print(f"         round trip {pct(r['round_trip'].get(key, []))}")
        ok &= r["pieces"][key] > 0 and r["warnings"][key] is None

    for mode in ("race", "marathon"):
        print(f"== {mode.title()} (human plus {', '.join(args.agents)} on Unlimited)")
        r = run_match(mode, [human, *[agent_spec(k) for k in args.agents]], args.seconds)
        per = ", ".join(f"{k} {v:.2f}s" for k, v in r["per_worker_startup"].items())
        print(f"pool startup {r['startup']:.2f}s (per worker: {per})")
        for key in args.agents:
            print(
                f"{key:8s} pieces {r['pieces'][key]:4d}  warning {r['warnings'][key]}"
                f"  round trip {pct(r['round_trip'].get(key, []))}"
            )
            ok &= r["pieces"][key] > 0 and r["warnings"][key] is None

    for label, sig in (
        ("hung worker (SIGSTOP)", signal.SIGSTOP),
        ("crash (SIGKILL)", signal.SIGKILL),
    ):
        print(f"== Fallback: DQN {label}, Medium preset, 2 s decision timeout")

        def hit(runner, sig=sig):
            os.kill(runner._workers["dqn"].process.pid, sig)

        r = run_match("watch", [agent_spec("dqn", "Medium")], 8.0, after_ready=hit)
        print(f"dqn pieces {r['pieces']['dqn']}  warning {r['warnings']['dqn']!r}")
        ok &= r["pieces"]["dqn"] > 0 and r["warnings"]["dqn"] is not None
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
