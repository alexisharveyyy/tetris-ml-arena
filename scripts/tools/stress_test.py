"""Frame-time stress test: human plus 3 Unlimited agents at 60 fps.

Runs the real Pygame app headless (SDL dummy driver) with process-pool agents.
The human is driven by synthetic key events (a Dellacherie bot plans each
piece outside the measured frame), so input goes through the normal keyboard
path. Reports pool startup time, frame time percentiles, and how soon after
the start of a frame each human input is applied.

Usage:
    python scripts/tools/stress_test.py [--mode race|battle] [--seconds 60]
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from tetris.agents.planner import plan_path  # noqa: E402
from tetris.agents.registry import create_agent  # noqa: E402
from tetris.engine.game import Move  # noqa: E402
from tetris.match.lobby import HUMAN_ID, Lobby, OpponentSlot  # noqa: E402
from tetris.match.rules import RaceRules  # noqa: E402
from tetris.ui.pygame.app import App, FramePacer  # noqa: E402
from tetris.ui.pygame.screens.match_screen import MatchScreen  # noqa: E402

KEYS = {
    Move.LEFT: pygame.K_LEFT,
    Move.RIGHT: pygame.K_RIGHT,
    Move.CW: pygame.K_UP,
    Move.CCW: pygame.K_z,
    Move.SOFT: pygame.K_DOWN,
    Move.HARD: pygame.K_SPACE,
    Move.HOLD: pygame.K_c,
}
HUMAN_MOVES_PER_SECOND = 12


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--mode", default="race", choices=["race", "battle"])
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--agents", nargs=3, default=["cem", "ga", "dellacherie"])
    args = parser.parse_args()

    app = App()
    app.settings.save = lambda *a, **k: None  # type: ignore[method-assign]
    lobby = Lobby(args.mode, [OpponentSlot(k, "Unlimited") for k in args.agents])
    lobby.race = RaceRules(time_limit=10_000.0)
    t0 = time.perf_counter()
    screen = MatchScreen(app, args.mode, lobby, seed=123)
    app.push(screen)
    match = screen.match
    while not match.ready:
        time.sleep(0.001)
    startup = time.perf_counter() - t0
    per_worker = screen.runner.startup_times()
    match.countdown = 0.0

    bot = create_agent("dellacherie")
    pending: list[Move] = []
    input_delays: list[float] = []
    frame_start = 0.0
    original_apply = match.apply_input

    def timed_apply(player_id, action):
        if player_id == HUMAN_ID:
            input_delays.append(time.perf_counter() - frame_start)
        return original_apply(player_id, action)

    match.apply_input = timed_apply  # type: ignore[method-assign]

    frame_times: list[float] = []
    human_resets = rematches = 0
    move_budget = 0.0
    pacer = FramePacer(60)
    end = time.perf_counter() + args.seconds
    while time.perf_counter() < end:
        dt = pacer.wait()
        # Bot planning for the human happens outside the measured frame.
        game = match.players[HUMAN_ID].game
        if match.finished or type(app.screen).__name__ == "ResultsScreen":
            if type(app.screen).__name__ == "ResultsScreen":
                app.pop()
            screen.rematch(same_seed=False)
            match.countdown = 0.0
            rematches += 1
            pending.clear()
            continue
        if game.game_over:
            human_resets += 1
        elif not pending and game.active is not None:
            pending = plan_path(game, bot.select_placement(game)) or [Move.HARD]
        move_budget += dt * HUMAN_MOVES_PER_SECOND
        while pending and move_budget >= 1:
            key = KEYS[pending.pop(0)]
            pygame.event.post(
                pygame.event.Event(pygame.KEYDOWN, key=key, mod=0, unicode="", scancode=0)
            )
            pygame.event.post(
                pygame.event.Event(pygame.KEYUP, key=key, mod=0, unicode="", scancode=0)
            )
            move_budget -= 1
        move_budget = min(move_budget, 1.0)

        frame_start = time.perf_counter()
        app.frame(dt)
        frame_times.append(time.perf_counter() - frame_start)

    snap = match.get_snapshot()
    match.close()
    ft = np.array(frame_times) * 1000
    delays = np.array(input_delays) * 1000
    print(f"mode {args.mode}, {args.seconds:g}s, agents {args.agents} on Unlimited")
    print(
        f"pool startup: {startup:.2f}s total; per worker "
        + ", ".join(f"{k} {v:.2f}s" for k, v in per_worker.items() if v is not None)
    )
    print(f"frames: {len(ft)}  ({len(ft) / args.seconds:.1f} fps achieved)")
    p50, p99 = np.percentile(ft, [50, 99])
    print(f"frame time ms: p50 {p50:.2f}  p99 {p99:.2f}  max {ft.max():.2f}")
    print(f"frames over 16.7 ms: {(ft > 16.7).sum()}")
    d50, d99 = np.percentile(delays, [50, 99])
    print(
        f"human inputs: {len(delays)}; applied after frame start ms: "
        f"p50 {d50:.3f}  p99 {d99:.3f}  max {delays.max():.3f}"
    )
    for p in snap["players"]:
        print(
            f"  {p['name']:32s} pieces {p['pieces']:5d}  pps {p['pps']:.1f}  warning {p['warning']}"
        )
    print(f"rematches {rematches}, human top-out frames {human_resets}")


if __name__ == "__main__":
    main()
