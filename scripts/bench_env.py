"""Per-piece timing for the placement-level env with a greedy linear policy.

Usage: python scripts/bench_env.py [--pieces N]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from tetris.env import FEATURE_NAMES, TetrisEnv, enumerate_afterstates

WEIGHTS = {"aggregate_height": -0.51, "holes": -0.72, "bumpiness": -0.18, "lines_cleared": 0.76}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pieces", type=int, default=1000)
    args = parser.parse_args()
    w = np.array([WEIGHTS.get(n, 0.0) for n in FEATURE_NAMES])

    env = TetrisEnv(max_pieces=args.pieces)
    env.reset(seed=3)
    t_after = t_step = 0.0
    n = 0
    while True:
        t0 = time.perf_counter()
        afs = env.afterstates()
        t1 = time.perf_counter()
        best = max(afs, key=lambda a: (not a.top_out, float(a.features @ w)))
        t2 = time.perf_counter()
        _, _, term, trunc, _ = env.step(best.placement.index)
        t3 = time.perf_counter()
        t_after += t1 - t0
        t_step += t3 - t2
        n += 1
        if term or trunc:
            break
    print(f"pieces={n}")
    print(f"afterstates   {t_after / n * 1000:.3f} ms/piece")
    print(f"env.step      {t_step / n * 1000:.3f} ms/piece")
    print(f"total         {(t_after + t_step) / n * 1000:.3f} ms/piece")

    from tetris.engine import Game

    g = Game(0)
    reps = 300
    t0 = time.perf_counter()
    for _ in range(reps):
        enumerate_afterstates(g)
    print(f"enumerate_afterstates (cold game) {(time.perf_counter() - t0) / reps * 1000:.3f} ms")


if __name__ == "__main__":
    main()
