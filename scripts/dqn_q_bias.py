"""Offline Q calibration of DQN checkpoints: predicted Q of the chosen placement
versus the discounted return the game actually delivered from that point.

Plays greedy games in the garbage setting (seeds 700000-700011, outside every
training, selection and held-out range) on the CPU.

Usage: python scripts/dqn_q_bias.py RUN CHECKPOINT [CHECKPOINT ...] [--games 12]
    e.g. python scripts/dqn_q_bias.py seed0 eval_0025min_1418880.pt
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool

import numpy as np
import torch

from tetris.agents.registry import CHECKPOINT_DIR

GAMMA = 0.99
#: A capped game's return is cut short; only steps at least this far from the cap
#: are used (0.99^300 = 0.05 of the value is missing at worst).
MIN_REMAINING = 300
SEED0 = 700_000


def _play(args: tuple[str, int]) -> dict:
    from tetris.agents.dqn.agent import DQNAgent
    from tetris.agents.evaluation import GARBAGE_SETTING, play_game
    from tetris.rl.encoding import reward_for_score

    torch.set_num_threads(1)
    path, seed = args
    agent = DQNAgent()
    agent.load(path)
    qs, rewards, gaps = [], [], []

    def policy(game):
        afs, q = agent.values(game)
        if not len(afs):
            return None
        order = np.argsort(-q)
        qs.append(float(q[order[0]]))
        gaps.append(float(q[order[0]] - q[order[1]]) if len(order) > 1 else 0.0)
        rewards.append(float(reward_for_score(afs[order[0]].score)))
        return afs[order[0]].placement.index

    stats = play_game(policy, seed, GARBAGE_SETTING)
    n = len(rewards)
    ret = np.zeros(n)
    acc = 0.0
    for t in range(n - 1, -1, -1):
        acc = rewards[t] + GAMMA * acc
        ret[t] = acc
    valid = np.ones(n, bool) if stats.topped_out else np.arange(n) <= n - MIN_REMAINING
    return {
        "pieces": stats.pieces,
        "topped_out": stats.topped_out,
        "q": np.array(qs)[valid],
        "ret": ret[valid],
        "gaps": np.array(gaps),
    }


def measure(run: str, checkpoint: str, games: int, pool: Pool) -> dict[str, float]:
    path = str(CHECKPOINT_DIR / "dqn" / run / checkpoint)
    res = pool.map(_play, [(path, SEED0 + i) for i in range(games)])
    q = np.concatenate([r["q"] for r in res])
    ret = np.concatenate([r["ret"] for r in res])
    near = [(r["q"] - r["ret"])[-20:] for r in res if r["topped_out"]]
    return {
        "pieces": float(np.mean([r["pieces"] for r in res])),
        "top_out": float(np.mean([r["topped_out"] for r in res])),
        "mean_q": float(q.mean()),
        "realized": float(ret.mean()),
        "bias": float(np.mean(q - ret)),
        "bias_last20": float(np.mean(np.concatenate(near))) if near else float("nan"),
        "gap": float(np.median(np.concatenate([r["gaps"] for r in res]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("run")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(
        "| Run | Checkpoint | Pieces | Top-out | Mean Q | Realized return | Bias "
        "| Bias, last 20 pieces before top-out | Median best - 2nd gap |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    with Pool(args.workers) as pool:
        for ck in args.checkpoints:
            m = measure(args.run, ck, args.games, pool)
            print(
                f"| {args.run} | {ck} | {m['pieces']:.0f} | {m['top_out']:.2f} | "
                f"{m['mean_q']:.0f} | {m['realized']:.0f} | {m['bias']:+.0f} | "
                f"{m['bias_last20']:+.0f} | {m['gap']:.2f} |",
                flush=True,
            )


if __name__ == "__main__":
    main()
