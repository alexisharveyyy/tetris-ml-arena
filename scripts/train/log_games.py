"""Log games from an agent for imitation learning (Phase 5) and offline RL.

Usage:
    python scripts/train/log_games.py --agent ga --episodes 100 [--settings garbage plain]
        [--out data/logs] [--workers N] [--checkpoint PATH]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from tetris.agents.evaluation import SETTINGS, default_workers
from tetris.agents.game_log import log_games
from tetris.agents.registry import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--agent", default="ga")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--settings", nargs="+", default=["garbage", "plain"], choices=list(SETTINGS)
    )
    parser.add_argument("--out", default=str(ROOT / "data" / "logs"))
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    t0 = time.time()
    manifest = log_games(
        args.agent,
        [SETTINGS[s] for s in args.settings],
        args.episodes,
        Path(args.out),
        args.checkpoint,
        args.workers,
    )
    print(
        f"logged {len(manifest['episodes'])} episodes, {manifest['total_steps']} steps "
        f"in {time.time() - t0:.0f}s -> {args.out}/{args.agent}"
    )


if __name__ == "__main__":
    main()
