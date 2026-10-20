"""DAgger data collection: the BC policy plays, the GA labels every state it visits.

Each game is written in the game-log format (tetris/agents/game_log.py) with
the GA's action as ``action``, so the same encoder turns it into training
data. The policy's own move is stored too (``played``), for agreement stats.
"""

from __future__ import annotations

import gzip
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

from tetris.agents.evaluation import GARBAGE_SETTING, PLAIN_SETTING, EvalSetting, play_game
from tetris.agents.game_log import mask_to_str
from tetris.engine.game import Game
from tetris.env.afterstates import action_mask


def dagger_episode(args: tuple[str, int, dict[str, Any], str]) -> dict[str, Any]:
    """Play one game with the BC checkpoint and label each state with the GA's choice."""
    from tetris.agents.registry import create_agent

    checkpoint, seed, setting_d, path_s = args
    setting = EvalSetting.from_dict(setting_d)
    policy_agent = create_agent("bc_dagger", checkpoint, seed=seed)
    expert = create_agent("ga", seed=seed)
    path = Path(path_s)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    agree = steps = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        header = {
            "type": "episode",
            "agent": "bc_dagger",
            "teacher": "ga",
            "seed": seed,
            "setting": setting.to_dict(),
            "checkpoint": checkpoint,
        }
        f.write(json.dumps(header) + "\n")

        def policy(game: Game) -> int | None:
            nonlocal agree, steps
            mask = action_mask(game, setting.include_hold)
            if not mask.any():
                return None
            played = policy_agent.select_placement(game).index
            label = expert.select_placement(game).index
            record = {
                "type": "step",
                "step": steps,
                "action": label,
                "played": played,
                "mask": mask_to_str(mask),
                "state": game.to_dict(),
            }
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            steps += 1
            agree += played == label
            return played

        stats = play_game(policy, seed, setting)
        f.write(json.dumps({"type": "result", **stats.to_dict()}) + "\n")
    tmp.replace(path)
    return {"path": str(path), "steps": steps, "agree": agree, **stats.to_dict()}


def collect_round(
    checkpoint: str,
    round_: int,
    games_per_setting: int,
    out_dir: Path,
    seed_base: int,
    workers: int,
) -> list[Path]:
    """Play ``games_per_setting`` games in each setting; returns the log files."""
    tasks = []
    for k, setting in enumerate((GARBAGE_SETTING, PLAIN_SETTING)):
        for i in range(games_per_setting):
            seed = seed_base + 500 * k + i
            path = out_dir / f"round{round_}" / setting.name / f"seed_{seed}.jsonl.gz"
            tasks.append((checkpoint, seed, setting.to_dict(), str(path)))
    if workers > 0:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(dagger_episode, tasks, chunksize=1)
    else:
        results = [dagger_episode(t) for t in tasks]
    steps = sum(r["steps"] for r in results)
    agree = sum(r["agree"] for r in results)
    pieces = sum(r["pieces"] for r in results) / max(len(results), 1)
    print(
        f"DAgger round {round_}: {len(results)} games, {steps} states, mean {pieces:.0f} pieces, "
        f"policy agreed with the GA on {agree / max(steps, 1):.1%}",
        flush=True,
    )
    return [Path(r["path"]) for r in results]
