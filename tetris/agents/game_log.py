"""Game logs for imitation learning and offline RL.

Layout::

    data/logs/<agent>/<setting>/seed_<seed>.jsonl.gz
    data/logs/<agent>/manifest.json

Each episode file is gzip-compressed JSON lines:

* ``{"type": "episode", "agent", "seed", "setting", "checkpoint"}``
* one ``{"type": "step", "step", "action", "mask", "state"}`` per placement, where
  ``action`` is the 80-space index, ``mask`` an 80-character ``0``/``1`` string of
  legal actions and ``state`` the full :meth:`Game.to_dict` before the move
* ``{"type": "result", ...GameStats}``
"""

from __future__ import annotations

import gzip
import json
import multiprocessing as mp
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from tetris.agents.evaluation import EvalSetting, GameStats, play_game
from tetris.engine.game import Game
from tetris.env.afterstates import action_mask

LOG_SEED_BASE = 600_000


def mask_to_str(mask: np.ndarray) -> str:
    return "".join("1" if m else "0" for m in mask)


def mask_from_str(s: str) -> np.ndarray:
    return np.frombuffer(s.encode(), dtype=np.uint8) == ord("1")


def episode_path(root: Path, agent: str, setting: str, seed: int) -> Path:
    return root / agent / setting / f"seed_{seed}.jsonl.gz"


def log_episode(
    agent_key: str,
    checkpoint: str | None,
    seed: int,
    setting: EvalSetting,
    path: Path,
) -> GameStats:
    """Play one game with a registry agent and write its log."""
    from tetris.agents.registry import create_agent

    agent = create_agent(agent_key, checkpoint, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        header = {
            "type": "episode",
            "agent": agent_key,
            "seed": seed,
            "setting": setting.to_dict(),
            "checkpoint": checkpoint,
        }
        f.write(json.dumps(header) + "\n")
        step = 0

        def policy(game: Game) -> int | None:
            if not action_mask(game, setting.include_hold).any():
                return None
            return agent.select_placement(game).index

        def on_step(game: Game, action: int) -> None:
            nonlocal step
            record = {
                "type": "step",
                "step": step,
                "action": action,
                "mask": mask_to_str(action_mask(game, setting.include_hold)),
                "state": game.to_dict(),
            }
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            step += 1

        stats = play_game(policy, seed, setting, on_step)
        f.write(json.dumps({"type": "result", **stats.to_dict()}) + "\n")
    tmp.replace(path)
    return stats


def _task(args: tuple[str, str | None, int, dict[str, Any], str]) -> dict[str, Any]:
    agent_key, checkpoint, seed, setting_d, path = args
    stats = log_episode(agent_key, checkpoint, seed, EvalSetting.from_dict(setting_d), Path(path))
    return {"path": path, **stats.to_dict()}


def log_games(
    agent_key: str,
    settings: list[EvalSetting],
    episodes: int,
    root: Path,
    checkpoint: str | None = None,
    workers: int = 0,
    seed_base: int = LOG_SEED_BASE,
) -> dict[str, Any]:
    """Log ``episodes`` games per setting; writes and returns the manifest."""
    tasks = []
    for setting in settings:
        for i in range(episodes):
            seed = seed_base + i
            path = episode_path(root, agent_key, setting.name, seed)
            tasks.append((agent_key, checkpoint, seed, setting.to_dict(), str(path)))
    if workers > 0:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_task, tasks, chunksize=1)
    else:
        results = [_task(t) for t in tasks]
    manifest = {
        "agent": agent_key,
        "checkpoint": checkpoint,
        "settings": {s.name: s.to_dict() for s in settings},
        "episodes": results,
        "total_steps": int(sum(r["pieces"] for r in results)),
    }
    out = root / agent_key / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    return manifest


def read_episode(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield every record (episode header, steps, result) of a log file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
