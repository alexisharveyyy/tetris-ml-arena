"""Trajectory collection for the Decision Transformer (CPU only).

Every source (an agent at one mistake rate) plays ``games_per_source``
garbage-setting games on its own block of seeds and writes them in the
game-log format (:mod:`tetris.agents.game_log`):

    data/logs/dt/<agent>_noise<rate>/garbage/seed_<seed>.jsonl.gz

Mistakes come from the difficulty system (``noise``: a plausible next-best
placement instead of the best), so weaker trajectories look like an agent
playing badly rather than at random. The spread of returns is what makes the
target-return knob work: the model can only reproduce returns it has seen.

A source can also name a ``checkpoint`` (relative to the repo root) and a
``label``: an agent at an earlier stage of its training, so skill varies in
ways the board shows rather than through injected mistakes (``config_v2.yaml``).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tetris.agents.evaluation import GARBAGE_SETTING, EvalSetting, default_workers
from tetris.agents.game_log import _task, episode_path
from tetris.agents.registry import ROOT
from tetris.rl.training import load_config

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


@dataclass(frozen=True)
class Source:
    agent: str
    noise: float
    games: int
    checkpoint: str | None = None  # relative to the repo root; None = the promoted one
    label: str | None = None

    @property
    def name(self) -> str:
        return self.label or f"{self.agent}_noise{self.noise:g}"


def sources(cfg: dict[str, Any]) -> list[Source]:
    """Every (agent, mistake rate) source in collection order."""
    c = cfg["collect"]
    out = [
        Source(agent, float(noise), int(c["games_per_source"]))
        for agent in c.get("agents", [])
        for noise in c.get("noise_levels", [])
    ]
    out += [
        Source(
            s["agent"],
            float(s.get("noise", 0.0)),
            int(s["games"]),
            s.get("checkpoint"),
            s.get("label"),
        )
        for s in c.get("extra_sources", [])
    ]
    names = [src.name for src in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate source names: {names}")
    return out


def collection_setting(cfg: dict[str, Any]) -> EvalSetting:
    """The garbage setting (its cap can be lowered with ``collect.max_pieces``, for tests)."""
    cap = cfg["collect"].get("max_pieces", GARBAGE_SETTING.max_pieces)
    return EvalSetting(**{**GARBAGE_SETTING.to_dict(), "max_pieces": int(cap)})


def collection_tasks(cfg: dict[str, Any], root: Path) -> list[tuple]:
    """Game-log tasks: each source gets seeds ``seed_base + 1000 * index + i``."""
    c = cfg["collect"]
    setting = collection_setting(cfg)
    tasks = []
    for k, src in enumerate(sources(cfg)):
        difficulty = {"noise": src.noise} if src.noise > 0 else None
        checkpoint = str(ROOT / src.checkpoint) if src.checkpoint else None
        for i in range(src.games):
            seed = int(c["seed_base"]) + 1000 * k + i
            path = episode_path(root, src.name, setting.name, seed)
            tasks.append((src.agent, checkpoint, seed, setting.to_dict(), str(path), difficulty))
    return tasks


def collect(cfg: dict[str, Any], workers: int | None = None, root: Path | None = None) -> dict:
    """Play every source's games (skipping games already on disk); writes a manifest."""
    root = root if root is not None else ROOT / cfg["collect"]["out_dir"]
    tasks = collection_tasks(cfg, root)
    todo = [t for t in tasks if not Path(t[4]).exists()]
    workers = default_workers() if workers is None else workers
    t0 = time.time()
    print(f"collecting {len(todo)} of {len(tasks)} games with {workers} workers", flush=True)
    if workers > 0 and todo:
        with mp.get_context("spawn").Pool(workers) as pool:
            for n, _ in enumerate(pool.imap_unordered(_task, todo, chunksize=1), 1):
                if n % 50 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} games, {time.time() - t0:.0f}s", flush=True)
    else:
        for t in todo:
            _task(t)
    manifest = summarize_collection(cfg, root)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["sources"], indent=2), flush=True)
    return manifest


def summarize_collection(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    """Per-source game count, states and mean return (in fitness units)."""
    from tetris.agents.evaluation import GameStats, game_fitness
    from tetris.agents.game_log import read_episode

    out: dict[str, Any] = {"sources": {}, "total_games": 0, "total_states": 0}
    for src in sources(cfg):
        files = sorted((root / src.name / GARBAGE_SETTING.name).glob("*.jsonl.gz"))
        stats = []
        for f in files:
            for rec in read_episode(f):
                if rec["type"] == "result":
                    stats.append(GameStats(**{k: v for k, v in rec.items() if k != "type"}))
        pieces = sum(s.pieces for s in stats)
        out["sources"][src.name] = {
            "games": len(stats),
            "states": pieces,
            "mean_fitness": sum(game_fitness(s) for s in stats) / max(len(stats), 1),
        }
        out["total_games"] += len(stats)
        out["total_states"] += pieces
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    collect(load_config(args.config), args.workers)


if __name__ == "__main__":
    main()
