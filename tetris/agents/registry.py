"""Agent registry.

Agents are referenced by key and imported lazily, so listing agents (or
spawning a worker process) never imports heavy dependencies it does not need.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from tetris.agents.base import Agent
from tetris.agents.difficulty import DifficultyParams

ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"


@dataclass(frozen=True)
class AgentSpec:
    key: str
    name: str
    description: str
    factory: str  # "module:Class"
    checkpoint: str | None  # file name under checkpoints/<key>/, None if not needed
    supports_epsilon: bool = True
    supports_target_return: bool = False

    @property
    def checkpoint_path(self) -> Path | None:
        return CHECKPOINT_DIR / self.key / self.checkpoint if self.checkpoint else None

    @property
    def available(self) -> bool:
        path = self.checkpoint_path
        return path is None or path.exists()


AGENTS: dict[str, AgentSpec] = {
    spec.key: spec
    for spec in (
        AgentSpec(
            "dellacherie",
            "Dellacherie (reference)",
            "Hand-tuned classic evaluation; untrained reference baseline.",
            "tetris.agents.linear:DellacherieAgent",
            None,
        ),
        AgentSpec(
            "ga",
            "GA heuristic",
            "Linear feature weights evolved with a genetic algorithm.",
            "tetris.agents.heuristic_ga.agent:GAAgent",
            "best.json",
        ),
        AgentSpec(
            "cem",
            "Cross-entropy method",
            "Linear feature weights optimized with the cross-entropy method.",
            "tetris.agents.cem.agent:CEMAgent",
            "best.json",
        ),
    )
}


def available_agents() -> list[AgentSpec]:
    """Agents that can play now (checkpoint present or none needed)."""
    return [spec for spec in AGENTS.values() if spec.available]


def create_agent(
    key: str,
    checkpoint: str | Path | None = None,
    difficulty: DifficultyParams | None = None,
    seed: int | None = None,
) -> Agent:
    """Instantiate an agent and load its checkpoint."""
    spec = AGENTS[key]
    module_name, cls_name = spec.factory.split(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    agent: Agent = cls(seed=seed)
    path = Path(checkpoint) if checkpoint else spec.checkpoint_path
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"no checkpoint for {key}: {path}")
        agent.load(path)
    if difficulty is not None:
        agent.set_difficulty(difficulty)
    return agent
