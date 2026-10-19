"""Agent registry.

Agents are referenced by key and imported lazily, so listing agents (or
spawning a worker process) never imports heavy dependencies it does not need.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from tetris.agents.base import Agent
from tetris.agents.difficulty import DifficultyParams

ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
DEFAULT_DECISION_TIMEOUT = 2.0
DEFAULT_STARTUP_TIMEOUT = 60.0


@dataclass(frozen=True)
class AgentSpec:
    key: str
    name: str
    description: str
    factory: str  # "module:Class"
    checkpoint: str | None  # file name under checkpoints/<key>/, None if not needed
    supports_epsilon: bool = True
    supports_target_return: bool = False
    #: Agent config YAML (relative to the repo root) holding a ``play`` section.
    config: str | None = None

    @property
    def play_config(self) -> dict:
        """The ``play`` section of the agent's YAML (decision and startup timeouts)."""
        if self.config is None:
            return {}
        path = ROOT / self.config
        if not path.exists():
            return {}
        return (yaml.safe_load(path.read_text()) or {}).get("play") or {}

    @property
    def decision_timeout(self) -> float:
        """Seconds a worker may take per decision before the fallback takes over."""
        return float(self.play_config.get("decision_timeout", DEFAULT_DECISION_TIMEOUT))

    @property
    def startup_timeout(self) -> float:
        return float(self.play_config.get("startup_timeout", DEFAULT_STARTUP_TIMEOUT))

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
            "Dellacherie (baseline)",
            "Hand-tuned baseline: classic Dellacherie evaluation, not trained.",
            "tetris.agents.linear:DellacherieAgent",
            None,
        ),
        AgentSpec(
            "ga",
            "GA heuristic",
            "Linear feature weights evolved with a genetic algorithm.",
            "tetris.agents.heuristic_ga.agent:GAAgent",
            "best.json",
            config="tetris/agents/heuristic_ga/config.yaml",
        ),
        AgentSpec(
            "cem",
            "Cross-entropy method",
            "Linear feature weights optimized with the cross-entropy method.",
            "tetris.agents.cem.agent:CEMAgent",
            "best.json",
            config="tetris/agents/cem/config.yaml",
        ),
        AgentSpec(
            "dqn",
            "DQN (afterstates)",
            "Double DQN with prioritized replay; a CNN values each afterstate.",
            "tetris.agents.dqn.agent:DQNAgent",
            "best.pt",
            config="tetris/agents/dqn/config.yaml",
        ),
        AgentSpec(
            "ppo",
            "PPO",
            "PPO policy that scores each placement's afterstate (masked 80-action space).",
            "tetris.agents.ppo.agent:PPOAgent",
            "best.pt",
            config="tetris/agents/ppo/config.yaml",
        ),
        AgentSpec(
            "ppo_raw",
            "PPO (raw actions)",
            "PPO policy over the 80 actions from the board alone; weak, kept for comparison.",
            "tetris.agents.ppo.agent:RawPPOAgent",
            "best.pt",
            config="tetris/agents/ppo/config_raw.yaml",
        ),
    )
}


def available_agents() -> list[AgentSpec]:
    """Agents that can play now (checkpoint present or none needed)."""
    return [spec for spec in AGENTS.values() if spec.available]


def register_agent(spec: AgentSpec) -> None:
    """Add or replace an agent spec (plugins, tests)."""
    AGENTS[spec.key] = spec


def create_agent_from_factory(
    factory: str,
    checkpoint: str | Path | None = None,
    difficulty: DifficultyParams | None = None,
    seed: int | None = None,
) -> Agent:
    """Instantiate ``"module:Class"`` and load ``checkpoint`` if given."""
    module_name, cls_name = factory.split(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    agent: Agent = cls(seed=seed)
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        agent.load(path)
    if difficulty is not None:
        agent.set_difficulty(difficulty)
    return agent


def resolve_checkpoint(key: str, checkpoint: str | Path | None = None) -> str | None:
    """Explicit checkpoint, else the spec's default path, else None."""
    if checkpoint is not None:
        return str(checkpoint)
    path = AGENTS[key].checkpoint_path
    return str(path) if path is not None else None


def create_agent(
    key: str,
    checkpoint: str | Path | None = None,
    difficulty: DifficultyParams | None = None,
    seed: int | None = None,
) -> Agent:
    """Instantiate a registry agent and load its checkpoint."""
    return create_agent_from_factory(
        AGENTS[key].factory, resolve_checkpoint(key, checkpoint), difficulty, seed
    )
