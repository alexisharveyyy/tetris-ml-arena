"""Demo mode settings (``python -m tetris.web --demo``), read from ``configs/demo.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tetris.agents.difficulty import PRESETS
from tetris.agents.identity import LOOKS
from tetris.agents.registry import AGENTS, ROOT
from tetris.match.lobby import MAX_OPPONENTS

DEFAULT_DEMO_PATH = ROOT / "configs" / "demo.yaml"


class DemoConfigError(ValueError):
    """The demo file names an unknown agent, preset or look."""


@dataclass(frozen=True)
class DemoConfig:
    look: str = "tetromino"
    #: (agent key, preset) per preselected opponent.
    opponents: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    seed: int = 0

    @classmethod
    def load(cls, path: Path = DEFAULT_DEMO_PATH) -> DemoConfig:
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data, source=str(path))

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str = "demo config") -> DemoConfig:
        look = data.get("look", "tetromino")
        if look not in LOOKS:
            raise DemoConfigError(f"{source}: unknown look {look!r} (one of {', '.join(LOOKS)})")
        opponents = []
        for item in data.get("opponents") or []:
            key, preset = item.get("agent"), item.get("preset", "Medium")
            spec = AGENTS.get(key)
            if spec is None or not spec.playable:
                raise DemoConfigError(f"{source}: {key!r} is not a playable agent")
            if preset not in PRESETS:
                raise DemoConfigError(f"{source}: unknown preset {preset!r} for {key}")
            opponents.append((key, preset))
        if not 1 <= len(opponents) <= MAX_OPPONENTS:
            raise DemoConfigError(f"{source}: list 1 to {MAX_OPPONENTS} opponents")
        seed = data.get("seed", 0)
        if not isinstance(seed, int) or seed < 0:
            raise DemoConfigError(f"{source}: seed must be a whole number, 0 or more")
        return cls(look=look, opponents=tuple(opponents), seed=seed)

    def lobby_slots(self) -> list[dict[str, Any] | None]:
        slots: list[dict[str, Any] | None] = [
            {"agent": a, "preset": p, "custom": None} for a, p in self.opponents
        ]
        return slots + [None] * (MAX_OPPONENTS - len(slots))

    def to_dict(self) -> dict[str, Any]:
        return {
            "look": self.look,
            "seed": self.seed,
            "opponents": [{"agent": a, "preset": p} for a, p in self.opponents],
        }
