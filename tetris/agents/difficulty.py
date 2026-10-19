"""Per-agent difficulty: speed caps, reaction delay and decision noise."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class DifficultyParams:
    """How fast and how well an agent plays.

    ``pps``: maximum pieces per second (None = unlimited).
    ``move_rate``: moves per second when executing a plan (None = instant).
    ``reaction_delay``: seconds after a piece spawns before the first move.
    ``epsilon``: probability of picking a random legal placement.
    ``target_return``: return-to-go for the Decision Transformer (None = agent default).
    """

    pps: float | None = 2.0
    move_rate: float | None = 15.0
    reaction_delay: float = 0.3
    epsilon: float = 0.0
    target_return: float | None = None

    def with_pps(self, pps: float | None) -> DifficultyParams:
        """Change the speed cap, scaling move rate and reaction delay to match."""
        if pps is None:
            return replace(self, pps=None, move_rate=None, reaction_delay=0.0)
        move_rate = max(self.move_rate or 0.0, pps * 8.0)
        reaction = min(self.reaction_delay, 0.5 / pps)
        return replace(self, pps=pps, move_rate=move_rate, reaction_delay=reaction)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DifficultyParams:
        return cls(**d)


PRESETS: dict[str, DifficultyParams] = {
    "Easy": DifficultyParams(pps=0.8, move_rate=8.0, reaction_delay=0.6, epsilon=0.15),
    "Medium": DifficultyParams(pps=1.5, move_rate=14.0, reaction_delay=0.35, epsilon=0.05),
    "Hard": DifficultyParams(pps=3.0, move_rate=30.0, reaction_delay=0.15, epsilon=0.0),
    "Unlimited": DifficultyParams(pps=None, move_rate=None, reaction_delay=0.0, epsilon=0.0),
}
PRESET_NAMES: tuple[str, ...] = tuple(PRESETS)
