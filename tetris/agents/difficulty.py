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
    ``noise``: probability of a plausible mistake: instead of the best placement,
    pick one of the next-best ranked placements (rank 2 most likely).
    ``epsilon``: probability of picking a uniformly random legal placement.
    ``target_return``: return-to-go for the Decision Transformer (None = agent default).
    """

    pps: float | None = 2.0
    move_rate: float | None = 15.0
    reaction_delay: float = 0.3
    noise: float = 0.0
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


#: Difficulty presets. Easy is meant to be beatable by a casual player: slow,
#: with frequent plausible mistakes. See ``scripts/eval/eval_difficulty.py`` for the
#: measured top-out rate and speed of each preset.
PRESETS: dict[str, DifficultyParams] = {
    "Easy": DifficultyParams(pps=0.5, move_rate=5.0, reaction_delay=0.9, noise=0.4, epsilon=0.03),
    "Medium": DifficultyParams(pps=1.0, move_rate=10.0, reaction_delay=0.45, noise=0.15),
    "Hard": DifficultyParams(pps=2.0, move_rate=25.0, reaction_delay=0.2, noise=0.03),
    "Unlimited": DifficultyParams(pps=None, move_rate=None, reaction_delay=0.0),
}
PRESET_NAMES: tuple[str, ...] = tuple(PRESETS)

#: Speed slider stops in pieces per second (None = unlimited).
PPS_STOPS: tuple[float | None, ...] = (
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
    15.0,
    20.0,
    30.0,
    None,
)


def nudge_pps(pps: float | None, step: int) -> float | None:
    """Move ``pps`` ``step`` stops along :data:`PPS_STOPS`."""
    stops = list(PPS_STOPS)
    if pps in stops:
        i = stops.index(pps)
    else:
        i = min(range(len(stops) - 1), key=lambda j: abs(stops[j] - pps))  # type: ignore[operator]
    return stops[max(0, min(len(stops) - 1, i + step))]
