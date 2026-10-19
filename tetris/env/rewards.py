"""Configurable reward shaping for the placement-level environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tetris.env.features import BOARD_FEATURES


@dataclass
class RewardConfig:
    """Reward = line reward + per-piece bonus + scaled score/attack + feature deltas.

    ``feature_deltas`` maps a board feature name to a weight applied to
    ``feature_after - feature_before`` (use negative weights to penalize growth).
    """

    line_clear: tuple[float, float, float, float, float] = (0.0, 1.0, 3.0, 5.0, 8.0)
    per_piece: float = 0.01
    game_over: float = -5.0
    score_scale: float = 0.0
    attack: float = 0.0
    feature_deltas: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.line_clear = tuple(float(x) for x in self.line_clear)  # type: ignore[assignment]
        if len(self.line_clear) != 5:
            raise ValueError("line_clear needs 5 entries (0 to 4 lines)")
        unknown = set(self.feature_deltas) - set(BOARD_FEATURES)
        if unknown:
            raise ValueError(f"unknown features in feature_deltas: {sorted(unknown)}")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RewardConfig:
        return cls(**(d or {}))

    @classmethod
    def from_yaml(cls, path: str | Path, key: str = "reward") -> RewardConfig:
        """Load from a YAML file; reads the ``reward`` section if present."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data.get(key, data))

    def compute(
        self,
        lines: int,
        score_delta: int,
        attack: int,
        feats_before: np.ndarray,
        feats_after: np.ndarray,
        terminated: bool,
    ) -> float:
        reward = self.line_clear[lines] + self.per_piece
        reward += self.score_scale * score_delta + self.attack * attack
        for name, weight in self.feature_deltas.items():
            i = BOARD_FEATURES.index(name)
            reward += weight * float(feats_after[i] - feats_before[i])
        if terminated:
            reward += self.game_over
        return float(reward)
