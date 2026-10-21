"""Target-return stops for the difficulty UI (reads the calibration; no torch import)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tetris.agents.registry import CHECKPOINT_DIR

CALIBRATION = CHECKPOINT_DIR / "dt" / "calibration.json"
#: Stops offered before calibration has run.
UNCALIBRATED_STOPS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4)


@dataclass(frozen=True)
class TargetStop:
    target: float
    achieved: float | None  # measured fitness at this target, None before calibration

    @property
    def label(self) -> str:
        if self.achieved is None:
            return f"{self.target:.2f} (uncalibrated)"
        return f"{self.target:.2f} (plays at about {self.achieved:.2f})"


def target_stops(path: Path | None = None) -> list[TargetStop]:
    """The calibrated targets with their measured fitness, lowest first."""
    path = path or CALIBRATION
    try:
        rows = json.loads(path.read_text())["rows"]
        return [TargetStop(float(r["target"]), float(r["achieved_fitness"])) for r in rows]
    except (OSError, KeyError, ValueError):
        return [TargetStop(t, None) for t in UNCALIBRATED_STOPS]


def default_label(path: Path | None = None) -> str:
    path = path or CALIBRATION
    try:
        target = json.loads(path.read_text())["leaderboard_target"]
        return f"default ({float(target):.2f}, strongest)"
    except (OSError, KeyError, ValueError):
        return "default"
