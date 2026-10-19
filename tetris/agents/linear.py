"""Linear afterstate evaluation shared by the GA, CEM and reference agents.

The agent enumerates every legal placement (including hold), computes the
feature vector of each afterstate, and picks the highest ``features @ weights``.
Afterstates that top out are never chosen unless nothing else is legal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tetris.agents.base import Agent, ScoredPlacement
from tetris.engine.game import Game
from tetris.env.afterstates import enumerate_afterstates
from tetris.env.features import FEATURE_NAMES, NUM_FEATURES

TOP_OUT_PENALTY = -1e9
CHECKPOINT_FORMAT = "linear-v1"

#: El-Tetris weights (Islam El-Ashi), a tuned version of Pierre Dellacherie's evaluation.
DELLACHERIE_WEIGHTS: dict[str, float] = {
    "landing_height": -4.500158825082766,
    "eroded_cells": 3.4181268101392694,
    "row_transitions": -3.2178882868487753,
    "col_transitions": -9.348695305445199,
    "holes": -7.899265427351652,
    "wells": -3.3855972247263626,
}


def weights_from_dict(d: dict[str, float]) -> np.ndarray:
    unknown = set(d) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown features: {sorted(unknown)}")
    return np.array([d.get(n, 0.0) for n in FEATURE_NAMES], dtype=np.float64)


def score_afterstates(game: Game, weights: np.ndarray, include_hold: bool = True):
    """Afterstates and their linear scores (top outs heavily penalized)."""
    afs = enumerate_afterstates(game, include_hold=include_hold, dedupe=True)
    if not afs:
        return afs, np.zeros(0)
    feats = np.stack([a.features for a in afs])
    scores = feats @ weights
    top = np.array([a.top_out for a in afs])
    scores = np.where(top, TOP_OUT_PENALTY + scores, scores)
    return afs, scores


def best_placement_index(game: Game, weights: np.ndarray, include_hold: bool = True) -> int | None:
    """Action index of the greedy placement, or None if no legal move."""
    afs, scores = score_afterstates(game, weights, include_hold)
    if not afs:
        return None
    return afs[int(np.argmax(scores))].placement.index


def save_linear_checkpoint(path: str | Path, weights: np.ndarray, **meta: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "format": CHECKPOINT_FORMAT,
        "feature_names": list(FEATURE_NAMES),
        "weights": [float(w) for w in weights],
        **meta,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def load_linear_checkpoint(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    if data.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    weights = weights_from_dict(dict(zip(data["feature_names"], data["weights"], strict=True)))
    return weights, data


class LinearAgent(Agent):
    """Greedy one-piece lookahead with a linear afterstate evaluation."""

    key = "linear"
    name = "Linear heuristic"
    description = "Scores each placement with a weighted sum of board features."

    def __init__(self, weights: np.ndarray | None = None, seed: int | None = None) -> None:
        super().__init__(seed)
        self.weights = (
            np.asarray(weights, dtype=np.float64)
            if weights is not None
            else weights_from_dict(DELLACHERIE_WEIGHTS)
        )
        if self.weights.shape != (NUM_FEATURES,):
            raise ValueError(f"expected {NUM_FEATURES} weights")
        self.include_hold = True

    def load(self, checkpoint: str | Path) -> None:
        self.weights, _ = load_linear_checkpoint(checkpoint)

    def rank_placements(self, game: Game) -> list[ScoredPlacement]:
        afs, scores = score_afterstates(game, self.weights, self.include_hold)
        order = np.argsort(-scores, kind="stable")
        return [ScoredPlacement(afs[i].placement, float(scores[i])) for i in order]


class DellacherieAgent(LinearAgent):
    key = "dellacherie"
    name = "Dellacherie (baseline)"
    description = "Hand-tuned baseline: classic Dellacherie evaluation, not trained."
