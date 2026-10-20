"""NEAT agent: an evolved network scores each afterstate from its 13 features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tetris.agents.base import Agent, ScoredPlacement
from tetris.agents.linear import TOP_OUT_PENALTY
from tetris.agents.neuroevolution.network import GenomeNet
from tetris.engine.game import Game
from tetris.env.afterstates import enumerate_afterstates
from tetris.rl.encoding import FEATURE_SCALE

CHECKPOINT_FORMAT = "neat-genome-v1"


def score_afterstates(game: Game, net: GenomeNet):
    """Afterstates and their network scores (top-outs ranked last, like the linear agents)."""
    afs = enumerate_afterstates(game, dedupe=True)
    if not afs:
        return afs, np.zeros(0)
    feats = np.stack([a.features for a in afs]) / FEATURE_SCALE
    scores = net(feats)
    top = np.array([a.top_out for a in afs])
    return afs, np.where(top, TOP_OUT_PENALTY + scores, scores)


def best_placement_index(game: Game, net: GenomeNet) -> int | None:
    afs, scores = score_afterstates(game, net)
    return afs[int(np.argmax(scores))].placement.index if afs else None


def save_genome_checkpoint(path: str | Path, net: GenomeNet, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"format": CHECKPOINT_FORMAT, "network": net.to_dict(), **meta}))
    tmp.replace(path)


def load_genome_checkpoint(path: str | Path) -> tuple[GenomeNet, dict]:
    data = json.loads(Path(path).read_text())
    if data.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    return GenomeNet.from_dict(data["network"]), data


class NEATAgent(Agent):
    """Ranks placements by the evolved network's score of each afterstate.

    Inputs are the same 13 features GA and CEM weigh linearly, scaled as for
    the learned agents. Difficulty mistakes pick among the next five placements.
    """

    key = "neat"
    name = "NEAT"
    description = "Evolved network scoring each afterstate from the same 13 features as GA and CEM."

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed)
        self.net: GenomeNet | None = None

    def load(self, checkpoint: str | Path) -> None:
        self.net, _ = load_genome_checkpoint(checkpoint)

    def rank_placements(self, game: Game) -> list[ScoredPlacement]:
        if self.net is None:
            raise RuntimeError("NEAT agent has no checkpoint loaded")
        afs, scores = score_afterstates(game, self.net)
        order = np.argsort(-scores, kind="stable")
        return [ScoredPlacement(afs[i].placement, float(scores[i])) for i in order]
