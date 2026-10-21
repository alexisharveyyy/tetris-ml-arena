"""Top-3 candidates: legal placements, ranked in order, for every agent type.

Learned agents use freshly initialized networks: ranking order and legality do
not depend on training.
"""

from __future__ import annotations

import pytest

from tetris.agents.neuroevolution.agent import save_genome_checkpoint
from tetris.agents.neuroevolution.network import GenomeNet
from tetris.agents.registry import AGENTS, create_agent
from tetris.match import MatchConfig, MatchController, PlayerSpec
from tetris.rl.nets import ActorCritic, AfterstateActorCritic, AfterstateValueNet, save_checkpoint

from .states import assert_legal, random_states
from .test_phase5 import neat_genomes  # noqa: F401 - fixture

#: Agents whose scores are policy probabilities.
PROBABILITY_AGENTS = {"ppo", "ppo_raw", "bc", "bc_you"}
KEYS = ["dellacherie", "ga", "cem", "dqn", "ppo", "ppo_raw", "bc", "bc_you", "neat"]


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory, neat_genomes):  # noqa: F811
    d = tmp_path_factory.mktemp("ckpt")
    save_checkpoint(d / "dqn.pt", "dqn", AfterstateValueNet(), gamma=0.99)
    for key in ("ppo", "bc", "bc_you"):
        save_checkpoint(d / f"{key}.pt", "ppo", AfterstateActorCritic())
    save_checkpoint(d / "ppo_raw.pt", "ppo_raw", ActorCritic())
    config, genomes = neat_genomes
    save_genome_checkpoint(d / "neat.json", GenomeNet.from_genome(genomes[0], config))
    paths = {k: d / f"{k}.pt" for k in ("dqn", "ppo", "ppo_raw", "bc", "bc_you")}
    paths["neat"] = d / "neat.json"
    paths["ga"] = AGENTS["ga"].checkpoint_path
    paths["cem"] = AGENTS["cem"].checkpoint_path
    paths["dellacherie"] = None
    return paths


@pytest.mark.parametrize("key", KEYS)
def test_top3_candidates_legal_and_ranked(key: str, checkpoints) -> None:
    agent = create_agent(key, checkpoints[key], seed=0)
    for game in random_states(6, seed=21):
        _, candidates = agent.decide(game, 3)
        assert 1 <= len(candidates) <= 3
        ranked = agent.rank_placements(game)
        assert len(candidates) == min(3, len(ranked))
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True), f"{key} candidates out of order"
        assert [c.placement for c in candidates] == [r.placement for r in ranked[:3]]
        for c in candidates:
            assert_legal(game, c.placement)
        if key in PROBABILITY_AGENTS:
            assert all(0.0 <= s <= 1.0 for s in scores)
        assert len({c.placement for c in candidates}) == len(candidates)


def test_snapshot_candidates_are_ranked_landing_cells() -> None:
    match = MatchController(
        MatchConfig(mode="watch", seed=4, lockstep=True),
        [PlayerSpec("ai1", "GA", "agent", "ga")],
        top_k=3,
    )
    for _ in range(30):
        match.step(1 / 60)
        cand = match.players["ai1"].driver.candidate_cells
        if cand:
            break
    assert [c["rank"] for c in cand] == [1, 2, 3]
    board = match.players["ai1"].game.board
    for c in cand:
        assert len(c["cells"]) == 4
        assert all(board[r, col] == 0 for r, col in c["cells"])
