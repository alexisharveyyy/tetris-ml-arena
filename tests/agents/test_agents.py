from __future__ import annotations

import json

import numpy as np
import pytest

from tetris.agents.base import NoLegalPlacementError
from tetris.agents.difficulty import DifficultyParams
from tetris.agents.evaluation import EvalSetting, GameStats, fitness, play_game
from tetris.agents.game_log import log_games, mask_from_str, read_episode
from tetris.agents.linear import (
    DELLACHERIE_WEIGHTS,
    LinearAgent,
    load_linear_checkpoint,
    save_linear_checkpoint,
    weights_from_dict,
)
from tetris.agents.registry import AGENTS, available_agents, create_agent
from tetris.engine import Game, Placement
from tetris.env import action_mask

from .states import assert_legal, random_states

DELL = weights_from_dict(DELLACHERIE_WEIGHTS)


def linear_agents(tmp_path):
    """Every linear agent type, with a temporary checkpoint where one is needed."""
    agents = []
    for key, spec in AGENTS.items():
        if not spec.factory.startswith("tetris."):
            continue  # test doubles registered by other test modules
        if spec.checkpoint and not spec.checkpoint.endswith(".json"):
            continue  # learned agents are covered in tests/rl
        if "neuroevolution" in spec.factory:
            continue  # NEAT genomes are covered in tests/agents/test_phase5.py
        ckpt = None
        if spec.checkpoint:
            ckpt = tmp_path / f"{key}.json"
            save_linear_checkpoint(ckpt, DELL + 0.01, agent=key)
        agents.append(create_agent(key, ckpt))
    return agents


def test_every_agent_returns_legal_placement(tmp_path) -> None:
    agents = linear_agents(tmp_path)
    checked = 0
    for game in random_states(40, seed=11):
        if not action_mask(game).any():
            for agent in agents:
                with pytest.raises(NoLegalPlacementError):
                    agent.select_placement(game)
            continue
        checked += 1
        for agent in agents:
            p = agent.select_placement(game)
            assert_legal(game, p)
            assert_legal(game, agent.select_placement(game.to_dict()))  # serialized state too
    assert checked >= 30


def test_epsilon_choice_is_legal(tmp_path) -> None:
    agent = LinearAgent(seed=0)
    agent.set_difficulty(DifficultyParams(epsilon=1.0))
    for game in random_states(15, seed=3):
        if action_mask(game).any():
            assert_legal(game, agent.select_placement(game))


def test_no_legal_placement_raises() -> None:
    game = Game(0)
    game.game_over = True
    with pytest.raises(NoLegalPlacementError):
        LinearAgent().select_placement(game)


def test_decide_returns_ranked_candidates() -> None:
    placement, ranked = LinearAgent().decide(Game(1), top_k=3)
    assert len(ranked) == 3 and ranked[0].placement == placement
    assert ranked[0].score >= ranked[1].score >= ranked[2].score


def test_checkpoint_round_trip(tmp_path) -> None:
    path = tmp_path / "w.json"
    save_linear_checkpoint(path, DELL, agent="ga", generation=3)
    w, meta = load_linear_checkpoint(path)
    np.testing.assert_allclose(w, DELL)
    assert meta["generation"] == 3


def test_registry_lists_dellacherie_without_checkpoint() -> None:
    assert "dellacherie" in {a.key for a in available_agents()}


def test_fitness_prefers_survival() -> None:
    survive = [GameStats(0, 500, 300, 300_000, 0, False, 500)]
    early = [GameStats(0, 250, 200, 300_000, 0, True, 500)]
    assert fitness(survive) > fitness(early)


def test_play_game_respects_cap_and_garbage() -> None:
    setting = EvalSetting("t", max_pieces=30, garbage_every=3)
    agent = LinearAgent()
    stats = play_game(lambda g: agent.select_placement(g).index, 5, setting)
    assert stats.pieces == 30 and not stats.topped_out


def test_game_log_format(tmp_path) -> None:
    settings = [EvalSetting("plain", 15), EvalSetting("garbage", 15, garbage_every=3)]
    manifest = log_games("dellacherie", settings, 2, tmp_path)
    assert manifest["total_steps"] == 4 * 15
    for ep in manifest["episodes"]:
        records = list(read_episode(ep["path"]))
        header, steps, result = records[0], records[1:-1], records[-1]
        assert header["type"] == "episode" and header["seed"] == ep["seed"]
        assert result["type"] == "result" and len(steps) == result["pieces"]
        for rec in steps:
            game = Game.from_dict(rec["state"])
            mask = mask_from_str(rec["mask"])
            np.testing.assert_array_equal(mask, action_mask(game))
            assert mask[rec["action"]]
            game.place(Placement.from_index(rec["action"]))  # replayable
    assert (
        json.loads((tmp_path / "dellacherie" / "manifest.json").read_text())["agent"]
        == "dellacherie"
    )
