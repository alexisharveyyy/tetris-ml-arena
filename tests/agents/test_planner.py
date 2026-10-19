from __future__ import annotations

import numpy as np
import pytest

from tetris.agents.planner import PlanExecutor, plan_path
from tetris.engine import Game, GameConfig, Move, Placement
from tetris.env import enumerate_afterstates

from .states import random_states


@pytest.mark.parametrize("seed", range(4))
def test_every_masked_action_is_reachable_and_matches_afterstate(seed: int) -> None:
    states = random_states(120, seed=seed)
    tall = sum(1 for g in states if (g.board[:8] != 0).any())
    in_spawn_rows = sum(1 for g in states if (g.board[:6] != 0).any())
    assert tall > 5 and in_spawn_rows > 5, "generator should include tall stacks"
    checked = 0
    for game in states:
        for after in enumerate_afterstates(game):
            moves = plan_path(game, after.placement)
            assert moves is not None, f"unreachable {after.placement}\n{game.to_dict()}"
            assert moves[-1] is Move.HARD
            sim = game.copy()
            for move in moves:
                assert sim.apply(move), f"move {move} failed for {after.placement}"
            assert sim.pieces_placed == game.pieces_placed + 1
            np.testing.assert_array_equal(sim.board, after.board)
            checked += 1
    assert checked > 1000


def test_plan_uses_hold_first() -> None:
    game = Game(3)
    moves = plan_path(game, Placement(0, 0, True))
    assert moves is not None and moves[0] is Move.HOLD


def test_unreachable_returns_none() -> None:
    game = Game(3)
    game.can_hold = False
    assert plan_path(game, Placement(0, 0, True)) is None


def test_executor_survives_gravity_and_replans() -> None:
    game = Game(5, GameConfig(gravity=30.0))
    target = Placement(1, 0)
    expected = next(a.board for a in enumerate_afterstates(game) if a.placement == target)
    executor = PlanExecutor(game, target)
    while not executor.done:
        game.tick(0.05)  # gravity moves the piece between moves
        executor.next_move(game)
    assert game.pieces_placed == 1
    np.testing.assert_array_equal(game.board, expected)
