from __future__ import annotations

import json

import pytest

from tetris.agents.difficulty import PRESETS, DifficultyParams
from tetris.engine import GameConfig, Move
from tetris.match import MatchConfig, MatchController, PlayerSpec


def watch(
    difficulty: DifficultyParams, seed: int = 0, top_k: int = 0, gravity=None
) -> MatchController:
    return MatchController(
        MatchConfig(mode="watch", seed=seed, game=GameConfig(gravity=gravity)),
        [PlayerSpec("ai", "AI", kind="agent", agent_key="dellacherie", difficulty=difficulty)],
        top_k=top_k,
    )


def run(match: MatchController, seconds: float, dt: float = 1 / 60) -> None:
    for _ in range(int(seconds / dt)):
        match.step(dt)


def test_unlimited_agent_plays_fast_and_clears_lines() -> None:
    match = watch(PRESETS["Unlimited"])
    run(match, 10.0)
    p = match.get_snapshot()["players"][0]
    assert p["pieces"] >= 300  # about one piece per frame
    assert p["lines"] >= 100
    assert not p["game_over"]


@pytest.mark.parametrize("pps", [1.0, 2.5])
def test_pps_cap_is_respected(pps: float) -> None:
    match = watch(DifficultyParams(pps=pps, move_rate=60.0, reaction_delay=0.0))
    run(match, 20.0)
    pieces = match.get_snapshot()["players"][0]["pieces"]
    assert pieces <= pps * 20 + 1
    assert pieces >= pps * 20 * 0.8


def test_reaction_delay_and_move_rate_slow_the_agent() -> None:
    match = watch(DifficultyParams(pps=None, move_rate=5.0, reaction_delay=1.0))
    run(match, 10.0)
    assert match.get_snapshot()["players"][0]["pieces"] <= 10


def test_candidates_overlay_in_snapshot() -> None:
    match = watch(DifficultyParams(pps=1.0, move_rate=10.0, reaction_delay=0.5), top_k=3)
    run(match, 0.2)
    snap = json.loads(json.dumps(match.get_snapshot()))
    cands = snap["players"][0]["candidates"]
    assert [c["rank"] for c in cands] == [1, 2, 3]
    assert all(len(c["cells"]) == 4 for c in cands)


def test_agent_survives_gravity_with_slow_moves() -> None:
    match = watch(PRESETS["Easy"], gravity=3.0)
    run(match, 30.0)
    p = match.get_snapshot()["players"][0]
    assert p["pieces"] >= 10 and not p["game_over"]  # Easy caps at 0.5 pieces/s


def test_human_input_ignored_on_agent_board() -> None:
    match = watch(PRESETS["Medium"])
    assert not match.apply_input("ai", Move.LEFT)


def test_epsilon_play_is_reproducible() -> None:
    def final_board() -> list[str]:
        m = watch(
            DifficultyParams(pps=None, move_rate=None, reaction_delay=0.0, epsilon=0.3), seed=4
        )
        run(m, 2.0)
        return m.get_snapshot()["players"][0]["board"]

    assert final_board() == final_board()


@pytest.mark.parametrize("seed", [950002, 950005])
def test_slow_agent_never_stalls_under_high_gravity(seed: int) -> None:
    """Regression: floor kicks plus gravity used to livelock a Medium agent."""
    match = MatchController(
        MatchConfig(
            mode="eval", seed=seed, lockstep=True, eval_garbage_every=3, eval_max_pieces=500
        ),
        [PlayerSpec("a", "GA", "agent", "ga", PRESETS["Medium"])],
    )
    last, since = 0, 0.0
    while not match.finished:
        match.step(1 / 60)
        pieces = match.players["a"].game.pieces_placed
        since = 0.0 if pieces != last else since + 1 / 60
        last = pieces
        assert since < 10.0, f"stalled at piece {pieces}"
