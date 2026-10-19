from __future__ import annotations

import json

import pytest

from tetris.engine import GameConfig, Move
from tetris.match import InputRepeater, MatchConfig, MatchController, PlayerSpec, RepeatSettings
from tetris.settings import Settings


def solo(seed: int = 0, gravity: float | None = None) -> MatchController:
    return MatchController(
        MatchConfig(mode="solo", seed=seed, game=GameConfig(gravity=gravity)),
        [PlayerSpec("human", "You")],
    )


def test_snapshot_is_json_serializable() -> None:
    match = solo()
    match.apply_input("human", Move.LEFT)
    match.step(0.1)
    snap = json.loads(json.dumps(match.get_snapshot()))
    player = snap["players"][0]
    assert player["id"] == "human"
    assert len(player["board"]) == 24 and len(player["queue"]) == 5
    assert len(player["active"]["cells"]) == 4


def test_pause_blocks_input_and_clock() -> None:
    match = solo(gravity=10.0)
    match.set_paused(True)
    assert not match.apply_input("human", Move.LEFT)
    match.step(1.0)
    assert match.time == 0.0
    match.set_paused(False)
    match.step(0.5)
    assert match.time == pytest.approx(0.5)


def test_solo_finishes_on_top_out() -> None:
    match = solo(gravity=0.0)
    for _ in range(200):
        match.apply_input("human", Move.HARD)
        match.step(0.0)
        if match.finished:
            break
    assert match.finished and match.winner == "human"
    assert any(e["type"] == "game_over" for e in match.pop_events())


def test_reset_restarts_same_sequence() -> None:
    match = solo(seed=5)
    first = match.get_snapshot()["players"][0]["queue"]
    match.apply_input("human", Move.HARD)
    match.reset()
    assert match.get_snapshot()["players"][0]["queue"] == first


def test_duplicate_player_ids_rejected() -> None:
    with pytest.raises(ValueError):
        MatchController(MatchConfig(), [PlayerSpec("a", "A"), PlayerSpec("a", "B")])


def test_das_and_arr() -> None:
    rep = InputRepeater(RepeatSettings(das=0.1, arr=0.02, soft_drop_rate=0))
    assert rep.press(Move.LEFT) == [Move.LEFT]
    assert rep.update(0.05) == []
    assert rep.update(0.06) == [Move.LEFT]  # DAS charged at 0.10, 0.01 into ARR
    assert rep.update(0.035) == [Move.LEFT, Move.LEFT]
    rep.release(Move.LEFT)
    assert rep.update(1.0) == []


def test_arr_zero_is_instant() -> None:
    rep = InputRepeater(RepeatSettings(das=0.1, arr=0.0, soft_drop_rate=0))
    rep.press(Move.RIGHT)
    assert len(rep.update(0.2)) >= 10


def test_last_pressed_direction_wins() -> None:
    rep = InputRepeater(RepeatSettings(das=0.0, arr=0.05, soft_drop_rate=0))
    rep.press(Move.LEFT)
    rep.press(Move.RIGHT)
    assert set(rep.update(0.2)) == {Move.RIGHT}
    rep.release(Move.RIGHT)
    assert set(rep.update(0.2)) == {Move.LEFT}


def test_soft_drop_repeat() -> None:
    rep = InputRepeater(RepeatSettings(soft_drop_rate=20))
    assert rep.press(Move.SOFT) == [Move.SOFT]
    assert rep.update(0.1).count(Move.SOFT) == 2
    rep.release(Move.SOFT)
    assert rep.update(0.1) == []


def test_settings_round_trip_and_rebind(tmp_path) -> None:
    s = Settings()
    s.bind("hard", "x")
    assert "x" not in s.bindings["cw"] and s.bindings["hard"][0] == "x"
    path = tmp_path / "s.yaml"
    s.das_ms = 90
    s.save(path)
    loaded = Settings.load(path)
    assert loaded.das_ms == 90 and loaded.key_to_action()["x"] == "hard"
