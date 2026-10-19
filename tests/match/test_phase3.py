from __future__ import annotations

import time

import numpy as np
import pytest

from tetris.agents.difficulty import PRESETS, DifficultyParams
from tetris.agents.registry import AgentSpec, register_agent
from tetris.engine import GARBAGE, HEIGHT, Game, GameConfig, Move, Piece
from tetris.env import action_mask
from tetris.match import MatchConfig, MatchController, PlayerSpec
from tetris.match.agent_runtime import InlineRunner, ProcessRunner
from tetris.match.rules import (
    FIRST_TO_LINES,
    LINES_IN_TIME,
    SCORE_IN_TIME,
    TARGET_LEADER,
    TARGET_RANDOM,
    TARGET_SPLIT,
    AttackRouter,
    BattleRules,
    RaceRules,
)

FAST = DifficultyParams(pps=None, move_rate=None, reaction_delay=0.0)

for key, cls in (
    ("crash_after", "CrashAfterAgent"),
    ("slow", "SlowAgent"),
    ("broken", "BrokenLoadAgent"),
):
    register_agent(AgentSpec(key, key, "test agent", f"tests.agents.fake_agents:{cls}", None))


def agents(*keys: str, difficulty: DifficultyParams = FAST) -> list[PlayerSpec]:
    return [PlayerSpec(f"a{i}", k, "agent", k, difficulty) for i, k in enumerate(keys)]


def run(match: MatchController, seconds: float, dt: float = 1 / 60) -> None:
    for _ in range(int(seconds / dt)):
        match.step(dt)
        if match.finished:
            break


# ------------------------------------------------------------ lockstep


def battle_result(runner_factory, seed: int) -> tuple:
    diff = DifficultyParams(pps=4.0, move_rate=40.0, reaction_delay=0.1, noise=0.2)
    match = MatchController(
        MatchConfig(mode="battle", seed=seed, lockstep=True, battle=BattleRules(TARGET_RANDOM)),
        agents("dellacherie", "ga", "cem", difficulty=diff),
        runner=runner_factory(),
    )
    try:
        run(match, 60.0)
        snap = match.get_snapshot()
        return (
            [(r["id"], r["lines"], r["attack_sent"], r["pieces"]) for r in match.standings],
            [p["board"] for p in snap["players"]],
        )
    finally:
        match.close()


def test_lockstep_same_seed_identical_results_inline() -> None:
    a = battle_result(InlineRunner, 5)
    assert a == battle_result(InlineRunner, 5)
    assert a != battle_result(InlineRunner, 6)


def test_lockstep_process_runner_matches_inline() -> None:
    assert battle_result(lambda: ProcessRunner(timeout=10.0), 5) == battle_result(InlineRunner, 5)


# ---------------------------------------------------------- worker pool


def test_pool_startup_and_realtime_play() -> None:
    runner = ProcessRunner()
    match = MatchController(
        MatchConfig(mode="watch", seed=1), agents("dellacherie", "ga", "cem"), runner
    )
    t0 = time.perf_counter()
    while not match.ready:
        assert time.perf_counter() - t0 < 30
        time.sleep(0.01)
    startup = runner.startup_times()
    assert all(v is not None and v < 30 for v in startup.values())
    for _ in range(120):
        match.step(1 / 60)
        time.sleep(1 / 240)
    assert all(p["pieces"] > 5 for p in match.get_snapshot()["players"])
    match.close()


@pytest.mark.parametrize(
    ("key", "expect"),
    [("crash_after", "crashed"), ("slow", "timed out"), ("broken", "failed to load")],
)
def test_worker_failure_falls_back_to_dellacherie(key: str, expect: str) -> None:
    runner = ProcessRunner(timeout=0.5, startup_timeout=20.0)
    match = MatchController(MatchConfig(mode="watch", seed=2), agents(key), runner)
    t0 = time.perf_counter()
    while match.get_snapshot()["players"][0]["pieces"] < 10:
        match.step(1 / 60)
        time.sleep(0.002)
        assert time.perf_counter() - t0 < 30, "match hung"
    warning = match.get_snapshot()["players"][0]["warning"]
    assert warning is not None and expect in warning and "Dellacherie" in warning
    match.close()


def test_lockstep_never_hangs_on_slow_worker() -> None:
    runner = ProcessRunner(timeout=0.5)
    match = MatchController(
        MatchConfig(mode="watch", seed=2, lockstep=True), agents("slow"), runner
    )
    t0 = time.perf_counter()
    run(match, 1.0)
    assert time.perf_counter() - t0 < 15
    assert match.get_snapshot()["players"][0]["pieces"] > 10
    match.close()


# --------------------------------------------------------------- seeds


def test_boards_share_pieces_but_not_garbage_holes() -> None:
    match = MatchController(
        MatchConfig(mode="battle", seed=9), [PlayerSpec("h", "H"), *agents("dellacherie")]
    )
    games = [p.game for p in match.players.values()]
    assert games[0].queue == games[1].queue
    for g in games:
        for _ in range(6):
            g.receive_garbage(1)
    holes = [tuple(int(np.flatnonzero(row == 0)[0]) for row in g.board[-6:]) for g in games]
    assert holes[0] != holes[1]
    match.reset()
    again = [p.game for p in match.players.values()]
    for g in again:
        for _ in range(6):
            g.receive_garbage(1)
    assert [tuple(int(np.flatnonzero(row == 0)[0]) for row in g.board[-6:]) for g in again] == holes


# ------------------------------------------------------------- battle


def test_router_modes_and_split_remainder() -> None:
    ids = ["a", "b", "c", "d"]
    alive = dict.fromkeys(ids, True)
    stats = {"a": (0, 0), "b": (5, 0), "c": (9, 10), "d": (9, 5)}
    r = AttackRouter(ids, 1, TARGET_LEADER)
    assert r.route("a", 3, alive, stats) == [("c", 3)]
    r.set_mode("a", TARGET_SPLIT)
    assert r.route("a", 4, alive, stats) == [("b", 2), ("c", 1), ("d", 1)]
    assert r.route("a", 1, alive, stats) == [("c", 1)]  # remainder rotates
    r.set_mode("a", TARGET_RANDOM)
    target = r.targets("a", alive, stats)
    assert len(target) == 1 and target[0] != "a"


def test_attacks_reroute_when_target_tops_out() -> None:
    ids = ["a", "b", "c"]
    stats = {"a": (0, 0), "b": (9, 0), "c": (1, 0)}
    r = AttackRouter(ids, 1, TARGET_LEADER)
    assert r.targets("a", dict.fromkeys(ids, True), stats) == ["b"]
    assert r.route("a", 2, {"a": True, "b": False, "c": True}, stats) == [("c", 2)]
    r.set_mode("a", TARGET_RANDOM)
    assert r.targets("a", {"a": True, "b": False, "c": True}, stats) == ["c"]


def test_battle_garbage_is_delivered_and_last_standing_wins() -> None:
    match = MatchController(
        MatchConfig(mode="battle", seed=3, lockstep=True, battle=BattleRules(TARGET_LEADER)),
        [PlayerSpec("h", "Human"), *agents("dellacherie")],
    )
    run(match, 5.0)
    snap = match.get_snapshot()
    human = next(p for p in snap["players"] if p["id"] == "h")
    assert human["targets"] == ["a0"] and human["targeted_by"] == ["a0"]
    # The idle human board receives the agent's attacks, then tops out.
    for _ in range(4000):
        match.step(1 / 60)
        if match.finished:
            break
    assert match.finished and match.winner == "a0"
    assert match.standings[1]["id"] == "h" and match.standings[1]["topped_out"]
    assert match.standings[1]["attack_received"] > 0


def test_human_can_switch_targeting_mid_match() -> None:
    match = MatchController(
        MatchConfig(mode="battle", seed=3, lockstep=True),
        [PlayerSpec("h", "Human"), *agents("dellacherie", "ga")],
    )
    match.set_targeting("h", TARGET_SPLIT)
    assert match.get_snapshot()["players"][0]["targets"] == ["a0", "a1"]


# --------------------------------------------------------------- race


def race(condition: str, **kw) -> MatchController:
    return MatchController(
        MatchConfig(mode="race", seed=4, lockstep=True, race=RaceRules(condition, **kw)),
        agents(
            "dellacherie",
            "cem",
            difficulty=DifficultyParams(pps=3.0, move_rate=30.0, reaction_delay=0.1),
        ),
    )


def test_race_first_to_lines() -> None:
    m = race(FIRST_TO_LINES, target_lines=10)
    run(m, 120.0)
    assert m.finished
    first = m.standings[0]
    assert first["lines"] >= 10 and not first["topped_out"]


@pytest.mark.parametrize("condition", [LINES_IN_TIME, SCORE_IN_TIME])
def test_race_time_limit(condition: str) -> None:
    m = race(condition, time_limit=20.0)
    run(m, 30.0)
    assert m.finished and m.time == pytest.approx(20.0, abs=0.05)
    key = "lines" if condition == LINES_IN_TIME else "score"
    values = [r[key] for r in m.standings]
    assert values == sorted(values, reverse=True)


def test_countdown_blocks_input() -> None:
    m = MatchController(MatchConfig(mode="race", seed=1, countdown=3.0), [PlayerSpec("h", "H")])
    assert not m.apply_input("h", Move.HARD)
    m.step(3.0)
    assert m.apply_input("h", Move.HARD)


# ------------------------------------------------------- 180 rotations


def test_strict_180_rule_only_limits_agent_actions_not_human_kicks() -> None:
    game = Game(0)
    game.board[6, 4] = GARBAGE  # blocks the in-place R and L states of a spawned I
    game.board[6, 5] = GARBAGE
    game.active = None
    game._spawn(Piece.I)
    mask = action_mask(game, include_hold=False)
    assert not mask[20:30].any(), "rotation 2 needs a kick-free intermediate for agents"
    assert game.apply(Move.CW) and game.apply(Move.CW)  # the engine still kicks
    assert game.active.rot == 2


def test_difficulty_noise_produces_legal_suboptimal_choices() -> None:
    from tetris.agents.linear import DellacherieAgent

    agent = DellacherieAgent(seed=0)
    agent.set_difficulty(DifficultyParams(noise=1.0))
    game = Game(3)
    best = DellacherieAgent().select_placement(game)
    picks = {agent.select_placement(game) for _ in range(30)}
    legal = set(np.flatnonzero(action_mask(game)))
    assert best not in picks and all(p.index in legal for p in picks)


def test_presets_have_expected_shape() -> None:
    assert PRESETS["Easy"].noise > PRESETS["Medium"].noise > PRESETS["Hard"].noise
    assert PRESETS["Unlimited"].pps is None and PRESETS["Unlimited"].noise == 0
    _ = GameConfig, HEIGHT


def test_decision_timeout_comes_from_agent_yaml(tmp_path, monkeypatch) -> None:
    from tetris.agents import registry

    cfg = tmp_path / "agent.yaml"
    cfg.write_text("play:\n  decision_timeout: 7.5\n")
    monkeypatch.setattr(registry, "ROOT", tmp_path)
    register_agent(
        AgentSpec(
            "cfg_agent",
            "cfg",
            "t",
            "tetris.agents.linear:DellacherieAgent",
            None,
            config="agent.yaml",
        )
    )
    assert registry.AGENTS["cfg_agent"].decision_timeout == 7.5
    assert registry.AGENTS["dellacherie"].decision_timeout == 2.0  # no YAML: default
    runner = ProcessRunner()
    try:
        runner.add_agent("p", "cfg_agent", FAST)
        runner.add_agent("q", "dellacherie", FAST)
        assert runner.timeouts() == {"p": 7.5, "q": 2.0}
    finally:
        runner.close()
    override = ProcessRunner(timeout=0.5)
    try:
        override.add_agent("p", "cfg_agent", FAST)
        assert override.timeouts() == {"p": 0.5}
    finally:
        override.close()
