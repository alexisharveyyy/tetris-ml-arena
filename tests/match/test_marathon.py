from __future__ import annotations

import json

from tetris.agents.difficulty import DifficultyParams
from tetris.engine import Move
from tetris.match import MatchConfig, MatchController, PlayerSpec
from tetris.match.history import MarathonHistory

DIFF = DifficultyParams(pps=3.0, move_rate=30.0, reaction_delay=0.1)


def marathon(seed: int = 1) -> MatchController:
    return MatchController(
        MatchConfig(mode="marathon", seed=seed, lockstep=True),
        [
            PlayerSpec("human", "You"),
            PlayerSpec("a1", "CEM", "agent", "cem", DIFF, preset="Hard"),
            PlayerSpec("a2", "Dell", "agent", "dellacherie", DIFF, preset="Hard"),
        ],
    )


def run_until_human_tops_out(m: MatchController) -> None:
    for i in range(20000):
        if i % 20 == 0:
            m.apply_input("human", Move.HARD)
        m.step(1 / 60)
        if m.finished:
            return


def test_marathon_ends_when_human_tops_out_and_freezes() -> None:
    m = marathon()
    run_until_human_tops_out(m)
    assert m.finished and m.players["human"].game.game_over
    frozen = json.loads(json.dumps(m.standings))
    scores = [r["score"] for r in frozen]
    assert scores == sorted(scores, reverse=True)
    for row in frozen:
        assert {"tetrises", "level", "survival_time", "pps"} <= set(row)
    agent_pieces = m.players["a1"].game.pieces_placed
    m.step(5.0)  # finished: nothing moves
    assert m.players["a1"].game.pieces_placed == agent_pieces
    m.keep_watching()
    for _ in range(300):
        m.step(1 / 60)
    assert m.players["a1"].game.pieces_placed > agent_pieces  # agents continue
    assert m.standings == frozen  # results stay frozen
    assert m.players["human"].game.pieces_placed == next(
        r["pieces"] for r in frozen if r["id"] == "human"
    )


def test_marathon_boards_share_pieces_and_never_interact() -> None:
    m = marathon()
    queues = {pid: p.game.queue[:7] for pid, p in m.players.items()}
    assert len({tuple(q) for q in queues.values()}) == 1
    run_until_human_tops_out(m)
    assert all(p.attack_received == 0 for p in m.players.values())
    assert all(p.game.pending_total == 0 for p in m.players.values())


def test_marathon_without_human_ends_when_all_done() -> None:
    m = MatchController(
        MatchConfig(mode="marathon", seed=2, lockstep=True, eval_max_pieces=40),
        [
            PlayerSpec(
                "a",
                "Dell",
                "agent",
                "dellacherie",
                DifficultyParams(pps=None, move_rate=None, reaction_delay=0),
            )
        ],
    )
    while not m.finished:
        m.step(1 / 60)
    assert m.standings[0]["pieces"] == 40 and not m.standings[0]["topped_out"]


def test_history_records_wins_losses_ties(tmp_path) -> None:
    h = MarathonHistory(tmp_path / "h.json")
    standings = [
        {"kind": "agent", "agent_key": "cem", "preset": "Hard", "score": 500},
        {"kind": "human", "score": 300},
        {"kind": "agent", "agent_key": "ga", "preset": "Easy", "score": 100},
        {"kind": "agent", "agent_key": "ga", "preset": "Medium", "score": 300},
    ]
    h.record_match(standings)
    h.record_match(standings)
    again = MarathonHistory(tmp_path / "h.json")
    assert again.get("cem", "Hard") == {"wins": 0, "losses": 2, "ties": 0}
    assert again.get("ga", "Easy")["wins"] == 2
    assert again.describe("ga", "Medium") == "0W 0L 2T"
    assert again.describe("dellacherie", "Hard") == "no games yet"
