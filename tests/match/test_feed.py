from __future__ import annotations

import time

import numpy as np

from tetris.agents.registry import AgentSpec, register_agent
from tetris.engine import Placement
from tetris.engine.board import HEIGHT, WIDTH
from tetris.engine.pieces import GARBAGE, Piece
from tetris.match import MatchConfig, MatchController, PlayerSpec
from tetris.match.agent_runtime import InlineRunner, ProcessRunner
from tetris.match.feed import FALLBACK_TEXT, FEED_TYPES


def marathon(players: list[PlayerSpec], runner=None, seed: int = 3) -> MatchController:
    return MatchController(MatchConfig(mode="marathon", seed=seed, lockstep=True), players, runner)


def humans(*ids: str) -> list[PlayerSpec]:
    return [PlayerSpec(pid, pid) for pid in ids]


def well_board(rows: int) -> np.ndarray:
    """``rows`` full garbage rows at the bottom except column 0."""
    board = np.zeros((HEIGHT, WIDTH), dtype=np.int8)
    board[HEIGHT - rows :, 1:] = GARBAGE
    return board


def drop_vertical_i(match: MatchController, pid: str) -> None:
    """Make ``pid``'s current piece an I and place it upright in column 0."""
    game = match.players[pid].game
    game._spawn(Piece.I)
    game.place(Placement(1, 0))
    match.step(0.0)


def test_tetris_and_back_to_back_events() -> None:
    match = marathon(humans("human", "other"))
    game = match.players["human"].game
    game.board = well_board(8)
    drop_vertical_i(match, "human")
    feed = match.pop_feed()
    types = [e["type"] for e in feed]
    assert "line_clear" in types and "tetris" in types and "b2b" not in types
    clear = next(e for e in feed if e["type"] == "line_clear")
    assert clear["count"] == 4 and clear["player"] == "human" and clear["lines"] == 4
    assert next(e for e in feed if e["type"] == "tetris")["nth"] == 1

    drop_vertical_i(match, "human")
    feed = match.pop_feed()
    assert [e["nth"] for e in feed if e["type"] == "tetris"] == [2]
    assert any(e["type"] == "b2b" and e["player"] == "human" for e in feed)


def test_single_line_clear_count() -> None:
    match = marathon(humans("human"))
    match.players["human"].game.board = well_board(1)
    drop_vertical_i(match, "human")
    clears = [e for e in match.pop_feed() if e["type"] == "line_clear"]
    assert [e["count"] for e in clears] == [1]


def test_rank_and_lead_change_events() -> None:
    match = marathon(humans("human", "b", "c"))
    assert match.live_standings() == ["human", "b", "c"]
    match.players["c"].game.board = well_board(8)
    drop_vertical_i(match, "c")
    feed = match.pop_feed()
    rank = next(e for e in feed if e["type"] == "rank_change")
    assert rank["player"] == "c" and rank["from"] == 3 and rank["to"] == 1
    assert rank["passed"] == "human"
    assert rank["gap_to_human"] == match.players["c"].game.score - match.players["human"].game.score
    lead = next(e for e in feed if e["type"] == "lead_change")
    assert lead["player"] == "c" and lead["previous"] == "human" and lead["gap"] > 0
    # No change, no new events.
    match.step(0.0)
    assert not [e for e in match.pop_feed() if e["type"] in ("rank_change", "lead_change")]


def test_top_out_and_match_end_events() -> None:
    match = marathon(humans("human", "other"))
    game = match.players["human"].game
    for _ in range(200):
        if game.game_over:
            break
        game.place(Placement(0, 4))
        match.step(0.0)
    feed = match.pop_feed()
    top = [e for e in feed if e["type"] == "top_out"]
    assert len(top) == 1 and top[0]["player"] == "human" and top[0]["reason"]
    end = [e for e in feed if e["type"] == "match_end"]
    assert len(end) == 1 and end[0]["player"] is None
    assert [r["id"] for r in end[0]["standings"]] == match.live_standings()
    assert all(e["type"] in FEED_TYPES for e in feed)


def test_human_hold_event() -> None:
    match = marathon(humans("human"))
    match.apply_input("human", "hold")
    holds = [e for e in match.pop_feed() if e["type"] == "hold"]
    assert len(holds) == 1 and holds[0]["piece"] in "IJLOSTZ"


class FlakyRunner(InlineRunner):
    """Inline runner that reports a timeout fallback after two decisions."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def submit(self, player_id, request_id, state, top_k):
        self.calls += 1
        super().submit(player_id, request_id, state, top_k)
        if self.calls > 2:
            self._results[player_id].fallback = True

    def fallback_reason(self, player_id):
        return "timeout" if self.calls > 2 else None


def test_fallback_event_once_and_candidates_dropped() -> None:
    runner = FlakyRunner()
    players = [PlayerSpec("human", "You"), PlayerSpec("ai1", "Bot", "agent", "dellacherie")]
    match = MatchController(
        MatchConfig(mode="marathon", seed=1, lockstep=True), players, runner, top_k=3
    )
    driver = match.players["ai1"].driver
    seen_candidates = False
    for _ in range(600):
        match.step(1 / 60)
        if runner.calls <= 2 and driver.candidate_cells:
            seen_candidates = True
        if runner.calls > 3:
            assert driver.candidate_cells == []
    assert seen_candidates
    fallbacks = [e for e in match.pop_feed() if e["type"] == "fallback"]
    assert len(fallbacks) == 1
    assert fallbacks[0]["player"] == "ai1" and fallbacks[0]["reason"] == "timeout"
    assert fallbacks[0]["text"] == FALLBACK_TEXT["timeout"]


def test_process_runner_timeout_reports_reason_and_no_candidates() -> None:
    register_agent(
        AgentSpec("slow", "slow", "test agent", "tests.agents.fake_agents:SlowAgent", None)
    )
    runner = ProcessRunner(timeout=0.3, startup_timeout=20.0)
    players = [PlayerSpec("ai1", "Slow", "agent", "slow")]
    match = MatchController(MatchConfig(mode="watch", seed=2), players, runner, top_k=3)
    t0 = time.perf_counter()
    feed = []
    while match.players["ai1"].game.pieces_placed < 3:
        match.step(1 / 60)
        feed += match.pop_feed()
        time.sleep(0.002)
        assert time.perf_counter() - t0 < 30, "match hung"
        if runner.fallback_reason("ai1"):
            assert match.players["ai1"].driver.candidate_cells == []
    assert runner.fallback_reason("ai1") == "timeout"
    assert [e["reason"] for e in feed if e["type"] == "fallback"] == ["timeout"]
    match.close()
