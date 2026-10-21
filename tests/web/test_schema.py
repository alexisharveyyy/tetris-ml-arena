"""Every snapshot of full matches validates against the protocol schema."""

from __future__ import annotations

import json

import pytest

from tetris.agents.difficulty import PRESETS
from tetris.match import MatchConfig, PlayerSpec, RepeatSettings
from tetris.match.agent_runtime import InlineRunner
from tetris.match.history import MarathonHistory
from tetris.match.rules import RaceRules
from tetris.web.protocol import Snapshot, dumps, json_schema, server_message
from tetris.web.session import MatchSession


def session(mode: str, tmp_path, n_agents: int = 3, race: RaceRules | None = None) -> MatchSession:
    players = [PlayerSpec("human", "You")] + [
        PlayerSpec(
            f"ai{i}", f"Bot {i}", "agent", "dellacherie", PRESETS["Unlimited"], preset="Unlimited"
        )
        for i in range(1, n_agents + 1)
    ]
    s = MatchSession(
        MatchConfig(mode=mode, seed=48213, race=race),
        players,
        InlineRunner(),
        RepeatSettings(),
        history=MarathonHistory(tmp_path / "h.json"),
    )
    s.attach(lambda text: None)
    return s


@pytest.mark.parametrize("mode", ["marathon", "race"])
def test_every_snapshot_validates_until_the_end(mode, tmp_path) -> None:
    s = session(mode, tmp_path, race=RaceRules(time_limit=20.0) if mode == "race" else None)
    seen_types: set[str] = set()
    had_candidates: set[str] = set()
    sizes = []
    for frame in range(60 * 60):
        if frame % 6 == 0:
            s.handle_input("hard", True, frame, float(frame))
            s.handle_input("hard", False, frame, float(frame))
        s.step(1 / 60)
        snap = s.snapshot()
        Snapshot.model_validate(snap)
        text = dumps(snap)
        server_message.validate_json(text)
        sizes.append(len(text))
        seen_types |= {e["type"] for e in snap["ev"]}
        had_candidates |= {p["id"] for p in snap["players"] if p["cand"]}
        if snap["finished"]:
            break
    assert snap["finished"] and snap["final"] and len(snap["standings"]) == 4
    assert {"line_clear", "match_end"} <= seen_types
    if mode == "marathon":
        assert "top_out" in seen_types and snap["history_id"]
    else:
        assert snap["race"]["time_remaining"] == 0.0
    # Unlimited agents place within one frame; the overlay still has something to show.
    assert had_candidates == {"ai1", "ai2", "ai3"}
    assert max(sizes) < 8000


def test_candidates_have_ranks_and_visible_coordinates(tmp_path) -> None:
    s = session("marathon", tmp_path, n_agents=1)
    for _ in range(20):
        s.step(1 / 60)
        agent = s.snapshot()["players"][1]
        if agent["cand"]:
            break
    assert [c["rank"] for c in agent["cand"]] == [1, 2, 3]
    for c in agent["cand"]:
        assert all(-4 <= r < 20 and 0 <= col < 10 for r, col in c["c"])


def test_schema_exports() -> None:
    schema = json_schema()
    json.dumps(schema)
    assert schema["protocol"] == 1 and "Snapshot" in json.dumps(schema["server"])
