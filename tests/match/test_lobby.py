from __future__ import annotations

from tetris.agents.difficulty import PRESETS
from tetris.match.lobby import CUSTOM, Lobby, OpponentSlot
from tetris.match.rules import FIRST_TO_LINES, TARGET_SPLIT


def test_duplicates_are_numbered_and_difficulty_applied() -> None:
    lobby = Lobby("race", [OpponentSlot("ga", "Easy"), OpponentSlot("ga", "Hard"), None])
    specs = lobby.player_specs()
    assert [s.player_id for s in specs] == ["human", "ai1", "ai2"]
    assert specs[1].name.startswith("GA heuristic #1") and specs[2].name.startswith(
        "GA heuristic #2"
    )
    assert specs[1].difficulty == PRESETS["Easy"] and specs[2].difficulty == PRESETS["Hard"]


def test_custom_difficulty_starts_from_preset() -> None:
    slot = OpponentSlot("cem", "Hard")
    slot.set_preset(CUSTOM)
    assert slot.difficulty == PRESETS["Hard"]
    slot.set_custom(noise=0.3)
    assert slot.difficulty.noise == 0.3 and slot.difficulty.pps == PRESETS["Hard"].pps


def test_round_trip_and_unavailable_agents_dropped() -> None:
    lobby = Lobby("battle", [OpponentSlot("cem", "Hard"), OpponentSlot("ga", CUSTOM), None])
    lobby.slots[1].set_custom(noise=0.25)
    lobby.battle.targeting = TARGET_SPLIT
    lobby.race.condition = FIRST_TO_LINES
    data = lobby.to_dict()
    restored = Lobby.from_dict("battle", data, ["cem", "ga", "dellacherie"])
    assert restored.to_dict() == data
    only_ga = Lobby.from_dict("battle", data, ["ga"])
    assert only_ga.slots[0] is None and only_ga.slots[1].agent_key == "ga"
    nothing = Lobby.from_dict("battle", data, ["dellacherie"])
    assert nothing.slots[0].agent_key == "dellacherie"  # always at least one opponent


def test_validity_requires_one_to_three_available() -> None:
    assert not Lobby("race", [None, None, None]).valid(["ga"])
    assert Lobby("race", [OpponentSlot("ga"), None, None]).valid(["ga"])
    assert not Lobby("race", [OpponentSlot("cem"), None, None]).valid(["ga"])
