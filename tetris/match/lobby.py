"""Opponent selection for Race and Battle, independent of any frontend.

A lobby holds 1 to 3 opponent slots (duplicates allowed), each with an agent
and a difficulty (a preset or custom values), plus the mode's rules. It
serializes to a plain dict so the last selection can be saved and restored.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

from tetris.agents.difficulty import PRESETS, DifficultyParams
from tetris.agents.registry import AGENTS
from tetris.match.controller import PlayerSpec
from tetris.match.rules import BattleRules, RaceRules

MAX_OPPONENTS = 3
CUSTOM = "Custom"
DIFFICULTY_CHOICES: tuple[str, ...] = (*PRESETS, CUSTOM)
HUMAN_ID = "human"


@dataclass
class OpponentSlot:
    agent_key: str
    preset: str = "Medium"
    custom: DifficultyParams | None = None  # used when preset == CUSTOM
    #: Decision Transformer target return (None = the agent's default).
    target_return: float | None = None

    @property
    def difficulty(self) -> DifficultyParams:
        base = (self.custom or PRESETS["Medium"]) if self.preset == CUSTOM else PRESETS[self.preset]
        if self.target_return is not None and AGENTS[self.agent_key].supports_target_return:
            return replace(base, target_return=self.target_return)
        return base

    def set_preset(self, preset: str) -> None:
        if preset == CUSTOM and self.custom is None:
            self.custom = self.difficulty  # start from the current preset's values
        self.preset = preset

    def set_custom(self, **changes: Any) -> None:
        self.custom = replace(self.difficulty, target_return=None, **changes)
        self.preset = CUSTOM

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_key,
            "preset": self.preset,
            "custom": self.custom.to_dict() if self.custom else None,
            "target_return": self.target_return,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OpponentSlot:
        custom = DifficultyParams.from_dict(d["custom"]) if d.get("custom") else None
        preset = d.get("preset", "Medium")
        if preset not in DIFFICULTY_CHOICES:
            preset = "Medium"
        target = d.get("target_return")
        return cls(d["agent"], preset, custom, float(target) if target is not None else None)


@dataclass
class Lobby:
    mode: str  # "race", "marathon" or "battle"
    slots: list[OpponentSlot | None] = field(default_factory=lambda: [None] * MAX_OPPONENTS)
    race: RaceRules = field(default_factory=RaceRules)
    battle: BattleRules = field(default_factory=BattleRules)

    @classmethod
    def default(cls, mode: str, available: list[str]) -> Lobby:
        lobby = cls(mode)
        if available:
            lobby.slots[0] = OpponentSlot(available[0])
        return lobby

    def opponents(self) -> list[OpponentSlot]:
        return [s for s in self.slots if s is not None]

    def valid(self, available: list[str]) -> bool:
        ops = self.opponents()
        return 1 <= len(ops) <= MAX_OPPONENTS and all(s.agent_key in available for s in ops)

    def player_specs(self, human_name: str = "You") -> list[PlayerSpec]:
        """Human first, then one agent player per filled slot (duplicates numbered)."""
        specs = [PlayerSpec(HUMAN_ID, human_name)]
        totals = Counter(s.agent_key for s in self.opponents())
        seen: Counter[str] = Counter()
        for i, slot in enumerate(self.opponents()):
            seen[slot.agent_key] += 1
            name = AGENTS[slot.agent_key].name
            if totals[slot.agent_key] > 1:
                name = f"{name} #{seen[slot.agent_key]}"
            name = f"{name} ({slot.preset})"
            specs.append(
                PlayerSpec(
                    f"ai{i + 1}", name, "agent", slot.agent_key, slot.difficulty, preset=slot.preset
                )
            )
        return specs

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots": [s.to_dict() if s else None for s in self.slots],
            "race": self.race.to_dict(),
            "battle": self.battle.to_dict(),
        }

    @classmethod
    def from_dict(cls, mode: str, d: dict[str, Any], available: list[str]) -> Lobby:
        """Restore a saved lobby, dropping agents whose checkpoints are gone."""
        lobby = cls(mode)
        slots = [OpponentSlot.from_dict(s) if s else None for s in d.get("slots", [])]
        slots = [s if s is not None and s.agent_key in available else None for s in slots]
        slots = (slots + [None] * MAX_OPPONENTS)[:MAX_OPPONENTS]
        if not any(slots) and available:
            slots[0] = OpponentSlot(available[0])
        lobby.slots = slots
        try:
            lobby.race = RaceRules.from_dict(d.get("race", {}))
            lobby.battle = BattleRules.from_dict(d.get("battle", {}))
        except TypeError:
            pass
        return lobby
