"""Race and Battle rules: win conditions, targeting and attack routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tetris.engine.rng import Rng

#: Race win conditions.
LINES_IN_TIME = "lines_in_time"  # most lines when the time limit expires
FIRST_TO_LINES = "first_to_lines"  # first to clear the target number of lines
SCORE_IN_TIME = "score_in_time"  # highest score when the time limit expires
RACE_CONDITIONS = (LINES_IN_TIME, FIRST_TO_LINES, SCORE_IN_TIME)
RACE_CONDITION_LABELS = {
    LINES_IN_TIME: "Most lines in time limit",
    FIRST_TO_LINES: "First to N lines",
    SCORE_IN_TIME: "Highest score at time limit",
}

#: Battle targeting rules.
TARGET_RANDOM = "random"
TARGET_LEADER = "leader"
TARGET_SPLIT = "split"
TARGET_MODES = (TARGET_RANDOM, TARGET_LEADER, TARGET_SPLIT)
TARGET_LABELS = {
    TARGET_RANDOM: "Random opponent",
    TARGET_LEADER: "The leader",
    TARGET_SPLIT: "All opponents (split)",
}


@dataclass
class RaceRules:
    condition: str = LINES_IN_TIME
    time_limit: float = 120.0  # seconds, for the time-limit conditions
    target_lines: int = 40  # for FIRST_TO_LINES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RaceRules:
        return cls(**d)

    def describe(self) -> str:
        if self.condition == FIRST_TO_LINES:
            return f"First to {self.target_lines} lines"
        minutes = self.time_limit / 60
        what = "lines" if self.condition == LINES_IN_TIME else "score"
        return f"Most {what} in {minutes:g} min"


@dataclass
class BattleRules:
    targeting: str = TARGET_LEADER  # default for every player; each can change their own

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BattleRules:
        return cls(**d)


class AttackRouter:
    """Decides who receives each player's attacks.

    * ``random``: one random living opponent, kept until it tops out or the
      player sends an attack (then a new one is drawn for the next attack).
    * ``leader``: the living opponent with the most lines (ties: higher score,
      then earlier seat).
    * ``split``: every living opponent; rows are split evenly and the remainder
      goes round-robin, starting after the last opponent that got a remainder.

    Targets are re-evaluated whenever someone tops out, so attacks are never
    sent to a dead board.
    """

    def __init__(self, player_ids: list[str], seed: int, default_mode: str) -> None:
        self.order = list(player_ids)
        self.rng = Rng(seed)
        self.modes = {pid: default_mode for pid in player_ids}
        self._random_target: dict[str, str | None] = {pid: None for pid in player_ids}
        self._split_cursor: dict[str, int] = {pid: 0 for pid in player_ids}

    def set_mode(self, player_id: str, mode: str) -> None:
        if mode not in TARGET_MODES:
            raise ValueError(f"unknown targeting mode {mode}")
        self.modes[player_id] = mode
        self._random_target[player_id] = None

    def targets(
        self, player_id: str, alive: dict[str, bool], stats: dict[str, tuple[int, int]]
    ) -> list[str]:
        """Current targets of a player (for routing and for display)."""
        opponents = [p for p in self.order if p != player_id and alive[p]]
        if not opponents:
            return []
        mode = self.modes[player_id]
        if mode == TARGET_SPLIT:
            return opponents
        if mode == TARGET_LEADER:
            return [max(opponents, key=lambda p: (stats[p][0], stats[p][1], -self.order.index(p)))]
        current = self._random_target[player_id]
        if current not in opponents:
            current = opponents[self.rng.randbelow(len(opponents))]
            self._random_target[player_id] = current
        return [current]

    def route(
        self, player_id: str, rows: int, alive: dict[str, bool], stats: dict[str, tuple[int, int]]
    ) -> list[tuple[str, int]]:
        """Split ``rows`` of attack into ``(target, rows)`` deliveries."""
        targets = self.targets(player_id, alive, stats)
        if not targets or rows <= 0:
            return []
        if self.modes[player_id] == TARGET_RANDOM:
            self._random_target[player_id] = None  # draw a fresh target next time
        if len(targets) == 1:
            return [(targets[0], rows)]
        base, extra = divmod(rows, len(targets))
        amounts = [base] * len(targets)
        start = self._split_cursor[player_id] % len(targets)
        for i in range(extra):
            amounts[(start + i) % len(targets)] += 1
        self._split_cursor[player_id] = start + extra
        return [(t, n) for t, n in zip(targets, amounts, strict=True) if n > 0]
