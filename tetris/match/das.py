"""Delayed auto shift (DAS), auto repeat rate (ARR) and soft drop repeat.

Frontend independent: feed it press/release of abstract actions and call
:meth:`InputRepeater.update` every frame to get the moves to apply.
"""

from __future__ import annotations

from dataclasses import dataclass

from tetris.engine.game import Move

#: Number of shifts that reach any wall; used for ARR 0 (instant).
_INSTANT_SHIFTS = 10


@dataclass
class RepeatSettings:
    das: float = 0.133  # seconds before auto shift starts
    arr: float = 0.010  # seconds between auto shifts; 0 is instant
    soft_drop_rate: float = 30.0  # rows per second while soft drop is held


class InputRepeater:
    """Turns held left/right/soft-drop keys into repeated moves.

    The most recently pressed horizontal direction wins; releasing it falls
    back to the other one if still held.
    """

    def __init__(self, settings: RepeatSettings | None = None) -> None:
        self.settings = settings or RepeatSettings()
        self.reset()

    def reset(self) -> None:
        self._held: list[Move] = []
        self._das_timer = 0.0
        self._arr_timer = 0.0
        self._soft_held = False
        self._soft_timer = 0.0

    def press(self, move: Move) -> list[Move]:
        """Register a key press. Returns moves to apply immediately."""
        if move in (Move.LEFT, Move.RIGHT):
            if move in self._held:
                self._held.remove(move)
            self._held.append(move)
            self._das_timer = 0.0
            self._arr_timer = 0.0
            return [move]
        if move is Move.SOFT:
            self._soft_held = True
            self._soft_timer = 0.0
            return [Move.SOFT]
        return [move]

    def release(self, move: Move) -> None:
        if move in self._held:
            was_active = self._held[-1] is move
            self._held.remove(move)
            if was_active:
                self._das_timer = 0.0
                self._arr_timer = 0.0
        elif move is Move.SOFT:
            self._soft_held = False

    def update(self, dt: float) -> list[Move]:
        """Advance timers by ``dt`` seconds and return repeated moves."""
        out: list[Move] = []
        if self._held:
            direction = self._held[-1]
            before = self._das_timer
            self._das_timer += dt
            das = self.settings.das
            if self._das_timer >= das:
                if self.settings.arr <= 0:
                    out.extend([direction] * _INSTANT_SHIFTS)
                else:
                    if before < das:
                        # DAS just charged: shift once now, then repeat every ARR.
                        out.append(direction)
                        self._arr_timer = self._das_timer - das
                    else:
                        self._arr_timer += dt
                    while self._arr_timer >= self.settings.arr:
                        self._arr_timer -= self.settings.arr
                        out.append(direction)
        if self._soft_held and self.settings.soft_drop_rate > 0:
            self._soft_timer += dt
            interval = 1.0 / self.settings.soft_drop_rate
            while self._soft_timer >= interval:
                self._soft_timer -= interval
                out.append(Move.SOFT)
        return out
