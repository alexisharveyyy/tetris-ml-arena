"""Misbehaving agents for worker-pool tests (imported inside worker processes)."""

from __future__ import annotations

import os
import time

from tetris.agents.linear import DellacherieAgent


class CrashAfterAgent(DellacherieAgent):
    """Plays normally, then kills its process on the 4th decision."""

    def rank_placements(self, game):
        self._calls = getattr(self, "_calls", 0) + 1
        if self._calls >= 4:
            os._exit(3)
        return super().rank_placements(game)


class SlowAgent(DellacherieAgent):
    """Never answers in time."""

    def rank_placements(self, game):
        time.sleep(30)
        return super().rank_placements(game)


class BrokenLoadAgent(DellacherieAgent):
    """Fails while being constructed."""

    def __init__(self, seed=None):
        raise RuntimeError("cannot load")
