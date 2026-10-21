"""Typed event feed: the match moments a frontend narrates.

The engine's raw events (spawn, lock, hold, garbage) stay on
:meth:`MatchController.pop_events`. This feed is a separate, smaller stream for
tickers and results screens:

* ``line_clear``: ``count`` rows cleared (1 to 4), with the player's running ``lines``.
* ``tetris``: a four-line clear; ``nth`` counts the player's Tetrises this match.
* ``b2b``: a back-to-back Tetris.
* ``rank_change``: a player moved up in the live standings, from place ``from``
  to ``to``. ``passed`` names the player they moved ahead of, and
  ``gap_to_human`` is their score minus the human's (None without a human).
* ``lead_change``: a new player is first; ``previous`` is the old leader and
  ``gap`` the score lead over second place.
* ``top_out``: a player's game ended by topping out (``reason`` from the engine).
* ``fallback``: an agent switched to the baseline; ``reason`` is a code and
  ``text`` a plain sentence for its board.
* ``hold``: the human held a piece.
* ``match_end``: the match finished; ``winner`` and ``standings``.

Every event carries ``type``, ``player`` (None for ``match_end``) and the match
``time`` in seconds.
"""

from __future__ import annotations

from typing import Any

FEED_TYPES: tuple[str, ...] = (
    "line_clear",
    "tetris",
    "b2b",
    "rank_change",
    "lead_change",
    "top_out",
    "fallback",
    "hold",
    "match_end",
)

#: Plain sentences for each fallback reason code, shown on the agent's board.
FALLBACK_TEXT: dict[str, str] = {
    "timeout": "The model missed its time limit, so the baseline is playing for it "
    "until this match ends.",
    "crash": "The model stopped responding, so the baseline is playing for it "
    "until this match ends.",
    "load_failed": "The model could not be loaded, so the baseline is playing for it.",
    "load_timeout": "The model took too long to load, so the baseline is playing for it.",
}


class FeedTracker:
    """Builds feed events from engine events and from standings between steps."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self._places: dict[str, int] = {}
        self._leader: str | None = None
        self._fallbacks: set[str] = set()
        self._tetrises: dict[str, int] = {}

    def reset(self, order: list[str]) -> None:
        """Start a new match with ``order`` as the initial standings."""
        self.events.clear()
        self._places = {pid: i + 1 for i, pid in enumerate(order)}
        self._leader = order[0] if order else None
        self._fallbacks.clear()
        self._tetrises.clear()

    def pop(self) -> list[dict[str, Any]]:
        events, self.events = self.events, []
        return events

    def _emit(self, type_: str, player: str | None, time: float, **data: Any) -> None:
        self.events.append({"type": type_, "player": player, "time": time, **data})

    # ------------------------------------------------------------ inputs

    def on_engine_event(
        self, player: str, event: dict[str, Any], time: float, lines: int, human: bool
    ) -> None:
        kind = event["type"]
        if kind == "lock" and event["lines"] > 0:
            self._emit("line_clear", player, time, count=event["lines"], lines=lines)
            if event["lines"] == 4:
                nth = self._tetrises.get(player, 0) + 1
                self._tetrises[player] = nth
                self._emit("tetris", player, time, nth=nth)
                if event["b2b"]:
                    self._emit("b2b", player, time)
        elif kind == "game_over":
            self._emit("top_out", player, time, reason=event.get("reason"))
        elif kind == "hold" and human:
            self._emit("hold", player, time, piece=event["piece"])

    def on_standings(self, rows: list[dict[str, Any]], time: float, human_id: str | None) -> None:
        """Compare live standings (best first) with the previous ones."""
        scores = {r["id"]: r["score"] for r in rows}
        order = [r["id"] for r in rows]
        human_score = scores.get(human_id) if human_id is not None else None
        for i, pid in enumerate(order):
            place = i + 1
            before = self._places.get(pid, place)
            if place < before:
                passed = order[i + 1] if i + 1 < len(order) else None
                self._emit(
                    "rank_change",
                    pid,
                    time,
                    **{"from": before, "to": place},
                    passed=passed,
                    gap_to_human=None if human_score is None else scores[pid] - human_score,
                )
        if order and order[0] != self._leader:
            gap = scores[order[0]] - scores[order[1]] if len(order) > 1 else 0
            self._emit("lead_change", order[0], time, previous=self._leader, gap=gap)
            self._leader = order[0]
        self._places = {pid: i + 1 for i, pid in enumerate(order)}

    def on_fallback(self, player: str, reason: str, time: float) -> None:
        if player in self._fallbacks:
            return
        self._fallbacks.add(player)
        self._emit(
            "fallback",
            player,
            time,
            reason=reason,
            text=FALLBACK_TEXT.get(reason, FALLBACK_TEXT["crash"]),
        )

    def on_match_end(
        self, winner: str | None, standings: list[dict[str, Any]], time: float
    ) -> None:
        self._emit("match_end", None, time, winner=winner, standings=standings)
