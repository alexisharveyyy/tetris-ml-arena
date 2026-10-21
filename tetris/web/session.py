"""One live match on the server: controller, human input and the 60 Hz loop.

The session owns a :class:`MatchController` in real-time mode and steps it
from an asyncio task. Human key presses arrive over the WebSocket and go
through the shared :class:`InputRepeater`, so DAS, ARR and soft drop behave
exactly as in the Pygame app. One client is attached at a time; a newer
connection replaces the older one.

If the client disconnects mid-match, the match pauses. When a client attaches
again it resumes after a short countdown, unless the player had paused it.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from tetris.engine.game import Move
from tetris.match.controller import MatchConfig, MatchController, PlayerSpec
from tetris.match.das import InputRepeater, RepeatSettings
from tetris.match.history import MarathonHistory
from tetris.match.recording import GameRecorder
from tetris.web.protocol import dumps, encode_snapshot

TICK = 1 / 60
#: Longest step after a stall (a slow frame never fast-forwards the game by more).
MAX_DT = 0.1
#: Countdown before a match resumes after a reconnect.
RESUME_COUNTDOWN = 3.0
#: Number of candidates every agent reports for the model-thinking overlay.
TOP_K = 3

Send = Callable[[str], Awaitable[None]]


class MatchSession:
    def __init__(
        self,
        config: MatchConfig,
        players: list[PlayerSpec],
        runner: Any,
        repeat: RepeatSettings,
        history: MarathonHistory | None = None,
        countdown: float = 0.0,
        record_setting: str | None = None,
    ) -> None:
        """``record_setting`` ("plain" or "garbage") records the human's game for
        behavior cloning (see :mod:`tetris.match.recording`)."""
        self.id = uuid.uuid4().hex[:12]
        config.countdown = countdown
        self.match = MatchController(config, players, runner=runner, top_k=TOP_K)
        self.match.set_paused(True)  # the clock starts when the first client attaches
        self.human_id = next((p.player_id for p in players if p.kind == "human"), None)
        self.repeater = InputRepeater(repeat)
        self.history = history
        self.history_id: str | None = None
        self.frame = 0
        self.ack: dict[str, Any] | None = None
        self.user_paused = False
        self.disconnected = False
        self._started = False  # a client has attached at least once
        self._client: Send | None = None
        self._client_token = 0
        self._events: list[dict[str, Any]] = []
        self._task: asyncio.Task | None = None
        self._closed = False
        self.bytes_sent = 0
        self.snapshots_sent = 0
        self.record_setting = record_setting
        self.recorder: GameRecorder | None = None
        self.saved_recording: str | None = None
        self._start_recorder()

    def _start_recorder(self) -> None:
        if self.record_setting is None or self.human_id is None:
            return
        self.recorder = GameRecorder(self.human_id, self.match.config.seed, self.record_setting)
        self.match.lock_listeners[:] = [self.recorder.on_lock]
        self.recorder.start(self.match.players[self.human_id].game)
        self.saved_recording = None

    def save_recording(self, root=None):
        """Write the finished recorded game; returns its path."""
        assert self.recorder is not None and self.human_id is not None
        path = self.recorder.save(self.match.players[self.human_id].game, root)
        self.saved_recording = str(path)
        return path

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def close(self) -> None:
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self.match.close()

    @property
    def closed(self) -> bool:
        return self._closed

    # --------------------------------------------------------------- client

    def attach(self, send: Send) -> int:
        """Attach a client. Returns a token for :meth:`detach`."""
        self._client_token += 1
        self._client = send
        if not self._started:
            self._started = True
            self.match.set_paused(self.user_paused)
        elif self.disconnected:
            self.disconnected = False
            if not self.match.finished and not self.user_paused:
                self.match.countdown = max(self.match.countdown, RESUME_COUNTDOWN)
                self.match.set_paused(False)
        return self._client_token

    def detach(self, token: int) -> None:
        """Forget the client if it is still the current one; pause the match."""
        if token != self._client_token:
            return
        self._client = None
        self.repeater.reset()
        if not self.match.finished:
            self.disconnected = True
            self.match.set_paused(True)

    @property
    def has_client(self) -> bool:
        return self._client is not None

    # ---------------------------------------------------------------- input

    def handle_input(self, action: str, down: bool, seq: int, ts: float) -> None:
        move = Move(action)
        if down:
            for m in self.repeater.press(move):
                self._apply(m)
        else:
            self.repeater.release(move)
        self.ack = {"seq": seq, "ts": ts}

    def set_paused(self, paused: bool) -> None:
        self.user_paused = paused
        if paused:
            self.repeater.reset()
        self.match.set_paused(paused or self.disconnected)

    def _apply(self, move: Move) -> None:
        if self.human_id is not None:
            self.match.apply_input(self.human_id, move)

    # ------------------------------------------------------------- control

    def rematch(self, seed: int | None = None) -> None:
        """Restart with the same agents (same seed unless ``seed`` is given)."""
        self.match.reset(seed)
        self.match.countdown = self.match.config.countdown
        self.history_id = None
        self.user_paused = False
        self.repeater.reset()
        self._events.clear()
        self._start_recorder()

    def keep_watching(self) -> None:
        self.match.keep_watching()

    def end(self) -> None:
        self.match.end_match()
        self._after_step()

    # ----------------------------------------------------------------- loop

    def step(self, dt: float) -> None:
        """Advance one frame (the loop calls this; tests may call it directly)."""
        if self.human_id is not None and not self.match.paused:
            for m in self.repeater.update(dt):
                self._apply(m)
        self.match.step(dt)
        self._after_step()

    def _after_step(self) -> None:
        self._events.extend(self.match.pop_feed())
        self.match.pop_events()  # raw engine events are not sent
        if (
            self.match.finished
            and self.history_id is None
            and self.history is not None
            and self.match.config.mode == "marathon"
            and self.human_id is not None
        ):
            self.history_id = self.history.record_match(
                self.match.standings, self.match.config.seed
            )

    def snapshot(self) -> dict[str, Any]:
        """The next snapshot; hands over the feed events gathered since the last one."""
        self.frame += 1
        events, self._events = self._events, []
        return encode_snapshot(
            self.match,
            self.frame,
            ack=self.ack,
            events=events,
            disconnected=self.disconnected,
            history_id=self.history_id,
        )

    async def _run(self) -> None:
        last = time.perf_counter()
        next_tick = last
        while not self._closed:
            now = time.perf_counter()
            self.step(min(now - last, MAX_DT))
            last = now
            send = self._client
            if send is not None:
                text = dumps(self.snapshot())
                try:
                    await send(text)
                    self.bytes_sent += len(text)
                    self.snapshots_sent += 1
                except Exception:  # noqa: BLE001 - the socket handler detaches it
                    pass
            next_tick += TICK
            delay = next_tick - time.perf_counter()
            if delay < -MAX_DT:
                next_tick = time.perf_counter()  # fell far behind; do not try to catch up
                delay = 0.0
            await asyncio.sleep(max(0.0, delay))
