"""Agent runners: where agent inference happens.

The controller talks to agents only through :class:`AgentRunner`, using an
asynchronous submit/poll protocol with serialized game states. The inline
runner below answers immediately in-process; a process-pool runner can
implement the same interface without any change to the controller.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from tetris.agents.difficulty import DifficultyParams
from tetris.engine.placement import Placement


@dataclass
class Decision:
    """An agent's answer to one decision request."""

    request_id: int
    placement: Placement | None
    candidates: list[tuple[Placement, float]] = field(default_factory=list)
    think_time: float = 0.0
    error: str | None = None


class AgentRunner(ABC):
    @abstractmethod
    def add_agent(
        self,
        player_id: str,
        agent_key: str,
        difficulty: DifficultyParams,
        checkpoint: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Create (and load) the agent that plays for ``player_id``."""

    @abstractmethod
    def set_difficulty(self, player_id: str, difficulty: DifficultyParams) -> None: ...

    @abstractmethod
    def submit(self, player_id: str, request_id: int, state: dict[str, Any], top_k: int) -> None:
        """Ask for a decision on a serialized game state."""

    @abstractmethod
    def poll(self, player_id: str) -> Decision | None:
        """Return the latest finished decision for a player, if any. Never blocks."""

    def wait(self, player_id: str) -> Decision:
        """Block until the outstanding decision is available (lockstep mode)."""
        decision = self.poll(player_id)
        if decision is None:
            raise RuntimeError(f"{type(self).__name__} has no pending decision for {player_id}")
        return decision

    def ready(self) -> bool:
        """True once every agent is loaded (or has fallen back)."""
        return True

    def status(self, player_id: str) -> str | None:
        """A warning to show for this player (for example after a fallback), or None."""
        return None

    def close(self) -> None:  # noqa: B027 - optional hook, default does nothing
        """Release resources."""


def decide(agent: Any, request_id: int, state: dict[str, Any], top_k: int) -> Decision:
    """Run one decision with error capture (shared by all runners)."""
    t0 = time.perf_counter()
    try:
        placement, ranked = agent.decide(state, top_k)
        return Decision(
            request_id,
            placement,
            [(c.placement, c.score) for c in ranked],
            time.perf_counter() - t0,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the driver, which falls back
        return Decision(request_id, None, [], time.perf_counter() - t0, error=repr(exc))


class InlineRunner(AgentRunner):
    """Runs agents synchronously in the calling thread. For tests and light agents."""

    def __init__(self) -> None:
        self._agents: dict[str, Any] = {}
        self._results: dict[str, Decision] = {}

    def add_agent(
        self,
        player_id: str,
        agent_key: str,
        difficulty: DifficultyParams,
        checkpoint: str | None = None,
        seed: int | None = None,
    ) -> None:
        from tetris.agents.registry import create_agent

        self._agents[player_id] = create_agent(agent_key, checkpoint, difficulty, seed)

    def set_difficulty(self, player_id: str, difficulty: DifficultyParams) -> None:
        self._agents[player_id].set_difficulty(difficulty)

    def submit(self, player_id: str, request_id: int, state: dict[str, Any], top_k: int) -> None:
        self._results[player_id] = decide(self._agents[player_id], request_id, state, top_k)

    def poll(self, player_id: str) -> Decision | None:
        return self._results.pop(player_id, None)


# ---------------------------------------------------------------- processes

#: Agent used when a worker crashes, fails to load or times out.
FALLBACK_AGENT = "dellacherie"
_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


@dataclass
class _Worker:
    player_id: str
    agent_key: str
    process: Any
    conn: Any
    started: float
    difficulty: DifficultyParams
    seed: int | None
    ready: bool = False
    startup_seconds: float | None = None
    pending: tuple[int, float, dict[str, Any], int] | None = None  # id, sent_at, state, top_k
    result: Decision | None = None
    fallback: Any = None  # in-process agent after a failure
    warning: str | None = None


class ProcessRunner(AgentRunner):
    """One persistent ``spawn`` worker process per agent slot.

    Workers are started by :meth:`add_agent` and load their model once. If a
    worker crashes, fails to load, or takes longer than ``timeout`` seconds on
    a decision (or ``startup_timeout`` to load), that slot switches to an
    in-process Dellacherie agent for the rest of the match and
    :meth:`status` reports a warning. Nothing here ever blocks indefinitely.
    """

    def __init__(self, timeout: float = 2.0, startup_timeout: float = 60.0) -> None:
        import multiprocessing as mp

        self._ctx = mp.get_context("spawn")
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self._workers: dict[str, _Worker] = {}
        self.created = time.perf_counter()

    # --------------------------------------------------------------- setup

    def add_agent(
        self,
        player_id: str,
        agent_key: str,
        difficulty: DifficultyParams,
        checkpoint: str | None = None,
        seed: int | None = None,
    ) -> None:
        import os

        from tetris.agents.registry import AGENTS, resolve_checkpoint
        from tetris.match.agent_worker import worker_main

        parent, child = self._ctx.Pipe()
        args = (child, AGENTS[agent_key].factory, resolve_checkpoint(agent_key, checkpoint))
        process = self._ctx.Process(
            target=worker_main,
            args=(*args, difficulty.to_dict(), seed),
            daemon=True,
            name=f"agent-{player_id}",
        )
        saved = {k: os.environ.get(k) for k in _THREAD_ENV}
        os.environ.update({k: "1" for k in _THREAD_ENV})  # one CPU thread per worker
        try:
            process.start()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        child.close()
        self._workers[player_id] = _Worker(
            player_id, agent_key, process, parent, time.perf_counter(), difficulty, seed
        )

    def _check_startup(self, w: _Worker, block: float = 0.0) -> None:
        if w.ready or w.fallback is not None:
            return
        try:
            if w.conn.poll(block):
                kind, payload = w.conn.recv()
                if kind == "ready":
                    w.ready = True
                    w.startup_seconds = time.perf_counter() - w.started
                else:
                    self._fall_back(w, "failed to load")
                return
        except (EOFError, OSError):
            self._fall_back(w, "crashed while loading")
            return
        if not w.process.is_alive():
            self._fall_back(w, "crashed while loading")
        elif time.perf_counter() - w.started > self.startup_timeout:
            self._fall_back(w, "load timed out")

    def ready(self) -> bool:
        for w in self._workers.values():
            self._check_startup(w)
        return all(w.ready or w.fallback is not None for w in self._workers.values())

    def wait_ready(self) -> dict[str, float | None]:
        """Block until all workers are loaded or fallen back; returns startup seconds."""
        deadline = time.perf_counter() + self.startup_timeout + 1.0
        while not self.ready() and time.perf_counter() < deadline:
            for w in self._workers.values():
                self._check_startup(w, block=0.01)
        return self.startup_times()

    def startup_times(self) -> dict[str, float | None]:
        return {pid: w.startup_seconds for pid, w in self._workers.items()}

    # ------------------------------------------------------------ fallback

    def _fall_back(self, w: _Worker, reason: str) -> None:
        from tetris.agents.registry import AGENTS, create_agent

        if w.fallback is not None:
            return
        w.fallback = create_agent(FALLBACK_AGENT, difficulty=w.difficulty, seed=w.seed)
        w.warning = f"{reason}; using {AGENTS[FALLBACK_AGENT].name}"
        try:
            w.process.kill()
        except Exception:  # noqa: BLE001 - already gone
            pass
        w.conn.close()
        if w.pending is not None:
            request_id, _, state, top_k = w.pending
            w.result = decide(w.fallback, request_id, state, top_k)
            w.pending = None

    def status(self, player_id: str) -> str | None:
        return self._workers[player_id].warning

    # ----------------------------------------------------------- requests

    def set_difficulty(self, player_id: str, difficulty: DifficultyParams) -> None:
        w = self._workers[player_id]
        w.difficulty = difficulty
        if w.fallback is not None:
            w.fallback.set_difficulty(difficulty)
            return
        try:
            w.conn.send(("difficulty", difficulty.to_dict()))
        except (OSError, BrokenPipeError):
            self._fall_back(w, "crashed")

    def submit(self, player_id: str, request_id: int, state: dict[str, Any], top_k: int) -> None:
        w = self._workers[player_id]
        w.result = None
        if w.fallback is not None:
            w.result = decide(w.fallback, request_id, state, top_k)
            return
        w.pending = (request_id, time.perf_counter(), state, top_k)
        if not w.ready:
            return  # sent once the worker reports ready (see _service)
        self._send(w)

    def _send(self, w: _Worker) -> None:
        assert w.pending is not None
        request_id, _, state, top_k = w.pending
        w.pending = (request_id, time.perf_counter(), state, top_k)
        try:
            w.conn.send(("decide", request_id, state, top_k))
        except (OSError, BrokenPipeError):
            self._fall_back(w, "crashed")

    def _service(self, w: _Worker, block: float = 0.0) -> None:
        """Advance one worker: startup, incoming decisions, crash and timeout checks."""
        if w.fallback is not None:
            return
        if not w.ready:
            self._check_startup(w, block)
            if w.ready and w.pending is not None:
                self._send(w)
            return
        if w.pending is None:
            return
        try:
            if w.conn.poll(block):
                kind, decision = w.conn.recv()
                if kind == "decision" and decision.request_id == w.pending[0]:
                    w.result = decision
                    w.pending = None
                return
        except (EOFError, OSError):
            self._fall_back(w, "crashed")
            return
        if not w.process.is_alive():
            self._fall_back(w, "crashed")
        elif time.perf_counter() - w.pending[1] > self.timeout:
            self._fall_back(w, f"timed out (>{self.timeout:g}s)")

    def poll(self, player_id: str) -> Decision | None:
        w = self._workers[player_id]
        self._service(w)
        result, w.result = w.result, None
        return result

    def wait(self, player_id: str) -> Decision:
        w = self._workers[player_id]
        while w.result is None:
            if w.pending is None and w.fallback is None and w.ready:
                raise RuntimeError(f"no pending decision for {player_id}")
            self._service(w, block=0.05)
        result, w.result = w.result, None
        return result

    def close(self) -> None:
        for w in self._workers.values():
            if w.fallback is None:
                try:
                    w.conn.send(("close",))
                except (OSError, BrokenPipeError):
                    pass
        for w in self._workers.values():
            w.process.join(timeout=1.0)
            if w.process.is_alive():
                w.process.kill()
        self._workers.clear()
