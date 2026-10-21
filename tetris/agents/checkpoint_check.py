"""Is an agent's promoted checkpoint safe to play?

A checkpoint counts as available only when the file exists, has not been
modified in the last :func:`settle_seconds` (so a file still being copied into
place is not read half-written), passes a cheap integrity check (complete zip
archive for ``.pt``, valid JSON for ``.json``), and actually loads. Loads run in
one spawned CPU-only worker process, so the caller never imports torch and a
crashing load cannot take it down. Results are cached per file fingerprint
(path, size, modification time) and redone when the file changes.

Checks only read files.
"""

from __future__ import annotations

import json
import os
import threading
import time
import zipfile
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tetris.agents.registry import AgentSpec

#: Default seconds a checkpoint must stay unmodified before it is read.
SETTLE_SECONDS = 10.0
#: Environment variable overriding :data:`SETTLE_SECONDS` (the tests set it to 0,
#: so a checkout made seconds ago still offers its committed checkpoints).
SETTLE_ENV = "TETRIS_CHECKPOINT_SETTLE_SECONDS"
_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")

#: Check states. Only ``ok`` is playable.
OK = "ok"
MISSING = "missing"
WRITING = "writing"
CORRUPT = "corrupt"
LOAD_FAILED = "load_failed"
CHECKING = "checking"


@dataclass(frozen=True)
class CheckResult:
    state: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.state == OK


def settle_seconds() -> float:
    """The stability window: :data:`SETTLE_ENV` if set, else :data:`SETTLE_SECONDS`."""
    value = os.environ.get(SETTLE_ENV, "").strip()
    if not value:
        return SETTLE_SECONDS
    try:
        seconds = float(value)
    except ValueError:
        raise ValueError(f"{SETTLE_ENV} must be a number of seconds, got {value!r}") from None
    if seconds < 0:
        raise ValueError(f"{SETTLE_ENV} must not be negative, got {value!r}")
    return seconds


def quick_check(spec: AgentSpec, now: float | None = None) -> CheckResult | None:
    """File-level checks. None means the file looks complete and needs a load test."""
    path = spec.checkpoint_path
    if path is None:
        return CheckResult(OK, "no checkpoint needed")
    try:
        stat = path.stat()
    except FileNotFoundError:
        return CheckResult(MISSING, f"{path} not found")
    now = time.time() if now is None else now
    if stat.st_size == 0 or now - stat.st_mtime < settle_seconds():
        return CheckResult(WRITING, f"{path} is still being written")
    if path.suffix == ".pt" and not zipfile.is_zipfile(path):
        return CheckResult(CORRUPT, f"{path} is not a complete checkpoint archive")
    if path.suffix == ".json":
        try:
            json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return CheckResult(CORRUPT, f"{path} is not valid JSON: {exc}")
    return None


def _init_worker() -> None:
    import os

    os.environ.update({k: "1" for k in _THREAD_ENV})
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parent = os.getppid()

    def exit_with_parent() -> None:
        while os.getppid() == parent:
            time.sleep(1.0)
        os._exit(0)

    threading.Thread(target=exit_with_parent, daemon=True).start()


def _load(factory: str, checkpoint: str) -> str:
    """Runs in the worker: build the agent and load the checkpoint on the CPU."""
    from tetris.agents.registry import create_agent_from_factory

    create_agent_from_factory(factory, checkpoint)
    return "loaded"


def _fingerprint(path: Path | None) -> tuple:
    if path is None:
        return (None,)
    try:
        st = path.stat()
    except FileNotFoundError:
        return (str(path), None)
    return (str(path), st.st_size, st.st_mtime_ns)


class CheckpointVerifier:
    """Caches :class:`CheckResult` per agent and runs load tests in the background."""

    def __init__(self, load_timeout: float = 60.0, inline: bool = False) -> None:
        """``inline`` runs load tests in this process (tests with light agents only)."""
        self.load_timeout = load_timeout
        self.inline = inline
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[tuple, CheckResult]] = {}
        self._pending: dict[str, tuple[tuple, Future, float]] = {}
        self._pool: ProcessPoolExecutor | None = None

    def _executor(self) -> ProcessPoolExecutor:
        if self._pool is None:
            import multiprocessing as mp

            self._pool = ProcessPoolExecutor(
                max_workers=1, mp_context=mp.get_context("spawn"), initializer=_init_worker
            )
        return self._pool

    def status(self, spec: AgentSpec) -> CheckResult:
        """Current result for ``spec``; starts a load test if one is needed."""
        path = spec.checkpoint_path
        fp = _fingerprint(path)
        with self._lock:
            cached = self._cache.get(spec.key)
            if cached is not None and cached[0] == fp:
                return cached[1]
            quick = quick_check(spec)
            if quick is not None:
                if quick.state != WRITING:
                    self._cache[spec.key] = (fp, quick)
                return quick
            assert path is not None
            pending = self._pending.get(spec.key)
            if pending is not None and pending[0] == fp:
                return self._collect(spec.key, *pending)
            if self.inline:
                result = self._run_inline(spec.factory, str(path))
                self._cache[spec.key] = (fp, result)
                return result
            future = self._executor().submit(_load, spec.factory, str(path))
            self._pending[spec.key] = (fp, future, time.perf_counter())
            return self._collect(spec.key, fp, future, self._pending[spec.key][2])

    def _collect(self, key: str, fp: tuple, future: Future, started: float) -> CheckResult:
        if not future.done():
            if time.perf_counter() - started > self.load_timeout:
                result = CheckResult(LOAD_FAILED, "load timed out")
                self._finish(key, fp, result)
                self._restart_pool()
                return result
            return CheckResult(CHECKING, "loading")
        exc = future.exception()
        result = CheckResult(OK) if exc is None else CheckResult(LOAD_FAILED, repr(exc))
        self._finish(key, fp, result)
        return result

    def _finish(self, key: str, fp: tuple, result: CheckResult) -> None:
        self._pending.pop(key, None)
        self._cache[key] = (fp, result)

    def _kill_pool(self) -> None:
        if self._pool is not None:
            for proc in list((getattr(self._pool, "_processes", None) or {}).values()):
                proc.kill()
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def _restart_pool(self) -> None:
        self._kill_pool()
        for key, (fp, _, _) in list(self._pending.items()):
            self._finish(key, fp, CheckResult(LOAD_FAILED, "load check interrupted"))

    @staticmethod
    def _run_inline(factory: str, checkpoint: str) -> CheckResult:
        try:
            _load(factory, checkpoint)
        except Exception as exc:  # noqa: BLE001 - any load failure means not available
            return CheckResult(LOAD_FAILED, repr(exc))
        return CheckResult(OK)

    def check_all(
        self, specs: list[AgentSpec], timeout: float | None = None
    ) -> dict[str, CheckResult]:
        """Check every spec, waiting up to ``timeout`` seconds for load tests."""
        deadline = time.perf_counter() + (self.load_timeout if timeout is None else timeout)
        results = {s.key: self.status(s) for s in specs}
        while any(r.state == CHECKING for r in results.values()) and time.perf_counter() < deadline:
            time.sleep(0.05)
            results = {s.key: self.status(s) for s in specs}
        return results

    def close(self) -> None:
        self._kill_pool()
