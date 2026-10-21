"""pytest plugin: keep test state in a temporary directory, and fail the run if
the real settings, Marathon history or recordings change.

Importing this module sets TETRIS_STATE_DIR, so it must be imported before any
tetris module (``tests/conftest.py`` does it first). The default settings,
history and recordings paths then all point into a temporary directory
(:mod:`tetris.state`).
"""

from __future__ import annotations

import os
import shutil
import tempfile

STATE_DIR = tempfile.mkdtemp(prefix="tetris-test-state-")
os.environ["TETRIS_STATE_DIR"] = STATE_DIR
# A fresh checkout's files were all just written; read checkpoints without
# waiting for them to settle (the settle test sets its own window).
os.environ["TETRIS_CHECKPOINT_SETTLE_SECONDS"] = "0"

import pytest  # noqa: E402

from tetris import state  # noqa: E402

#: The files the run must leave alone.
GUARDED = (state.REAL_SETTINGS, state.REAL_HISTORY, state.REAL_RECORDINGS)
MESSAGE = "Tests changed the real user state files"

_before: dict[str, str] = {}


def pytest_sessionstart(session) -> None:
    _before.update(state.fingerprint(GUARDED))


def pytest_sessionfinish(session, exitstatus) -> None:
    shutil.rmtree(STATE_DIR, ignore_errors=True)
    touched = state.changed(_before, state.fingerprint(GUARDED))
    if not touched:
        return
    message = f"{MESSAGE}: {', '.join(touched)}"
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter:
        reporter.write_line(message, red=True, bold=True)
    else:
        print(message)
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
