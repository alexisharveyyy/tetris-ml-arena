"""Tests never use the real settings, history or recordings."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tetris import state
from tetris.match import history, recording
from tetris.settings import DEFAULT_SETTINGS_PATH
from tetris.web.__main__ import _config
from tetris.web.app import WebConfig

ROOT = Path(__file__).resolve().parent.parent


def _inside(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def test_defaults_point_into_the_temporary_state_dir() -> None:
    root = state.state_dir()
    assert root is not None and not _inside(root, ROOT)
    for path in (
        DEFAULT_SETTINGS_PATH,
        history.DEFAULT_HISTORY_PATH,
        recording.RECORDINGS_DIR,
        WebConfig().settings_path,
        WebConfig().history_path,
        WebConfig().recordings_dir,
    ):
        assert _inside(path, root), path
    assert {state.REAL_SETTINGS, state.REAL_HISTORY, state.REAL_RECORDINGS}.isdisjoint(
        {DEFAULT_SETTINGS_PATH, history.DEFAULT_HISTORY_PATH, recording.RECORDINGS_DIR}
    )


def test_without_the_variable_the_real_paths_are_used() -> None:
    env = {k: v for k, v in os.environ.items() if k != state.STATE_ENV}
    out = subprocess.run(
        [sys.executable, "-c", "from tetris.settings import DEFAULT_SETTINGS_PATH as p; print(p)"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout.strip()
    assert Path(out) == state.REAL_SETTINGS


def test_state_dir_flag_keeps_everything_there(tmp_path) -> None:
    config = _config(None, str(tmp_path / "s"))
    for path in (config.settings_path, config.history_path, config.recordings_dir):
        assert _inside(path, tmp_path / "s")


def test_fingerprint_notices_changed_added_and_removed_files(tmp_path) -> None:
    settings, rec = tmp_path / "settings.yaml", tmp_path / "rec"
    settings.write_text("a")
    (rec / "human").mkdir(parents=True)
    (rec / "human" / "one.jsonl.gz").write_bytes(b"1")
    paths = (settings, tmp_path / "history.json", rec)
    before = state.fingerprint(paths)
    assert state.changed(before, state.fingerprint(paths)) == []
    settings.write_text("b")
    (tmp_path / "history.json").write_text("{}")
    (rec / "human" / "two.jsonl.gz").write_bytes(b"2")
    assert state.changed(before, state.fingerprint(paths)) == sorted(
        [str(settings), str(tmp_path / "history.json"), str(rec / "human" / "two.jsonl.gz")]
    )


def _run_guarded(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    """Run pytest on one scratch test with the guard watching a scratch file."""
    watched = tmp_path / "watched.json"
    watched.write_text("{}")
    (tmp_path / "conftest.py").write_text(
        "from pathlib import Path\n"
        "import tests.state_guard as guard\n"
        f"guard.GUARDED = (Path({str(watched)!r}),)\n"
    )
    (tmp_path / "test_scratch.py").write_text(
        f"from pathlib import Path\n\ndef test_it():\n    {body}\n"
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "tests.state_guard",
            "-p",
            "no:cacheprovider",
            str(tmp_path / "test_scratch.py"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_the_run_fails_when_a_test_writes_real_state(tmp_path) -> None:
    out = _run_guarded(tmp_path, f"Path({str(tmp_path / 'watched.json')!r}).write_text('changed')")
    assert out.returncode == 1, out.stdout
    assert "Tests changed the real user state files" in out.stdout
    assert "1 passed" in out.stdout  # the test itself passed; the guard failed the run


def test_the_run_passes_when_tests_leave_real_state_alone(tmp_path) -> None:
    out = _run_guarded(tmp_path, "assert True")
    assert out.returncode == 0, out.stdout
