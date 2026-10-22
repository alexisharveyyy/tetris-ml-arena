"""Queue steps run the pinned worktree's code, or abort (scripts/pinned_step.py)."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pinned_step = _load("pinned_step")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def worktree(tmp_path) -> Path:
    """A git checkout with a stand-in ``tetris`` package and a copy of the launcher."""
    root = tmp_path / "wt"
    (root / "tetris").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "tetris" / "__init__.py").write_text("")
    (root / "tetris" / "step.py").write_text(
        "import sys, tetris\nprint('ran', tetris.__file__, sys.argv[1:])\n"
    )
    (root / "scripts" / "pinned_step.py").write_text((SCRIPTS / "pinned_step.py").read_text())
    (root / "scripts" / "hello.py").write_text("import sys\nprint('script', sys.argv[1:])\n")
    (root / ".gitignore").write_text("__pycache__/\nlogs/\n")  # as in the real repo
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return root


def _run(root: Path, commit: str, *args: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        "TETRIS_PINNED_ROOT": str(root),
        "TETRIS_PINNED_COMMIT": commit,
    }
    return subprocess.run(
        [sys.executable, str(root / "scripts" / "pinned_step.py"), *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )


def test_step_runs_the_worktree_code_not_the_installed_copy(worktree) -> None:
    # This interpreter's own `tetris` (the editable install) is another checkout;
    # the step must still import the worktree's.
    head = _git(worktree, "rev-parse", "HEAD")
    proc = _run(worktree, head, "-m", "tetris.step", "--x", "1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    expected = worktree.resolve() / "tetris" / "__init__.py"
    assert f"pinned step: tetris from {expected}, commit {head}" in proc.stdout
    assert f"ran {expected} ['--x', '1']" in proc.stdout
    proc = _run(worktree, head, "scripts/hello.py", "a")
    assert proc.returncode == 0 and "script ['a']" in proc.stdout


def test_step_aborts_on_another_commit_or_changed_code(worktree) -> None:
    head = _git(worktree, "rev-parse", "HEAD")
    proc = _run(worktree, "0" * 40, "-m", "tetris.step")
    assert proc.returncode == pinned_step.EXIT_NOT_PINNED
    assert "not the pinned commit" in proc.stdout and "ran" not in proc.stdout
    (worktree / "tetris" / "step.py").write_text("print('edited')\n")
    proc = _run(worktree, head, "-m", "tetris.step")
    assert proc.returncode == pinned_step.EXIT_NOT_PINNED
    assert "code differs" in proc.stdout and "tetris/step.py" in proc.stdout


def test_pin_problems_names_a_foreign_package(worktree, tmp_path) -> None:
    head = _git(worktree, "rev-parse", "HEAD")
    assert pinned_step.pin_problems(worktree, head, str(worktree / "tetris" / "__init__.py")) == []
    other = str(tmp_path / "main" / "tetris" / "__init__.py")
    problems = pinned_step.pin_problems(worktree, head, other)
    assert len(problems) == 1 and "tetris resolves to" in problems[0]
    assert "cannot be found" in pinned_step.pin_problems(worktree, head, None)[0]


def test_queue_helper_runs_steps_pinned_and_stops_when_the_commit_moves(
    worktree, monkeypatch
) -> None:
    run_phase4 = _load("run_phase4")
    monkeypatch.setattr(run_phase4, "ROOT", worktree)
    monkeypatch.setattr(run_phase4, "LOGS", worktree / "logs")
    monkeypatch.setattr(run_phase4, "_pinned_commit", None)
    (worktree / "logs").mkdir()
    head = _git(worktree, "rev-parse", "HEAD")
    run_phase4.run("first", ["-m", "tetris.step"])
    log = (worktree / "logs" / "first.log").read_text()
    assert f"commit {head}" in log and "ran" in log
    queue = (worktree / "logs" / "phase4_queue.log").read_text()
    assert f"queue pinned to {worktree} at {head}" in queue
    # A new commit after the queue started: the next step refuses to run.
    (worktree / "tetris" / "new.py").write_text("")
    _git(worktree, "add", ".")
    _git(worktree, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "moved")
    with pytest.raises(SystemExit):
        run_phase4.run("second", ["-m", "tetris.step"])
    assert "ABORTED second" in (worktree / "logs" / "phase4_queue.log").read_text()
    assert "not the pinned commit" in (worktree / "logs" / "second.log").read_text()
