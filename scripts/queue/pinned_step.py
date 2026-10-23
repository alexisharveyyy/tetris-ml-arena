"""Run one queue step only if it will execute the pinned worktree's code.

Usage (from the queue helper, scripts/queue/run_phase4.py):
    TETRIS_PINNED_ROOT=<worktree> TETRIS_PINNED_COMMIT=<sha> \\
        python scripts/queue/pinned_step.py -m package.module [args...]
    ... python scripts/queue/pinned_step.py scripts/eval/evaluate.py [args...]

Before running the step, in the same interpreter that will run it, this
checks that the ``tetris`` package resolves to the pinned worktree (not, for
example, the checkout an editable install points at), that the worktree's
HEAD is the pinned commit, and that nothing under ``tetris/`` or ``scripts/``
differs from that commit. It logs the package path and commit, then runs the
step, or exits with status 3 and the reasons.
"""

from __future__ import annotations

import importlib.util
import os
import runpy
import subprocess
import sys
from pathlib import Path

ROOT_ENV = "TETRIS_PINNED_ROOT"
COMMIT_ENV = "TETRIS_PINNED_COMMIT"
#: Paths whose contents must match the pinned commit (the code steps load).
CODE_PATHS = ("tetris", "scripts")
EXIT_NOT_PINNED = 3


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def pin_problems(root: Path, commit: str, package_file: str | None) -> list[str]:
    """Reasons the step would not run the pinned code (empty when it would)."""
    problems = []
    root = root.resolve()
    expected = root / "tetris" / "__init__.py"
    if package_file is None:
        problems.append("the tetris package cannot be found")
    elif Path(package_file).resolve() != expected:
        problems.append(f"tetris resolves to {package_file}, not {expected}")
    head = git(root, "rev-parse", "HEAD")
    if head != commit:
        problems.append(f"{root} is at {head[:12]}, not the pinned commit {commit[:12]}")
    changed = git(root, "status", "--porcelain", "--untracked-files=all", "--", *CODE_PATHS)
    if changed:
        files = ", ".join(line.split(maxsplit=1)[1] for line in changed.splitlines()[:5])
        problems.append(f"code differs from {commit[:12]}: {files}")
    return problems


def main(argv: list[str]) -> None:
    if not argv:
        sys.exit("usage: pinned_step.py (-m module | script.py) [args...]")
    root = Path(os.environ[ROOT_ENV])
    commit = os.environ[COMMIT_ENV]
    spec = importlib.util.find_spec("tetris")  # locates without importing
    package_file = spec.origin if spec else None
    problems = pin_problems(root, commit, package_file)
    print(f"pinned step: tetris from {package_file}, commit {commit}", flush=True)
    if problems:
        for p in problems:
            print(f"NOT PINNED: {p}", flush=True)
        sys.exit(EXIT_NOT_PINNED)
    if argv[0] == "-m":
        sys.argv = [argv[1], *argv[2:]]
        sys.path[0] = str(root)
        runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
    else:
        script = root / argv[0]
        sys.argv = [str(script), *argv[1:]]
        sys.path[0] = str(script.parent)
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
