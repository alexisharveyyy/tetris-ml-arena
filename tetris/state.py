"""Where per-user state lives: settings, Marathon history and human recordings.

By default that is ``configs/user_settings.yaml``, ``configs/user_history.json``
and ``data/recordings/``. When ``TETRIS_STATE_DIR`` is set, all three move under
it (``user_settings.yaml``, ``user_history.json``, ``recordings/``), so tests,
rehearsals and test servers never touch the real files. The Python and
Playwright test runs set it to a temporary directory and fail if the real files
change while they run (:func:`fingerprint`).

The defaults are read when :mod:`tetris.settings`, :mod:`tetris.match.history`
and :mod:`tetris.match.recording` are imported, so set the variable first.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

STATE_ENV = "TETRIS_STATE_DIR"
ROOT = Path(__file__).resolve().parent.parent

#: The real state files, for :func:`fingerprint`.
REAL_SETTINGS = ROOT / "configs" / "user_settings.yaml"
REAL_HISTORY = ROOT / "configs" / "user_history.json"
REAL_RECORDINGS = ROOT / "data" / "recordings"


def state_dir() -> Path | None:
    value = os.environ.get(STATE_ENV)
    return Path(value).expanduser() if value else None


def settings_path(root: Path | None = None) -> Path:
    root = root or state_dir()
    return root / "user_settings.yaml" if root else REAL_SETTINGS


def history_path(root: Path | None = None) -> Path:
    root = root or state_dir()
    return root / "user_history.json" if root else REAL_HISTORY


def recordings_dir(root: Path | None = None) -> Path:
    root = root or state_dir()
    return root / "recordings" if root else REAL_RECORDINGS


def fingerprint(
    paths: tuple[Path, ...] = (REAL_SETTINGS, REAL_HISTORY, REAL_RECORDINGS),
) -> dict[str, str]:
    """Content hash of each file under ``paths`` (``"missing"`` if absent)."""
    out: dict[str, str] = {}
    for path in paths:
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        if not files:
            out[str(path)] = "missing" if not path.exists() else "empty"
        for f in files:
            out[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest() if f.exists() else "missing"
    return out


def changed(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Paths whose fingerprint differs, including files added or removed."""
    return sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
