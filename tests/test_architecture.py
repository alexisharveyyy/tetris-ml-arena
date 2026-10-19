"""Pygame must not leak outside tetris/ui/pygame."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "tetris"
PYGAME_IMPORT = re.compile(r"^\s*(import|from)\s+pygame\b", re.MULTILINE)


def test_no_pygame_outside_ui() -> None:
    offenders = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*.py")
        if "ui/pygame" not in path.as_posix() and PYGAME_IMPORT.search(path.read_text())
    ]
    assert offenders == []
