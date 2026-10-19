"""User settings: key bindings and input timing, persisted as YAML.

Key names are frontend-neutral strings (Pygame's ``pygame.key.name`` values,
which also map cleanly to browser ``KeyboardEvent.key`` names).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tetris.match.das import RepeatSettings

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "configs" / "user_settings.yaml"

#: Actions that can be bound, in display order.
ACTIONS: tuple[str, ...] = ("left", "right", "soft", "hard", "cw", "ccw", "hold", "pause")

ACTION_LABELS: dict[str, str] = {
    "left": "Move left",
    "right": "Move right",
    "soft": "Soft drop",
    "hard": "Hard drop",
    "cw": "Rotate clockwise",
    "ccw": "Rotate counterclockwise",
    "hold": "Hold",
    "pause": "Pause",
}


def default_bindings() -> dict[str, list[str]]:
    return {
        "left": ["left"],
        "right": ["right"],
        "soft": ["down"],
        "hard": ["space"],
        "cw": ["up", "x"],
        "ccw": ["z"],
        "hold": ["c", "left shift", "right shift"],
        "pause": ["escape"],
    }


@dataclass
class Settings:
    bindings: dict[str, list[str]] = field(default_factory=default_bindings)
    das_ms: int = 133
    arr_ms: int = 10
    soft_drop_rate: float = 30.0
    show_ghost: bool = True
    fps: int = 60
    #: Last opponent selection and rules per mode ("race", "battle"), restored next time.
    last_match: dict[str, Any] = field(default_factory=dict)

    def repeat_settings(self) -> RepeatSettings:
        return RepeatSettings(self.das_ms / 1000.0, self.arr_ms / 1000.0, self.soft_drop_rate)

    def key_to_action(self) -> dict[str, str]:
        """Reverse lookup from key name to action."""
        return {key: action for action, keys in self.bindings.items() for key in keys}

    def bind(self, action: str, key: str) -> None:
        """Bind ``key`` to ``action`` as its primary key, removing it from other actions."""
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action}")
        for keys in self.bindings.values():
            if key in keys:
                keys.remove(key)
        self.bindings.setdefault(action, [])
        self.bindings[action].insert(0, key)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Settings:
        s = cls()
        for k, v in d.items():
            if hasattr(s, k):
                setattr(s, k, v)
        for action, keys in default_bindings().items():
            s.bindings.setdefault(action, keys)
        return s

    @classmethod
    def load(cls, path: Path = DEFAULT_SETTINGS_PATH) -> Settings:
        if not path.exists():
            return cls()
        return cls.from_dict(yaml.safe_load(path.read_text()) or {})

    def save(self, path: Path = DEFAULT_SETTINGS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
