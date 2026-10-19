"""Settings: key rebinding, DAS, ARR, soft drop speed, ghost piece."""

from __future__ import annotations

import pygame

from tetris.settings import ACTION_LABELS, ACTIONS, Settings
from tetris.ui.pygame.render import ACCENT, BG, MUTED, TEXT, draw_text
from tetris.ui.pygame.screens.base import Screen

# (attribute, label, step, minimum, maximum)
NUMERIC = (
    ("das_ms", "DAS (ms)", 5, 0, 500),
    ("arr_ms", "ARR (ms, 0 = instant)", 1, 0, 200),
    ("soft_drop_rate", "Soft drop (rows/s)", 5, 5, 200),
)


class SettingsScreen(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.settings: Settings = app.settings
        self.rows: list[tuple[str, str]] = (
            [("action", a) for a in ACTIONS]
            + [("numeric", n[0]) for n in NUMERIC]
            + [("toggle", "show_ghost"), ("reset", ""), ("back", "")]
        )
        self.selected = 0
        self.capturing: str | None = None

    def _leave(self) -> None:
        self.settings.save()
        self.app.pop()

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if self.capturing is not None:
            if event.key != pygame.K_ESCAPE or self.capturing == "pause":
                self.settings.bind(self.capturing, pygame.key.name(event.key))
            self.capturing = None
            return
        kind, key = self.rows[self.selected]
        if event.key == pygame.K_ESCAPE:
            self._leave()
        elif event.key == pygame.K_DOWN:
            self.selected = (self.selected + 1) % len(self.rows)
        elif event.key == pygame.K_UP:
            self.selected = (self.selected - 1) % len(self.rows)
        elif event.key in (pygame.K_LEFT, pygame.K_RIGHT) and kind == "numeric":
            _, _, step, lo, hi = next(n for n in NUMERIC if n[0] == key)
            delta = step if event.key == pygame.K_RIGHT else -step
            setattr(self.settings, key, max(lo, min(hi, getattr(self.settings, key) + delta)))
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            if kind == "action":
                self.capturing = key
            elif kind == "toggle":
                setattr(self.settings, key, not getattr(self.settings, key))
            elif kind == "reset":
                fresh = Settings()
                self.settings.__dict__.update(fresh.__dict__)
            elif kind == "back":
                self._leave()
        elif event.key in (pygame.K_LEFT, pygame.K_RIGHT) and kind == "toggle":
            setattr(self.settings, key, not getattr(self.settings, key))

    def _row_text(self, kind: str, key: str) -> tuple[str, str]:
        s = self.settings
        if kind == "action":
            value = (
                "press a key..." if self.capturing == key else ", ".join(s.bindings.get(key, []))
            )
            return ACTION_LABELS[key], value
        if kind == "numeric":
            label = next(n[1] for n in NUMERIC if n[0] == key)
            value = getattr(s, key)
            return label, f"< {value:g} >"
        if kind == "toggle":
            return "Ghost piece", "on" if s.show_ghost else "off"
        if kind == "reset":
            return "Reset to defaults", ""
        return "Save and back", ""

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG)
        w, h = surface.get_size()
        draw_text(surface, "SETTINGS", (w // 2, 60), 64, ACCENT, "center")
        top = 120
        spacing = min(40, (h - 200) // len(self.rows))
        for i, (kind, key) in enumerate(self.rows):
            label, value = self._row_text(kind, key)
            y = top + i * spacing
            color = ACCENT if i == self.selected else TEXT
            draw_text(surface, label, (w // 2 - 40, y), 32, color, "topright")
            draw_text(surface, value, (w // 2 + 40, y), 32, color if value else MUTED)
        help_text = (
            "Press the new key (Esc to cancel)"
            if self.capturing
            else "Enter: rebind or select    Left/Right: adjust    Esc: save and back"
        )
        draw_text(surface, help_text, (w // 2, h - 30), 24, MUTED, "center")
