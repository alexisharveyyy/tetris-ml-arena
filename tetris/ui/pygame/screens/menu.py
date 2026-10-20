"""Main menu."""

from __future__ import annotations

from collections.abc import Callable

import pygame

from tetris.ui.pygame.render import ACCENT, BG, MUTED, draw_menu, draw_text
from tetris.ui.pygame.screens.base import Screen


class MenuScreen(Screen):
    def __init__(self, app) -> None:
        super().__init__(app)
        from tetris.ui.pygame.screens.opponents import OpponentSelectScreen
        from tetris.ui.pygame.screens.record import RecordScreen
        from tetris.ui.pygame.screens.settings_screen import SettingsScreen
        from tetris.ui.pygame.screens.solo import SoloScreen
        from tetris.ui.pygame.screens.watch import WatchSelectScreen

        # (label, action or None if not available yet, note for unavailable items)
        self.items: list[tuple[str, Callable[[], None] | None, str]] = [
            ("Solo practice", lambda: app.push(SoloScreen(app)), ""),
            ("Watch AI", lambda: app.push(WatchSelectScreen(app)), ""),
            ("Race", lambda: app.push(OpponentSelectScreen(app, "race")), ""),
            ("Marathon", lambda: app.push(OpponentSelectScreen(app, "marathon")), ""),
            ("Record Gameplay", lambda: app.push(RecordScreen(app)), ""),
            ("Leaderboard", None, "Available in phase 7"),
            ("Settings", lambda: app.push(SettingsScreen(app)), ""),
            ("Quit", app.quit, ""),
        ]
        self.selected = 0

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_DOWN, pygame.K_s):
            self.selected = (self.selected + 1) % len(self.items)
        elif event.key in (pygame.K_UP, pygame.K_w):
            self.selected = (self.selected - 1) % len(self.items)
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            action = self.items[self.selected][1]
            if action is not None:
                action()
        elif event.key == pygame.K_ESCAPE:
            self.app.quit()

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG)
        w, h = surface.get_size()
        draw_text(surface, "TETRIS ML ARENA", (w // 2, 120), 84, ACCENT, "center")
        draw_text(surface, "Human versus machine learning", (w // 2, 175), 30, MUTED, "center")
        draw_menu(
            surface,
            [(label, action is not None) for label, action, _ in self.items],
            self.selected,
            (w // 2, h // 2 + 60),
        )
        note = self.items[self.selected][2]
        if note:
            draw_text(surface, note, (w // 2, h - 60), 26, MUTED, "center")
        draw_text(
            surface, "Arrows to choose, Enter to select", (w // 2, h - 30), 22, MUTED, "center"
        )
