"""Screen base class and the app interface screens rely on."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

if TYPE_CHECKING:
    from tetris.ui.pygame.app import App


class Screen:
    """One full-window state (menu, game, settings). Screens are stacked by the app."""

    def __init__(self, app: App) -> None:
        self.app = app

    def handle_event(self, event: pygame.event.Event) -> None:
        """Process one Pygame event."""

    def update(self, dt: float) -> None:
        """Advance by ``dt`` seconds of real time."""

    def draw(self, surface: pygame.Surface) -> None:
        """Render the screen."""

    def on_focus_lost(self) -> None:
        """Called when the window loses focus."""
