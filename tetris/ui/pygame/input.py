"""Keyboard handling: Pygame key events to abstract moves via settings and DAS."""

from __future__ import annotations

from collections.abc import Callable

import pygame

from tetris.engine.game import Move
from tetris.match.das import InputRepeater
from tetris.settings import Settings


class KeyboardInput:
    """Maps key events to moves for one human player.

    ``on_move`` receives every move to apply; ``on_pause`` fires for the pause
    binding.
    """

    def __init__(
        self,
        settings: Settings,
        on_move: Callable[[Move], None],
        on_pause: Callable[[], None],
    ) -> None:
        self.keymap = settings.key_to_action()
        self.repeater = InputRepeater(settings.repeat_settings())
        self.on_move = on_move
        self.on_pause = on_pause

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type not in (pygame.KEYDOWN, pygame.KEYUP):
            return
        action = self.keymap.get(pygame.key.name(event.key))
        if action is None:
            return
        if action == "pause":
            if event.type == pygame.KEYDOWN:
                self.on_pause()
            return
        move = Move(action)
        if event.type == pygame.KEYDOWN:
            for m in self.repeater.press(move):
                self.on_move(m)
        else:
            self.repeater.release(move)

    def update(self, dt: float) -> None:
        for m in self.repeater.update(dt):
            self.on_move(m)

    def reset(self) -> None:
        """Forget held keys (after pause or focus loss)."""
        self.repeater.reset()
