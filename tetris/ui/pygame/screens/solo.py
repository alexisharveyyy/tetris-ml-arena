"""Solo practice: one human board, endless marathon."""

from __future__ import annotations

import random

import pygame

from tetris.match import MatchConfig, MatchController, PlayerSpec
from tetris.ui.pygame.input import KeyboardInput
from tetris.ui.pygame.render import (
    ACCENT,
    BG,
    MUTED,
    TEXT,
    draw_menu,
    draw_overlay,
    draw_player_large,
    draw_text,
    large_player_size,
)
from tetris.ui.pygame.screens.base import Screen

PLAYER_ID = "human"
PAUSE_ITEMS = ("Resume", "Restart", "Main menu")


class SoloScreen(Screen):
    def __init__(self, app, seed: int | None = None) -> None:
        super().__init__(app)
        self.seed = seed if seed is not None else random.randrange(2**31)
        self.match = MatchController(
            MatchConfig(mode="solo", seed=self.seed), [PlayerSpec(PLAYER_ID, "You")]
        )
        self.input = KeyboardInput(app.settings, self._on_move, self._toggle_pause)
        self.pause_index = 0

    # ------------------------------------------------------------- input

    def _on_move(self, move) -> None:
        self.match.apply_input(PLAYER_ID, move)

    def _toggle_pause(self) -> None:
        if self.match.finished:
            return
        self.match.set_paused(not self.match.paused)
        self.pause_index = 0
        self.input.reset()

    def _restart(self, new_seed: bool = True) -> None:
        if new_seed:
            self.seed = random.randrange(2**31)
            self.match.config.seed = self.seed
        self.match.reset()
        self.input.reset()

    def on_focus_lost(self) -> None:
        if not self.match.finished and not self.match.paused:
            self._toggle_pause()

    def handle_event(self, event: pygame.event.Event) -> None:
        if self.match.finished:
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_r):
                    self._restart()
                elif event.key == pygame.K_ESCAPE:
                    self.app.pop()
            return
        if self.match.paused:
            if event.type != pygame.KEYDOWN:
                return
            if event.key == pygame.K_ESCAPE:
                self._toggle_pause()
            elif event.key == pygame.K_DOWN:
                self.pause_index = (self.pause_index + 1) % len(PAUSE_ITEMS)
            elif event.key == pygame.K_UP:
                self.pause_index = (self.pause_index - 1) % len(PAUSE_ITEMS)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                choice = PAUSE_ITEMS[self.pause_index]
                if choice == "Resume":
                    self._toggle_pause()
                elif choice == "Restart":
                    self._restart(new_seed=False)
                else:
                    self.app.pop()
            return
        self.input.handle_event(event)

    def update(self, dt: float) -> None:
        if not self.match.paused and not self.match.finished:
            self.input.update(dt)
        self.match.step(dt)
        self.match.pop_events()

    # ------------------------------------------------------------- draw

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG)
        snap = self.match.get_snapshot()
        player = snap["players"][0]
        w, h = surface.get_size()
        cell = max(16, min(34, (h - 90) // 22))
        pw, ph = large_player_size(cell)
        origin = ((w - pw) // 2, (h - ph) // 2 + 20)
        draw_player_large(
            surface, player, origin, cell, snap["hidden_rows"], self.app.settings.show_ghost
        )
        minutes, seconds = divmod(int(snap["time"]), 60)
        draw_text(surface, f"{minutes:02d}:{seconds:02d}", (w - 20, 16), 32, TEXT, "topright")
        draw_text(surface, f"Seed {self.seed}", (20, 16), 22, MUTED)
        draw_text(surface, "Esc: pause", (20, h - 28), 22, MUTED)

        if snap["paused"]:
            draw_overlay(surface)
            draw_text(surface, "PAUSED", (w // 2, h // 2 - 110), 64, ACCENT, "center")
            draw_menu(
                surface, [(i, True) for i in PAUSE_ITEMS], self.pause_index, (w // 2, h // 2 + 10)
            )
        elif snap["finished"]:
            draw_overlay(surface)
            draw_text(surface, "GAME OVER", (w // 2, h // 2 - 90), 72, ACCENT, "center")
            draw_text(
                surface,
                f"Score {player['score']:,}    Lines {player['lines']}    "
                f"Pieces {player['pieces']}",
                (w // 2, h // 2 - 20),
                34,
                TEXT,
                "center",
            )
            draw_text(
                surface,
                "Enter: play again    Esc: main menu",
                (w // 2, h // 2 + 40),
                28,
                MUTED,
                "center",
            )
