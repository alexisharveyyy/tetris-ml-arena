"""Pygame application shell: window, main loop and screen stack."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pygame

from tetris.settings import Settings
from tetris.ui.pygame.render import clear_font_cache
from tetris.ui.pygame.screens.base import Screen

WINDOW_SIZE = (1280, 800)
MAX_DT = 0.1
SCREENSHOT_DIR = Path("screenshots")


class App:
    def __init__(self, size: tuple[int, int] = WINDOW_SIZE) -> None:
        pygame.init()
        clear_font_cache()
        pygame.display.set_caption("Tetris ML Arena")
        self.surface = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.settings = Settings.load()
        self.screens: list[Screen] = []
        self.running = True

    # ------------------------------------------------------- screen stack

    def push(self, screen: Screen) -> None:
        self.screens.append(screen)

    def pop(self) -> None:
        self.screens.pop()
        if not self.screens:
            self.running = False

    def replace(self, screen: Screen) -> None:
        self.screens[-1] = screen

    def quit(self) -> None:
        self.running = False

    @property
    def screen(self) -> Screen:
        return self.screens[-1]

    # ------------------------------------------------------------ loop

    def save_screenshot(self) -> Path:
        SCREENSHOT_DIR.mkdir(exist_ok=True)
        path = SCREENSHOT_DIR / f"arena_{time.strftime('%Y%m%d_%H%M%S')}.png"
        pygame.image.save(self.surface, str(path))
        return path

    def frame(self, dt: float) -> None:
        """Process events, update and draw one frame."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.WINDOWFOCUSLOST:
                self.screen.on_focus_lost()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F12:
                self.save_screenshot()
            else:
                self.screen.handle_event(event)
            if not self.running:
                return
        self.screen.update(min(dt, MAX_DT))
        self.screen.draw(self.surface)
        pygame.display.flip()

    def run(self) -> None:
        self.clock.tick()
        while self.running and self.screens:
            dt = self.clock.tick(self.settings.fps) / 1000.0
            self.frame(dt)
        pygame.quit()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tetris ML Arena")
    parser.add_argument("--solo", action="store_true", help="start directly in solo practice")
    parser.add_argument("--seed", type=int, default=None, help="piece sequence seed for --solo")
    args = parser.parse_args(argv)

    from tetris.ui.pygame.screens.menu import MenuScreen
    from tetris.ui.pygame.screens.solo import SoloScreen

    app = App()
    app.push(MenuScreen(app))
    if args.solo:
        app.push(SoloScreen(app, seed=args.seed))
    app.run()


if __name__ == "__main__":
    main()
