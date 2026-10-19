from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402
import pytest  # noqa: E402

from tetris.engine import Move  # noqa: E402


@pytest.fixture
def app(tmp_path, monkeypatch):
    from tetris.ui.pygame import app as app_module

    monkeypatch.setattr(app_module, "SCREENSHOT_DIR", tmp_path)
    instance = app_module.App()
    yield instance
    pygame.quit()


def press(key: int) -> None:
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0, unicode="", scancode=0))


def test_menu_solo_pause_and_back(app) -> None:
    from tetris.ui.pygame.screens.menu import MenuScreen
    from tetris.ui.pygame.screens.solo import SoloScreen

    app.push(MenuScreen(app))
    app.frame(1 / 60)
    press(pygame.K_RETURN)  # Solo practice is the first item
    app.frame(1 / 60)
    assert isinstance(app.screen, SoloScreen)
    solo = app.screen
    for i in range(60):
        if i % 10 == 9:
            solo._on_move(Move.HARD)
        app.frame(1 / 60)
    assert solo.match.get_snapshot()["players"][0]["pieces"] >= 5
    press(pygame.K_ESCAPE)
    app.frame(1 / 60)
    assert solo.match.paused
    press(pygame.K_DOWN)
    press(pygame.K_DOWN)
    press(pygame.K_RETURN)  # Main menu
    app.frame(1 / 60)
    assert isinstance(app.screen, MenuScreen)


def test_settings_screen_renders(app, tmp_path, monkeypatch) -> None:
    from tetris.ui.pygame.screens.settings_screen import SettingsScreen

    monkeypatch.setattr(app.settings, "save", lambda *a, **k: None)
    app.push(SettingsScreen(app))
    press(pygame.K_RETURN)  # rebind "Move left"
    press(pygame.K_a)
    app.frame(1 / 60)
    assert app.settings.bindings["left"][0] == "a"
