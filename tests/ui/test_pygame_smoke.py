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


def test_watch_ai_flow(app) -> None:
    from tetris.agents.difficulty import PRESETS
    from tetris.agents.registry import AGENTS
    from tetris.ui.pygame.screens.menu import MenuScreen
    from tetris.ui.pygame.screens.watch import WatchScreen, WatchSelectScreen

    app.push(MenuScreen(app))
    press(pygame.K_DOWN)  # Watch AI
    press(pygame.K_RETURN)
    app.frame(1 / 60)
    assert isinstance(app.screen, WatchSelectScreen)

    watch = WatchScreen(app, AGENTS["dellacherie"], seed=1)
    app.push(watch)
    press(pygame.K_4)  # Unlimited
    press(pygame.K_t)  # top 3 overlay
    for _ in range(30):
        app.frame(1 / 60)
    assert watch.pps is None and watch.show_candidates
    assert watch.match.get_snapshot()["players"][0]["pieces"] >= 20
    press(pygame.K_1)  # Easy
    press(pygame.K_RIGHT)  # nudge speed up from the preset
    app.frame(1 / 60)
    assert watch.preset == "Easy" and watch.pps > PRESETS["Easy"].pps
    press(pygame.K_ESCAPE)
    app.frame(1 / 60)
    assert isinstance(app.screen, WatchSelectScreen)


@pytest.mark.parametrize("mode", ["race", "battle"])
def test_opponent_selection_match_and_results(app, monkeypatch, mode: str) -> None:
    from tetris.match.agent_runtime import InlineRunner
    from tetris.match.rules import RaceRules
    from tetris.ui.pygame.screens import match_screen
    from tetris.ui.pygame.screens.match_screen import MatchScreen
    from tetris.ui.pygame.screens.opponents import OpponentSelectScreen
    from tetris.ui.pygame.screens.results import ResultsScreen

    monkeypatch.setattr(match_screen, "ProcessRunner", InlineRunner)  # keep the UI test fast
    monkeypatch.setattr(InlineRunner, "startup_times", lambda self: {}, raising=False)
    saved = {}
    monkeypatch.setattr(app.settings, "save", lambda *a, **k: saved.update(app.settings.last_match))

    select = OpponentSelectScreen(app, mode)
    app.push(select)
    press(pygame.K_RIGHT)  # cycle opponent 1
    app.frame(1 / 60)
    if mode == "race":
        select.lobby.race = RaceRules(time_limit=5.0)
    rows = select.rows()
    select.selected = rows.index(("start", 0))
    press(pygame.K_RETURN)
    app.frame(1 / 60)
    assert isinstance(app.screen, MatchScreen) and mode in saved
    screen = app.screen
    if mode == "battle":
        before = screen.match.router.modes["human"]
        press(pygame.K_TAB)
        app.frame(1 / 60)
        assert screen.match.router.modes["human"] != before
    for _ in range(60 * 60):
        screen._on_move(Move.HARD)
        app.frame(1 / 30)
        if isinstance(app.screen, ResultsScreen):
            break
    assert isinstance(app.screen, ResultsScreen)
    seed = screen.match.config.seed
    press(pygame.K_r)  # rematch, same seed
    app.frame(1 / 60)
    assert app.screen is screen and screen.match.config.seed == seed and not screen.match.finished

    # A new opponent selection screen restores the saved lobby.
    again = OpponentSelectScreen(app, mode)
    assert again.lobby.to_dict()["slots"] == saved[mode]["slots"]
