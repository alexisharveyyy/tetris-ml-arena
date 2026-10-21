"""Race, Marathon and Battle: the human against 1 to 3 agents in real time."""

from __future__ import annotations

import random
import time

import pygame

from tetris.match import MatchConfig, MatchController
from tetris.match.agent_runtime import ProcessRunner
from tetris.match.history import MarathonHistory
from tetris.match.lobby import HUMAN_ID, Lobby
from tetris.match.rules import FIRST_TO_LINES, TARGET_LABELS, TARGET_MODES
from tetris.ui.pygame.input import KeyboardInput
from tetris.ui.pygame.render import (
    ACCENT,
    BG,
    DANGER,
    MUTED,
    TEXT,
    draw_menu,
    draw_overlay,
    draw_player_large,
    draw_player_small,
    draw_text,
    large_player_size,
    small_player_size,
)
from tetris.ui.pygame.screens.base import Screen

COUNTDOWN = 3.0
PAUSE_ITEMS = ("Resume", "End match", "Main menu")


class MatchScreen(Screen):
    def __init__(self, app, mode: str, lobby: Lobby, seed: int | None = None) -> None:
        super().__init__(app)
        self.mode = mode
        self.lobby = lobby
        self.runner = ProcessRunner()
        config = MatchConfig(
            mode=mode,
            seed=seed if seed is not None else random.randrange(2**31),
            race=lobby.race if mode == "race" else None,
            battle=lobby.battle if mode == "battle" else None,
            countdown=COUNTDOWN,
        )
        self.match = MatchController(config, lobby.player_specs(), self.runner)
        self.input = KeyboardInput(app.settings, self._on_move, self._toggle_pause)
        self.pause_index = 0
        self.load_started = time.perf_counter()
        self.startup_report: str | None = None
        self._results_shown = False
        self._blink = 0.0

    # ----------------------------------------------------------- control

    def _on_move(self, move) -> None:
        self.match.apply_input(HUMAN_ID, move)

    def _toggle_pause(self) -> None:
        if self.match.finished or self.match.countdown > 0:
            return
        self.match.set_paused(not self.match.paused)
        self.pause_index = 0
        self.input.reset()

    def _cycle_targeting(self) -> None:
        current = self.match.router.modes[HUMAN_ID]
        nxt = TARGET_MODES[(TARGET_MODES.index(current) + 1) % len(TARGET_MODES)]
        self.match.set_targeting(HUMAN_ID, nxt)

    def rematch(self, same_seed: bool) -> None:
        """Restart with the same agents (already loaded) and the same or a new seed."""
        self.match.reset(None if same_seed else random.randrange(2**31))
        self.match.countdown = COUNTDOWN
        self.input.reset()
        self._results_shown = False

    def leave(self) -> None:
        self.match.close()
        self.app.pop()

    def on_focus_lost(self) -> None:
        if not self.match.paused and not self.match.finished and self.match.countdown <= 0:
            self._toggle_pause()

    def handle_event(self, event: pygame.event.Event) -> None:
        m = self.match
        if m.paused:
            if event.type != pygame.KEYDOWN:
                return
            if event.key == pygame.K_ESCAPE:
                self._toggle_pause()
            elif event.key in (pygame.K_DOWN, pygame.K_UP):
                step = 1 if event.key == pygame.K_DOWN else -1
                self.pause_index = (self.pause_index + step) % len(PAUSE_ITEMS)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                choice = PAUSE_ITEMS[self.pause_index]
                if choice == "Resume":
                    self._toggle_pause()
                elif choice == "End match":
                    m.end_match()
                else:
                    self.leave()
            return
        if m.spectating:
            if event.type == pygame.KEYDOWN and event.key in (
                pygame.K_RETURN,
                pygame.K_KP_ENTER,
                pygame.K_ESCAPE,
            ):
                self.show_results()
            return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB and self.mode == "battle":
            self._cycle_targeting()
            return
        if (
            event.type == pygame.KEYDOWN
            and event.key == pygame.K_ESCAPE
            and (m.countdown > 0 or not m.ready)
        ):
            self.leave()
            return
        self.input.handle_event(event)

    def update(self, dt: float) -> None:
        m = self.match
        self._blink = (self._blink + dt) % 1.0
        if self.startup_report is None and m.ready:
            times = [t for t in self.runner.startup_times().values() if t is not None]
            self.startup_report = f"agents loaded in {max(times):.2f}s" if times else "agents ready"
        if m.ready and m.countdown <= 0 and not m.paused and not m.finished:
            self.input.update(dt)
        m.step(dt)
        m.pop_events()
        if m.finished and not self._results_shown:
            self._results_shown = True
            self.input.reset()
            if self.mode == "marathon":
                MarathonHistory(self.app.history_path).record_match(m.standings, m.config.seed)
            self.show_results()

    def show_results(self) -> None:
        from tetris.ui.pygame.screens.results import ResultsScreen

        self.app.push(ResultsScreen(self.app, self))

    def keep_watching(self) -> None:
        self.match.keep_watching()

    # -------------------------------------------------------------- draw

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG)
        snap = self.match.get_snapshot()
        w, h = surface.get_size()
        players = snap["players"]
        names = {p["id"]: ("You" if p["id"] == HUMAN_ID else p["name"]) for p in players}
        human = next(p for p in players if p["id"] == HUMAN_ID)
        agents = [p for p in players if p["id"] != HUMAN_ID]
        battle = self.mode == "battle"

        cell = max(14, min(28, (h - 120) // 22))
        pw, ph = large_player_size(cell)
        origin = (24, 70)
        draw_player_large(
            surface, human, origin, cell, snap["hidden_rows"], self.app.settings.show_ghost, battle
        )

        small = max(10, min(18, (w - pw - 80) // (len(agents) * 11 + 1)))
        sw, _ = small_player_size(small)
        x = origin[0] + pw + 30
        for p in agents:
            draw_player_small(
                surface, p, (x, 70), small, snap["hidden_rows"], names, battle, self._blink < 0.6
            )
            x += sw + 16

        self._draw_header(surface, snap, human, names)
        if snap["loading"]:
            draw_overlay(surface, 150)
            draw_text(surface, "Loading agents...", (w // 2, h // 2), 56, ACCENT, "center")
            elapsed = time.perf_counter() - self.load_started
            draw_text(surface, f"{elapsed:.1f}s", (w // 2, h // 2 + 50), 28, MUTED, "center")
        elif snap["countdown"] > 0:
            draw_overlay(surface, 100)
            n = int(snap["countdown"]) + 1
            draw_text(surface, str(n), (w // 2, h // 2), 160, ACCENT, "center")
            if self.startup_report:
                draw_text(surface, self.startup_report, (w // 2, h // 2 + 90), 24, MUTED, "center")
        elif snap["paused"]:
            draw_overlay(surface)
            draw_text(surface, "PAUSED", (w // 2, h // 2 - 110), 64, ACCENT, "center")
            draw_menu(
                surface, [(i, True) for i in PAUSE_ITEMS], self.pause_index, (w // 2, h // 2 + 10)
            )
        elif snap["spectating"]:
            draw_text(
                surface,
                "Watching the agents. Results are frozen (Enter: back to results)",
                (w // 2, h - 30),
                26,
                ACCENT,
                "center",
            )
        elif human["game_over"] and not snap["finished"]:
            draw_text(
                surface,
                "You topped out. Watching the others (Esc: pause menu)",
                (w // 2, h - 30),
                26,
                DANGER,
                "center",
            )

    def _draw_header(self, surface, snap, human, names) -> None:
        w, _ = surface.get_size()
        if self.mode == "marathon":
            draw_text(surface, "MARATHON: highest score when you top out", (24, 14), 32, ACCENT)
            leader = max(snap["players"], key=lambda p: p["score"])
            draw_text(
                surface,
                f"Level {human['level']}   Leader: {names[leader['id']]}",
                (w - 24, 14),
                28,
                TEXT,
                "topright",
            )
        elif self.mode == "race":
            race = snap["race"]
            draw_text(surface, f"RACE: {race['description']}", (24, 14), 32, ACCENT)
            if race["condition"] == FIRST_TO_LINES:
                status = f"Your lines {human['lines']} / {race['target_lines']}"
            else:
                remaining = race["time_remaining"]
                status = f"{int(remaining // 60)}:{int(remaining % 60):02d} left"
            draw_text(surface, status, (w - 24, 14), 32, TEXT, "topright")
        else:
            draw_text(surface, "BATTLE: last board standing", (24, 14), 32, ACCENT)
            mode = TARGET_LABELS[human["targeting"]] if human.get("targeting") else ""
            target = ", ".join(names[t] for t in human.get("targets", [])) or "-"
            draw_text(
                surface, f"Targeting: {mode} -> {target}  (Tab)", (w - 24, 14), 26, TEXT, "topright"
            )
            attackers = [names[a] for a in human.get("targeted_by", [])]
            if attackers:
                draw_text(
                    surface,
                    "Targeting you: " + ", ".join(attackers),
                    (w - 24, 40),
                    22,
                    DANGER,
                    "topright",
                )
        minutes, seconds = divmod(int(snap["time"]), 60)
        draw_text(surface, f"{minutes:02d}:{seconds:02d}", (w // 2, 14), 30, MUTED, "midtop")
