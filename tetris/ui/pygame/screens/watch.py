"""Watch AI: pick an agent, then watch it play with adjustable speed."""

from __future__ import annotations

import random

import pygame

from tetris.agents.difficulty import PRESET_NAMES, PRESETS
from tetris.agents.registry import AgentSpec, available_agents
from tetris.match import MatchConfig, MatchController, PlayerSpec
from tetris.ui.pygame.render import (
    ACCENT,
    BG,
    CANDIDATE_COLORS,
    MUTED,
    TEXT,
    draw_menu,
    draw_overlay,
    draw_player_large,
    draw_text,
    large_player_size,
)
from tetris.ui.pygame.screens.base import Screen

PLAYER_ID = "ai"
#: Free speed slider stops, in pieces per second (None = unlimited).
PPS_STOPS: tuple[float | None, ...] = (
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    8.0,
    10.0,
    15.0,
    20.0,
    30.0,
    None,
)


class WatchSelectScreen(Screen):
    """Choose which agent to watch."""

    def __init__(self, app) -> None:
        super().__init__(app)
        self.agents: list[AgentSpec] = available_agents()
        self.selected = 0

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.app.pop()
        elif not self.agents:
            return
        elif event.key == pygame.K_DOWN:
            self.selected = (self.selected + 1) % len(self.agents)
        elif event.key == pygame.K_UP:
            self.selected = (self.selected - 1) % len(self.agents)
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            self.app.push(WatchScreen(self.app, self.agents[self.selected]))

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG)
        w, h = surface.get_size()
        draw_text(surface, "WATCH AI", (w // 2, 80), 64, ACCENT, "center")
        if not self.agents:
            draw_text(
                surface,
                "No trained agents found in checkpoints/",
                (w // 2, h // 2),
                32,
                MUTED,
                "center",
            )
        else:
            draw_menu(
                surface, [(a.name, True) for a in self.agents], self.selected, (w // 2, h // 2 - 40)
            )
            desc = self.agents[self.selected].description
            draw_text(surface, desc, (w // 2, h - 90), 28, MUTED, "center")
        draw_text(surface, "Enter: watch    Esc: back", (w // 2, h - 30), 22, MUTED, "center")


class WatchScreen(Screen):
    """One agent playing at real-time speed with a speed control and candidate overlay."""

    def __init__(self, app, spec: AgentSpec, seed: int | None = None) -> None:
        super().__init__(app)
        self.spec = spec
        self.preset = "Medium"
        self.pps = PRESETS[self.preset].pps
        self.show_candidates = False
        self.seed = seed if seed is not None else random.randrange(2**31)
        self.match = self._new_match()

    def _difficulty(self):
        return (
            PRESETS[self.preset].with_pps(self.pps)
            if self.pps != PRESETS[self.preset].pps
            else PRESETS[self.preset]
        )

    def _new_match(self) -> MatchController:
        return MatchController(
            MatchConfig(mode="watch", seed=self.seed),
            [PlayerSpec(PLAYER_ID, self.spec.name, "agent", self.spec.key, self._difficulty())],
            top_k=3 if self.show_candidates else 0,
        )

    def _apply_difficulty(self) -> None:
        self.match.set_difficulty(PLAYER_ID, self._difficulty())

    def _set_preset(self, name: str) -> None:
        self.preset = name
        self.pps = PRESETS[name].pps
        self._apply_difficulty()

    def _nudge_speed(self, step: int) -> None:
        stops = list(PPS_STOPS)
        if self.pps in stops:
            i = stops.index(self.pps)
        else:
            i = min(range(len(stops) - 1), key=lambda j: abs((stops[j] or 1e9) - self.pps))
        self.pps = stops[max(0, min(len(stops) - 1, i + step))]
        self._apply_difficulty()

    def on_focus_lost(self) -> None:
        pass  # watching continues in the background

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        key = event.key
        if key == pygame.K_ESCAPE:
            self.match.close()
            self.app.pop()
        elif key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
            self._set_preset(PRESET_NAMES[key - pygame.K_1])
        elif key in (pygame.K_RIGHT, pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self._nudge_speed(1)
        elif key in (pygame.K_LEFT, pygame.K_MINUS, pygame.K_KP_MINUS):
            self._nudge_speed(-1)
        elif key == pygame.K_t:
            self.show_candidates = not self.show_candidates
            self.match.set_top_k(3 if self.show_candidates else 0)
        elif key in (pygame.K_SPACE, pygame.K_p):
            self.match.set_paused(not self.match.paused)
        elif key == pygame.K_r:
            self.match.close()
            self.seed = random.randrange(2**31)
            self.match = self._new_match()

    def update(self, dt: float) -> None:
        self.match.step(dt)
        self.match.pop_events()

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG)
        snap = self.match.get_snapshot()
        player = snap["players"][0]
        w, h = surface.get_size()
        cell = max(16, min(32, (h - 90) // 22))
        pw, ph = large_player_size(cell)
        origin = (40, (h - ph) // 2 + 20)
        draw_player_large(surface, player, origin, cell, snap["hidden_rows"], True)

        x = origin[0] + pw + 50
        y = origin[1]
        draw_text(surface, self.spec.name, (x, y), 44, ACCENT)
        draw_text(surface, self.spec.description, (x, y + 40), 24, MUTED)
        y += 90
        diff = self._difficulty()
        speed = "unlimited" if diff.pps is None else f"{diff.pps:g} pieces/s"
        elapsed = max(snap["time"], 1e-6)
        rows = [
            (
                "Preset",
                self.preset if self.pps == PRESETS[self.preset].pps else f"{self.preset} (custom)",
            ),
            ("Speed cap", speed),
            ("Actual speed", f"{player['pieces'] / elapsed:.2f} pieces/s"),
            ("Reaction delay", f"{diff.reaction_delay * 1000:.0f} ms"),
            ("Noise (epsilon)", f"{diff.epsilon:g}"),
            ("Top 3 overlay", "on" if self.show_candidates else "off"),
            ("Seed", str(self.seed)),
        ]
        for label, value in rows:
            draw_text(surface, label, (x, y), 26, MUTED)
            draw_text(surface, value, (x + 210, y), 26, TEXT)
            y += 32
        indicator = "AI is thinking..." if player["thinking"] else ""
        draw_text(surface, indicator, (x, y + 8), 26, ACCENT)
        y += 50
        if self.show_candidates:
            for i, color in enumerate(CANDIDATE_COLORS):
                draw_text(
                    surface, f"{i + 1}: candidate rank {i + 1} (H = uses hold)", (x, y), 22, color
                )
                y += 24
            y += 10
        help_lines = (
            "1-4: Easy / Medium / Hard / Unlimited",
            "Left/Right or -/+: adjust speed",
            "T: toggle top 3 candidates",
            "Space: pause    R: new game    Esc: back",
        )
        for line in help_lines:
            draw_text(surface, line, (x, y), 22, MUTED)
            y += 24

        if snap["paused"]:
            draw_overlay(surface, 120)
            draw_text(surface, "PAUSED", (w // 2, h // 2), 64, ACCENT, "center")
        elif player["game_over"]:
            draw_overlay(surface, 120)
            draw_text(surface, "TOPPED OUT", (w // 2, h // 2 - 20), 64, ACCENT, "center")
            draw_text(
                surface, "R: new game    Esc: back", (w // 2, h // 2 + 40), 28, MUTED, "center"
            )
