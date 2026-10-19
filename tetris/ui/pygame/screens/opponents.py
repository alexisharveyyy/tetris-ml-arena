"""Opponent selection for Race and Battle."""

from __future__ import annotations

import pygame

from tetris.agents.difficulty import nudge_pps
from tetris.agents.registry import AGENTS, available_agents
from tetris.match.history import MarathonHistory
from tetris.match.lobby import CUSTOM, DIFFICULTY_CHOICES, MAX_OPPONENTS, Lobby, OpponentSlot
from tetris.match.rules import (
    FIRST_TO_LINES,
    RACE_CONDITION_LABELS,
    RACE_CONDITIONS,
    TARGET_LABELS,
    TARGET_MODES,
)
from tetris.ui.pygame.render import ACCENT, BG, MUTED, TEXT, WARNING, draw_text
from tetris.ui.pygame.screens.base import Screen

LINE_TARGETS = (10, 20, 40, 100, 150)
MINUTE_LIMITS = (1, 2, 3, 5, 10)


def _cycle(options: list, current, step: int):
    i = options.index(current) if current in options else 0
    return options[(i + step) % len(options)]


class OpponentSelectScreen(Screen):
    def __init__(self, app, mode: str) -> None:
        super().__init__(app)
        self.mode = mode
        self.available = [a.key for a in available_agents()]
        saved = app.settings.last_match.get(mode)
        self.lobby = (
            Lobby.from_dict(mode, saved, self.available)
            if saved
            else Lobby.default(mode, self.available)
        )
        self.selected = 0
        self.message = ""
        self.history = MarathonHistory(app.history_path)

    # ---------------------------------------------------------------- rows

    def rows(self) -> list[tuple[str, int]]:
        rows: list[tuple[str, int]] = []
        for i, slot in enumerate(self.lobby.slots):
            rows.append(("agent", i))
            if slot is not None:
                rows.append(("difficulty", i))
                if slot.preset == CUSTOM:
                    rows += [("pps", i), ("reaction", i), ("noise", i)]
        if self.mode == "race":
            rows += [("condition", 0), ("goal", 0)]
        elif self.mode == "battle":
            rows.append(("targeting", 0))
        rows += [("start", 0), ("back", 0)]
        return rows

    def _row_text(self, kind: str, i: int) -> tuple[str, str]:
        lobby = self.lobby
        slot = (
            lobby.slots[i] if kind in ("agent", "difficulty", "pps", "reaction", "noise") else None
        )
        if kind == "agent":
            return f"Opponent {i + 1}", AGENTS[slot.agent_key].name if slot else "(empty)"
        if kind == "difficulty":
            if self.mode == "marathon":
                record = self.history.describe(slot.agent_key, slot.preset)
                return "    Difficulty", f"{slot.preset}   (your record: {record})"
            return "    Difficulty", slot.preset
        if kind == "pps":
            pps = slot.difficulty.pps
            return "      Speed cap", "unlimited" if pps is None else f"{pps:g} pieces/s"
        if kind == "reaction":
            return "      Reaction delay", f"{slot.difficulty.reaction_delay * 1000:.0f} ms"
        if kind == "noise":
            return "      Mistake rate", f"{slot.difficulty.noise:.2f}"
        if kind == "condition":
            return "Win condition", RACE_CONDITION_LABELS[lobby.race.condition]
        if kind == "goal":
            if lobby.race.condition == FIRST_TO_LINES:
                return "Target lines", str(lobby.race.target_lines)
            return "Time limit", f"{lobby.race.time_limit / 60:g} min"
        if kind == "targeting":
            return "Default targeting", TARGET_LABELS[lobby.battle.targeting]
        if kind == "start":
            return "Start match", ""
        return "Back", ""

    # --------------------------------------------------------------- input

    def _adjust(self, kind: str, i: int, step: int) -> None:
        lobby = self.lobby
        slot = lobby.slots[i] if i < len(lobby.slots) else None
        if kind == "agent":
            choices: list = list(self.available)
            if i > 0:
                choices = [None, *choices]  # opponent 1 is required
            current = slot.agent_key if slot else None
            new = _cycle(choices, current, step)
            if new is None:
                lobby.slots[i] = None
            elif slot is None:
                lobby.slots[i] = OpponentSlot(new)
            else:
                slot.agent_key = new
        elif kind == "difficulty" and slot:
            slot.set_preset(_cycle(list(DIFFICULTY_CHOICES), slot.preset, step))
        elif kind == "pps" and slot:
            pps = nudge_pps(slot.difficulty.pps, step)
            slot.set_custom(**slot.difficulty.with_pps(pps).to_dict())
        elif kind == "reaction" and slot:
            value = max(0.0, min(1.5, slot.difficulty.reaction_delay + 0.05 * step))
            slot.set_custom(reaction_delay=round(value, 2))
        elif kind == "noise" and slot:
            value = max(0.0, min(0.6, slot.difficulty.noise + 0.05 * step))
            slot.set_custom(noise=round(value, 2))
        elif kind == "condition":
            lobby.race.condition = _cycle(list(RACE_CONDITIONS), lobby.race.condition, step)
        elif kind == "goal":
            if lobby.race.condition == FIRST_TO_LINES:
                lobby.race.target_lines = _cycle(list(LINE_TARGETS), lobby.race.target_lines, step)
            else:
                minutes = _cycle(list(MINUTE_LIMITS), int(lobby.race.time_limit // 60), step)
                lobby.race.time_limit = 60.0 * minutes
        elif kind == "targeting":
            lobby.battle.targeting = _cycle(list(TARGET_MODES), lobby.battle.targeting, step)

    def _start(self) -> None:
        from tetris.ui.pygame.screens.match_screen import MatchScreen

        if not self.lobby.valid(self.available):
            self.message = "Pick at least one opponent with a checkpoint."
            return
        self.app.settings.last_match[self.mode] = self.lobby.to_dict()
        self.app.settings.save()
        self.app.push(MatchScreen(self.app, self.mode, self.lobby))

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        rows = self.rows()
        self.selected = min(self.selected, len(rows) - 1)
        kind, i = rows[self.selected]
        if event.key == pygame.K_ESCAPE:
            self.app.pop()
        elif event.key == pygame.K_DOWN:
            self.selected = (self.selected + 1) % len(rows)
        elif event.key == pygame.K_UP:
            self.selected = (self.selected - 1) % len(rows)
        elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
            self._adjust(kind, i, 1 if event.key == pygame.K_RIGHT else -1)
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
            if kind == "start":
                self._start()
            elif kind == "back":
                self.app.pop()
            else:
                self._adjust(kind, i, 1)

    # ---------------------------------------------------------------- draw

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG)
        w, h = surface.get_size()
        title = self.mode.upper()
        draw_text(surface, f"{title}: CHOOSE OPPONENTS", (w // 2, 50), 56, ACCENT, "center")
        rows = self.rows()
        spacing = min(36, (h - 220) // max(1, len(rows)))
        top = 100
        for n, (kind, i) in enumerate(rows):
            label, value = self._row_text(kind, i)
            color = ACCENT if n == self.selected else TEXT
            y = top + n * spacing
            draw_text(surface, label, (w // 2 - 30, y), 30, color, "topright")
            shown = (
                f"< {value} >"
                if value and n == self.selected and kind not in ("start", "back")
                else value
            )
            draw_text(surface, shown, (w // 2 + 30, y), 30, color if value != "(empty)" else MUTED)
        kind, i = rows[min(self.selected, len(rows) - 1)]
        slot = (
            self.lobby.slots[i]
            if kind in ("agent", "difficulty", "pps", "reaction", "noise")
            else None
        )
        if slot is not None:
            draw_text(
                surface, AGENTS[slot.agent_key].description, (w // 2, h - 95), 26, MUTED, "center"
            )
        if self.message:
            draw_text(surface, self.message, (w // 2, h - 65), 26, WARNING, "center")
        draw_text(
            surface,
            f"Up/Down: choose    Left/Right: change    Enter: select    Esc: back"
            f"    (up to {MAX_OPPONENTS} opponents, duplicates allowed)",
            (w // 2, h - 30),
            22,
            MUTED,
            "center",
        )
