"""End-of-match standings with rematch options."""

from __future__ import annotations

import pygame

from tetris.ui.pygame.render import ACCENT, BG, DANGER, MUTED, TEXT, draw_text, draw_text_fit
from tetris.ui.pygame.screens.base import Screen

COLUMNS = (
    ("Place", 80, lambda r: f"{r['place']}"),
    ("Player", 400, lambda r: r["name"]),
    ("Lines", 90, lambda r: f"{r['lines']}"),
    ("Attack sent", 140, lambda r: f"{r['attack_sent']}"),
    ("Pieces/s", 120, lambda r: f"{r['pps']:.2f}"),
    ("Score", 140, lambda r: f"{r['score']:,}"),
    ("Status", 150, lambda r: "topped out" if r["topped_out"] else "finished"),
)


class ResultsScreen(Screen):
    def __init__(self, app, match_screen) -> None:
        super().__init__(app)
        self.match_screen = match_screen
        self.standings = list(match_screen.match.standings)
        self.seed = match_screen.match.config.seed

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_r, pygame.K_RETURN, pygame.K_KP_ENTER):
            self.app.pop()
            self.match_screen.rematch(same_seed=True)
        elif event.key == pygame.K_n:
            self.app.pop()
            self.match_screen.rematch(same_seed=False)
        elif event.key == pygame.K_ESCAPE:
            self.app.pop()
            self.match_screen.leave()

    def draw(self, surface: pygame.Surface) -> None:
        surface.fill(BG)
        w, h = surface.get_size()
        winner = self.standings[0]["name"] if self.standings else "-"
        human_won = bool(self.standings) and self.standings[0]["kind"] == "human"
        title = "YOU WIN" if human_won else f"{winner.upper()} WINS"
        color = ACCENT if human_won else DANGER
        draw_text_fit(surface, title, (w // 2, 80), w - 80, 72, color, "center")
        total = sum(c[1] for c in COLUMNS)
        x0 = (w - total) // 2
        y = 170
        x = x0
        for label, width, _ in COLUMNS:
            draw_text(surface, label, (x, y), 26, MUTED)
            x += width
        y += 40
        for row in self.standings:
            x = x0
            color = ACCENT if row["kind"] == "human" else TEXT
            for _, width, fmt in COLUMNS:
                draw_text_fit(surface, fmt(row), (x, y), width - 12, 32, color)
                x += width
            y += 44
        draw_text(surface, f"Seed {self.seed}", (w // 2, h - 80), 24, MUTED, "center")
        draw_text(
            surface,
            "R: rematch (same seed)    N: rematch (new seed)    Esc: main menu",
            (w // 2, h - 40),
            28,
            TEXT,
            "center",
        )
