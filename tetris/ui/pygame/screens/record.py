"""Record Gameplay: a Solo game whose placements are saved as an imitation dataset."""

from __future__ import annotations

import pygame

from tetris.match import MatchConfig, MatchController, PlayerSpec
from tetris.match.recording import RECORD_SETTINGS, GameRecorder, recording_totals
from tetris.ui.pygame.render import ACCENT, MUTED, TEXT, draw_overlay, draw_text
from tetris.ui.pygame.screens.solo import PLAYER_ID, SoloScreen

#: Recording settings in menu order; plain (no garbage) is the default.
SETTING_ORDER = ("plain", "garbage")
SETTING_LABELS = {
    "plain": "Plain (level gravity, no garbage)",
    "garbage": "Garbage (1 row every 3 pieces, the agents' training setting)",
}
REC_RED = (230, 70, 70)


class RecordScreen(SoloScreen):
    """Choose a setting, play, then save or discard the game.

    Each locked piece is recorded with the state it spawned in and the matching
    80-space action (see :mod:`tetris.match.recording`).
    """

    def __init__(self, app, seed: int | None = None) -> None:
        self.setting = "plain"
        self.phase = "setup"  # setup -> playing -> review
        self.saved_path: str | None = None
        self.totals = recording_totals()
        super().__init__(app, seed)

    def _make_match(self) -> MatchController:
        garbage = RECORD_SETTINGS[self.setting].garbage_every
        match = MatchController(
            MatchConfig(mode="solo", seed=self.seed, eval_garbage_every=garbage),
            [PlayerSpec(PLAYER_ID, "You")],
        )
        self.recorder = GameRecorder(PLAYER_ID, self.seed, self.setting)
        match.lock_listeners.append(self.recorder.on_lock)
        self.recorder.start(match.players[PLAYER_ID].game)
        return match

    def _begin(self) -> None:
        self.match = self._make_match()
        self.saved_path = None
        self.phase = "playing"
        self.input.reset()

    def _restart(self, new_seed: bool = True) -> None:
        super()._restart(new_seed)
        self.recorder = GameRecorder(PLAYER_ID, self.seed, self.setting)
        self.match.lock_listeners[:] = [self.recorder.on_lock]
        self.recorder.start(self.match.players[PLAYER_ID].game)
        self.phase = "playing"

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN and self.phase != "playing":
            return
        if self.phase == "setup":
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN):
                i = SETTING_ORDER.index(self.setting)
                self.setting = SETTING_ORDER[(i + 1) % len(SETTING_ORDER)]
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._begin()
            elif event.key == pygame.K_ESCAPE:
                self.app.pop()
            return
        if self.phase == "review":
            if event.key == pygame.K_s and self.saved_path is None:
                path = self.recorder.save(self.match.players[PLAYER_ID].game)
                self.saved_path = str(path)
                self.totals = recording_totals()
            elif event.key in (pygame.K_d, pygame.K_RETURN, pygame.K_KP_ENTER):
                self.phase = "setup"
                self.seed = self.seed + 1
            elif event.key == pygame.K_ESCAPE:
                self.app.pop()
            return
        super().handle_event(event)

    def update(self, dt: float) -> None:
        if self.phase != "playing":
            return
        super().update(dt)
        if self.match.finished:
            self.phase = "review"

    def _draw_overlays(self, surface: pygame.Surface, snap: dict, player: dict) -> None:
        if self.phase == "playing":
            super()._draw_overlays(surface, snap, player)
        w, h = surface.get_size()
        if self.phase == "playing" and not self.match.paused:
            draw_text(surface, "REC", (w - 20, 56), 32, REC_RED, "topright")
            draw_text(
                surface,
                f"{self.recorder.on_space} placements"
                + (f" ({self.recorder.off_space} off-space)" if self.recorder.off_space else ""),
                (w - 20, 92),
                22,
                MUTED,
                "topright",
            )
            return
        if self.phase == "setup":
            draw_overlay(surface)
            draw_text(surface, "RECORD GAMEPLAY", (w // 2, h // 2 - 150), 60, ACCENT, "center")
            draw_text(
                surface, SETTING_LABELS[self.setting], (w // 2, h // 2 - 60), 30, TEXT, "center"
            )
            draw_text(
                surface,
                "Arrows: change setting    Enter: start    Esc: back",
                (w // 2, h // 2 - 10),
                24,
                MUTED,
                "center",
            )
            t = self.totals
            draw_text(
                surface,
                f"Saved so far: {t['games']} games, {t['on_space']} placements "
                f"({t['off_space']} off-space, not used for training)",
                (w // 2, h // 2 + 50),
                24,
                MUTED,
                "center",
            )
            draw_text(
                surface,
                "Tucks, spins and slides are saved but marked off-space: the agents only use "
                "hard-drop placements.",
                (w // 2, h // 2 + 85),
                20,
                MUTED,
                "center",
            )
        elif self.phase == "review":
            draw_overlay(surface)
            rec = self.recorder
            draw_text(surface, "GAME OVER", (w // 2, h // 2 - 120), 64, ACCENT, "center")
            draw_text(
                surface,
                f"{rec.on_space} placements recorded, {rec.off_space} off-space",
                (w // 2, h // 2 - 50),
                30,
                TEXT,
                "center",
            )
            msg = (
                f"Saved to {self.saved_path}"
                if self.saved_path
                else "S: save    D or Enter: discard and play again    Esc: back"
            )
            draw_text(surface, msg, (w // 2, h // 2 + 10), 22, MUTED, "center")
            if self.saved_path:
                draw_text(
                    surface,
                    "Enter: play again    Esc: back",
                    (w // 2, h // 2 + 45),
                    22,
                    MUTED,
                    "center",
                )
