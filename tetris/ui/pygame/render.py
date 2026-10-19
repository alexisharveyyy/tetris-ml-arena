"""Draw match snapshots. Reads only the JSON snapshot, never engine objects."""

from __future__ import annotations

from typing import Any

import pygame

from tetris.engine.board import HEIGHT, WIDTH
from tetris.engine.pieces import CELLS, Piece

BG = (16, 18, 24)
PANEL = (28, 31, 40)
GRID = (38, 42, 54)
TEXT = (226, 230, 238)
MUTED = (140, 148, 164)
ACCENT = (98, 180, 255)
DANGER = (235, 88, 88)

PIECE_COLORS: dict[str, tuple[int, int, int]] = {
    "I": (64, 208, 224),
    "O": (240, 208, 64),
    "T": (168, 88, 224),
    "S": (96, 200, 96),
    "Z": (232, 80, 80),
    "J": (72, 112, 232),
    "L": (240, 144, 48),
    "G": (120, 124, 136),
}

#: Hidden rows drawn above the playfield so freshly spawned pieces are visible.
SHOWN_HIDDEN_ROWS = 2

_font_cache: dict[int, pygame.font.Font] = {}


def clear_font_cache() -> None:
    """Drop cached fonts (they become invalid after ``pygame.quit``)."""
    _font_cache.clear()


def font(size: int) -> pygame.font.Font:
    if size not in _font_cache:
        _font_cache[size] = pygame.font.Font(None, size)
    return _font_cache[size]


def draw_text(
    surface: pygame.Surface,
    text: str,
    pos: tuple[int, int],
    size: int = 28,
    color: tuple[int, int, int] = TEXT,
    anchor: str = "topleft",
) -> pygame.Rect:
    img = font(size).render(text, True, color)
    rect = img.get_rect(**{anchor: pos})
    surface.blit(img, rect)
    return rect


def _shade(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(c * factor))) for c in color)  # type: ignore[return-value]


def draw_cell(
    surface: pygame.Surface, x: int, y: int, size: int, color: tuple[int, int, int]
) -> None:
    rect = pygame.Rect(x, y, size, size)
    pygame.draw.rect(surface, color, rect)
    if size >= 10:
        pygame.draw.rect(surface, _shade(color, 1.3), rect.inflate(-size // 2, -size // 2))
    pygame.draw.rect(surface, _shade(color, 0.6), rect, 1)


def draw_mini_piece(
    surface: pygame.Surface,
    letter: str | None,
    center: tuple[int, int],
    cell: int,
    dim: bool = False,
) -> None:
    """Draw a piece in its spawn orientation centered at ``center``."""
    if letter is None:
        return
    cells = CELLS[Piece.from_letter(letter)][0]
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    w = (max(cols) - min(cols) + 1) * cell
    h = (max(rows) - min(rows) + 1) * cell
    ox = center[0] - w // 2 - min(cols) * cell
    oy = center[1] - h // 2 - min(rows) * cell
    color = PIECE_COLORS[letter]
    if dim:
        color = _shade(color, 0.4)
    for r, c in cells:
        draw_cell(surface, ox + c * cell, oy + r * cell, cell, color)


def board_size(cell: int) -> tuple[int, int]:
    """Pixel size of the playfield including the shown hidden rows."""
    visible = HEIGHT - 4 + SHOWN_HIDDEN_ROWS
    return WIDTH * cell, visible * cell


def draw_board(
    surface: pygame.Surface,
    player: dict[str, Any],
    origin: tuple[int, int],
    cell: int,
    hidden_rows: int,
    show_ghost: bool = True,
) -> pygame.Rect:
    """Draw the playfield. ``origin`` is the top-left of the shown hidden rows."""
    x0, y0 = origin
    first_row = hidden_rows - SHOWN_HIDDEN_ROWS
    field_top = y0 + SHOWN_HIDDEN_ROWS * cell
    field = pygame.Rect(x0, field_top, WIDTH * cell, (HEIGHT - hidden_rows) * cell)
    pygame.draw.rect(surface, PANEL, field)
    for c in range(1, WIDTH):
        pygame.draw.line(surface, GRID, (x0 + c * cell, field.top), (x0 + c * cell, field.bottom))
    for r in range(1, HEIGHT - hidden_rows):
        y = field.top + r * cell
        pygame.draw.line(surface, GRID, (x0, y), (field.right, y))

    def cell_pos(r: int, c: int) -> tuple[int, int]:
        return x0 + c * cell, y0 + (r - first_row) * cell

    dead = player.get("game_over", False)
    for r, row in enumerate(player["board"]):
        if r < first_row:
            continue
        for c, ch in enumerate(row):
            if ch != ".":
                color = PIECE_COLORS[ch]
                draw_cell(surface, *cell_pos(r, c), cell, _shade(color, 0.45) if dead else color)

    active = player.get("active")
    if active:
        color = PIECE_COLORS[active["piece"]]
        if show_ghost:
            for r, c in active["ghost"]:
                if r >= first_row:
                    gx, gy = cell_pos(r, c)
                    pygame.draw.rect(
                        surface, _shade(color, 0.55), (gx, gy, cell, cell), max(1, cell // 12)
                    )
        for r, c in active["cells"]:
            if r >= first_row:
                draw_cell(surface, *cell_pos(r, c), cell, color)

    pygame.draw.rect(surface, MUTED, field, 2)
    return field


def draw_garbage_meter(
    surface: pygame.Surface, field: pygame.Rect, pending: int, cell: int
) -> None:
    """Vertical bar left of the field showing incoming garbage rows."""
    width = max(4, cell // 4)
    bar = pygame.Rect(field.left - width - 3, field.top, width, field.height)
    pygame.draw.rect(surface, GRID, bar)
    if pending > 0:
        h = min(field.height, pending * cell)
        pygame.draw.rect(surface, DANGER, (bar.left, bar.bottom - h, bar.width, h))


def draw_player_large(
    surface: pygame.Surface,
    player: dict[str, Any],
    origin: tuple[int, int],
    cell: int,
    hidden_rows: int,
    show_ghost: bool = True,
    show_garbage: bool = False,
) -> pygame.Rect:
    """Hold panel, playfield, next queue and stats. Returns the playfield rect."""
    x0, y0 = origin
    side = cell * 5
    board_x = x0 + side + cell // 2
    field = draw_board(surface, player, (board_x, y0), cell, hidden_rows, show_ghost)
    if show_garbage:
        draw_garbage_meter(surface, field, player.get("pending_garbage", 0), cell)

    mini = max(8, int(cell * 0.7))
    # Hold
    draw_text(surface, "HOLD", (x0, field.top), 26, MUTED)
    hold_box = pygame.Rect(x0, field.top + 24, side, mini * 3)
    pygame.draw.rect(surface, PANEL, hold_box, border_radius=6)
    draw_mini_piece(
        surface, player.get("hold"), hold_box.center, mini, dim=not player.get("can_hold", True)
    )

    # Stats
    sy = hold_box.bottom + 24
    for label, key in (
        ("SCORE", "score"),
        ("LINES", "lines"),
        ("LEVEL", "level"),
        ("PIECES", "pieces"),
    ):
        draw_text(surface, label, (x0, sy), 22, MUTED)
        draw_text(surface, f"{player.get(key, 0):,}", (x0, sy + 18), 34)
        sy += 58
    if player.get("combo", -1) > 0:
        draw_text(surface, f"COMBO {player['combo']}", (x0, sy), 26, ACCENT)
        sy += 26
    if player.get("b2b"):
        draw_text(surface, "B2B", (x0, sy), 26, ACCENT)

    # Next queue
    nx = field.right + cell // 2
    draw_text(surface, "NEXT", (nx, field.top), 26, MUTED)
    qy = field.top + 24
    queue_box = pygame.Rect(nx, qy, side, mini * 3 * max(1, len(player.get("queue", []))))
    pygame.draw.rect(surface, PANEL, queue_box, border_radius=6)
    for i, letter in enumerate(player.get("queue", [])):
        draw_mini_piece(
            surface, letter, (queue_box.centerx, qy + mini * 3 * i + mini * 3 // 2), mini
        )

    draw_text(surface, player.get("name", ""), (field.centerx, y0 - 4), 30, TEXT, "midbottom")
    return field


def large_player_size(cell: int) -> tuple[int, int]:
    bw, bh = board_size(cell)
    return cell * 5 * 2 + cell + bw, bh


def draw_overlay(surface: pygame.Surface, alpha: int = 170) -> None:
    shade = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    shade.fill((0, 0, 0, alpha))
    surface.blit(shade, (0, 0))


def draw_menu(
    surface: pygame.Surface,
    items: list[tuple[str, bool]],
    selected: int,
    center: tuple[int, int],
    size: int = 40,
    spacing: int = 48,
) -> None:
    """Vertical menu. ``items`` are ``(label, enabled)`` pairs."""
    cx, cy = center
    top = cy - spacing * (len(items) - 1) // 2
    for i, (label, enabled) in enumerate(items):
        color = TEXT if enabled else MUTED
        if i == selected:
            color = ACCENT if enabled else _shade(ACCENT, 0.6)
            label = f"> {label} <"
        draw_text(surface, label, (cx, top + i * spacing), size, color, "center")
