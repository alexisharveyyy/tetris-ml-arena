"""Agent identity: every registry agent has one, and its colors are readable."""

from __future__ import annotations

import pytest

from tetris.agents.identity import HUMAN, IDENTITIES, PIECE_CELLS, identity, lineup
from tetris.agents.registry import AGENTS

#: Look backgrounds the identity colors sit on.
A_PANEL, A_BG, C_PANEL = "#FFFFFF", "#ECEEF2", "#121C80"


def _lum(color: str) -> float:
    c = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_every_registered_agent_has_an_identity() -> None:
    real = [k for k, s in AGENTS.items() if s.factory.startswith("tetris.")]
    assert set(real) <= set(IDENTITIES)


def test_home_lineup_is_the_ten_main_entries() -> None:
    keys = [i.key for i in lineup("main")]
    assert keys == ["dqn", "cem", "ga", "ppo", "bc", "bc_you", "neat", "dt", "dellacherie", "llm"]
    assert [i.key for i in lineup("variant")] == ["ppo_raw"]
    assert {i.key for i in lineup("leaderboard")} == {"bc_pure", "bc_dagger"}
    assert {i.key for i in IDENTITIES.values() if i.planned} == {"dt", "llm"}


def test_variants_share_parent_piece_and_colors() -> None:
    for key, parent, letter in (
        ("ppo_raw", "ppo", "l"),
        ("bc_pure", "bc", "J"),
        ("bc_dagger", "bc", "J"),
    ):
        v, p = IDENTITIES[key], IDENTITIES[parent]
        assert (v.piece, v.a_fill, v.c_fill, v.d_field, v.d_text) == (
            p.piece,
            p.a_fill,
            p.c_fill,
            p.d_field,
            p.d_text,
        )
        assert v.letter == letter


def test_pieces_are_known_shapes() -> None:
    for i in IDENTITIES.values():
        assert i.piece in PIECE_CELLS
    assert len(PIECE_CELLS["P"]) == 5 and len(PIECE_CELLS["mono"]) == 1
    assert identity("not_a_key").piece == "mono"


@pytest.mark.parametrize("ident", [*IDENTITIES.values(), HUMAN], ids=lambda i: i.key)
def test_text_contrast(ident) -> None:
    # Poster: names and numbers sit on the column color.
    assert contrast(ident.d_text, ident.d_field) >= 4.5
    # Tetromino: agent card headers are filled with the agent color.
    assert contrast(ident.a_text, ident.a_fill) >= 4.5


def test_bc_you_passes_every_look() -> None:
    i = IDENTITIES["bc_you"]
    assert i.piece == "J" and i.letter == "j"
    assert contrast(i.a_text, i.a_fill) >= 4.5
    assert contrast(i.a_fill, A_PANEL) >= 3.0 and contrast(i.a_fill, A_BG) >= 3.0  # icon
    assert contrast(i.c_fill, C_PANEL) >= 3.0
    assert contrast(i.d_text, i.d_field) >= 4.5
