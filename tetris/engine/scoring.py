"""Score and attack tables.

Attack (garbage sent) rules for v1:

* Lines: single 0, double 1, triple 2, tetris 4.
* Combo bonus by combo count (0 for the first clear in a chain):
  0-1: 0, 2-3: 1, 4-5: 2, 6-7: 3, 8-10: 4, 11+: 5.
* Back-to-back: +1 for a Tetris that follows a Tetris (non-clearing pieces
  do not break the chain; any other line clear does).
* No T-spins in v1.
"""

from __future__ import annotations

LINE_ATTACK = (0, 0, 1, 2, 4)
LINE_SCORE = (0, 100, 300, 500, 800)
COMBO_SCORE = 50
B2B_SCORE_MULTIPLIER = 1.5
B2B_ATTACK_BONUS = 1


def combo_attack(combo: int) -> int:
    """Combo garbage bonus for a combo count (0 = first clear in the chain)."""
    if combo <= 1:
        return 0
    if combo <= 3:
        return 1
    if combo <= 5:
        return 2
    if combo <= 7:
        return 3
    if combo <= 10:
        return 4
    return 5


def attack_for(lines: int, combo: int, b2b: bool) -> int:
    """Outgoing attack for a lock that cleared ``lines`` lines.

    ``combo`` is the combo count after this clear, ``b2b`` whether this clear is
    a back-to-back Tetris.
    """
    if lines <= 0:
        return 0
    return LINE_ATTACK[lines] + combo_attack(combo) + (B2B_ATTACK_BONUS if b2b else 0)


def score_for(lines: int, level: int, combo: int, b2b: bool) -> int:
    """Points awarded for a line clear (excluding drop points)."""
    if lines <= 0:
        return 0
    base = LINE_SCORE[lines] * level
    if b2b:
        base = int(base * B2B_SCORE_MULTIPLIER)
    return base + (COMBO_SCORE * combo * level if combo > 0 else 0)


def gravity_rows_per_second(level: int) -> float:
    """Guideline gravity curve, in rows per second."""
    level = max(1, min(level, 20))
    seconds_per_row = (0.8 - (level - 1) * 0.007) ** (level - 1)
    return 1.0 / seconds_per_row
