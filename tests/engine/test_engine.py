from __future__ import annotations

import json
import random

import numpy as np
import pytest

from tetris.engine import (
    GARBAGE,
    HEIGHT,
    WIDTH,
    Game,
    GameConfig,
    IllegalPlacementError,
    Move,
    Piece,
    Placement,
)
from tetris.engine.board import fits, piece_cells
from tetris.engine.pieces import CELLS, kicks
from tetris.engine.placement import legal_targets
from tetris.engine.scoring import attack_for, combo_attack


def set_active(game: Game, piece: Piece, rot: int, row: int, col: int) -> None:
    game.active.piece = piece
    game.active.rot = rot
    game.active.row = row
    game.active.col = col


def fill_rows(game: Game, rows: int, gap_col: int) -> None:
    """Fill the bottom ``rows`` rows except ``gap_col``."""
    for r in range(HEIGHT - rows, HEIGHT):
        game.board[r, :] = GARBAGE
        game.board[r, gap_col] = 0


def random_play(game: Game, rng: random.Random, pieces: int) -> None:
    for _ in range(pieces):
        if game.game_over:
            break
        targets = legal_targets(game.board, game.active.piece)
        t = rng.choice(targets)
        game.place(Placement(t.rotation, t.column))


# ----------------------------------------------------------------- rotation


@pytest.mark.parametrize("piece", list(Piece))
def test_four_rotations_return_to_start(piece: Piece) -> None:
    game = Game(0)
    set_active(game, piece, 0, 10, 3)
    before = sorted(piece_cells(piece, 0, 10, 3))
    for _ in range(4):
        assert game.apply(Move.CW)
    a = game.active
    assert sorted(piece_cells(a.piece, a.rot, a.row, a.col)) == before


def test_cw_then_ccw_is_identity() -> None:
    game = Game(0)
    set_active(game, Piece.T, 0, 10, 3)
    game.apply(Move.CW)
    game.apply(Move.CCW)
    a = game.active
    assert (a.rot, a.row, a.col) == (0, 10, 3)


def test_rotation_states_have_four_cells() -> None:
    for piece in Piece:
        for rot in range(4):
            assert len(CELLS[piece][rot]) == 4


def test_t_wall_kick_off_left_wall() -> None:
    game = Game(0)
    # T in state R hugging the left wall: occupied columns 0..1.
    set_active(game, Piece.T, 1, 10, -1)
    assert fits(game.board, Piece.T, 1, 10, -1)
    assert game.apply(Move.CW)  # R -> 2 needs the (+1, 0) kick
    a = game.active
    assert (a.rot, a.row, a.col) == (2, 10, 0)


def test_i_wall_kick_off_right_wall() -> None:
    game = Game(0)
    # Vertical I (state R) in the rightmost column.
    set_active(game, Piece.I, 1, 10, 7)
    assert {c for _, c in piece_cells(Piece.I, 1, 10, 7)} == {9}
    assert game.apply(Move.CW)  # R -> 2: first kick (0,0) fails, (-1,0) succeeds
    a = game.active
    assert (a.rot, a.row, a.col) == (2, 10, 6)


def test_kick_table_uses_guideline_offsets() -> None:
    # Guideline 0->R JLSTZ test 3 is (-1, +1) in y-up, i.e. one row up.
    assert kicks(Piece.T, 0, 1)[2] == (-1, -1)
    assert kicks(Piece.O, 0, 1) == ((0, 0),)


def test_rotation_blocked_when_no_kick_fits() -> None:
    game = Game(0)
    game.board[:, :] = GARBAGE
    game.board[10:13, 3:6] = 0
    game.board[10, 3] = GARBAGE
    game.board[10, 5] = GARBAGE
    set_active(game, Piece.T, 0, 10, 3)
    assert fits(game.board, Piece.T, 0, 10, 3)
    # With the surroundings packed solid every kick test collides.
    game.board[12, :] = GARBAGE
    assert not game.apply(Move.CW)
    assert game.active.rot == 0


# --------------------------------------------------------------- movement


def test_lock_delay_locks_grounded_piece() -> None:
    game = Game(0, GameConfig(gravity=0.0, lock_delay=0.5))
    while game.apply(Move.SOFT):
        pass
    game.tick(0.3)
    assert game.pieces_placed == 0
    game.tick(0.3)
    assert game.pieces_placed == 1
    assert game.active is not None and game.board.any()


def test_gravity_moves_piece_down() -> None:
    game = Game(0, GameConfig(gravity=10.0))
    row = game.active.row
    game.tick(0.25)
    assert game.active.row == row + 2


def test_hold_swaps_once_per_piece() -> None:
    game = Game(5)
    first = game.active.piece
    nxt = game.queue[0]
    assert game.apply(Move.HOLD)
    assert game.hold is first and game.active.piece is nxt
    assert not game.apply(Move.HOLD)
    game.apply(Move.HARD)
    assert game.apply(Move.HOLD)
    assert game.active.piece is first


# ----------------------------------------------------------- line clears


def test_single_line_clear() -> None:
    game = Game(0)
    fill_rows(game, 1, gap_col=0)
    game.board[HEIGHT - 1, 0:4] = 0
    game.active.piece = Piece.I
    game.place(Placement(0, 0))
    assert game.lines == 1
    assert not game.board[HEIGHT - 1].any()


def test_tetris_scores_and_attacks() -> None:
    game = Game(0)
    fill_rows(game, 4, gap_col=9)
    game.active.piece = Piece.I
    result = game.place(Placement(1, 9))
    assert result.lines == 4
    assert result.attack == 4
    assert result.score == 800
    assert not game.board.any()


def test_back_to_back_tetris_bonus() -> None:
    game = Game(0)
    fill_rows(game, 4, gap_col=9)
    game.active.piece = Piece.I
    first = game.place(Placement(1, 9))
    assert (first.b2b, first.attack) == (False, 4)
    # A non-clearing piece breaks the combo but not the back-to-back chain.
    game.active.piece = Piece.O
    game.place(Placement(0, 0))
    game.board[:] = 0
    fill_rows(game, 4, gap_col=9)
    game.active.piece = Piece.I
    second = game.place(Placement(1, 9))
    assert (second.b2b, second.attack) == (True, 5)
    assert second.score == 1200


def test_combo_attack_table() -> None:
    expected = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4, 10: 4, 11: 5, 20: 5}
    for combo, bonus in expected.items():
        assert combo_attack(combo) == bonus
    assert attack_for(1, 0, False) == 0
    assert attack_for(2, 0, False) == 1
    assert attack_for(3, 0, False) == 2
    assert attack_for(4, 0, True) == 5
    assert attack_for(2, 4, False) == 3


def test_combo_counter_increments_and_resets() -> None:
    game = Game(0)
    for i in range(3):
        game.board[:] = 0
        fill_rows(game, 1, gap_col=0)
        game.board[HEIGHT - 1, 0:4] = 0
        game.active.piece = Piece.I
        result = game.place(Placement(0, 0))
        assert result.combo == i
    game.active.piece = Piece.O
    game.place(Placement(0, 4))
    assert game.combo == -1


# --------------------------------------------------------------- 7-bag


def test_seven_bag_fairness() -> None:
    game = Game(123)
    seq = [game.active.piece]
    while len(seq) < 7 * 200:
        game.active = None
        game._spawn(game._next_piece())
        seq.append(game.active.piece)
    for i in range(0, len(seq), 7):
        assert sorted(seq[i : i + 7]) == sorted(Piece)


def test_queue_shows_five() -> None:
    assert len(Game(0).next_queue) == 5


# -------------------------------------------------------------- garbage


def test_receive_garbage_inserts_rows_with_one_hole() -> None:
    game = Game(9)
    game.board[HEIGHT - 1, 0] = int(Piece.T)
    game.receive_garbage(3)
    bottom = game.board[HEIGHT - 3 :]
    assert (bottom == GARBAGE).sum(axis=1).tolist() == [WIDTH - 1] * 3
    holes = [int(np.flatnonzero(row == 0)[0]) for row in bottom]
    assert len(set(holes)) == 1
    assert game.board[HEIGHT - 4, 0] == int(Piece.T)


def test_garbage_overflow_tops_out() -> None:
    game = Game(0)
    game.board[0, 0] = GARBAGE
    game.receive_garbage(1)
    assert game.game_over


def test_pending_garbage_is_cancelled_by_attack() -> None:
    game = Game(0)
    game.queue_garbage(3)
    fill_rows(game, 4, gap_col=9)
    game.active.piece = Piece.I
    result = game.place(Placement(1, 9))
    assert result.attack == 4 and result.sent == 1
    assert game.pending_total == 0


def test_pending_garbage_inserted_on_non_clearing_lock() -> None:
    game = Game(0)
    game.queue_garbage(2)
    game.active.piece = Piece.O
    game.place(Placement(0, 0))
    assert game.pending_total == 0
    assert (game.board[HEIGHT - 2 :] == GARBAGE).sum() == 2 * (WIDTH - 1)


# ----------------------------------------------- determinism and serialization


def test_determinism_under_fixed_seed() -> None:
    a, b = Game(42), Game(42)
    random_play(a, random.Random(7), 150)
    random_play(b, random.Random(7), 150)
    assert a.to_dict() == b.to_dict()
    assert Game(1).queue != Game(2).queue


def test_live_play_determinism() -> None:
    def run() -> dict:
        g = Game(11)
        rng = random.Random(3)
        moves = list(Move)
        for _ in range(2000):
            if g.game_over:
                break
            g.apply(rng.choice(moves))
            g.tick(1 / 60)
        return g.to_dict()

    assert run() == run()


def test_serialization_round_trip() -> None:
    game = Game(77)
    random_play(game, random.Random(1), 40)
    game.queue_garbage(2)
    game.apply(Move.HOLD)
    game.apply(Move.LEFT)
    game.tick(0.1)
    data = json.loads(json.dumps(game.to_dict()))
    restored = Game.from_dict(data)
    assert restored.to_dict() == game.to_dict()
    # Continuing play from both yields identical states.
    random_play(game, random.Random(2), 30)
    random_play(restored, random.Random(2), 30)
    assert restored.to_dict() == game.to_dict()


def test_copy_is_independent() -> None:
    game = Game(3)
    clone = game.copy()
    random_play(clone, random.Random(0), 5)
    assert game.pieces_placed == 0
    assert clone.to_dict() != game.to_dict()


# ------------------------------------------------------------- placements


def test_illegal_placement_raises() -> None:
    game = Game(0)
    with pytest.raises(IllegalPlacementError):
        game.place(Placement(0, 9))  # every piece is wider than one column in rotation 0


def test_placement_index_round_trip() -> None:
    for i in range(80):
        assert Placement.from_index(i).index == i


def test_lock_out_ends_game() -> None:
    game = Game(0)
    game.board[4:, 1:] = GARBAGE  # column 0 open so no row is full
    game.active.piece = Piece.O
    game.place(Placement(0, 4))  # lands entirely in the hidden rows
    assert game.game_over
    assert game.pop_events()[-1].data["reason"] == "lock_out"


def test_block_out_ends_game() -> None:
    game = Game(0)
    game.board[2:4, 3:7] = GARBAGE  # occupy the spawn area under the active piece
    game.apply(Move.HOLD)
    assert game.game_over
    assert game.pop_events()[-1].data["reason"] == "block_out"
