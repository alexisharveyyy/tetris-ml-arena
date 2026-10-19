from __future__ import annotations

import numpy as np
import pytest

from tetris.engine import GARBAGE, HEIGHT, Game, Piece, Placement
from tetris.env import (
    FEATURE_NAMES,
    RewardConfig,
    TetrisEnv,
    action_mask,
    board_features,
    enumerate_afterstates,
)
from tetris.env.features import BOARD_FEATURES

from ..engine.test_engine import respawn


def feat(board: np.ndarray, name: str) -> float:
    return float(board_features(board != 0)[BOARD_FEATURES.index(name)])


def test_features_on_known_board() -> None:
    board = np.zeros((HEIGHT, 10), dtype=np.int8)
    board[HEIGHT - 1, :9] = GARBAGE  # bottom row, column 9 open
    board[HEIGHT - 3, 0] = GARBAGE  # column 0 has height 3 with one hole under it
    assert feat(board, "holes") == 1
    assert feat(board, "rows_with_holes") == 1
    assert feat(board, "max_height") == 3
    assert feat(board, "aggregate_height") == 3 + 8
    assert feat(board, "bumpiness") == 2 + 1
    # Column 9 is a depth-1 well (wall on the right, filled on the left).
    assert feat(board, "max_well_depth") >= 1


def test_batch_features_match_single() -> None:
    rng = np.random.default_rng(0)
    boards = rng.random((5, HEIGHT, 10)) < 0.3
    batch = board_features(boards)
    for i in range(5):
        np.testing.assert_array_equal(batch[i], board_features(boards[i]))


def test_reset_observation_in_space() -> None:
    env = TetrisEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert info["action_mask"].shape == (80,)
    assert info["action_mask"].any()


def test_mask_matches_legality() -> None:
    env = TetrisEnv()
    env.reset(seed=4)
    rng = np.random.default_rng(1)
    for _ in range(30):
        mask = env.action_masks()
        for idx in range(80):
            clone = env.game.copy()
            try:
                clone.place(Placement.from_index(idx))
                legal = True
            except ValueError:
                legal = False
            assert legal == bool(mask[idx]), idx
        _, _, term, trunc, _ = env.step(int(rng.choice(np.flatnonzero(mask))))
        if term or trunc:
            break


def test_illegal_action_terminates() -> None:
    env = TetrisEnv()
    _, info = env.reset(seed=0)
    illegal = int(np.flatnonzero(~info["action_mask"])[0])
    _, reward, terminated, _, info = env.step(illegal)
    assert terminated and reward == RewardConfig().game_over
    assert info["illegal_action"] == illegal


def test_afterstates_match_engine_place() -> None:
    game = Game(21)
    rng = np.random.default_rng(2)
    for _ in range(25):
        afs = enumerate_afterstates(game)
        assert {a.placement.index for a in afs} == set(np.flatnonzero(action_mask(game)))
        choice = afs[int(rng.integers(len(afs)))]
        before = game.score
        game.place(choice.placement)
        np.testing.assert_array_equal(game.board, choice.board)
        assert game.score - before == choice.score
        assert len(choice.features) == len(FEATURE_NAMES)
        if game.game_over:
            break


def test_dedupe_reduces_symmetric_pieces() -> None:
    game = Game(0)
    respawn(game, Piece.O)
    plain = [a for a in enumerate_afterstates(game, include_hold=False)]
    unique = enumerate_afterstates(game, include_hold=False, dedupe=True)
    assert len(plain) == 4 * 9 and len(unique) == 9


def test_hold_actions_masked_after_hold() -> None:
    env = TetrisEnv()
    env.reset(seed=0)
    env.game.can_hold = False
    assert not env.action_masks()[40:].any()


def test_truncation_at_max_pieces() -> None:
    env = TetrisEnv(max_pieces=3)
    _, info = env.reset(seed=0)
    for _ in range(3):
        _, _, term, trunc, info = env.step(int(np.flatnonzero(info["action_mask"])[0]))
    assert trunc and not term


def test_garbage_injection_every_n_pieces() -> None:
    env = TetrisEnv(garbage_every=2, garbage_rows=1, include_hold=False)
    _, info = env.reset(seed=0)
    for _ in range(2):
        _, _, _, _, info = env.step(int(np.flatnonzero(info["action_mask"])[0]))
    assert (env.game.board == GARBAGE).any()


def test_reward_config_from_yaml(tmp_path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "reward:\n  per_piece: 0.5\n  line_clear: [0, 2, 4, 6, 10]\n"
        "  feature_deltas: {holes: -1.0}\n"
    )
    cfg = RewardConfig.from_yaml(path)
    assert cfg.per_piece == 0.5 and cfg.line_clear[4] == 10
    before = np.zeros(len(BOARD_FEATURES), dtype=np.float32)
    after = before.copy()
    after[BOARD_FEATURES.index("holes")] = 2
    assert cfg.compute(1, 0, 0, before, after, False) == pytest.approx(2 + 0.5 - 2)
    with pytest.raises(ValueError):
        RewardConfig(feature_deltas={"nope": 1.0})


def test_env_seed_determinism() -> None:
    def run() -> list[float]:
        env = TetrisEnv(max_pieces=50)
        _, info = env.reset(seed=8)
        rewards = []
        rng = np.random.default_rng(0)
        while True:
            a = int(rng.choice(np.flatnonzero(info["action_mask"])))
            _, r, term, trunc, info = env.step(a)
            rewards.append(r)
            if term or trunc:
                return rewards

    assert run() == run()


def test_piece_pushed_up_by_garbage_still_has_moves() -> None:
    game = Game(0)
    spawn_row = game.active.row
    game.board[HEIGHT - 19 :, :] = GARBAGE  # stack reaching row 5, one hole per row
    for r in range(HEIGHT - 19, HEIGHT):
        game.board[r, r % 10] = 0
    game.receive_garbage(1)  # pushes the stack into the live piece
    assert not game.game_over and game.active.row < spawn_row
    mask = action_mask(game)
    assert mask.any()
    game.place(Placement.from_index(int(np.flatnonzero(mask)[0])))
