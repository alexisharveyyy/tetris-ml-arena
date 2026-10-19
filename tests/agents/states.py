"""Random game-state generators for agent and planner tests."""

from __future__ import annotations

import random

import numpy as np

from tetris.engine import GARBAGE, HEIGHT, WIDTH, Game, Move, Piece, Placement
from tetris.env import legal_placements


def random_play_state(seed: int, rng: random.Random) -> Game:
    """A game advanced by random legal placements; often tall and messy."""
    game = Game(seed)
    for _ in range(rng.randint(0, 80)):
        options = legal_placements(game)
        if not options:
            break
        game.place(rng.choice(options)[0])
        if rng.random() < 0.1 and not game.game_over:
            game.receive_garbage(rng.randint(1, 3))
        if game.game_over:
            return random_play_state(seed + 1000, rng)
    return game


def jagged_state(
    seed: int, rng: random.Random, max_height: int = 19, spawn_rows: bool = False
) -> Game:
    """A board with random column heights, holes and overhangs."""
    game = Game(seed)
    board = np.zeros((HEIGHT, WIDTH), dtype=np.int8)
    for c in range(WIDTH):
        h = rng.randint(0, max_height)
        for r in range(HEIGHT - h, HEIGHT):
            board[r, c] = GARBAGE if rng.random() > 0.15 else 0
    for r in range(HEIGHT):
        if board[r].all():
            board[r, rng.randrange(WIDTH)] = 0
    board[:4] = 0  # keep the spawn rows clear
    if spawn_rows:
        # Let some columns poke into the hidden rows, leaving the spawn cells free.
        for c in range(WIDTH):
            if rng.random() < 0.3:
                board[rng.randint(2, 4) : 6, c] = GARBAGE
        board[2:4, 3:7] = 0
        for r in range(HEIGHT):
            if board[r].all():
                board[r, rng.randrange(WIDTH)] = 0
    game.board = board
    game.active = None
    game._spawn(rng.choice(list(Piece)))  # place the new piece at its real spawn position
    if rng.random() < 0.5:
        game.hold = rng.choice(list(Piece))
    game.can_hold = rng.random() < 0.8
    return game


def pushed_up_state(seed: int, rng: random.Random) -> Game:
    """Garbage arrives after spawn and pushes the stack into the live piece."""
    game = jagged_state(seed, rng, max_height=rng.randint(15, 19))
    for _ in range(4):
        if game.game_over:
            break
        game.receive_garbage(1)
    return game


def moved_state(seed: int, rng: random.Random) -> Game:
    """The live piece has already been shifted, rotated or dropped by a player."""
    game = random_play_state(seed, rng) if rng.random() < 0.5 else jagged_state(seed, rng)
    for _ in range(rng.randint(1, 6)):
        game.apply(rng.choice([Move.LEFT, Move.RIGHT, Move.CW, Move.CCW, Move.SOFT]))
    return game


def random_states(n: int, seed: int = 0) -> list[Game]:
    rng = random.Random(seed)
    states = []
    for i in range(n):
        kind = i % 6
        s = seed * 7919 + i
        if kind == 0:
            states.append(random_play_state(s, rng))
        elif kind == 1:
            states.append(jagged_state(s, rng))
        elif kind == 2:
            states.append(jagged_state(s, rng, max_height=rng.randint(14, 19)))
        elif kind == 3:
            states.append(jagged_state(s, rng, spawn_rows=True))
        elif kind == 4:
            states.append(pushed_up_state(s, rng))
        else:
            states.append(moved_state(s, rng))
    return [g for g in states if not g.game_over and g.active is not None]


def assert_legal(game: Game, placement: Placement) -> None:
    legal = {p.index for p, _ in legal_placements(game)}
    assert placement.index in legal, f"{placement} not legal"
