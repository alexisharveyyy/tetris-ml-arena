# Tetris ML Arena

Play Tetris against machine learning and deep learning agents in real time.
The game engine is headless and deterministic. A `MatchController` owns every
board and the shared clock. The Pygame desktop app is one frontend over that
controller, and a web frontend can wrap the same controller later.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Engine, Gymnasium env, tests, Solo mode | Done |
| 2 | Agent interface, path planner, GA heuristic, CEM, Watch AI | Planned |
| 3 | Race and Battle, opponent selection, difficulty, worker pool | Planned |
| 4 | DQN, PPO | Planned |
| 5 | Imitation learning, recording mode, NEAT | Planned |
| 6 | Decision Transformer, optional LLM agent | Planned |
| 7 | Evaluation harness, leaderboard, docs, web plan | Planned |

## Setup

Requires Python 3.11 or 3.12 (3.12 recommended; some dependencies lack wheels
for newer versions).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Play

```bash
python -m tetris            # main menu
python -m tetris --solo     # jump straight into solo practice
```

Default controls (rebindable in Settings):

| Action | Keys |
|---|---|
| Move | Left / Right |
| Soft drop | Down |
| Hard drop | Space |
| Rotate clockwise | Up, X |
| Rotate counterclockwise | Z |
| Hold | C, Shift |
| Pause | Esc |

DAS, ARR, soft drop speed and the ghost piece are set in Settings and saved to
`configs/user_settings.yaml`. F12 saves a screenshot to `screenshots/`.

## Tests

```bash
pytest
ruff check tetris tests && ruff format --check tetris tests
```

## Layout

```
tetris/
  engine/     headless rules: SRS, 7-bag, hold, gravity, lock delay, scoring, garbage
  env/        Gymnasium env, afterstate enumeration, features, reward shaping
  match/      MatchController, DAS/ARR input repeat
  ui/pygame/  the only package that imports Pygame
  settings.py key bindings and input timing
configs/      default env and reward config
tests/
```

## Engine rules (v1)

- 10x20 visible board plus 4 hidden spawn rows; SRS rotation with standard wall kicks.
- 7-bag randomizer, 5 visible next pieces, hold once per piece.
- Guideline gravity curve, 0.5 s lock delay with up to 15 move resets.
- Scoring: 100/300/500/800 times level, 1.5x for back-to-back Tetrises, 50 times combo times level.
- Attack: single 0, double 1, triple 2, Tetris 4; combo bonus 0-1: 0, 2-3: 1,
  4-5: 2, 6-7: 3, 8-10: 4, 11+: 5; +1 for back-to-back Tetris. No T-spins.
- Incoming garbage waits in a pending meter. Outgoing attack cancels it first.
  Whatever is left is inserted the next time a piece locks without clearing a line.
  Each batch shares one random hole column.
- All randomness comes from seeded SplitMix64 streams, so the full state
  serializes to JSON (`Game.to_dict` / `Game.from_dict`).

## Action space

Agents choose a placement: `(rotation, column, use_hold)`, which gives 4 x 10 x 2 = 80
discrete actions with a legality mask. A placement is legal when the piece can
rotate in place at spawn, shift sideways at spawn height to the target column,
and hard drop. Tucks and spins are out of scope for v1.
`tetris.env.enumerate_afterstates(game)` returns every legal placement with its
resulting board and features.
