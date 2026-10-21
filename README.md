# Tetris ML Arena

Play Tetris against machine learning and deep learning agents in real time.
The game engine is headless and deterministic. A `MatchController` owns every
board and the shared clock. The Pygame desktop app is one frontend over that
controller, and a web frontend can wrap the same controller later.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Engine, Gymnasium env, tests, Solo mode | Done |
| 2 | Agent interface, path planner, GA heuristic, CEM, Watch AI | Done |
| 3 | Race and Battle, opponent selection, difficulty, worker pool | Done |
| 4 | DQN, PPO | Done |
| 5 | Imitation learning, recording mode, NEAT | Done |
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

### Watch AI

Pick any agent with a checkpoint and watch it play in real time.

| Key | Action |
|---|---|
| 1 to 4 | Difficulty preset: Easy, Medium, Hard, Unlimited |
| Left/Right or -/+ | Free speed slider (pieces per second, up to unlimited) |
| T | Show the agent's top 3 candidate placements as outlines (H = via hold) |
| Space | Pause |
| R | New game |
| Esc | Back |

### Race and Battle

Choose 1 to 3 opponents (duplicates allowed), each with a difficulty preset or
custom settings. Only agents with checkpoints are listed, and the last
selection is remembered.

- **Race**: all boards get the same piece sequence. Pick a win condition: most
  lines in a time limit, first to N lines, or highest score at the time limit.
  Topping out ends that player's game.
- **Battle**: line clears send garbage (see the attack table below) to opponents
  by targeting rule: a random opponent, the leader (most lines), or all
  opponents with the rows split. Each board has its own garbage hole sequence,
  seeded from the match seed. Incoming garbage waits in the red meter beside
  each board and is cancelled by your own attacks first. Press Tab to change
  your targeting mode during the match. The header shows who you are
  targeting and who is targeting you. Last board standing wins.
- The results screen shows placement, lines, attack sent and pieces per
  second. Press R for a rematch with the same seed or N for a new one.

Agents run in persistent worker processes (one per opponent, model loaded
once, CPU inference). If a worker crashes or takes more than 2 s to decide,
that opponent switches to the Dellacherie baseline for the rest of the match
and its board shows a warning.

| Difficulty | Speed cap | Moves | Reaction | Mistake rate | Random moves |
|---|---|---|---|---|---|
| Easy | 0.5 pieces/s | 5/s | 900 ms | 0.40 | 0.03 |
| Medium | 1 piece/s | 10/s | 450 ms | 0.15 | 0 |
| Hard | 2 pieces/s | 25/s | 200 ms | 0.03 | 0 |
| Unlimited | none | instant | 0 | 0 | 0 |

The mistake rate is the chance of picking one of the next five best placements
instead of the best one, so mistakes stay plausible. Measured results are in
`reports/difficulty.md`. Easy clears about 10 lines per minute in a race.

## Train

```bash
python -m tetris.agents.heuristic_ga.train     # GA heuristic, about 15 minutes on 12 cores
python -m tetris.agents.cem.train              # cross-entropy method, about 15 minutes
python scripts/log_games.py --agent ga         # logs games for imitation learning
tensorboard --logdir runs
```

Every trainer takes `--resume`, reads its `config.yaml`, seeds everything, and
writes checkpoints under `checkpoints/<agent>/`.

## Evaluate

```bash
python scripts/evaluate.py                 # leaderboard, 500 held-out seeds, paired bootstrap
python scripts/eval_difficulty.py          # difficulty presets in real-time headless matches
python scripts/stress_test.py --mode battle  # frame times with 3 Unlimited agents
```

`evaluate.py` plays each available agent on 500 fixed held-out seeds and writes
`reports/leaderboard.md` plus CSVs, including paired bootstrap intervals
between agents. The primary leaderboard is the garbage
setting (one garbage row every 3 pieces, 500-piece cap). The plain setting
(no garbage, 1,000-piece cap) is a secondary sanity check.

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
  agents/     agent interface, registry, path planner, difficulty, evaluation
    heuristic_ga/, cem/   one subpackage per agent: agent.py, train.py, config.yaml, README.md
  match/      MatchController (real time or lockstep), rules (race, battle targeting),
              agent driver, worker pool runner, lobby, DAS/ARR input repeat
  ui/pygame/  the only package that imports Pygame
  settings.py key bindings and input timing
configs/      default env and reward config
scripts/      evaluate.py, eval_difficulty.py, stress_test.py, log_games.py, bench_env.py
reports/      leaderboard, difficulty and stress test results
tests/
```

## Adding an agent

1. Create `tetris/agents/<name>/` with `agent.py`, `train.py`, `config.yaml`
   and `README.md`.
2. In `agent.py`, subclass `tetris.agents.base.Agent`:
   - set `key`, `name` and `description`;
   - implement `rank_placements(game)` to return legal placements best first
     as `ScoredPlacement` objects;
   - implement `load(checkpoint)`.

   `select_placement` and `decide` (which adds the top-k candidates for the
   overlay) come for free, including epsilon noise from the difficulty
   settings. Use `tetris.env.enumerate_afterstates(game)` or
   `tetris.env.action_mask(game)` to see the legal moves.
3. Register it in `tetris/agents/registry.py` with an `AgentSpec` naming the
   factory (`"module:Class"`) and the checkpoint file. It appears in Watch AI
   once that file exists under `checkpoints/<key>/`.
4. Agents never move pieces themselves. The shared planner
   (`tetris/agents/planner.py`) finds real SRS move sequences to the chosen
   placement, and the match's agent driver plays them at the difficulty's
   speed.

## Engine rules (v1)

- 10x20 visible board plus 4 hidden spawn rows; SRS rotation with standard wall kicks.
- Lock delay uses extended placement: up to 15 moves or rotations reset it, and
  a move counts even when a floor kick lifts the piece. Once the 15 are used,
  the piece locks as soon as it touches down (no infinite spinning).
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
rotate in place at spawn (every intermediate orientation must also fit, so no
kick is needed), shift sideways at spawn height to the target column, and hard
drop. Tucks and spins are out of scope for v1.

A test runs 480 random states, about 21,800 legal actions, including tall,
jagged and spawn-row stacks. It checks that the path planner can reach every
action the mask allows, and that executing the plan reproduces the enumerated
board exactly.
`tetris.env.enumerate_afterstates(game)` returns every legal placement with its
resulting board and features.
