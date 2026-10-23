# Tetris ML Arena

Play Tetris against machine learning and deep learning agents in real time,
on the desktop (Pygame) or in the browser. Eleven agents (a hand-tuned
heuristic, a genetic algorithm, the cross-entropy method, DQN, PPO, behavior
cloning with and without DAgger, BC on your own games, NEAT, a Decision
Transformer, and an optional LLM) play the same deterministic engine. They
are compared on 500 held-out games under one protocol.

The full write-up is `reports/FINAL_REPORT.md`; the code structure is in
`docs/architecture.md`.

![Match screen, Tetromino look](docs/screenshots/A3-match.png)

| | |
|---|---|
| ![Home, Poster look](docs/screenshots/D1-home.png) | ![Leaderboard, Broadcast look](docs/screenshots/C5-leaderboard.png) |
| ![Match, Terminal look](docs/screenshots/B3-match.png) | ![Results, Tetromino look](docs/screenshots/A4-results.png) |

Screenshots: `cd web && npx playwright test screens` regenerates every
screen in every look in `web/screenshots/` (mocked data; see Tests).

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Engine, Gymnasium env, tests, Solo mode | Done |
| 2 | Agent interface, path planner, GA heuristic, CEM, Watch AI | Done |
| 3 | Race and Battle, opponent selection, difficulty, worker pool | Done |
| 4 | DQN, PPO | Done |
| 5 | Imitation learning, recording mode, NEAT | Done |
| 6 | Decision Transformer (v1, v2), optional LLM agent | Done (LLM API runs pending a key) |
| 7 | LLM follow-ups, final report, docs | Done |
| 8 | Web UI | Done |

## Main findings

From the held-out garbage leaderboard (500 seeds, paired bootstrap;
`reports/leaderboard/leaderboard.md`). The full discussion is in
`reports/FINAL_REPORT.md`.

- **The top 8 agents are statistically tied in sequence.** Every adjacent
  pair from DQN (1.345) down to BC + DAgger (1.280) has a fitness interval
  that includes 0. DQN and PPO are significantly ahead of the bottom of the
  group, but the whole spread is 0.065. These agents share the same action
  space (one hard-drop placement per piece, chosen by scoring afterstates),
  and the linear ones use the same 13 board features. The ceiling appears to
  come from that shared setup rather than from the learning method: a
  linear weighting (GA, CEM), a deep network on the raw board (DQN, PPO), a
  cloned policy (BC), an evolved network (NEAT) and a Decision Transformer
  trained offline on their games all land in the same band. Raw-action PPO,
  which must learn placements from scratch, is far below (0.103).
- **The Decision Transformer's target return is not a difficulty knob.** v1
  ranks 4th (1.309), but asking for a return of 0.2 still plays at 1.32 (rank
  correlation between target and achieved fitness +0.43). v2 tested whether
  training data with skill the board shows would fix that; it did not (rank
  correlation +0.00) and played worse held out (-0.090 [-0.127, -0.054]), so
  v1 stays. The app plays the DT at its default target
  (`reports/analysis/dt_v2.md`).
- **DAgger added nothing when BC already agreed 95% with the teacher.** Pure
  BC matched the GA on 95.1% of held-out GA states and played as well as the
  GA (+0.007, tie). Two DAgger rounds raised agreement to 96.3% but not play
  (pure BC minus DAgger +0.017 [-0.017, +0.049]).
- **NEAT evolved a nearly linear network.** The selected network kept 2
  hidden nodes, and its output is dominated by a linear term in four
  features. It ties the GA and is significantly behind CEM.
- **Selection on 30 seeds overestimates.** Picking the best of many noisy
  checkpoints picks lucky ones: NEAT dropped 0.115 from selection to held
  out, the others 0.04 to 0.07 (`reports/analysis/neat_selection.md`).
- **Small local LLMs cannot play this game.** qwen3:8b and llama3.1:8b
  answer every move but top out in every game, after about a quarter of the
  GA's pieces; the features prompt helps (+16.2 pieces for qwen3)
  (`reports/llm/llm_agent.md`).
- **Two bugs were diagnosed and fixed along the way:**
  - *DQN overestimation:* greedy play peaked at 25 minutes and then
    declined as Q-values drifted upward; a soft target update and an LR
    schedule slowed the decline, but its best (1.253) stayed below the
    first run's 1.340 (`reports/training/dqn_overestimation.md`).
  - *MPS gradient bug:* on Apple MPS, convolution backward returned wrong
    gradients (relative error about 6e8) at about 115k afterstates per
    batch, so the first afterstate PPO run never trained. The scorer now
    runs in chunks, and every GPU run starts with a pre-flight check against
    CPU gradients and runs under a stalled-training watchdog
    (`reports/training/ppo_mps_bug.md`).

## Setup

Requires Python 3.11 or 3.12 (3.12 recommended; some dependencies lack wheels
for newer versions). Training the deep agents uses Apple MPS when available;
playing uses only the CPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,web]"
```

Node is needed only to work on the web frontend (`cd web && npm install`);
the built frontend is committed.

Trained checkpoints are not in git. Train the agents (below), or point the
apps at an existing `checkpoints/` directory with `--checkpoint-dir` (web) or
the `TETRIS_CHECKPOINT_DIR` environment variable (both apps). An agent is
offered only when its promoted checkpoint is complete and loads.

## Train

Each agent has a package under `tetris/agents/<name>/` with `train.py`,
`config.yaml` and a README describing its method and results. Every trainer
reads its `config.yaml`, seeds everything, takes `--resume`, chooses its
checkpoint on the 30 selection seeds, and promotes it to
`checkpoints/<agent>/best.*`.

```bash
python -m tetris.agents.heuristic_ga.train     # GA heuristic, about 15 minutes on 12 cores
python -m tetris.agents.cem.train              # cross-entropy method, about 15 minutes
python -m tetris.agents.dqn.train              # DQN, 3 hours on MPS
python -m tetris.agents.ppo.train              # PPO (afterstates), 3 hours on MPS
python scripts/train/log_games.py --agent ga         # GA games for imitation learning
python -m tetris.agents.imitation.train pure   # BC on the GA's games (also: dagger, choose, you)
python -m tetris.agents.neuroevolution.train   # NEAT, 3 hours on CPU
python scripts/queue/run_phase6.py                   # Decision Transformer: collect, train, calibrate, leaderboard
python scripts/queue/run_phase6b.py                  # DT v2 (collect, train, calibrate, choose, evaluate), then the local LLM evaluation
tensorboard --logdir runs
```

The queue scripts (`scripts/queue/run_phase*.py`) run their steps one at a time,
each through `scripts/queue/pinned_step.py`, which refuses to run unless the code
is the pinned commit's. Run them from a worktree pinned to the commit you
want results for.

BC (you) trains on your own games: record them with Record mode (desktop or
web), then run `python -m tetris.agents.imitation.train you`.

## Play: desktop

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
once, CPU inference). If a worker crashes or misses its decision deadline
(2 s for most agents, set in each agent's `config.yaml` under `play`), that
opponent switches to the Dellacherie baseline for the rest of the match and
its board shows a warning.

| Difficulty | Speed cap | Moves | Reaction | Mistake rate | Random moves |
|---|---|---|---|---|---|
| Easy | 0.5 pieces/s | 5/s | 900 ms | 0.40 | 0.03 |
| Medium | 1 piece/s | 10/s | 450 ms | 0.15 | 0 |
| Hard | 2 pieces/s | 25/s | 200 ms | 0.03 | 0 |
| Unlimited | none | instant | 0 | 0 | 0 |

The mistake rate is the chance of picking one of the next five best placements
instead of the best one, so mistakes stay plausible. Measured results are in
`reports/analysis/difficulty.md`. Easy clears about 10 lines per minute in a race.

### The LLM opponent (optional, off by default)

A novelty opponent: a language model picks each placement from a text
description of the board, and the GA's move is played when it times out or
answers badly. It is weak (see Main findings) and slow: 8 s per move, capped
at one piece per 8 s at every preset. To offer it, install Ollama, run
`ollama pull qwen3:8b`, and set `enabled: true` in
`tetris/agents/llm/config.yaml`. Its board shows how many of its moves fell
back to the GA. See `tetris/agents/llm/README.md` for the API backends and
the spending cap.

## Play: web

```bash
python -m tetris.web                     # serves http://127.0.0.1:8765/ and opens the browser
python -m tetris.web --checkpoint-dir ~/tetris-ml-arena/checkpoints
```

The same modes, agents and rules as the desktop app, in four looks
(Tetromino, Terminal, Broadcast, Poster; switch in Settings). The server
wraps the same `MatchController`, so game rules, DAS and ARR stay in Python;
the page sends key presses and draws snapshots at 60 fps. The leaderboard
screen reads `reports/` and marks tied ranks.

`--checkpoint-dir` (or `TETRIS_CHECKPOINT_DIR`) names a read-only directory
searched before `checkpoints/` for each agent's promoted file. An agent is
offered only when that file is complete (unchanged for 10 s, a full archive
or valid JSON) and loads in a CPU worker; otherwise it shows as "Not
available yet". `TETRIS_CHECKPOINT_SETTLE_SECONDS` changes the 10 s window
(the test suite sets it to 0). Trainers always write to this checkout's
`checkpoints/`.

To work on the frontend: `cd web && npm install && npm run dev` (with the
server running), then `npm run build` before committing, which updates
`tetris/web/static`. `tests/web/test_build.py` fails if `web/src` has
changed since that build, and the launcher rebuilds it when Node is
installed. `/debug` is a bare page that exercises the whole loop;
`python -m tetris.web.protocol` prints the WebSocket protocol's JSON schema;
`scripts/tools/web_latency.py` measures snapshot size and input latency against a
running server (`reports/ui/web_performance.md`).

## Demo mode

```bash
python -m tetris.web --demo --checkpoint-dir ~/tetris-ml-arena/checkpoints
```

For a class demo: the look, the preselected opponents and a fixed seed come
from `configs/demo.yaml`; F9 ends a match and shows the results; F10 cycles
the four looks on any screen. `docs/demo_checklist.md` has the steps for the
day before and at the projector. `python -m tetris.web --state-dir DIR`
keeps settings, history and recordings in `DIR` for a rehearsal.

## Evaluate

```bash
python scripts/eval/evaluate.py                 # leaderboard, 500 held-out seeds, paired bootstrap, Marathon
python scripts/eval/eval_dt_versions.py         # DT v1 vs v2 held out
python scripts/eval/eval_llm.py --dry-run       # LLM evaluation plan (needs Ollama or an API key)
python scripts/eval/seed_spread.py              # 1-hour DQN and PPO runs on 3 seeds
python scripts/eval/eval_difficulty.py          # difficulty presets in real-time headless matches
python scripts/tools/stress_test.py --mode battle  # frame times with 3 Unlimited agents
```

`evaluate.py` plays each available agent on 500 fixed held-out seeds and
writes `reports/leaderboard/leaderboard.md` plus CSVs: per-agent rows, every game
(`leaderboard_games_<setting>.csv`), paired bootstrap intervals between
agents, and the checkpoint behind each result
(`leaderboard_sources.json`). The primary leaderboard is the garbage setting
(one garbage row every 3 pieces, 500-piece cap). The plain setting (no
garbage, 1,000-piece cap) is a secondary sanity check, and the Marathon
section plays each agent at the Easy, Medium and Hard presets.

## Tests

```bash
pytest
ruff check . && ruff format --check .
cd web && npm test && npx playwright test     # web UI: unit tests, screenshots, audit, LLM fallback line
LIVE=1 TETRIS_CHECKPOINT_DIR=~/tetris-ml-arena/checkpoints npx playwright test keyboard live
```

Tests never touch your settings, Marathon history or recordings. pytest and
the Playwright servers keep them in a temporary directory (`TETRIS_STATE_DIR`,
see `tetris/state.py`), and both runs fail if the real files change while they
run. No test calls a real model backend: `tests/conftest.py` fails any test
that sends a request through the LLM backends.

## Adding an agent

1. Create `tetris/agents/<name>/` with `agent.py`, `train.py`, `config.yaml`
   and `README.md`.
2. In `agent.py`, subclass `tetris.agents.base.Agent`:
   - set `key`, `name` and `description`;
   - implement `rank_placements(game)` to return legal placements best first
     as `ScoredPlacement` objects;
   - implement `load(checkpoint)`.

   `select_placement` and `decide` (which adds the top-k candidates for the
   overlay) come for free, including the noise from the difficulty
   settings. Use `tetris.env.enumerate_afterstates(game)` or
   `tetris.env.action_mask(game)` to see the legal moves.
3. Register it in `tetris/agents/registry.py` with an `AgentSpec` naming the
   factory (`"module:Class"`), the checkpoint file and the config. It is
   offered in Watch AI, Race and Marathon, on both desktop and web, once
   that file exists under `checkpoints/<key>/` and loads, and it appears on
   the leaderboard at the next `scripts/eval/evaluate.py` run.
4. Optional, in the config's `play` section: `decision_timeout` and
   `startup_timeout` for its worker, and `max_pps` to cap its speed at every
   preset. An agent that reports each move (a `move_info()` method, as the
   LLM agent does) gets a fallback count on its board.
5. Agents never move pieces themselves. The shared planner
   (`tetris/agents/planner.py`) finds real SRS move sequences to the chosen
   placement, and the match's agent driver plays them at the difficulty's
   speed.

## Layout

```
tetris/
  engine/     headless rules: SRS, 7-bag, hold, gravity, lock delay, scoring, garbage
  env/        Gymnasium env, afterstate enumeration, features, reward shaping
  agents/     agent interface, registry, planner, difficulty, evaluation, checkpoint checks
    heuristic_ga/ cem/ dqn/ ppo/ imitation/ neuroevolution/ decision_transformer/ llm/
  rl/         shared networks, vectorized env, selection, training loop helpers
  match/      MatchController (real time or lockstep), rules, agent driver,
              worker processes, lobby, DAS/ARR, feed, history, recording
  ui/pygame/  the desktop app (the only package that imports Pygame)
  web/        FastAPI server, WebSocket protocol, committed frontend build (static/)
web/          React + TypeScript frontend source, four looks, Vitest and Playwright tests
configs/      env defaults, demo.yaml
scripts/      evaluation, queues (run_phase*.py, pinned_step.py), diagnostics
reports/      FINAL_REPORT.md and every result, indexed in reports/README.md
docs/         architecture.md, demo_checklist.md
logs/         queue logs of the runs behind the reports
tests/
```

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
