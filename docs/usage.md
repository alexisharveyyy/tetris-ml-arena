# Using the arena

How to play, train, evaluate and test, and how to add an agent. Setup and
the quick start are in the [README](../README.md); the code structure is in
[architecture.md](architecture.md).

## Checkpoints

Trained checkpoints are not in git, except the small JSON weights of the GA,
CEM and NEAT agents. Train the other agents (below), or point the apps at an
existing `checkpoints/` directory with `--checkpoint-dir` (web) or the
`TETRIS_CHECKPOINT_DIR` environment variable (both apps).

That directory is read-only and searched before `checkpoints/` for each
agent's promoted file. An agent is offered only when that file is complete
(unchanged for 10 s, a full archive or valid JSON) and loads in a CPU worker;
otherwise it shows as "Not available yet". `TETRIS_CHECKPOINT_SETTLE_SECONDS`
changes the 10 s window (the test suite sets it to 0). Trainers always write
to this checkout's `checkpoints/`.

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
- **Battle**: line clears send garbage (see the attack table in
  [architecture.md](architecture.md#game-rules)) to opponents by targeting
  rule: a random opponent, the leader (most lines), or all opponents with the
  rows split. Each board has its own garbage hole sequence, seeded from the
  match seed. Incoming garbage waits in the red meter beside each board and
  is cancelled by your own attacks first. Press Tab to change your targeting
  mode during the match. The header shows who you are targeting and who is
  targeting you. Last board standing wins.
- The results screen shows placement, lines, attack sent and pieces per
  second. Press R for a rematch with the same seed or N for a new one.

Agents run in persistent worker processes (one per opponent, model loaded
once, CPU inference). If a worker crashes or misses its decision deadline
(2 s for most agents, set in each agent's `config.yaml` under `play`), that
opponent switches to the Dellacherie baseline for the rest of the match and
its board shows a warning.

### Difficulty

| Difficulty | Speed cap | Moves | Reaction | Mistake rate | Random moves |
|---|---|---|---|---|---|
| Easy | 0.5 pieces/s | 5/s | 900 ms | 0.40 | 0.03 |
| Medium | 1 piece/s | 10/s | 450 ms | 0.15 | 0 |
| Hard | 2 pieces/s | 25/s | 200 ms | 0.03 | 0 |
| Unlimited | none | instant | 0 | 0 | 0 |

The mistake rate is the chance of picking one of the next five best placements
instead of the best one, so mistakes stay plausible. Measured results are in
`reports/analysis/difficulty.md`. Easy clears about 10 lines per minute in a
race.

### The LLM opponent (optional, off by default)

A novelty opponent: a language model picks each placement from a text
description of the board, and the GA's move is played when it times out or
answers badly. It is weak (`reports/llm/llm_agent.md`) and slow: 8 s per
move, capped at one piece per 8 s at every preset. To offer it, install
Ollama, run `ollama pull qwen3:8b`, and set `enabled: true` in
`tetris/agents/llm/config.yaml`. Its board shows how many of its moves fell
back to the GA. See `tetris/agents/llm/README.md` for the API backends and
the spending cap.

## Play: web

```bash
python -m tetris.web                     # serves http://127.0.0.1:8765/ and opens the browser
python -m tetris.web --checkpoint-dir DIR
```

The same modes, agents and rules as the desktop app, in four looks
(Tetromino, Terminal, Broadcast, Poster; switch in Settings). The server
wraps the same `MatchController`, so game rules, DAS and ARR stay in Python;
the page sends key presses and draws snapshots at 60 fps. The leaderboard
screen reads `reports/leaderboard/` and marks tied ranks.

To work on the frontend: `cd web && npm install && npm run dev` (with the
server running), then `npm run build` before committing, which updates
`tetris/web/static`. `tests/web/test_build.py` fails if `web/src` has
changed since that build, and the launcher rebuilds it when Node is
installed. `/debug` is a bare page that exercises the whole loop;
`python -m tetris.web.protocol` prints the WebSocket protocol's JSON schema;
`scripts/tools/web_latency.py` measures snapshot size and input latency
against a running server (`reports/ui/web_performance.md`).

## Demo mode

```bash
python -m tetris.web --demo
```

For a class demo: the look, the preselected opponents and a fixed seed come
from `configs/demo.yaml`; F9 ends a match and shows the results; F10 cycles
the four looks on any screen. `docs/demo_checklist.md` has the steps for the
day before and at the projector. `python -m tetris.web --state-dir DIR`
keeps settings, history and recordings in `DIR` for a rehearsal.

## Train

Each agent has a package under `tetris/agents/<name>/` with `train.py`,
`config.yaml` and a README describing its method and results. Every trainer
reads its `config.yaml`, seeds everything, takes `--resume`, chooses its
checkpoint on the 30 selection seeds, and promotes it to
`checkpoints/<agent>/best.*`. Learning curves go to
`reports/training/curves/`.

```bash
python -m tetris.agents.heuristic_ga.train     # GA heuristic, about 15 minutes on 12 cores
python -m tetris.agents.cem.train              # cross-entropy method, about 15 minutes
python -m tetris.agents.dqn.train              # DQN, 3 hours on MPS
python -m tetris.agents.ppo.train              # PPO (afterstates), 3 hours on MPS
python scripts/train/log_games.py --agent ga   # GA games for imitation learning
python -m tetris.agents.imitation.train pure   # BC on the GA's games (also: dagger, choose, you)
python -m tetris.agents.neuroevolution.train   # NEAT, 3 hours on CPU
python scripts/queue/run_phase6.py             # Decision Transformer: collect, train, calibrate, leaderboard
python scripts/queue/run_phase6b.py            # DT v2 (collect, train, calibrate, choose, evaluate), then the local LLM evaluation
tensorboard --logdir runs
```

The queue scripts (`scripts/queue/run_phase*.py`) run their steps one at a
time, each through `scripts/queue/pinned_step.py`, which refuses to run
unless the code is the pinned commit's. Run them from a worktree pinned to
the commit you want results for. `scripts/README.md` lists every script.

BC (you) trains on your own games: record them with Record mode (desktop or
web), then run `python -m tetris.agents.imitation.train you`.

## Evaluate

```bash
python scripts/eval/evaluate.py                    # leaderboard, 500 held-out seeds, paired bootstrap, Marathon
python scripts/eval/eval_dt_versions.py            # DT v1 vs v2 held out
python scripts/eval/eval_llm.py --dry-run          # LLM evaluation plan (needs Ollama or an API key)
python scripts/eval/seed_spread.py                 # 1-hour DQN and PPO runs on 3 seeds
python scripts/eval/eval_difficulty.py             # difficulty presets in real-time headless matches
python scripts/tools/stress_test.py --mode battle  # frame times with 3 Unlimited agents
```

`evaluate.py` plays each available agent on 500 fixed held-out seeds and
writes `reports/leaderboard/leaderboard.md` plus CSVs: per-agent rows, every
game (`leaderboard_games_<setting>.csv`), paired bootstrap intervals between
agents, and the checkpoint behind each result (`leaderboard_sources.json`).
The primary leaderboard is the garbage setting (one garbage row every 3
pieces, 500-piece cap). The plain setting (no garbage, 1,000-piece cap) is a
secondary sanity check, and the Marathon section plays each agent at the
Easy, Medium and Hard presets.

## Tests

```bash
pytest
ruff check . && ruff format --check .
cd web && npm test && npx playwright test     # web UI: unit tests, screenshots, audit, LLM fallback line
LIVE=1 npx playwright test keyboard live      # against a real server in demo mode
```

Playwright writes every screen in every look to `web/screenshots/` (not in
git); the README uses five of them, copied to `docs/screenshots/`.

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
