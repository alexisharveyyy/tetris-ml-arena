# Architecture

How Tetris ML Arena is built. For setup, see `README.md`; for usage,
`docs/usage.md`; for results, `reports/FINAL_REPORT.md`.

## Overview

```
                     tetris.engine  (rules, pure Python, deterministic)
                           |
          +----------------+------------------+
          |                                   |
     tetris.env                        tetris.match
 (Gymnasium env, afterstates,    (MatchController: every board, the shared
  features, rewards)              clock, Race/Battle rules, agent drivers,
          |                       worker processes, DAS/ARR, feed, history)
     tetris.agents                     |                    |
 (Agent interface, registry,     tetris.ui.pygame      tetris.web
  one package per agent,         (desktop frontend)    (FastAPI server, WebSocket
  evaluation, difficulty,                               protocol, built React app
  checkpoint checks)                                    in tetris/web/static)
          |                                                 |
     tetris.rl                                         web/  (React + TypeScript
 (shared nets, vectorized env,                          source, four looks,
  selection, pre-flight, watchdog)                      Vitest and Playwright)
```

**Design rules:**
- **One engine, one controller.** Game rules, gravity, lock delay, DAS and
  ARR live only in Python (`tetris.engine` and `tetris.match`).
  - Both frontends drive the same `MatchController` and only draw what it
    reports.
  - The web page never simulates the game. It sends key presses and draws
    snapshots.
- **Pygame stays in `tetris/ui/pygame`.** `tests/test_architecture.py` fails
  if any other module imports it.
- **Determinism.** Every random choice comes from seeded SplitMix64 streams
  (`tetris/engine/rng.py`).
  - A game state serializes to JSON (`Game.to_dict`/`from_dict`). This is
    how a state crosses into agent worker processes.
  - A seed fixes a match's piece sequence and every board's garbage holes.

## Engine (`tetris/engine/`)

- **`game.py`:** a headless `Game`.
  - Pieces: SRS rotation with standard kicks, 7-bag, 5 next pieces, hold.
  - Timing: guideline gravity; 0.5 s lock delay with 15 move resets.
  - Scoring, attack and a pending garbage meter.
  - `apply(move)` for real-time input, `tick(dt)` for gravity, and
    `place(placement)` for an instant placement in headless play.
- **`placement.py`:** the 80 placement actions (rotation × column × hold) and
  their legality rule.
- **`board.py`, `pieces.py`, `scoring.py`:** the board array, piece shapes and
  kicks, and the scoring and attack tables.

## Environment (`tetris/env/`)

- **`TetrisEnv`:** a Gymnasium env over placements, with an action mask.
- **`afterstates.py`:** enumerates every legal placement with the board after
  line clears. The afterstate agents score these boards.
- **`features.py`:** the 13 board and placement features shared by the
  linear agents and NEAT.
- **`rewards.py`:** reward shaping from `configs/env_default.yaml`.

## Agents (`tetris/agents/`)

**Interface (`base.py`).** An agent implements
`rank_placements(game) -> [ScoredPlacement]`, best first.
- `select_placement` and `decide(state, top_k)` are built on it. They add
  difficulty noise and return the top candidates for the "model thinking"
  overlay.
- Agents never move pieces. The shared planner (`planner.py`) turns a
  chosen placement into real key moves.

**Registry (`registry.py`).** An `AgentSpec` per agent gives:
- display name, description, factory and promoted checkpoint file;
- flags: `playable` and `leaderboard`; `opt_in`, used by the LLM agent,
  which is off unless its config enables it; and `supports_target_return`,
  now off for the DT;
- the agent's YAML `play` section:
  - `decision_timeout` and `startup_timeout` for its worker;
  - an optional `max_pps` speed cap applied over every preset.

**One package per agent**, each with `agent.py`, `train.py`, `config.yaml`
and `README.md`:
- `heuristic_ga/`, `cem/`, `neuroevolution/` (NEAT);
- `dqn/`, `ppo/`, `imitation/` (BC, DAgger, BC (you));
- `decision_transformer/`, `llm/`.
- The linear heuristics, Dellacherie included, share `linear.py`.

**Evaluation (`evaluation.py`):**
- the garbage and plain settings, fitness, and the held-out and selection
  seeds;
- parallel headless play, Wilson intervals and the paired bootstrap.
- `scripts/eval/evaluate.py` builds the leaderboard from it and writes
  per-game results.

**Difficulty (`difficulty.py`):** the presets (speed cap, move rate,
reaction delay, decision noise).

**Checkpoint safety and provenance:**
- **`checkpoint_check.py`** decides whether a promoted checkpoint is safe to
  offer:
  - the file must be complete and unchanged for a settle window;
  - it must load in a CPU worker within the startup timeout.
- **`provenance.py`** records which checkpoint (path and SHA-256) produced
  each leaderboard result (`reports/leaderboard/leaderboard_sources.json`). The web
  leaderboard marks a result "pending rerun" when the promoted checkpoint
  has changed since.

## Training infrastructure (`tetris/rl/`, `scripts/`)

**Shared training code:**
- `nets.py`: the CNN afterstate scorer, run in chunks of 4,096 afterstates
  because of the MPS gradient bug.
- `vec_env.py`: vectorized environments in worker processes.
- `training.py`: selection on the 30 selection seeds, checkpoint promotion
  and resume.

**Safeguards for GPU runs:**
- `scripts/train/preflight.py` compares MPS gradients with CPU gradients before
  any GPU run.
- A watchdog stops runs whose learning has stalled.

**Long jobs run as queues** (`scripts/queue/run_phase*.py`, built on the helper in
`run_phase4.py`):
- **Isolation:** each queue runs from a detached git worktree pinned to one
  commit.
- **Order:** steps run one at a time, so nothing runs in parallel with GPU
  training.
- **Pinning check:** each step starts through `scripts/queue/pinned_step.py`. It
  logs where `tetris` was imported from and the commit, and refuses to run
  unless both are the pinned ones and no code has changed.

## Matches (`tetris/match/`)

**`MatchController`** owns every board, the shared clock, the mode rules
(Solo, Watch, Race, Battle, Marathon) and the final standings.
- **Real-time mode:** used by both frontends. Agents answer asynchronously
  while gravity runs.
- **Lockstep mode:** used for headless Marathon evaluation and tests. The
  clock waits for each decision, so results are deterministic.

**Agent players:**
- **Agents run in worker processes** (`agent_runtime.py`, `agent_worker.py`),
  one per opponent, with the model loaded once and inference on the CPU.
  `ProcessRunner` serves live play and `InlineRunner` serves tests.
- **Fallback:** if a worker crashes, fails to load or misses its decision
  deadline, the slot switches to the Dellacherie baseline for the rest of
  the match, and the snapshot says why.
- **`AgentDriver`** plays one agent's board in real time:
  - it requests a decision when a piece spawns;
  - it waits out the reaction delay;
  - it feeds the planner's moves at the preset's move rate;
  - it holds the hard drop back to respect the speed cap.
- **The LLM agent's per-move record:** each decision carries the agent's
  report, `Decision.info`, with the outcome and seconds waited. The driver
  also records "late" when a piece locks while the model is still thinking.
  Result rows and Marathon history keep this record, and snapshots carry the
  count.

**Supporting modules:**
- `rules.py`: Race win conditions and Battle targeting and attack routing.
- `das.py`: DAS, ARR and soft-drop repeat, shared by both frontends.
- `feed.py`: typed match events (line clears, rank changes, fallbacks) for
  narration.
- `lobby.py`: opponent slots and presets.
- `history.py`: the human's Marathon record against each agent and preset.
- `recording.py`: turns a human game into an imitation-learning log.
- `marathon_eval.py`: the headless Marathon benchmark.

**Per-user state** (`tetris/state.py`): settings, history and recordings.
They live under `configs/` and `data/`, or under `TETRIS_STATE_DIR`, which
the tests and rehearsal servers use so they never touch the real files.

## Desktop frontend (`tetris/ui/pygame/`)

- **Screens:** menu, solo, watch, opponents, match, results, record and
  settings.
- **Drawing:** a renderer that draws controller snapshots.
- **Input:** goes through the shared DAS/ARR repeater.

## Web frontend

### Server (`tetris/web/`, FastAPI)

- **`app.py`:** the app. It serves the REST API, the WebSocket and the
  built pages, and starts the checkpoint verifier.
- **`api.py`** (`/api`):
  - reads: agents (with availability and status), presets and lobby;
  - settings: read and update;
  - leaderboard and history;
  - matches: create, end, rematch, pause, keep watching and change an
    agent's difficulty;
  - recordings and the demo-mode settings.
- **`session.py`:** one live match, stepped at 60 Hz by an asyncio task.
  Human key presses arrive over the WebSocket and go through the same
  `InputRepeater` as the desktop app. If the client disconnects, the match
  pauses and resumes after a countdown on reconnect.
- **`protocol.py`:** version 1 of the WebSocket protocol, as Pydantic models.
  - The server sends `welcome`, then about 60 snapshots per second (about
    3.4 KB each).
  - Each snapshot holds every board, the standings, feed events and an
    input acknowledgement for latency measurement.
  - `python -m tetris.web.protocol` prints the JSON schema.
- **`ws.py`:** the WebSocket endpoint.
- **`leaderboard.py`:** reads `reports/` for the leaderboard screen.
- **`demo.py`:** demo mode (`configs/demo.yaml`). It sets the look, the
  opponents and the seed, and enables the F9 results and F10 look-cycling
  shortcuts.
- **`build.py`:** checks that the committed build in `tetris/web/static`
  matches `web/src`.
  - The server runs offline and without Node.
  - `tests/web/test_build.py` fails on a stale build.
  - The launcher rebuilds the frontend when Node is available.

### Client (`web/`, React 19 + TypeScript, Vite)

- **`src/api/`:** the REST client, the WebSocket client with reconnects,
  and the protocol types.
- **`src/state/`:** app state (look, settings, demo) and match state
  (snapshots, ticker, input).
- **`src/screens/`:** Home, Opponents, Match, Results, Leaderboard, People,
  Comparisons, Watch, Solo, Record and Settings.
  - Screens build a view model. A look's layout renders it, and screens
    without a layout fall back to a default.
- **`src/themes/`:** the four looks: Tetromino, Terminal, Broadcast and
  Poster.
  - Each look has its own piece colors, canvas board renderer and layouts
    for Home, Opponents, Match, Results and Leaderboard.
  - `registry.ts` lists them; F10 cycles them in demo mode.
- **`src/components/`:** shared pieces: the canvas board, panels, overlays,
  the LLM fallback line (`ModelFallbacks`), and the significance note
  (adjacent ranks that are statistically tied).
- **`src/lib/`:**
  - `copy.ts` holds all user-facing text, written in sentences rather than
    abbreviations;
  - `fit.tsx` sizes boards to the window;
  - `perf.ts` is the `?perf` frame-time readout.
- **Fonts** are self-hosted and copied from npm packages
  (`npm run fonts`).

**Tests** (`web/tests/`):
- Vitest unit tests for copy, availability and components.
- Playwright:
  - screenshots of every screen in every look, against mocked API and
    WebSocket data, saved to `web/screenshots/` (not in git; the README uses five of them, in `docs/screenshots/`);
  - an accessibility audit with axe (`audit.spec.ts`);
  - the LLM fallback line in all four looks (`model.spec.ts`).
- Live specs (`LIVE=1`) start a real server in demo mode with temporary
  state: keyboard, live play and frame-time measurement.
  - Measured: 60 fps in every look; input-to-render p50 of about 25 ms
    (`reports/ui/web_performance.md`).

## Where results come from

```
train.py (per agent)      --> checkpoints/<agent>/<run>/eval_*.pt --(selection seeds)--> best.pt
                              reports/training/curves/curve_<agent>_<run>.png
scripts/eval/evaluate.py  --> reports/leaderboard/: leaderboard.md, leaderboard_*.csv,
                              leaderboard_sources.json, leaderboard_games_*.csv (per game)
scripts/eval/eval_*.py    --> reports/analysis/ (difficulty, DT versions), reports/llm/
reports/leaderboard/      --> web leaderboard screen (tetris/web/leaderboard.py)
```

`checkpoints/`, `data/` and `runs/` are not in git. The reports name the
checkpoint behind each result, and `reports/FINAL_REPORT.md` Section 6 gives
the command and commit for each one.

## Game rules

- 10x20 visible board plus 4 hidden spawn rows; SRS rotation with standard
  wall kicks.
- Lock delay uses extended placement: up to 15 moves or rotations reset it,
  and a move counts even when a floor kick lifts the piece. Once the 15 are
  used, the piece locks as soon as it touches down (no infinite spinning).
- 7-bag randomizer, 5 visible next pieces, hold once per piece.
- Guideline gravity curve, 0.5 s lock delay with up to 15 move resets.
- Scoring: 100/300/500/800 times level, 1.5x for back-to-back Tetrises, 50
  times combo times level.
- Attack: single 0, double 1, triple 2, Tetris 4; combo bonus 0-1: 0, 2-3: 1,
  4-5: 2, 6-7: 3, 8-10: 4, 11+: 5; +1 for back-to-back Tetris. No T-spins.
- Incoming garbage waits in a pending meter. Outgoing attack cancels it
  first. Whatever is left is inserted the next time a piece locks without
  clearing a line. Each batch shares one random hole column.
- All randomness comes from seeded SplitMix64 streams, so the full state
  serializes to JSON (`Game.to_dict` / `Game.from_dict`).

## Action space

Agents choose a placement: `(rotation, column, use_hold)`, which gives
4 x 10 x 2 = 80 discrete actions with a legality mask. A placement is legal
when the piece can rotate in place at spawn (every intermediate orientation
must also fit, so no kick is needed), shift sideways at spawn height to the
target column, and hard drop. Tucks and spins are out of scope.

A test runs 480 random states, about 21,800 legal actions, including tall,
jagged and spawn-row stacks. It checks that the path planner can reach every
action the mask allows, and that executing the plan reproduces the enumerated
board exactly. `tetris.env.enumerate_afterstates(game)` returns every legal
placement with its resulting board and features.
