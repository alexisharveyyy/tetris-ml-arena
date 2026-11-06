# Tetris ML Arena in the browser

A static build of the arena for GitHub Pages
(<https://alexisalineharvey.com/tetris-ml-arena/>). The engine and the agents
run in the visitor's browser; there is no server.

## How it fits together

- `src/engine/`: the TypeScript port of `tetris/engine` (SRS with kicks,
  7-bag from SplitMix64, scoring, gravity, lock delay). Gravity uses the
  float64 values exported from Python, because `pow` can differ in the last
  bit between languages.
- `src/agents/`: afterstate enumeration, the 13 features, the network input
  encodings, the path planner, and every agent of the web version. The LLM
  agent is not included.
- `src/nn/`: a small inference module. `onnx.ts` reads the float32
  initializers of an ONNX file by name; `layers.ts` and `models.ts` rebuild
  the networks of `tetris/rl/nets.py` and the Decision Transformer.
- `src/match/`: the match controller, agent driver, DAS/ARR, event feed and
  snapshot encoder, ported from `tetris/match` and `tetris/web`.
- `src/worker/`: one Web Worker per agent board. A worker fetches its model
  files only when that agent is chosen. A decision that takes longer than
  2 s, or a worker that fails, hands the board to the Dellacherie baseline,
  as the local app does.
- `src/local/`: the REST API and socket of the local server, answered in the
  page. `vite.config.ts` swaps these modules in for `web/src/api/client.ts`
  and `web/src/api/socket.ts` at build time, so the web client's screens and
  looks run without changes to `web/src`. Settings and the Marathon record
  are kept in `localStorage`.
- `src/shell/`: the entry point, the About page, touch controls, phone
  layouts (CSS only) and a copy of `web/src/lib/fit.tsx` that lets boards
  shrink further on phones.

## Model files

`python scripts/tools/web_export.py` (needs `pip install -e ".[export]"` and
the trained checkpoints) writes `public/models/*.onnx` and `src/data/*.json`.
It checks every exported initializer against its checkpoint tensor and
compares onnxruntime with PyTorch on real afterstates. PPO and the BC agents
export only the afterstate scorer (the critic is not used in play). The
Decision Transformer exports two graphs: the afterstate embedder and the
sequence model. The trained checkpoints themselves are not changed.

## Parity tests

```bash
npm run fixtures:engine   # python scripts/tools/web_fixtures.py engine
npm test
```

- **Engine:** 50 games from the Python engine (25 with random inputs, 20
  played by the GA agent through the planner in real time, 5 with instant
  placements), 55,284 steps in total. After every move or gravity tick the
  TypeScript state, including the RNG states, timers and emitted events,
  must equal Python's.
- **Agents:** on recorded board states, each agent must pick the same
  placement as its Python original. A different pick is allowed only at a
  near-tie: Python's top two scores are within 1e-4 of each other (relative to
  the top score's size, at least 1), and the TypeScript pick is one of
  Python's placements within that distance of the top. Near-ties are counted
  per agent in `tests/parity-report.json`.

  | Agent | States | Same pick | Near-ties | Of which the same board |
  |---|---|---|---|---|
  | Dellacherie | 180 | 180 | 0 | 0 |
  | GA heuristic | 180 | 180 | 0 | 0 |
  | CEM | 180 | 180 | 0 | 0 |
  | NEAT | 180 | 180 | 0 | 0 |
  | DQN | 180 | 180 | 0 | 0 |
  | PPO | 180 | 173 | 7 | 5 |
  | PPO (raw actions) | 122 | 122 | 0 | 0 |
  | BC | 180 | 172 | 8 | 7 |
  | BC + DAgger | 180 | 176 | 4 | 4 |
  | BC (you) | 180 | 173 | 7 | 7 |
  | Decision Transformer | 300 | 300 | 0 | 0 |

  "The same board" means both placements leave an identical board (for
  example the four rotations of the O piece): the network sees identical
  inputs, PyTorch's batched arithmetic gives them scores a few millionths
  apart, and TypeScript gives them equal scores and takes the first. The
  other three near-ties have Python scores within 1e-5 of each other. The
  Decision Transformer states are two whole games (it keeps a history), with
  the history fed the placement Python actually played.
- **Planner:** every legal placement on the recorded states is reachable with
  real moves and gives the same board as an instant placement.

The neural agents' fixtures need the trained checkpoints, so they are
recorded locally (`python scripts/tools/web_fixtures.py agents`) and
committed. CI recomputes the linear and NEAT fixtures from the committed
weights and checks them against the committed files.

## Browser checks

```bash
npx playwright test                       # against a local preview
SITE_URL=https://alexisalineharvey.com/tetris-ml-arena/ npx playwright test
```

At desktop and phone sizes: every look loads without console errors and never
scrolls sideways; every agent places pieces; the frame rate with three
neural agents; the transfer size of the first load and of each model; the
touch controls.
