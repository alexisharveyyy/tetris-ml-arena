# GA-tuned heuristic

## Method

A linear evaluation over hand-crafted afterstate features, with weights evolved
by a genetic algorithm. For each piece the agent enumerates every legal
placement in the 80-action space (including hold), computes the features of
the resulting board, and picks the placement with the highest
`features @ weights`. Placements that top out are only chosen if nothing else
is legal.

GA details (`config.yaml`):

- Population 50, 40 generations, weights kept at unit length (a linear
  evaluation is invariant to positive scaling).
- Elitism (top 4 carried over), tournament selection (size 3).
- Fitness-weighted blend crossover (rate 0.9).
- Per-gene Gaussian mutation (rate 0.3, sigma 0.3 decaying by 0.96 per
  generation to a floor of 0.05).

## Features

`tetris/env/features.py`, 13 values in this order: aggregate height, max
height, holes, rows with holes, hole depth, bumpiness, wells (cumulative well
depth, Dellacherie style), max well depth, row transitions, column
transitions, lines cleared, landing height and eroded piece cells.

## Fitness and seeds

Plain capped games saturate: a simple greedy agent cleared 799 of a possible
800 lines in 2,000 pieces, so all good weights would tie. Fitness is computed
on the **garbage setting**: one garbage row is inserted every 3 pieces and the
game is capped at 500 pieces. Each batch's hole column moves with probability
0.3.

```
fitness = mean(pieces / 500) + 0.25 * mean(score per piece) / 1000
```

Survival dominates. The efficiency term separates agents that all survive and
rewards multi-line clears.

Calibration (40 seeds each, 500-piece cap; Dellacherie is a strong hand-tuned
reference, greedy4 a weak four-feature heuristic):

| Garbage every | Dellacherie top-out | greedy4 top-out |
|---|---|---|
| 3 pieces | 0.33 | 1.00 |
| 4 pieces | 0.05 | 0.93 |
| 5 pieces | 0.00 | 0.50 |
| 6 pieces | 0.00 | 0.17 |

Every 3 pieces was chosen because a strong agent still tops out in a third of
games, which leaves room above it. Every 4 pieces would likely saturate once
the GA beats the reference. Weak agents still separate on pieces survived.

Seed protocol:

- Every individual in a generation plays the same 10 seeds (common random
  numbers), and each generation draws new seeds from 0 to 500,000.
- After training, the best individual of each of the last 10 generations is
  re-evaluated on 30 fixed selection seeds (800,000+) to pick the final
  weights.
- Final numbers come from 100 held-out seeds (900,000+) that are never used
  for any decision.

Rollouts run in parallel worker processes (`cpu_count - 1` by default).

## Train

```bash
python -m tetris.agents.heuristic_ga.train                # full run (config.yaml)
python -m tetris.agents.heuristic_ga.train --resume       # continue from checkpoints/ga/state.json
python -m tetris.agents.heuristic_ga.train --generations 10 --population 20 --games 5
tensorboard --logdir runs/ga
```

Outputs in `checkpoints/ga/`:

- `best.json`: the weights. Interim weights are written every generation and
  replaced by the selected final weights at the end.
- `state.json`: resume state, saved every generation.
- `report.json`: held-out results.

## Logging games for imitation learning

```bash
python scripts/log_games.py --agent ga --episodes 100 --settings garbage plain
```

Writes `data/logs/ga/<setting>/seed_<seed>.jsonl.gz`. Each step stores the
80-space action index, the legal-action mask and the full game state. See
`tetris/agents/game_log.py`.

## Results

Held-out seeds 900000 to 900099. Garbage setting: one row every 3 pieces, 500-piece cap.

| Agent | Pieces survived | Mean lines | Score/piece | Lines/piece | Top-out rate | Fitness |
|---|---|---|---|---|---|---|
| Cross-entropy method | 474.5 | 325.8 | 1544 | 0.682 | 0.13 | 1.335 |
| GA heuristic | 450.4 | 307.6 | 1473 | 0.672 | 0.20 | 1.269 |
| Dellacherie (reference) | 443.0 | 301.6 | 1419 | 0.672 | 0.25 | 1.241 |

Paired bootstrap on the same 100 seeds (95% intervals):

| Comparison | Fitness difference | Top-out difference |
|---|---|---|
| CEM minus GA | +0.066 [-0.007, +0.143] | -0.07 [-0.17, +0.03] |
| GA minus Dellacherie | +0.028 [-0.062, +0.118] | -0.05 [-0.16, +0.06] |
| CEM minus Dellacherie | +0.094 [+0.029, +0.162] | -0.12 [-0.22, -0.02] |

Only CEM's improvement over the hand-tuned reference is significant at 100
games. CEM ranks above the GA, but that gap is not significant: the
interval includes zero.

Plain setting (no garbage, 1,000-piece cap, secondary sanity check): all three
agents survive every game, with 398.5 to 398.8 lines out of a possible 400.
This saturation is why the garbage setting is the primary measure.

Training time on 12 cores: about 15 minutes each, with 40 generations of 50
candidates times 10 games.

The GA's final weights (unit length) put most mass on column transitions (-0.78),
holes (-0.31), rows with holes (-0.26), landing height (-0.24) and row
transitions (-0.23). The eroded-cells weight came out negative, which is
compensated by the positive lines-cleared weight (the two are strongly
correlated).
