# Cross-entropy method

## Method

Same agent and feature space as the GA heuristic (`tetris/agents/linear.py`):
a greedy one-piece search scoring each afterstate with `features @ weights`.
Only the optimizer differs.

The weights are optimized with the noisy cross-entropy method (Szita and
Lorincz, 2006):

1. Sample 50 weight vectors from a diagonal Gaussian `N(mean, std^2)`,
   starting from mean 0 and std 0.5.
2. Evaluate the samples plus the current mean.
3. Refit the mean and std to the top 20% of samples.
4. Add extra variance (0.05, decaying linearly to 0 over 30 iterations) to
   avoid early collapse.

Run for 40 iterations.

## Fitness and seeds

Identical to the GA so the two are directly comparable: the garbage setting
(one row every 3 pieces, 500-piece cap) and the same fitness
`mean(pieces / 500) + 0.25 * mean(score per piece) / 1000`. Each iteration's
samples share one fresh seed set, the final weights are chosen on the
selection seeds, and results are reported on the same 100 held-out seeds. See
the GA README for the calibration of the garbage setting.

## Train

```bash
python -m tetris.agents.cem.train
python -m tetris.agents.cem.train --resume
tensorboard --logdir runs/cem
```

Outputs `checkpoints/cem/best.json`, `state.json` and `report.json`.

## Results

Held-out seeds 900000 to 900499 (500 games). Garbage setting: one row every 3
pieces, inserted directly (not cancellable), 500-piece cap.

| Agent | Pieces survived | Mean lines | Score/piece | Lines/piece | Top-out rate | Fitness |
|---|---|---|---|---|---|---|
| Cross-entropy method | 468.7 | 321.2 | 1522 | 0.681 | 0.15 | 1.318 |
| GA heuristic | 457.9 | 313.1 | 1495 | 0.676 | 0.18 | 1.289 |
| Dellacherie (baseline) | 427.5 | 289.7 | 1362 | 0.665 | 0.31 | 1.196 |

Paired bootstrap on the same 500 seeds (95% intervals):

| Comparison | Fitness difference | Top-out difference |
|---|---|---|
| CEM minus GA | +0.028 [-0.004, +0.061] | -0.032 [-0.078, +0.014] |
| CEM minus Dellacherie | +0.122 [+0.085, +0.159] | -0.160 [-0.210, -0.110] |
| GA minus Dellacherie | +0.094 [+0.055, +0.133] | -0.128 [-0.180, -0.076] |

Both trained agents beat the hand-tuned baseline significantly. CEM ranks
above the GA, but even at 500 games that gap is not significant: the interval
includes zero. (At 100 games, only CEM versus the baseline was significant.)

Plain setting (no garbage, 1,000-piece cap, secondary sanity check): all three
agents survive every game, with 398.4 to 398.8 lines out of a possible 400.
This saturation is why the garbage setting is the primary measure.

Training time on 12 cores: about 15 minutes each, with 40 generations of 50
candidates times 10 games.

CEM's final weights put most weight on column transitions, rows with holes,
landing height, lines cleared (negative), holes and row transitions. The
negative lines-cleared weight combined with a positive eroded-cells weight
favors clears that use more of the placed piece.

## GA versus CEM

Same features, fitness, seeds and compute budget (2,000 evaluations of 10
games each). Both clearly improve on random initial weights within about 10
generations. CEM converged to a population whose mean fitness nearly matched
its best (1.32 versus 1.41 by iteration 33), while the GA population kept more
diversity (mean 1.0 versus best 1.41). On held-out seeds CEM ended higher,
but not significantly so, even at 500 games (see the table above).
