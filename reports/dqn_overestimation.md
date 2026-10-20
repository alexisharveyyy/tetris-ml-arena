# DQN v1: decline after 25 minutes and value overestimation

The first 3-hour DQN run (v1, `checkpoints/dqn/seed0`) peaked early and then
declined. Its greedy evaluations on the 30 selection seeds, in the garbage
setting:

- 25 min: fitness **1.340**, top-out 0.17 (the peak).
- 35 min: 1.336.
- 105 to 145 min: 0.49 to 0.77, top-out 0.70 to 0.87.

The full curve is in `checkpoints/dqn/seed0/selection_curve.csv` and
`reports/curve_dqn_seed0.png`. This report gives the diagnosis and the fix
(v2), and later a v1 versus v2 comparison.

## Training scalars over time (v1)

TensorBoard `runs/dqn/seed0`, averaged per 0.5M environment steps (about 9
minutes):

| Steps | Time | Epsilon | Training pieces | Training top-out | Loss | Mean V (batch) |
|---|---|---|---|---|---|---|
| 0.5M | ~9 min | 0.15 | 53 | 1.00 | 1.14 | 26 |
| 1.5M | ~27 min | 0.09 | 92 | 1.00 | 2.31 | 50 |
| 3.0M | ~54 min | 0.005 | 257 | 0.77 | 2.20 | 116 |
| 5.0M | ~90 min | 0.005 | 190 | 0.88 | 1.39 | 117 |
| 8.0M | ~145 min | 0.005 | 180 | 0.89 | 0.86 | 114 |

The loss fell from 2.7 to 0.86 while play got worse. This means the network
was fitting its targets well, but the targets themselves were inflated (see
below).

## Offline Q calibration

**Method** (`scripts/dqn_q_bias.py`):

- For each checkpoint, play 12 greedy games in the garbage setting (seeds
  700000 to 700011, outside every training, selection and held-out range), on
  the CPU.
- At each move, record the Q of the chosen placement, `r + 0.99 * V(afterstate)`.
- Compare it with the discounted return (gamma 0.99) that the game actually
  delivered from that move on.
- Steps within 300 pieces of the 500-piece cap are left out of games that
  reached the cap, because their return is cut short. Steps from games that
  topped out are all used.
- "Bias, last 20" is the mean of Q minus the actual return over the 20 moves
  before each top-out.
- "Gap" is the median Q difference between the best and second-best placement.

| Run | Checkpoint | Pieces | Top-out | Mean Q | Actual return | Bias | Bias, last 20 pieces before top-out | Best vs 2nd-best gap |
|---|---|---|---|---|---|---|---|---|
| v1 | 25 min | 499 | 0.08 | 64 | 130 | -65 | +1 | 0.71 |
| v1 | 65 min | 284 | 0.75 | 127 | 92 | +34 | +63 | 0.30 |
| v1 | 135 min | 225 | 0.83 | 128 | 86 | +42 | +66 | 0.19 |

- **25 min:** the checkpoint underestimated its values but ranked placements
  well.
- **65 and 135 min:** these checkpoints value boards 20 pieces before a top-out
  at about 65 above what those boards actually deliver, so the network cannot
  tell when it is in danger.
- **Ranking signal:** the gap between the best and second-best placement shrank
  from 0.71 to 0.19 while Q doubled, so placement choice became mostly noise.

## Suspects checklist

| Suspect | v1 setting | Finding | Cause? |
|---|---|---|---|
| Target update interval | Hard sync every 500 updates (about every 17 s, or 32k transitions) | The target follows the online network quickly, so the overestimate feeds into its own targets | **Yes** |
| Learning rate | 2.5e-4, constant | No decay, so the drift continued after epsilon reached its floor at 54 min | **Yes** |
| Replay buffer age | 400k transitions, at about 920 transitions/s, which is about 7 minutes of play. Each transition is sampled about 8 times | After 54 min, the buffer held only recent, near-greedy play | **Yes** |
| Q-value magnitude | Mean V rose from 26 to 116 and stayed there. Offline Q reached up to 168. v1 did not log the maximum | The offline measurement confirms an optimistic bias of +34 to +42 overall, and +63 to +66 near a top-out | Symptom |
| Loss function | Huber (smooth L1, beta 1) | Loss is low and stable. Fitting is not the problem; the targets are | No |
| Gradient clipping | Global norm 10 | Same as above | No |
| n-step bias with epsilon (n = 5, no off-policy correction) | The chance that a 5-step target includes a random move is 1 - (1 - eps)^4: 59% at eps 0.2, 2% at eps 0.005 (from 54 min) | The bias is pessimistic, because random moves lower returns, and it was largest when the agent was best. It cannot explain an optimistic drift late in the run. Shortening n would make the target lean more on the inflated bootstrap value | No |

## Fix (v2)

Only the three settings identified as causes change:

| Setting | v1 | v2 |
|---|---|---|
| Target network | Hard sync every 500 updates | Polyak update after every gradient step, tau 0.001. Average target lag goes from about 250 to about 1000 updates |
| Learning rate | 2.5e-4, constant | 1e-4, cosine-decayed to 1e-5 over the budget |
| Replay capacity | 400k (about 7 minutes of play) | 2M (about 36 minutes of play, about 1.7 GB) |
| Logging | Loss, mean V | Also `train/v_max` and `train/lr` |

Unchanged:

- Huber loss and gradient clipping at 10.
- n = 5.
- The epsilon schedule.
- The network and the reward.
- 64 environments, 2 updates per vector step, batch 256.

v2 is a fresh 3-hour run (`checkpoints/dqn/seed0_v2`), with greedy evaluations
on the selection seeds every 10 minutes, logged like v1.

The leaderboard DQN is v2 only if its best selection fitness is strictly higher
than v1's 1.340. Otherwise v1 stays. The choice is recorded in
`checkpoints/dqn/version_choice.json`.

## v1 versus v2

Filled in after the v2 run.
