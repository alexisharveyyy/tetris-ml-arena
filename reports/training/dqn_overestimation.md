# DQN v1: decline after 25 minutes and value overestimation

The first 3-hour DQN run (v1, `checkpoints/dqn/seed0`) peaked early and then
declined. Its greedy evaluations on the 30 selection seeds, in the garbage
setting:

- 25 min: fitness **1.340**, top-out 0.17 (the peak).
- 35 min: 1.336.
- 105 to 145 min: 0.49 to 0.77, top-out 0.70 to 0.87.

The full curve is in `checkpoints/dqn/seed0/selection_curve.csv` and
`reports/training/curves/curve_dqn_seed0.png`. This report gives the diagnosis and the fix
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

**Method** (`scripts/analysis/dqn_q_bias.py`):

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

### Selection curves (30 selection seeds, garbage setting)

| Minutes | 5 | 15 | 25 | 35 | 45 | 55 | 65 | 75 | 85 | 95 | 105 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 fitness | 0.17 | 1.28 | **1.34** | 1.34 | 1.07 | 1.09 | 0.75 | 0.91 | 0.83 | 0.97 | 0.62 |
| v2 fitness | 0.15 | 0.82 | 1.25 | 1.19 | 1.20 | **1.25** | 1.13 | 1.16 | 1.03 | 0.64 | 0.86 |

| Minutes | 115 | 125 | 135 | 145 | 155 | 165 | 175 | 180 |
|---|---|---|---|---|---|---|---|---|
| v1 fitness | 0.64 | 0.77 | 0.49 | 0.75 | 0.37 | 0.57 | 0.98 | 0.44 |
| v2 fitness | 0.93 | 0.69 | 0.84 | 1.10 | 0.76 | 0.72 | 0.74 | 0.94 |

v2 held a plateau of about 1.13 to 1.25 until 75 minutes, while v1 had
dropped to 0.75 by 65 minutes. v2 then declined from 85 minutes and spent the
rest of the run between 0.64 and 1.10.

**Result: v2 did not beat v1.** v2's best selection fitness is 1.253 (55
minutes), below v1's 1.340 (25 minutes). v1's 25-minute checkpoint stays the
leaderboard DQN (`checkpoints/dqn/version_choice.json`). v2's later evaluations
were higher on average than v1's over the same period, but that does not
matter for the choice, because only the best checkpoint of each run is used.

### Training scalars (v2)

| Steps | Time | Training pieces | Training top-out | Loss | Mean V | Max V (batch) | LR |
|---|---|---|---|---|---|---|---|
| 1.0M | ~17 min | 59 | 1.00 | 1.04 | 26 | 58 | 9.7e-5 |
| 2.0M | ~37 min | 140 | 0.99 | 1.69 | 41 | 127 | 8.8e-5 |
| 3.0M | ~60 min | 374 | 0.57 | 2.23 | 88 | 165 | 7.4e-5 |
| 4.0M | ~83 min | 268 | 0.76 | 1.35 | 124 | 171 | 5.6e-5 |
| 5.0M | ~105 min | 218 | 0.81 | 1.06 | 126 | 173 | 4.1e-5 |

Mean V rose more slowly than in v1, but ended at the same level, about 125
(v1: 111 to 124). The training games peaked at 3.0M steps and then declined,
as in v1.

### Offline Q calibration (same method and seeds as above)

| Run | Checkpoint | Pieces | Top-out | Mean Q | Actual return | Bias | Bias, last 20 pieces before top-out | Best vs 2nd-best gap |
|---|---|---|---|---|---|---|---|---|
| v1 | 25 min | 499 | 0.08 | 64 | 130 | -65 | +1 | 0.71 |
| v1 | 65 min | 284 | 0.75 | 127 | 92 | +34 | +63 | 0.30 |
| v1 | 135 min | 225 | 0.83 | 128 | 86 | +42 | +66 | 0.19 |
| v2 | 25 min | 430 | 0.50 | 49 | 112 | -63 | +11 | 0.63 |
| v2 | 55 min (peak) | 434 | 0.25 | 94 | 120 | -26 | +10 | 0.97 |
| v2 | 75 min | 450 | 0.25 | 119 | 121 | -2 | +36 | 0.67 |
| v2 | 95 min | 228 | 1.00 | 128 | 86 | +42 | +53 | 0.45 |
| v2 | 145 min (late high, 1.10) | 407 | 0.25 | 128 | 122 | +6 | +40 | 0.20 |
| v2 | 180 min (final) | 376 | 0.33 | 130 | 117 | +13 | +60 | 0.18 |

v2 reproduces v1's pattern, about 50 minutes later:

- **Good checkpoints underestimate.** The best checkpoints of both runs
  underestimate their overall value and are roughly calibrated near a top-out.
- **Mean Q climbs regardless of play.** Mean Q rises steadily to about 125-128
  whether or not play improves.
- **Decline starts when Q passes the actual return.** It begins once mean Q
  reaches the actual return. That point is 75 minutes in v2, with bias -2 and
  near-top-out bias +36.
- **Same end state as v1.** By 95 minutes v2 has the same overall bias as v1
  at 135 minutes (+42), with +53 near a top-out.

The fix slowed the drift but did not remove it.

### Replay composition hypothesis

The hypothesis is that the decline starts when the replay buffer holds only
data from after epsilon reached its floor:

- **Floor.** Epsilon reaches its floor of 0.005 at 54 minutes (about step
  2.75M). It is already below 0.04 from about 45 minutes.
- **Buffer age.** v2 ran at about 730 environment steps per second, slower than
  the 920 assumed when sizing the buffer. The 2M buffer therefore holds about
  45 minutes of play, not 36.
- **Post-floor share.** The share of the buffer collected after the floor was
  about 50% at 75 minutes and about 70% at 85 minutes. It was about 90% at
  95 minutes, and 100% from about 100 minutes (step 4.75M).

The decline (85 minutes) starts while the buffer is still about 30%
pre-floor, and it gets steep (95 minutes) as the buffer approaches 100%
post-floor. The timing fits the hypothesis, but one run cannot separate it from
a second explanation that fits equally well: Q reaches the actual return at
the same time. These measurements cannot tell the two apart. Neither the
lower learning rate (still 5.6e-5 at 85 minutes) nor the slower target stopped
mean V from rising.

## Future work: clipped double Q (not run)

The v2 changes slowed the overestimation but did not remove it. Both runs
reached the same overestimated values (mean Q about 125 to 130 against actual
returns of 86 to 122), and play declined once Q passed the actual return. The
leaderboard DQN is v1's 25-minute checkpoint: checkpoint selection keeps the
peak, and that checkpoint is statistically tied with CEM.

**Proposal:**

- **Clipped double Q.** Train two value networks with separate target networks
  and bootstrap from the smaller of the two target values, as in TD3, so that
  overestimation in one network is not copied into the targets.
- **Higher epsilon floor.** Raise it from 0.005 to 0.02, so the replay buffer
  keeps boards close to a top-out after the decay ends. This addresses the
  replay composition explanation above.
- **Logging.** Log the offline Q bias (`scripts/analysis/dqn_q_bias.py`) at every
  selection evaluation, to see whether the bias stays negative or near zero.

**Expected cost:**

- **Time:** one 3-hour MPS run, plus about 2 minutes of CPU for the Q-bias
  checks.
- **Updates:** each update evaluates two networks, about 30% slower. The run
  would complete about 25% fewer gradient updates in the same 3 hours.
- **Payoff:** expected to be modest. The goal is a run that holds its peak
  rather than a higher peak. If it held, DQN could train longer than 25
  minutes usefully.
