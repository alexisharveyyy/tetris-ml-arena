# Afterstate PPO: MPS gradient bug

The 3-hour afterstate PPO run (`checkpoints/ppo/seed0`) stayed at selection
fitness 0.061 to 0.064 for the whole run. That is random play, about 17
pieces per game. The three 1-hour seed runs (`short0` to `short2`) did the
same. These runs are invalid: the policy never trained. Their numbers in
`reports/leaderboard.md` and `reports/seed_spread.md` describe an untrained
network.

## Checks

| Check | Finding |
|---|---|
| PPO ratios at the first update epoch | Exactly 1. On a recorded batch, log-probabilities recomputed in training mode match the rollout ones (max deviation 0, CPU and MPS). Action indices and afterstate rows line up. |
| Is the policy changing? | No. Approximate KL was about 1e-5 (noise), clip fraction 0 and entropy 4.06 to 4.07 for 3 hours. The uniform distribution over the legal actions has entropy about 4.2. After 3 hours, the scorer's weights were essentially at their initial values. The first convolution's weights had a relative change of exactly 0, and the head's under 1e-3. Two convolution biases, however, had moved by 3x and 37x. |
| Scorer gradients | Correct on the CPU. On MPS, at the size of one training minibatch (2,048 states, about 115k legal afterstates scored in one call), the scorer gradient had a relative error of 6e8 against the CPU. Up to about 57k afterstates (1,024 states) MPS matched the CPU (1e-5). The training logs also show Metal command-buffer errors ("Impacting Interactivity"). |
| Critic explained variance | -0.01 to -0.05, measured offline on CPU rollouts of the 5-minute and selected checkpoints. Mean predicted value 0.1 against a mean return of 0.7. The critic barely trained either. Gradient clipping is global (norm 0.5), so the corrupted scorer gradients shrank the critic's updates to almost nothing. |
| Advantage sign and normalization | Correct: GAE with truncation bootstrapping (tested), normalized per minibatch, standard clipped objective. The mean advantage was +0.6 only because the critic stayed near 0. |

**Why the other learned agents were not affected:** raw PPO (2,048 boards per
minibatch) and DQN (about 2,300 boards per update) stay below the size where
MPS goes wrong.

## Fix

`AfterstateActorCritic.forward` now scores the legal afterstates in chunks of
`SCORER_CHUNK` = 4,096 (`tetris/rl/nets.py`). Chunked and single-pass scoring
give the same logits and gradients.

Tests (`tests/rl/test_rl.py`):

- `test_afterstate_policy_chunked_scoring_matches_single_pass` (CPU).
- `test_afterstate_policy_mps_gradients_match_cpu_at_training_scale`: compares
  MPS and CPU scorer gradients for a full 2,048-state minibatch. It passes with
  the fix (relative difference 6e-5) and fails without it (6e8).

PPO training now also logs:

- the first-minibatch ratio deviation;
- the gradient norm of the policy parameters;
- explained variance;
- the advantage mean and standard deviation;
- the clip fraction.

## Check run after the fix (12 minutes, MPS)

| Minutes | Entropy | KL | Clip fraction | Explained variance | Training pieces |
|---|---|---|---|---|---|
| 1.9 | 4.07 | 0.0002 | 0.000 | 0.89 | 16.7 |
| 3.8 | 3.80 | 0.0109 | 0.119 | 0.95 | 20.7 |
| 5.6 | 3.31 | 0.0092 | 0.104 | 0.97 | 25.3 |
| 7.5 | 2.74 | 0.0042 | 0.059 | 0.96 | 29.8 |
| 9.4 | 2.27 | 0.0040 | 0.041 | 0.95 | 33.9 |
| 11.3 | 1.98 | 0.0010 | 0.004 | 0.94 | 38.0 |

Selection fitness rose from 0.077 at 5 minutes to 0.146 at 12 minutes.
For comparison, raw PPO's 1-hour run scored 0.103 on the held-out seeds.
Throughput is about 360 environment steps per second; the broken runs logged
about 480.
