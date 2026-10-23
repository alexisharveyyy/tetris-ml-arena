# Afterstate PPO: MPS gradient bug

The 3-hour afterstate PPO run (`checkpoints/ppo/seed0`) stayed at selection
fitness 0.061 to 0.064 for the whole run. That is random play, about 17
pieces per game. The three 1-hour seed runs (`short0` to `short2`) did the
same. These runs are invalid: the policy never trained. When this report
was written, their numbers in `reports/leaderboard/leaderboard.md` and
`reports/training/seed_spread.md` described an untrained network. Both were later
replaced by reruns at the fixed code (9c4936e): PPO now ranks 2nd on the
leaderboard, and the seed spread shows the rerun's three 1-hour runs.

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

## Safeguards

These were added before the rerun (`tetris/rl/safeguards.py`).

**Pre-flight check.** Every DQN and PPO training run on a non-CPU device
starts with a pre-flight check:

- The run's network is built twice from the same weights.
- The run's loss is computed on one batch of the run's real size, once on the
  device and once on the CPU.
- The two full gradient vectors are compared.
- If the relative difference exceeds 1e-2, the check prints a loud warning.
  It then tries smaller scorer chunks (2,048, 1,024, 512) for the afterstate
  policy, and finally falls back to the CPU.
- The result is saved as `preflight.json` in the run directory.

`scripts/preflight.py` reports the check for all three trainers.

Results on MPS, at real batch sizes:

| Trainer | Batch | Relative gradient difference |
|---|---|---|
| DQN | 256 transitions x 8 candidates | 1.2e-7 |
| PPO (raw actions) | 2,048 states | 3.2e-7 |
| PPO (afterstate), chunk 4,096 | 2,048 states (about 115k afterstates) | 8.1e-7 to 1.0e-6 |
| PPO (afterstate), chunking disabled (the broken run) | same | 1.5e7, fails |

**Watchdog.** A run aborts with `StalledTrainingError` when it stops making
progress:

- **PPO.** Per-update KL is measured with the `(r - 1) - log r` estimator,
  which is always non-negative. The run aborts if KL stays below 1e-6 for 30
  consecutive updates. Because KL per update shrinks roughly with the square
  of the step size, the threshold scales with `(lr / lr0)^2`.
  - Calibrated on MPS: the reintroduced bug settles at about 1.2e-7; a healthy
    run starts at 3e-6 and rises to about 2e-4.
  - With the bug reintroduced and the pre-flight off, the watchdog aborted the
    run after its patience (5 updates in the demonstration).
- **DQN.** The run aborts if the relative change of the online network's
  parameters between log lines (every 30 s) stays below 1e-6 for 10
  consecutive checks.
