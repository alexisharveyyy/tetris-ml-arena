# PPO

Two PPO agents share one trainer (`train.py`). They differ only in how the
policy produces its logits. Both output a distribution over the same masked
80-action placement space (rotation x column x hold), so the shared
action-space rule holds for both.

| Agent | Key | Config | Budget | Role |
|---|---|---|---|---|
| PPO | `ppo` | `config.yaml` | 3 hours | The playable PPO opponent |
| PPO (raw actions) | `ppo_raw` | `config_raw.yaml` | 1 hour | The originally specified design, kept for comparison |

## Design choice: afterstate-scored logits

The originally specified design maps the observation (board, pieces) straight
to 80 logits through a CNN and MLP encoder. In a 15-minute sanity run it
stayed at about 26 pieces per game in the garbage setting (a random policy
gets about 20) while its entropy collapsed from 4.0 to 0.65. To be useful,
the policy has to learn from scratch which column and rotation each output
means for the current board, which is the hard part of Tetris.

The playable `ppo` agent therefore computes **each action's logit by scoring
that action's afterstate**:

- the board after the placement;
- the 13 features;
- the piece that spawns next and the hold slot after the placement;
- the level.

One shared network does the scoring, the same shape as the DQN value net.
Illegal actions still get a logit of -1e9. The policy is a softmax over the
80 actions, trained with the standard PPO clipped objective. The critic is a
separate network over the ordinary observation.

This is the same afterstate representation the DQN uses, so PPO and DQN
differ in the learning algorithm (on-policy policy gradient versus off-policy
value learning), not in what they can see. The raw-action variant stays on
the leaderboard, with its learning curve, to show the effect of this choice.
This change was approved before the 3-hour run.

## Shared details

**Observation (critic, and the raw policy):**

- the 24x10 board;
- current piece, hold, can-hold flag and the next 5 pieces (one-hot);
- the 10 board features (scaled);
- the level.

**Masking:** illegal actions get a logit of -1e9 (not `-inf`). Entropy is
computed over legal actions only (`masked_log_probs` in `tetris/rl/nets.py`),
so neither the loss nor its gradient can become NaN, even with a single legal
action (tested for both policies).

**Reward:** the same as DQN, `1 + 0.25 * score_delta / 1000` per piece (the
per-step form of the leaderboard fitness), scaled by 0.1 for the value head.

**Truncation:** a top-out ends the return. At the piece cap, the step's reward
is augmented with `gamma * V(final observation)` and the advantage chain is
cut, so unfinished games are bootstrapped rather than treated as terminal
(tested).

**Training:**

- The garbage evaluation setting.
- 64 games in 8 worker processes, 128-step rollouts (8,192 transitions per
  update).
- 4 epochs, minibatch 2,048.
- Clip 0.2, value clip 10.
- Learning rate decayed linearly; gamma 0.99, GAE lambda 0.95.
- Training on MPS.

**Play:** the most probable legal action (deterministic). Difficulty mistakes
pick among the next five legal actions by probability.

## Device choice

`scripts/bench_device.py` on an Apple M2 Max (raw policy shapes):

| Workload | CPU (4 threads) | MPS |
|---|---|---|
| Inference, 64 observations | 17 ms | 6 ms |
| Update, minibatch 1,024 | 218 ms | 17 ms |

Training uses MPS. Live play uses the CPU.

## Train

```bash
python -m tetris.agents.ppo.train                                  # afterstate PPO, 3 hours
python -m tetris.agents.ppo.train --config tetris/agents/ppo/config_raw.yaml   # raw PPO, 1 hour
python -m tetris.agents.ppo.train --budget-minutes 60 --seed 1 --run-name seed1 --no-promote
tensorboard --logdir runs/ppo
```

## Results

See the end of this file (filled in after training).
