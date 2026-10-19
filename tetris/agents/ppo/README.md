# PPO

## Method

An actor-critic trained with PPO over the masked 80-action placement space
(rotation x column x hold).

**Observation:**

- the 24x10 board;
- the current piece, hold slot, can-hold flag and the next 5 pieces (one-hot);
- the 10 board features (scaled);
- the level.

**Network:**

- A CNN board encoder (same design as DQN, 128 units) and a 128-unit MLP over
  the extras, concatenated into a shared 256-unit layer.
- Heads: 80 policy logits and a scalar value.

**Masking:** illegal actions get a logit of -1e9 (not `-inf`). Entropy is
computed over legal actions only (`masked_log_probs` in `tetris/rl/nets.py`),
so neither the loss nor its gradient can become NaN, even with a single legal
action (tested).

**Reward:** the same as DQN, `1 + 0.25 * score_delta / 1000` per piece, scaled
by 0.1 for the value head.

**Truncation:** a top-out ends the return. At the piece cap, the step's reward
is augmented with `gamma * V(final observation)` and the advantage chain is
cut, so unfinished games are bootstrapped rather than treated as terminal
(tested).

**Hyperparameters:**

- 64 games in 8 worker processes, 128-step rollouts (8,192 transitions per
  update).
- 4 epochs, minibatch 2,048.
- Clip 0.2, value clip 10, entropy 0.01.
- Learning rate 3e-4, decayed linearly.
- gamma 0.99, GAE lambda 0.95.
- Training on MPS.

**Play:** the most probable legal action (deterministic). Difficulty mistakes
pick among the next five legal actions by probability.

## Device choice

`scripts/bench_device.py` on an Apple M2 Max:

| Workload | CPU (4 threads) | MPS |
|---|---|---|
| Inference, 64 observations | 17 ms | 6 ms |
| Update, minibatch 1,024 | 218 ms | 17 ms |

Training uses MPS. Live play uses the CPU, at about 0.2 ms per decision.

## Train

```bash
python -m tetris.agents.ppo.train
python -m tetris.agents.ppo.train --budget-minutes 60 --seed 1 --run-name seed1 --no-promote
tensorboard --logdir runs/ppo
```

## Results

See the end of this file (filled in after training).
