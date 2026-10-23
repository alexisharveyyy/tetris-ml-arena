# DQN over afterstates

## Method

Double DQN with prioritized replay, learning a value for **afterstates**: the
board right after a placement, before the next piece appears. At each
decision the agent enumerates every legal placement in the 80-action space
(including hold) and plays

```
argmax_a  r(s, a) + gamma * V(after(s, a))        (V = 0 if the placement tops out)
```

**Input per afterstate:**

- the 24x10 board after the placement and any line clears;
- the 13 hand-crafted features (scaled to about [0, 1]);
- the piece that spawns next and the hold slot as they will be after the
  placement (one-hot), so a hold placement is represented correctly;
- the level after the placement.

**Network:** a small CNN over the board (3x3 convolutions, 16/32/32 channels,
two with stride 2, then 128 units). An MLP over the extras (64 units) is
concatenated with it, followed by a 128-unit head that outputs V.

**Reward** (shared with PPO, `tetris/rl/encoding.py`): `1 + 0.25 * score_delta / 1000`
per placed piece. This is the per-step form of the leaderboard fitness:
survival counts 1 per piece and the score term matches the fitness's
efficiency term. There is no extra top-out penalty; with gamma = 0.99,
topping out forfeits about 100 of future reward.

**Targets:** n-step (n = 5) Double DQN. Each stored transition keeps the chosen
afterstate, the discounted rewards of the next 4 actions actually taken, and
the top 8 candidate afterstates of the bootstrap state (ranked by the online
network when collected, bit-packed boards):

```
y = R + gamma^k * [ r(c*) + gamma * V_target(c*) ],   c* = argmax_c r(c) + gamma * V_online(c)
```

- A top-out ends the return (no bootstrap).
- Reaching the piece cap is **truncation**: the target bootstraps from the
  final state with the discount reached so far, and is never treated as
  terminal (tested in `tests/rl/test_rl.py`).

The top-8 pre-filter is an approximation of the full max. The online network
ranks candidates at collection time, which can be slightly stale by the time
the transition is replayed.

**Training setting:** the garbage evaluation setting (one direct garbage row
every 3 pieces, 500-piece cap), the same one used for the leaderboard.

**Infrastructure:**

- 64 games in 8 worker processes.
- Learning on MPS, overlapped with environment stepping.
- Proportional prioritized replay (alpha 0.6, beta 0.4 to 1.0), 2M
  transitions (about 36 minutes of play).
- Epsilon decays from 0.2 to 0.005 over the first 30% of the budget.
- Adam, learning rate cosine-decayed from 1e-4 to 1e-5 over the budget.
- Soft target update: `target += 0.001 * (online - target)` after every
  gradient step.
- Huber loss, gradient-norm clipping at 10.

## Stability fix (v2)

The first 3-hour run (v1, `seed0`) peaked at 25 minutes (selection fitness
1.340) and then declined to 0.49-0.77. Its value function became
overestimated: offline, boards 20 pieces before a top-out were valued about 65
above the discounted return they actually delivered, and the gap between the
best and second-best placement shrank to about 0.2. v1 used a hard target sync
every 500 updates (about 17 s), a constant learning rate of 2.5e-4 and a 400k
replay buffer (about 7 minutes of play). v2 changes only those three settings,
as listed above. The diagnosis and a v1/v2 comparison are in
`reports/training/dqn_overestimation.md`.

The leaderboard DQN is whichever run has the higher selection fitness (v1 if
v2 does not beat it); `checkpoints/dqn/version_choice.json` records the choice.

## Device choice

`scripts/train/bench_device.py` on an Apple M2 Max:

| Workload | CPU (4 threads) | MPS |
|---|---|---|
| Inference, 3,200 afterstates | 111 ms | 5 ms |
| Update, batch 256 x 8 candidates | 324 ms | 10 ms |

Training uses MPS. Live play uses the CPU: one decision (about 50
afterstates) takes about 1 ms on one thread. The CPU numbers above already
include a workaround for this PyTorch build's slow CPU convolutions
(`Conv3x3` in `tetris/rl/nets.py`, about 5x faster).

## Train

```bash
python -m tetris.agents.dqn.train                      # 3-hour budget, seed 0, promotes to best.pt
python -m tetris.agents.dqn.train --budget-minutes 60 --seed 1 --run-name seed1 --no-promote
python -m tetris.agents.dqn.train --resume             # replay buffer refills after resume
tensorboard --logdir runs/dqn
```

Every 10 minutes a checkpoint is evaluated on the 30 selection seeds in
background processes. The best one by selection fitness becomes
`checkpoints/dqn/best.pt`. Held-out results come from `scripts/eval/evaluate.py`.

## Results

See the end of this file (filled in after training).
