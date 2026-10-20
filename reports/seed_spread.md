# Seed variance (1-hour runs)

Each run's checkpoint chosen on the 30 selection seeds, evaluated on 500 held-out seeds in the garbage setting (1 direct garbage row every 3 pieces, 500-piece cap).

The short runs have a 1-hour budget, not the main runs' 3 hours. Every schedule is defined as a fraction of the budget, so it is scaled to 1 hour: epsilon decays over the first 30% (18 minutes), the DQN learning rate follows the same cosine decay (1e-4 to 1e-5) over the hour, and the PPO learning rate decays linearly over the hour (to 5% of its start value). DQN short runs use the v2 config (soft target update, LR schedule, 2M replay; see reports/dqn_overestimation.md).

| Agent | Run | Selected at | Selection fitness | Held-out fitness | Top-out rate | Pieces |
|---|---|---|---|---|---|---|
| DQN (afterstates) | short0 | 15 min | 1.177 | 1.097 | 0.46 | 386 |
| DQN (afterstates) | short1 | 15 min | 1.179 | 1.204 | 0.32 | 423 |
| DQN (afterstates) | short2 | 15 min | 1.241 | 1.217 | 0.34 | 428 |
| DQN (afterstates) | mean +/- std | | | 1.173 +/- 0.066 | | |
| PPO | short0 | 25 min | 0.062 | 0.061 | 1.00 | 27 |
| PPO | short1 | 15 min | 0.033 | 0.031 | 1.00 | 13 |
| PPO | short2 | 15 min | 0.048 | 0.047 | 1.00 | 21 |
| PPO | mean +/- std | | | 0.047 +/- 0.015 | | |
