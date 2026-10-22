# Decision Transformer v2: target-return calibration

The model chooses actions by afterstate matching (each placement's logit is the match between the transformer's output and an embedding of that placement's afterstate) rather than with an 80-way action classifier. Raw-action PPO already shows the raw action space failing here: held-out fitness 0.103, against 1.280 to 1.345 for the seven top agents, all of which score afterstates.

The selected model played 30 selection seeds (garbage setting) at each target return. Returns are in fitness units: a game's return equals its fitness when it reaches the 500-piece cap.

| Target | Achieved fitness | Top-out rate | Mean pieces | Score/piece |
|---|---|---|---|---|
| 0.20 | 1.350 | 0.10 | 480 | 1559 |
| 0.40 | 1.284 | 0.20 | 458 | 1476 |
| 0.60 | 1.263 | 0.23 | 450 | 1453 |
| 0.80 | 1.240 | 0.37 | 441 | 1432 |
| 1.00 | 1.284 | 0.30 | 456 | 1489 |
| 1.20 | 1.304 | 0.20 | 465 | 1498 |
| 1.40 | 1.346 | 0.17 | 479 | 1552 |

Rank correlation between target and achieved fitness: +0.00 (+1 means a higher target always played better).

Default target (highest achieved fitness, used on the leaderboard): 0.20.

![calibration](dt_calibration_v2.png)
