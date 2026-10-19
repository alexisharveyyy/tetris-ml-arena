# Leaderboard

Held-out seeds 900000..900499 (500 games per agent and setting). Attack/min assumes 2 pieces per second.

Evaluation garbage is inserted directly after every scheduled piece. It does not go through the pending meter and cannot be cancelled by the agent's attacks, so the garbage setting measures survival under a fixed load. Placements are instant (no gravity or speed limits).

## Primary: garbage (1 garbage row every 3 pieces, 500-piece cap)

| Rank | Agent | Mean lines | Mean score | Pieces survived | Attack/min | Lines/piece | Score/piece | Top-out rate | Fitness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Cross-entropy method | 321.2 | 737,539 | 468.7 | 40.8 | 0.681 | 1522 | 0.15 | 1.318 |
| 2 | GA heuristic | 313.1 | 720,348 | 457.9 | 39.9 | 0.676 | 1495 | 0.18 | 1.289 |
| 3 | Dellacherie (baseline) | 289.7 | 633,186 | 427.5 | 35.2 | 0.665 | 1362 | 0.31 | 1.196 |

Paired bootstrap on the same seeds (10,000 resamples), mean difference and 95% interval. Intervals that exclude 0 are marked *.

| Comparison | Fitness difference | Top-out rate difference |
|---|---|---|
| Cross-entropy method minus GA heuristic | +0.028 [-0.004, +0.061] | -0.032 [-0.078, +0.014] |
| Cross-entropy method minus Dellacherie (baseline) | +0.122 [+0.085, +0.159]* | -0.160 [-0.210, -0.110]* |
| GA heuristic minus Dellacherie (baseline) | +0.094 [+0.055, +0.133]* | -0.128 [-0.180, -0.076]* |

## Secondary: plain (no garbage, 1000-piece cap)

| Rank | Agent | Mean lines | Mean score | Pieces survived | Attack/min | Lines/piece | Score/piece | Top-out rate | Fitness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Cross-entropy method | 398.4 | 819,050 | 1000.0 | 10.8 | 0.398 | 819 | 0.00 | 1.205 |
| 2 | GA heuristic | 398.6 | 781,992 | 1000.0 | 6.4 | 0.399 | 782 | 0.00 | 1.195 |
| 3 | Dellacherie (baseline) | 398.8 | 755,690 | 1000.0 | 4.4 | 0.399 | 756 | 0.00 | 1.189 |
