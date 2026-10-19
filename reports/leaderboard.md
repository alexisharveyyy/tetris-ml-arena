# Leaderboard

Held-out seeds 900000..900099 (100 games per agent and setting). Attack/min assumes 2 pieces per second.

## Primary: garbage (1 garbage row every 3 pieces, 500-piece cap)

| Rank | Agent | Mean lines | Mean score | Pieces survived | Attack/min | Lines/piece | Score/piece | Top-out rate | Fitness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Cross-entropy method | 325.8 | 754,612 | 474.5 | 40.9 | 0.682 | 1544 | 0.13 | 1.335 |
| 2 | GA heuristic | 307.6 | 708,463 | 450.4 | 39.7 | 0.672 | 1473 | 0.20 | 1.269 |
| 3 | Dellacherie (reference) | 301.6 | 668,786 | 443.0 | 36.1 | 0.672 | 1419 | 0.25 | 1.241 |

## Secondary: plain (no garbage, 1000-piece cap)

| Rank | Agent | Mean lines | Mean score | Pieces survived | Attack/min | Lines/piece | Score/piece | Top-out rate | Fitness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Cross-entropy method | 398.5 | 819,310 | 1000.0 | 10.8 | 0.399 | 819 | 0.00 | 1.205 |
| 2 | GA heuristic | 398.6 | 782,172 | 1000.0 | 6.4 | 0.399 | 782 | 0.00 | 1.196 |
| 3 | Dellacherie (reference) | 398.8 | 756,195 | 1000.0 | 4.5 | 0.399 | 756 | 0.00 | 1.189 |
