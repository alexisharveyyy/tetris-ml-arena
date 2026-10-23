# Difficulty presets

Real-time headless matches, 30 games per cell, seeds 950000+. Garbage evaluation: 1 direct (non-cancellable) garbage row every 3 pieces, 500-piece cap, normal gravity curve.

| Preset | pps cap | Move rate | Reaction | Noise | Epsilon |
|---|---|---|---|---|---|
| Easy | 0.5 | 5/s | 900 ms | 0.4 | 0.03 |
| Medium | 1 | 10/s | 450 ms | 0.15 | 0 |
| Hard | 2 | 25/s | 200 ms | 0.03 | 0 |
| Unlimited | none | instant | 0 ms | 0 | 0 |

Top-out rate, pieces, lines and measured pieces/s come from the garbage evaluation. The last column is lines per minute in a separate 2-minute race with no garbage, for comparison with human play.

| Preset | Agent | Top-out rate | Pieces | Lines | Measured pps | Race lines/min |
|---|---|---|---|---|---|---|
| Easy | Dellacherie (baseline) | 1.00 | 57 | 23 | 0.50 | 10.0 |
| Easy | GA heuristic | 1.00 | 61 | 26 | 0.50 | 9.5 |
| Easy | Cross-entropy method | 1.00 | 59 | 25 | 0.50 | 9.9 |
| Medium | Dellacherie (baseline) | 1.00 | 154 | 89 | 0.98 | 22.6 |
| Medium | GA heuristic | 1.00 | 164 | 96 | 0.98 | 22.6 |
| Medium | Cross-entropy method | 1.00 | 172 | 102 | 0.98 | 22.4 |
| Hard | Dellacherie (baseline) | 0.90 | 279 | 177 | 1.94 | 46.0 |
| Hard | GA heuristic | 0.93 | 293 | 187 | 1.94 | 45.9 |
| Hard | Cross-entropy method | 0.87 | 326 | 211 | 1.95 | 45.8 |
| Unlimited | Dellacherie (baseline) | 0.77 | 365 | 239 | 119.91 | 2879.7 |
| Unlimited | GA heuristic | 0.47 | 420 | 281 | 119.87 | 2879.6 |
| Unlimited | Cross-entropy method | 0.57 | 428 | 286 | 119.95 | 2879.6 |
