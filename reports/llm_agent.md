# LLM agent (novelty opponent)

Garbage setting (1 row every 3 pieces), 100-piece cap, the first 20 held-out seeds. The GA heuristic played the same seeds with the same cap as the reference; it reached the cap in 19 of 20 games.

**Main metrics: pieces survived, top-out rate and lines per piece.** Leaderboard fitness is not used here: with a 100-piece cap the GA reaches the cap in almost every game, so fitness would mostly measure score per piece. Pieces survived and top-out rate measure whether the model keeps the stack alive under garbage; lines per piece measures how efficiently it clears.

**20 games per run gives wide intervals.** The 95% intervals below (bootstrap over games) are often wider than the differences between runs; treat small differences as ties. The LLM agent is not on the main leaderboard.

Every decision falls back to the GA heuristic's move on a timeout, an unparseable or unlisted answer, or a backend error, so a run with many fallbacks partly measures the GA. The API model is gpt-4.1-mini, a small API model, not a frontier model. The local models are 8-billion-parameter models quantized to 4 bits. The timeout column is the per-decision limit the run used; live play in the app uses 3 s.

| Run | Games | Timeout | Pieces survived [95% interval] | Top-out rate [95% interval] | Lines per piece [95% interval] | Fallback rate | Timeouts / invalid / errors | Seconds per piece | Tokens per game | Cost |
|---|---|---|---|---|---|---|---|---|---|---|
| GA heuristic (reference) | 20 | | 99.3 [97.9, 100.0] | 0.05 [0.00, 0.15] | 0.637 [0.614, 0.656] | | | | | |
| Ollama qwen3:8b, features | 20 | 15 s | 26.8 [25.1, 28.7] | 1.00 [1.00, 1.00] | 0.143 [0.108, 0.180] | 0.0% | 0 / 0 / 0 | 3.48 | 39,029 | $0.00 |
| Ollama qwen3:8b, bare | 20 | 3 s | 13.2 [12.3, 13.9] | 1.00 [1.00, 1.00] | 0.044 [0.023, 0.069] | 22.4% | 59 / 0 / 0 | 2.39 | 10,103 | $0.00 |
| Ollama llama3.1:8b, features | 20 | 15 s | 23.9 [22.1, 25.8] | 1.00 [1.00, 1.00] | 0.092 [0.062, 0.122] | 0.0% | 0 / 0 / 0 | 3.54 | 34,676 | $0.00 |
| gpt-4.1-mini (small API model, not a frontier model), features | 0 | | Pending (no valid API key) | | | | | | | |
| gpt-4.1-mini (small API model, not a frontier model), bare | 0 | | Pending (no valid API key) | | | | | | | |

Paired differences on the same seeds (mean, 95% bootstrap interval over games):

| Comparison | Games | Pieces survived | Top-out rate | Lines per piece |
|---|---|---|---|---|
| Ollama qwen3:8b, features minus GA | 20 | -72.5 [-74.9, -69.2] | +0.95 [+0.85, +1.00] | -0.494 [-0.535, -0.447] |
| Ollama qwen3:8b, bare minus GA | 20 | -86.2 [-87.4, -84.7] | +0.95 [+0.85, +1.00] | -0.593 [-0.621, -0.563] |
| Ollama llama3.1:8b, features minus GA | 20 | -75.3 [-77.3, -73.5] | +0.95 [+0.85, +1.00] | -0.545 [-0.581, -0.511] |
| Ollama qwen3:8b, features minus Ollama llama3.1:8b, features | 20 | +2.9 [+0.2, +5.8] | +0.00 [+0.00, +0.00] | +0.051 [+0.012, +0.094] |
| Ollama qwen3:8b, features minus Ollama qwen3:8b, bare (timeouts 15 s vs 3 s) | 20 | +13.7 [+11.7, +15.8] | +0.00 [+0.00, +0.00] | +0.099 [+0.050, +0.149] |

Comparisons marked with two timeouts pair runs made with different per-decision limits; part of the difference can come from the timeout (a shorter limit means more fallbacks to the GA).

API spending to date: $5.00 of the $5.00 cap.
