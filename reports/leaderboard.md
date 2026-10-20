# Leaderboard

Held-out seeds 900000..900499 (500 games per agent and setting). Attack/min assumes 2 pieces per second.

Evaluation garbage is inserted directly after every scheduled piece. It does not go through the pending meter and cannot be cancelled by the agent's attacks, so the garbage setting measures survival under a fixed load. Placements are instant (no gravity or speed limits).

## Primary: garbage (1 garbage row every 3 pieces, 500-piece cap)

| Rank | Agent | Mean lines | Mean score | Pieces survived | Attack/min | Lines/piece | Score/piece | Top-out rate | Fitness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | DQN (afterstates) | 325.8 | 768,213 | 475.6 | 41.2 | 0.681 | 1574 | 0.13 | 1.345 |
| 2 | Cross-entropy method | 321.2 | 737,539 | 468.7 | 40.8 | 0.681 | 1522 | 0.15 | 1.318 |
| 3 | GA heuristic | 313.1 | 720,348 | 457.9 | 39.9 | 0.676 | 1495 | 0.18 | 1.289 |
| 4 | Dellacherie (baseline) | 289.7 | 633,186 | 427.5 | 35.2 | 0.665 | 1362 | 0.31 | 1.196 |
| 5 | PPO (raw actions) | 11.9 | 2,663 | 44.4 | 9.9 | 0.266 | 59 | 1.00 | 0.103 |
| 6 | PPO | 2.8 | 861 | 27.4 | 1.3 | 0.094 | 30 | 1.00 | 0.062 |

Paired bootstrap on the same seeds (10,000 resamples), mean difference and 95% interval. Intervals that exclude 0 are marked *.

| Comparison | Fitness difference | Top-out rate difference |
|---|---|---|
| DQN (afterstates) minus Cross-entropy method | +0.027 [-0.002, +0.055] | -0.024 [-0.068, +0.020] |
| DQN (afterstates) minus GA heuristic | +0.055 [+0.024, +0.086]* | -0.056 [-0.100, -0.012]* |
| DQN (afterstates) minus Dellacherie (baseline) | +0.149 [+0.114, +0.184]* | -0.184 [-0.232, -0.136]* |
| DQN (afterstates) minus PPO (raw actions) | +1.241 [+1.222, +1.259]* | -0.874 [-0.902, -0.844]* |
| DQN (afterstates) minus PPO | +1.282 [+1.263, +1.300]* | -0.874 [-0.902, -0.844]* |
| Cross-entropy method minus GA heuristic | +0.028 [-0.004, +0.061] | -0.032 [-0.078, +0.014] |
| Cross-entropy method minus Dellacherie (baseline) | +0.122 [+0.085, +0.159]* | -0.160 [-0.210, -0.110]* |
| Cross-entropy method minus PPO (raw actions) | +1.215 [+1.192, +1.235]* | -0.850 [-0.882, -0.818]* |
| Cross-entropy method minus PPO | +1.256 [+1.234, +1.276]* | -0.850 [-0.882, -0.818]* |
| GA heuristic minus Dellacherie (baseline) | +0.094 [+0.055, +0.133]* | -0.128 [-0.180, -0.076]* |
| GA heuristic minus PPO (raw actions) | +1.186 [+1.160, +1.211]* | -0.818 [-0.852, -0.784]* |
| GA heuristic minus PPO | +1.227 [+1.201, +1.252]* | -0.818 [-0.852, -0.784]* |
| Dellacherie (baseline) minus PPO (raw actions) | +1.092 [+1.061, +1.123]* | -0.690 [-0.730, -0.650]* |
| Dellacherie (baseline) minus PPO | +1.133 [+1.102, +1.164]* | -0.690 [-0.730, -0.650]* |
| PPO (raw actions) minus PPO | +0.041 [+0.040, +0.042]* | +0.000 [+0.000, +0.000] |

Adjacent ranks whose fitness interval includes 0 are statistically tied; their order in the table is not a result: DQN (afterstates) and Cross-entropy method (interval -0.002 to +0.055); Cross-entropy method and GA heuristic (interval -0.004 to +0.061).

## Secondary: plain (no garbage, 1000-piece cap)

| Rank | Agent | Mean lines | Mean score | Pieces survived | Attack/min | Lines/piece | Score/piece | Top-out rate | Fitness |
|---|---|---|---|---|---|---|---|---|---|
| 1 | DQN (afterstates) | 398.3 | 827,497 | 1000.0 | 10.3 | 0.398 | 827 | 0.00 | 1.207 |
| 2 | Cross-entropy method | 398.4 | 819,050 | 1000.0 | 10.8 | 0.398 | 819 | 0.00 | 1.205 |
| 3 | GA heuristic | 398.6 | 781,992 | 1000.0 | 6.4 | 0.399 | 782 | 0.00 | 1.195 |
| 4 | Dellacherie (baseline) | 398.8 | 755,690 | 1000.0 | 4.4 | 0.399 | 756 | 0.00 | 1.189 |
| 5 | PPO (raw actions) | 21.4 | 6,344 | 97.2 | 7.1 | 0.218 | 64 | 1.00 | 0.113 |
| 6 | PPO | 5.8 | 1,619 | 50.1 | 0.8 | 0.101 | 30 | 1.00 | 0.058 |

## Marathon (no garbage, level gravity, 2000-piece cap)

100 headless games per agent and preset on held-out seeds 900000..900099, lockstep with a simulated 60 fps clock, so speed caps, reaction delays, move rates and mistakes apply exactly as in real-time play. Runtime: 9 minutes.

A Marathon match ends when the human tops out, so the numbers that matter are the agent's score and lines at the moment you top out. The table gives the agent's mean score and lines after 1, 3, 5 and 10 minutes of play (after the countdown). An agent that has already topped out keeps its final score. If you top out at 3 minutes with more points than the 3-minute score, you beat the average game of that agent.

At Easy, the heuristic agents (Dellacherie, GA, CEM) and DQN are all limited by the preset's speed cap and reaction delay rather than by their placement choices, so they score almost the same.

Time to top-out is the mean time until the agent tops out. Games that reach the 2000-piece cap count at the time they reached it, so where the cap-hit rate is above 0 the mean is a lower bound (marked >=).

| Agent | Preset | Score 1 min | Score 3 min | Score 5 min | Score 10 min | Lines 1 / 3 / 5 / 10 min | Time to top-out | Cap-hit rate |
|---|---|---|---|---|---|---|---|---|
| Dellacherie (baseline) | Easy | 1,994 | 10,816 | 26,339 | 77,121 | 9 / 31 / 53 / 98 | 8:49 | 0.00 |
| Dellacherie (baseline) | Medium | 6,241 | 39,031 | 96,883 | 357,141 | 22 / 69 / 116 / 233 | >= 33:21 | 0.96 |
| Dellacherie (baseline) | Hard | 19,317 | 133,010 | 347,028 | 924,188 | 45 / 139 / 232 / 461 | >= 17:04 | 0.99 |
| GA heuristic | Easy | 1,980 | 11,216 | 26,776 | 81,923 | 9 / 31 / 53 / 99 | 8:53 | 0.00 |
| GA heuristic | Medium | 6,354 | 40,173 | 99,446 | 363,607 | 22 / 69 / 116 / 232 | >= 32:53 | 0.93 |
| GA heuristic | Hard | 19,609 | 136,888 | 357,613 | 950,483 | 45 / 139 / 233 / 464 | >= 17:11 | 1.00 |
| Cross-entropy method | Easy | 2,009 | 11,050 | 27,116 | 80,900 | 9 / 31 / 53 / 98 | 8:49 | 0.00 |
| Cross-entropy method | Medium | 6,444 | 41,075 | 102,766 | 373,106 | 22 / 69 / 116 / 233 | >= 33:14 | 0.97 |
| Cross-entropy method | Hard | 20,385 | 142,420 | 366,886 | 975,626 | 45 / 139 / 233 / 465 | >= 17:11 | 1.00 |
| DQN (afterstates) | Easy | 1,899 | 10,916 | 26,443 | 78,116 | 8 / 31 / 53 / 98 | 8:51 | 0.00 |
| DQN (afterstates) | Medium | 6,296 | 41,008 | 103,124 | 370,889 | 21 / 68 / 115 / 231 | >= 33:32 | 0.96 |
| DQN (afterstates) | Hard | 20,256 | 145,391 | 371,554 | 983,144 | 44 / 139 / 232 / 462 | >= 17:17 | 1.00 |
| PPO | Easy | 697 | 833 | 833 | 833 | 1 / 2 / 2 / 2 | 1:12 | 0.00 |
| PPO | Medium | 1,224 | 1,277 | 1,277 | 1,277 | 4 / 4 / 4 / 4 | 0:46 | 0.00 |
| PPO | Hard | 1,489 | 1,489 | 1,489 | 1,489 | 5 / 5 / 5 / 5 | 0:24 | 0.00 |
| PPO (raw actions) | Easy | 909 | 1,314 | 1,314 | 1,314 | 2 / 4 / 4 / 4 | 1:39 | 0.00 |
| PPO (raw actions) | Medium | 2,586 | 3,347 | 3,347 | 3,347 | 10 / 13 / 13 / 13 | 1:17 | 0.00 |
| PPO (raw actions) | Hard | 5,865 | 5,873 | 5,873 | 5,873 | 20 / 20 / 20 / 20 | 0:48 | 0.00 |
