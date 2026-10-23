# Leaderboard

Held-out seeds 900000..900499 (500 games per agent and setting). Attack/min assumes 2 pieces per second.

Evaluation garbage is inserted directly after every scheduled piece. It does not go through the pending meter and cannot be cancelled by the agent's attacks, so the garbage setting measures survival under a fixed load. Placements are instant (no gravity or speed limits).

Top-out intervals are 95% Wilson score intervals, which stay inside 0 to 100% and have nonzero width when every game or no game topped out.

## Primary: garbage (1 garbage row every 3 pieces, 500-piece cap)

| Rank | Agent | Mean lines | Mean score | Pieces survived | Attack/min | Lines/piece | Score/piece | Top-out rate | Top-out 95% interval (Wilson) | Fitness |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | DQN (afterstates) | 325.8 | 768,213 | 475.6 | 41.2 | 0.681 | 1574 | 0.13 | [0.10, 0.16] | 1.345 |
| 2 | PPO | 324.1 | 767,929 | 472.8 | 44.4 | 0.680 | 1573 | 0.12 | [0.10, 0.16] | 1.339 |
| 3 | Cross-entropy method | 321.2 | 737,539 | 468.7 | 40.8 | 0.681 | 1522 | 0.15 | [0.12, 0.18] | 1.318 |
| 4 | Decision Transformer | 319.2 | 730,141 | 466.0 | 38.9 | 0.679 | 1509 | 0.16 | [0.13, 0.19] | 1.309 |
| 5 | Imitation (pure BC) | 314.0 | 725,266 | 459.7 | 40.4 | 0.675 | 1508 | 0.20 | [0.17, 0.24] | 1.296 |
| 6 | GA heuristic | 313.1 | 720,348 | 457.9 | 39.9 | 0.676 | 1495 | 0.18 | [0.15, 0.22] | 1.289 |
| 7 | NEAT | 311.6 | 709,113 | 456.1 | 38.4 | 0.674 | 1472 | 0.19 | [0.16, 0.23] | 1.280 |
| 8 | Imitation (BC + DAgger) | 310.1 | 713,000 | 453.9 | 40.1 | 0.675 | 1489 | 0.21 | [0.17, 0.24] | 1.280 |
| 9 | Dellacherie (baseline) | 289.7 | 633,186 | 427.5 | 35.2 | 0.665 | 1362 | 0.31 | [0.27, 0.35] | 1.196 |
| 10 | BC (you) | 23.4 | 10,321 | 56.5 | 17.4 | 0.360 | 135 | 1.00 | [0.99, 1.00] | 0.147 |
| 11 | PPO (raw actions) | 11.9 | 2,663 | 44.4 | 9.9 | 0.266 | 59 | 1.00 | [0.99, 1.00] | 0.103 |

Paired bootstrap on the same seeds (10,000 resamples), mean difference and 95% interval. Intervals that exclude 0 are marked *.

| Comparison | Fitness difference | Top-out rate difference |
|---|---|---|
| DQN (afterstates) minus PPO | +0.006 [-0.023, +0.035] | +0.002 [-0.040, +0.044] |
| DQN (afterstates) minus Cross-entropy method | +0.027 [-0.002, +0.055] | -0.024 [-0.068, +0.020] |
| DQN (afterstates) minus Decision Transformer | +0.035 [+0.006, +0.065]* | -0.032 [-0.076, +0.012] |
| DQN (afterstates) minus Imitation (pure BC) | +0.048 [+0.017, +0.080]* | -0.076 [-0.120, -0.030]* |
| DQN (afterstates) minus GA heuristic | +0.055 [+0.024, +0.086]* | -0.056 [-0.100, -0.012]* |
| DQN (afterstates) minus NEAT | +0.064 [+0.033, +0.097]* | -0.064 [-0.110, -0.018]* |
| DQN (afterstates) minus Imitation (BC + DAgger) | +0.065 [+0.032, +0.098]* | -0.080 [-0.126, -0.034]* |
| DQN (afterstates) minus Dellacherie (baseline) | +0.149 [+0.114, +0.184]* | -0.184 [-0.232, -0.136]* |
| DQN (afterstates) minus BC (you) | +1.198 [+1.177, +1.217]* | -0.874 [-0.902, -0.844]* |
| DQN (afterstates) minus PPO (raw actions) | +1.241 [+1.222, +1.259]* | -0.874 [-0.902, -0.844]* |
| PPO minus Cross-entropy method | +0.021 [-0.008, +0.051] | -0.026 [-0.068, +0.016] |
| PPO minus Decision Transformer | +0.030 [-0.001, +0.060] | -0.034 [-0.076, +0.008] |
| PPO minus Imitation (pure BC) | +0.043 [+0.011, +0.074]* | -0.078 [-0.124, -0.032]* |
| PPO minus GA heuristic | +0.050 [+0.016, +0.083]* | -0.058 [-0.102, -0.014]* |
| PPO minus NEAT | +0.059 [+0.025, +0.092]* | -0.066 [-0.110, -0.022]* |
| PPO minus Imitation (BC + DAgger) | +0.059 [+0.026, +0.093]* | -0.082 [-0.126, -0.038]* |
| PPO minus Dellacherie (baseline) | +0.143 [+0.106, +0.181]* | -0.186 [-0.236, -0.138]* |
| PPO minus BC (you) | +1.192 [+1.169, +1.213]* | -0.876 [-0.904, -0.846]* |
| PPO minus PPO (raw actions) | +1.236 [+1.213, +1.256]* | -0.876 [-0.904, -0.846]* |
| Cross-entropy method minus Decision Transformer | +0.009 [-0.022, +0.038] | -0.008 [-0.052, +0.036] |
| Cross-entropy method minus Imitation (pure BC) | +0.021 [-0.009, +0.053] | -0.052 [-0.098, -0.006]* |
| Cross-entropy method minus GA heuristic | +0.028 [-0.004, +0.061] | -0.032 [-0.078, +0.014] |
| Cross-entropy method minus NEAT | +0.038 [+0.004, +0.072]* | -0.040 [-0.086, +0.006] |
| Cross-entropy method minus Imitation (BC + DAgger) | +0.038 [+0.006, +0.071]* | -0.056 [-0.102, -0.010]* |
| Cross-entropy method minus Dellacherie (baseline) | +0.122 [+0.085, +0.159]* | -0.160 [-0.210, -0.110]* |
| Cross-entropy method minus BC (you) | +1.171 [+1.148, +1.193]* | -0.850 [-0.882, -0.818]* |
| Cross-entropy method minus PPO (raw actions) | +1.215 [+1.192, +1.235]* | -0.850 [-0.882, -0.818]* |
| Decision Transformer minus Imitation (pure BC) | +0.013 [-0.018, +0.044] | -0.044 [-0.090, +0.002] |
| Decision Transformer minus GA heuristic | +0.020 [-0.013, +0.053] | -0.024 [-0.070, +0.022] |
| Decision Transformer minus NEAT | +0.029 [-0.004, +0.063] | -0.032 [-0.078, +0.014] |
| Decision Transformer minus Imitation (BC + DAgger) | +0.029 [-0.004, +0.063] | -0.048 [-0.094, -0.002]* |
| Decision Transformer minus Dellacherie (baseline) | +0.114 [+0.077, +0.151]* | -0.152 [-0.204, -0.100]* |
| Decision Transformer minus BC (you) | +1.163 [+1.139, +1.186]* | -0.842 [-0.874, -0.810]* |
| Decision Transformer minus PPO (raw actions) | +1.206 [+1.183, +1.228]* | -0.842 [-0.874, -0.810]* |
| Imitation (pure BC) minus GA heuristic | +0.007 [-0.026, +0.040] | +0.020 [-0.028, +0.066] |
| Imitation (pure BC) minus NEAT | +0.016 [-0.020, +0.051] | +0.012 [-0.036, +0.060] |
| Imitation (pure BC) minus Imitation (BC + DAgger) | +0.017 [-0.017, +0.049] | -0.004 [-0.052, +0.044] |
| Imitation (pure BC) minus Dellacherie (baseline) | +0.101 [+0.060, +0.142]* | -0.108 [-0.162, -0.054]* |
| Imitation (pure BC) minus BC (you) | +1.150 [+1.123, +1.174]* | -0.798 [-0.832, -0.764]* |
| Imitation (pure BC) minus PPO (raw actions) | +1.193 [+1.168, +1.217]* | -0.798 [-0.832, -0.764]* |
| GA heuristic minus NEAT | +0.009 [-0.025, +0.045] | -0.008 [-0.056, +0.038] |
| GA heuristic minus Imitation (BC + DAgger) | +0.010 [-0.026, +0.044] | -0.024 [-0.070, +0.022] |
| GA heuristic minus Dellacherie (baseline) | +0.094 [+0.055, +0.133]* | -0.128 [-0.180, -0.076]* |
| GA heuristic minus BC (you) | +1.143 [+1.116, +1.169]* | -0.818 [-0.852, -0.784]* |
| GA heuristic minus PPO (raw actions) | +1.186 [+1.160, +1.211]* | -0.818 [-0.852, -0.784]* |
| NEAT minus Imitation (BC + DAgger) | +0.000 [-0.037, +0.037] | -0.016 [-0.066, +0.034] |
| NEAT minus Dellacherie (baseline) | +0.085 [+0.046, +0.123]* | -0.120 [-0.170, -0.070]* |
| NEAT minus BC (you) | +1.133 [+1.106, +1.160]* | -0.810 [-0.844, -0.776]* |
| NEAT minus PPO (raw actions) | +1.177 [+1.150, +1.202]* | -0.810 [-0.844, -0.776]* |
| Imitation (BC + DAgger) minus Dellacherie (baseline) | +0.084 [+0.046, +0.122]* | -0.104 [-0.154, -0.054]* |
| Imitation (BC + DAgger) minus BC (you) | +1.133 [+1.105, +1.160]* | -0.794 [-0.828, -0.758]* |
| Imitation (BC + DAgger) minus PPO (raw actions) | +1.177 [+1.149, +1.203]* | -0.794 [-0.828, -0.758]* |
| Dellacherie (baseline) minus BC (you) | +1.049 [+1.017, +1.080]* | -0.690 [-0.730, -0.650]* |
| Dellacherie (baseline) minus PPO (raw actions) | +1.092 [+1.061, +1.123]* | -0.690 [-0.730, -0.650]* |
| BC (you) minus PPO (raw actions) | +0.043 [+0.037, +0.050]* | +0.000 [+0.000, +0.000] |

Adjacent ranks whose fitness interval includes 0 are statistically tied; their order in the table is not a result: DQN (afterstates) and PPO (interval -0.023 to +0.035); PPO and Cross-entropy method (interval -0.008 to +0.051); Cross-entropy method and Decision Transformer (interval -0.022 to +0.038); Decision Transformer and Imitation (pure BC) (interval -0.018 to +0.044); Imitation (pure BC) and GA heuristic (interval -0.026 to +0.040); GA heuristic and NEAT (interval -0.025 to +0.045); NEAT and Imitation (BC + DAgger) (interval -0.037 to +0.037).

## Secondary: plain (no garbage, 1000-piece cap)

| Rank | Agent | Mean lines | Mean score | Pieces survived | Attack/min | Lines/piece | Score/piece | Top-out rate | Top-out 95% interval (Wilson) | Fitness |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | PPO | 397.2 | 900,201 | 1000.0 | 18.2 | 0.397 | 900 | 0.00 | [0.00, 0.01] | 1.225 |
| 2 | DQN (afterstates) | 398.3 | 827,497 | 1000.0 | 10.3 | 0.398 | 827 | 0.00 | [0.00, 0.01] | 1.207 |
| 3 | Cross-entropy method | 398.4 | 819,050 | 1000.0 | 10.8 | 0.398 | 819 | 0.00 | [0.00, 0.01] | 1.205 |
| 4 | Imitation (pure BC) | 398.5 | 786,553 | 1000.0 | 6.8 | 0.399 | 787 | 0.00 | [0.00, 0.01] | 1.197 |
| 5 | Decision Transformer | 398.7 | 785,497 | 1000.0 | 6.9 | 0.399 | 785 | 0.00 | [0.00, 0.01] | 1.196 |
| 6 | GA heuristic | 398.6 | 781,992 | 1000.0 | 6.4 | 0.399 | 782 | 0.00 | [0.00, 0.01] | 1.195 |
| 7 | Imitation (BC + DAgger) | 398.6 | 780,125 | 1000.0 | 6.2 | 0.399 | 780 | 0.00 | [0.00, 0.01] | 1.195 |
| 8 | NEAT | 398.8 | 755,862 | 1000.0 | 4.8 | 0.399 | 756 | 0.00 | [0.00, 0.01] | 1.189 |
| 9 | Dellacherie (baseline) | 398.8 | 755,690 | 1000.0 | 4.4 | 0.399 | 756 | 0.00 | [0.00, 0.01] | 1.189 |
| 10 | BC (you) | 359.5 | 722,392 | 907.8 | 9.1 | 0.388 | 745 | 0.15 | [0.13, 0.19] | 1.094 |
| 11 | PPO (raw actions) | 21.4 | 6,344 | 97.2 | 7.1 | 0.218 | 64 | 1.00 | [0.99, 1.00] | 0.113 |

## Marathon (no garbage, level gravity, 2000-piece cap)

100 headless games per agent and preset on held-out seeds 900000..900099, lockstep with a simulated 60 fps clock, so speed caps, reaction delays, move rates and mistakes apply exactly as in real-time play. Runtime: 50 minutes.

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
| PPO | Easy | 1,996 | 11,873 | 29,056 | 96,286 | 8 / 32 / 55 / 109 | 9:35 | 0.00 |
| PPO | Medium | 6,210 | 42,415 | 108,305 | 392,798 | 20 / 68 / 115 / 234 | >= 33:21 | 0.99 |
| PPO | Hard | 20,753 | 154,075 | 393,328 | 1,041,008 | 44 / 138 / 232 / 465 | >= 17:08 | 1.00 |
| PPO (raw actions) | Easy | 909 | 1,314 | 1,314 | 1,314 | 2 / 4 / 4 / 4 | 1:39 | 0.00 |
| PPO (raw actions) | Medium | 2,586 | 3,347 | 3,347 | 3,347 | 10 / 13 / 13 / 13 | 1:17 | 0.00 |
| PPO (raw actions) | Hard | 5,865 | 5,873 | 5,873 | 5,873 | 20 / 20 / 20 / 20 | 0:48 | 0.00 |
| Imitation (pure BC) | Easy | 2,074 | 11,701 | 28,876 | 95,759 | 9 / 32 / 56 / 112 | 9:47 | 0.00 |
| Imitation (pure BC) | Medium | 6,286 | 39,418 | 98,414 | 358,794 | 22 / 69 / 116 / 232 | >= 33:26 | 0.98 |
| Imitation (pure BC) | Hard | 19,661 | 137,154 | 356,190 | 948,700 | 45 / 139 / 233 / 464 | >= 17:12 | 1.00 |
| Imitation (BC + DAgger) | Easy | 2,043 | 11,582 | 28,378 | 93,453 | 9 / 32 / 56 / 111 | 9:42 | 0.00 |
| Imitation (BC + DAgger) | Medium | 6,230 | 39,307 | 97,203 | 357,416 | 22 / 69 / 116 / 232 | >= 33:35 | 0.98 |
| Imitation (BC + DAgger) | Hard | 19,608 | 136,485 | 354,974 | 944,044 | 45 / 139 / 233 / 465 | >= 17:11 | 1.00 |
| BC (you) | Easy | 1,970 | 9,946 | 20,823 | 44,902 | 8 / 28 / 43 / 64 | 6:16 | 0.00 |
| BC (you) | Medium | 6,322 | 37,795 | 88,989 | 276,234 | 21 / 65 / 104 / 187 | >= 15:09 | 0.12 |
| BC (you) | Hard | 19,487 | 130,971 | 325,221 | 742,581 | 44 / 130 / 211 / 369 | >= 11:26 | 0.33 |
| Decision Transformer | Easy | 2,132 | 11,974 | 28,832 | 94,999 | 9 / 33 / 57 / 112 | 9:47 | 0.00 |
| Decision Transformer | Medium | 6,413 | 39,716 | 98,966 | 363,272 | 22 / 69 / 116 / 234 | >= 33:40 | 1.00 |
| Decision Transformer | Hard | 19,680 | 137,237 | 355,809 | 950,449 | 45 / 139 / 233 / 465 | >= 17:09 | 1.00 |
| NEAT | Easy | 1,999 | 11,164 | 27,055 | 75,710 | 9 / 31 / 54 / 96 | 8:43 | 0.00 |
| NEAT | Medium | 6,329 | 39,345 | 97,609 | 356,801 | 22 / 69 / 116 / 233 | >= 33:35 | 0.98 |
| NEAT | Hard | 19,231 | 133,382 | 346,714 | 923,554 | 45 / 139 / 233 / 464 | >= 17:12 | 1.00 |
