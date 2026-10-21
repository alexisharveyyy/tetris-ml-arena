# NEAT selection: winner's-curse bias

NEAT's selected genome scored 1.395 on the 30 selection seeds and 1.280 on
the 500 held-out seeds, a drop of 0.115. This report explains the drop. The
selection rule is not changed: the held-out results have been seen, and
re-selecting now would tune on them.

## What happened

Every 5 generations, that generation's best genome (by training fitness on
10 seeds) was evaluated on the same 30 selection seeds. The run made 46 such
evaluations, and the highest one was kept.

From generation 20 on, training fitness was flat at about 1.41. The 42
candidates evaluated from then on are therefore of roughly equal quality,
but their selection fitness varied widely:

| Candidates from generation 20 on | Value |
|---|---|
| Count | 42 |
| Mean selection fitness | 1.269 |
| Median | 1.280 |
| Standard deviation | 0.067 |
| Range | 1.070 to 1.395 |
| Selected (generation 60) | 1.395 |
| Selected genome, held-out (500 seeds) | 1.280 |

A 30-game mean is noisy, so picking the maximum of 42 similar candidates
mostly picks the candidate with the luckiest draw. For 42 draws from a
normal distribution the expected maximum is about 2.2 standard deviations
above the mean (about 1.41 here). The selected value, 1.395, is consistent
with that. The held-out score, 1.280, equals the plateau's median: the
selected genome is a typical plateau genome, not an unusually good one.

This is winner's-curse bias: the selection score of the chosen candidate is
biased upward, by more when there are more candidates and when each
candidate's score is noisier.

## Comparison with the other agents

| Agent | Candidates evaluated | Selection fitness | Held-out fitness | Drop |
|---|---|---|---|---|
| DQN | 18 (plus the final checkpoint) | 1.340 | 1.345 | -0.005 (none) |
| CEM | 10 (last 10 generations) | 1.356 | 1.318 | 0.038 |
| PPO | 18 (plus the final checkpoint) | 1.392 | 1.339 | 0.053 |
| Imitation (pure BC) | 10 | 1.358 | 1.296 | 0.062 |
| Imitation (BC + DAgger) | 7 | 1.343 | 1.280 | 0.063 |
| GA | 10 (last 10 generations) | 1.354 | 1.289 | 0.065 |
| NEAT | 46 | 1.395 | 1.280 | 0.115 |

DQN shows no drop. Its selection fitness rose to a single peak (1.276,
1.340, 1.336 at 15, 25 and 35 minutes) and then fell off (0.37 to 1.09
afterwards, see `reports/dqn_overestimation.md`), so only about three
candidates competed for the top. With few close competitors, the maximum
carries little luck.

The other agents drop by 0.04 to 0.07, which is the same bias at a smaller
scale. NEAT drops by about twice as much because it had the most candidates
(46) and about 42 of them were near-identical in quality, the worst case for
this bias.

Held-out results are unaffected: every leaderboard number comes from the
500 held-out seeds, never from the selection seeds. The bias only means that
selection fitness overstates how good the chosen model is.

## Future work

For population methods (GA, CEM, NEAT), use more selection seeds, for
example 100. That cuts the noise of each candidate's selection score by
about 45% (the standard error scales with 1/sqrt(games)), and with it the
upward bias. It is cheap: GA and CEM evaluated 300 selection games in 15
seconds, so 70 more games per candidate cost about 3 to 4 seconds, or about
3 minutes over NEAT's 46 candidates and under a minute for GA and CEM.
