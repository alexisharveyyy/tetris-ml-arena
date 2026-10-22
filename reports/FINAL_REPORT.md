# Tetris ML Arena: final report

Eleven agents, from a hand-tuned heuristic to deep RL, imitation learning,
neuroevolution, offline RL with a Decision Transformer, and an LLM, all play
the same deterministic Tetris engine. They are compared on 500 held-out
games under one protocol, and they play against people in real time through
a desktop and a web interface.

**Main result.** Eight agents built by six different methods land within
0.065 fitness of each other and form a chain of statistical ties. Their
shared choice of action space (one hard-drop placement per piece, chosen by
scoring the board after the placement) appears to set the ceiling, not the
learning method. The only agent that learns raw actions is far below them.

Every number here comes from a report in `reports/`, named next to it.
Section 6 gives the command and commit for each result.

---

## 1. Setup

### 1.1 Engine

- **Board:** 10 wide, 20 visible rows plus 4 hidden rows (24 in total).
- **Pieces:** 7-bag randomizer, 5 visible next pieces, hold once per piece.
  All randomness comes from seeded SplitMix64 streams, so a seed fixes the
  whole piece sequence and every garbage hole (`tetris/engine/`).
- **Gravity and lock:** the guideline gravity curve, with the level going up
  every 10 lines up to 20. Lock delay is 0.5 s with at most 15 resets.
- **Scoring:** the guideline values (100/300/500/800 × level, ×1.5 for a
  back-to-back Tetris, a combo bonus, 1 point per row of soft drop and 2 per
  row of hard drop). There are no T-spins.
- **Top-out** happens in three ways:
  - the spawn position is blocked;
  - a piece locks entirely in the hidden rows;
  - garbage pushes cells off the top.

  In headless evaluation, a state with no legal placement also ends the game.

### 1.2 Action space

- **The 80 actions:** each is a placement: rotation (4) × column (10) × use
  hold (2). A placement is legal if the piece can rotate in place at spawn
  with no wall kicks, shift sideways at spawn height, and hard drop.
- **Out of scope:** tucks, spins and soft-drop slides.
- **Interface:** `TetrisEnv` (Gymnasium) exposes the action mask.
- **Afterstates:** `enumerate_afterstates` returns every legal placement
  with the board after line clears and 13 features: 10 board features (height,
  holes, bumpiness, wells, transitions and similar) and 3 placement features
  (lines cleared, landing height, eroded cells).
- **Path check:** a test checks that the real-time path planner reaches
  every action the mask allows, over 480 random states (about 21,800
  actions).

### 1.3 Agents

| Agent | Method | Decides by |
|---|---|---|
| Dellacherie | Hand-tuned linear weights (baseline) | Scoring afterstates (13 features) |
| GA heuristic | Genetic algorithm over the linear weights | Scoring afterstates (13 features) |
| Cross-entropy method (CEM) | CEM over the linear weights | Scoring afterstates (13 features) |
| NEAT | Neuroevolution, starting from linear genomes | Scoring afterstates (13 features) |
| DQN | Deep Q-learning with a CNN on the raw board | Scoring afterstates |
| PPO | PPO with a CNN scorer; the policy is a softmax over afterstate scores | Scoring afterstates |
| PPO (raw actions) | PPO with an 80-way action head | Raw actions |
| Imitation (pure BC) | Behavior cloning of the GA's games | Scoring afterstates |
| Imitation (BC + DAgger) | BC with 2 DAgger rounds, the GA as teacher | Scoring afterstates |
| BC (you) | Pure BC fine-tuned on 4 recorded human games | Scoring afterstates |
| Decision Transformer (DT) | Return-conditioned transformer trained offline on the other agents' games | Matching afterstates (each placement's logit is the match between the model's output and that afterstate's embedding) |
| LLM (novelty, not on the leaderboard) | Local or API language model choosing from a text list of placements | Falls back to the GA's move |

### 1.4 Hardware

- One Apple M2 Max laptop with 12 CPU cores.
- Deep models train on the Apple GPU (MPS). Evaluation and live play run on
  CPU worker processes: 11 for evaluation, 1 per opponent in live play.

## 2. Protocol

### 2.1 Evaluation settings (`tetris/agents/evaluation.py`)

- **Primary: garbage.**
  - After every 3rd piece, one garbage row is inserted directly into the
    board. It skips the pending meter and cannot be cancelled by attacks,
    so the setting measures survival under a fixed load.
  - 500-piece cap. Placements are instant: no gravity or speed limits.
- **Secondary: plain.** No garbage and a 1000-piece cap. It is a sanity check:
  every afterstate agent survives every game.
- **Fitness per game** = pieces / cap + 0.25 × (score per piece) / 1000.
  The leaderboard reports the mean over games. Survival dominates the
  fitness, and score per piece breaks ties among agents that survive.
- **Marathon:** headless real-time games run in lockstep with a simulated
  60 fps clock.
  - Level gravity, no garbage, a 2000-piece cap.
  - Each agent plays at the Easy, Medium and Hard presets. The presets
    apply real speed caps, reaction delays, move rates and decision noise
    (`tetris/agents/difficulty.py`): Easy 0.5 pieces/s with a 0.9 s
    reaction, Medium 1.0 pieces/s with 0.45 s, Hard 2.0 pieces/s with 0.2 s.
  - Scores are recorded at 1, 3, 5 and 10 minutes, the numbers a human
    opponent compares against.

### 2.2 Seed splits

| Use | Seeds |
|---|---|
| Training games (GA, CEM, NEAT, DQN, PPO) | drawn from [0, 500,000) |
| GA game logs (BC data) | 600,000-600,099 (garbage and plain) |
| DAgger rounds | 651,000-652,549 |
| DT v1 trajectories | 700,000-721,049 |
| DT v2 trajectories | 720,000-733,069 |
| **Selection** (checkpoint choice) | **800,000-800,029** (30) |
| **Held-out** (leaderboard) | **900,000-900,499** (500) |
| Marathon | 900,000-900,099 |
| LLM evaluation | 900,000-900,019 |
| Difficulty tuning | 950,000 and up |

No training or selection data uses a held-out seed. Two overlaps outside the
evaluation seeds are listed in Section 5.

### 2.3 Budgets

| Agent | Budget |
|---|---|
| GA, CEM | 40 generations × 50 candidates × 10 games each (the same 2,000 evaluations), about 15 minutes on 12 cores |
| DQN, PPO | 180 minutes of wall clock on MPS |
| PPO (raw actions) | 60 minutes, plus two 15-minute tuning runs |
| Pure BC | 30 minutes |
| BC + DAgger | 2 rounds of 10 minutes |
| BC (you) | at most 10 minutes, early stopping (it stopped at 9.1 minutes) |
| NEAT | 180 minutes on CPU, population 150 (222 generations) |
| DT v1, DT v2 | 120 minutes each on MPS |
| Seed-spread runs | 60 minutes each |

The budgets are wall clock on one machine, not equal compute (Section 5).

### 2.4 Checkpoint selection

- **The rule:** every learned agent writes evaluation checkpoints during
  training. Each is played greedily on the 30 selection seeds (garbage
  setting), and the one with the highest selection fitness is promoted.
  - DQN and PPO: at 5 minutes, then every 10 minutes.
  - BC: every 3 minutes.
  - NEAT: every 5 generations, that generation's best genome.
  - GA and CEM: the last 10 hall-of-fame entries.
  - DT: every 10 minutes, at its selection target (the 90th percentile of
    returns in its data).
- **Version choices** use only the selection seeds. A new version replaces
  the old one only if its selection fitness is strictly higher:
  - DQN v1 vs v2;
  - DT v1 vs v2;
  - pure BC vs DAgger (ties go to pure BC).

  The held-out seeds play no part in any choice.
- **BC (you)** is selected by top-1 agreement on a held-out human
  validation game.

### 2.5 Statistics

- **Top-out rates:** 95% Wilson score intervals (z = 1.96). They stay inside
  [0, 1] and are not zero-width at 0% or 100%.
- **Paired bootstrap:** for each pair of agents (or versions), on the same
  seeds in the same order: per-game differences in fitness and in top-out,
  10,000 resamples of game indices with replacement (seed 0), the mean
  difference and the 2.5th/97.5th percentiles.
  - **Significant:** an interval that excludes 0.
  - **Statistically tied:** adjacent ranks whose interval includes 0. Their
    order is not a result.
  - **No correction:** the leaderboard reports all 55 pairs in the garbage
    setting without a multiple-comparison correction.
- **LLM evaluation:** 20 games per run, with bootstrap intervals over games
  and paired differences against the GA on the same seeds.

## 3. Results

### 3.1 Garbage leaderboard (`reports/leaderboard.md`)

500 held-out games per agent. In the group column, agents that share a
letter are not significantly different by the paired bootstrap; agents with
no letter in common are.

| Rank | Agent | Fitness | Pieces survived | Lines/piece | Top-out rate [95% Wilson] | Group |
|---|---|---|---|---|---|---|
| 1 | DQN (afterstates) | 1.345 | 475.6 | 0.681 | 0.13 [0.10, 0.16] | a |
| 2 | PPO | 1.339 | 472.8 | 0.680 | 0.12 [0.10, 0.16] | ab |
| 3 | Cross-entropy method | 1.318 | 468.7 | 0.681 | 0.15 [0.12, 0.18] | abc |
| 4 | Decision Transformer (v1) | 1.309 | 466.0 | 0.679 | 0.16 [0.13, 0.19] | bcd |
| 5 | Imitation (pure BC) | 1.296 | 459.7 | 0.675 | 0.20 [0.17, 0.24] | cd |
| 6 | GA heuristic | 1.289 | 457.9 | 0.676 | 0.18 [0.15, 0.22] | cd |
| 7 | NEAT | 1.280 | 456.1 | 0.674 | 0.19 [0.16, 0.23] | d |
| 8 | Imitation (BC + DAgger) | 1.280 | 453.9 | 0.675 | 0.21 [0.17, 0.24] | d |
| 9 | Dellacherie (baseline) | 1.196 | 427.5 | 0.665 | 0.31 [0.27, 0.35] | e |
| 10 | BC (you) | 0.147 | 56.5 | 0.360 | 1.00 [0.99, 1.00] | f |
| 11 | PPO (raw actions) | 0.103 | 44.4 | 0.266 | 1.00 [0.99, 1.00] | g |

**How to read it:**
- The tie chain covers ranks 1 to 8: every adjacent pair's interval includes
  0. The ends of the chain do differ:
  - DQN is ahead of DT (+0.035 [+0.006, +0.065]) and of every agent below;
  - PPO is ahead of pure BC (+0.043 [+0.011, +0.074]) and below;
  - CEM is ahead of NEAT (+0.038 [+0.004, +0.072]) and DAgger.
- Every one of the eight is significantly ahead of Dellacherie: the smallest
  margin is BC + DAgger's, +0.084 [+0.046, +0.122].

**Plain setting:** every afterstate agent except BC (you) reaches the
1000-piece cap in every game.
- They differ only in score per piece (fitness 1.189 to 1.225, PPO first).
- BC (you) tops out in 15% of games; raw-action PPO in all of them.

### 3.2 Marathon (`reports/leaderboard.md`)

Mean score at 1, 3, 5 and 10 minutes of play, 100 games per preset. An agent
that has topped out keeps its final score.

| Agent | Easy 1 / 3 / 5 / 10 min | Medium 1 / 3 / 5 / 10 min | Hard 1 / 3 / 5 / 10 min |
|---|---|---|---|
| Dellacherie | 1,994 / 10,816 / 26,339 / 77,121 | 6,241 / 39,031 / 96,883 / 357,141 | 19,317 / 133,010 / 347,028 / 924,188 |
| GA heuristic | 1,980 / 11,216 / 26,776 / 81,923 | 6,354 / 40,173 / 99,446 / 363,607 | 19,609 / 136,888 / 357,613 / 950,483 |
| CEM | 2,009 / 11,050 / 27,116 / 80,900 | 6,444 / 41,075 / 102,766 / 373,106 | 20,385 / 142,420 / 366,886 / 975,626 |
| DQN | 1,899 / 10,916 / 26,443 / 78,116 | 6,296 / 41,008 / 103,124 / 370,889 | 20,256 / 145,391 / 371,554 / 983,144 |
| PPO | 1,996 / 11,873 / 29,056 / 96,286 | 6,210 / 42,415 / 108,305 / 392,798 | 20,753 / 154,075 / 393,328 / 1,041,008 |
| PPO (raw actions) | 909 / 1,314 / 1,314 / 1,314 | 2,586 / 3,347 / 3,347 / 3,347 | 5,865 / 5,873 / 5,873 / 5,873 |
| Imitation (pure BC) | 2,074 / 11,701 / 28,876 / 95,759 | 6,286 / 39,418 / 98,414 / 358,794 | 19,661 / 137,154 / 356,190 / 948,700 |
| Imitation (BC + DAgger) | 2,043 / 11,582 / 28,378 / 93,453 | 6,230 / 39,307 / 97,203 / 357,416 | 19,608 / 136,485 / 354,974 / 944,044 |
| BC (you) | 1,970 / 9,946 / 20,823 / 44,902 | 6,322 / 37,795 / 88,989 / 276,234 | 19,487 / 130,971 / 325,221 / 742,581 |
| Decision Transformer | 2,132 / 11,974 / 28,832 / 94,999 | 6,413 / 39,716 / 98,966 / 363,272 | 19,680 / 137,237 / 355,809 / 950,449 |
| NEAT | 1,999 / 11,164 / 27,055 / 75,710 | 6,329 / 39,345 / 97,609 / 356,801 | 19,231 / 133,382 / 346,714 / 923,554 |

**What this shows:**
- **Easy:** no agent reaches the 2000-piece cap; the afterstate agents top
  out after 8:43 to 9:47 on average (BC (you) at 6:16).
- **Medium and Hard:** 93-100% of the afterstate agents' games reach the cap
  (BC (you) 12% and 33%).
- **The presets, not the method, set the opponent's strength.** Within a
  preset the afterstate agents' scores at 1 to 5 minutes are within about
  15% of each other (BC (you) aside).
- **Easy:** speed caps and reaction delays dominate, so placement quality
  barely shows.
- **Exceptions:** BC (you) falls behind from 5 minutes on, and raw-action PPO
  tops out in under 2 minutes.

### 3.3 Seed spread (`reports/seed_spread.md`)

The main runs use one seed each. To measure run-to-run variation, DQN (v2
config) and PPO were each trained three more times for 1 hour (seeds 0-2).
Each run was selected on the selection seeds and evaluated on all 500
held-out seeds:

| Agent | Held-out fitness per run | Mean ± SD |
|---|---|---|
| DQN | 1.097, 1.204, 1.217 | 1.173 ± 0.066 |
| PPO | 0.497, 0.544, 1.045 | 0.695 ± 0.304 |

- **What it means:** with a 1-hour budget, PPO's outcome depends heavily on
  the seed: one run in three learned to survive.
- **Size of the variation:** the spread between seeds is larger than any
  gap inside the 8-agent tie. A single seed per agent therefore cannot rank
  methods within that group (Section 5).

## 4. Findings

### 4.1 The 8-way tie: shared features and action space set the ceiling

- **The methods:**
  - linear weights tuned by a GA and by CEM;
  - an evolved network (NEAT) on the same 13 features;
  - CNNs on the raw board trained by Q-learning (DQN) and by policy gradient
    (PPO);
  - two cloned policies (BC, BC + DAgger);
  - a transformer trained offline on all their games (DT).
- **The result:** all eight land between 1.280 and 1.345 (Section 3.1), all
  clear 0.674-0.681 lines per piece, and all fail on the same kind of
  garbage-heavy game.
- **What they share** is the decision structure: one hard-drop placement per
  piece, chosen by scoring the board after the placement. The four linear
  and neuroevolution agents also share the same features.
- **The one agent without it:** raw-action PPO must learn placements from an
  80-way output, with no afterstate scoring. It reaches 0.103 and tops out
  in every game.
- **Interpretation (a hypothesis, not a tested result):**
  - Under this protocol, the action-space choice (placements scored as
    afterstates, no tucks or spins) and the information the scorer can use
    limit play more than the learning method does.
  - Raising the ceiling would likely need a richer action space or
    lookahead over the 5 visible next pieces. None of the agents uses
    lookahead: each scores only the current piece's afterstates.
  - The experiment that would test this (the same methods with lookahead or
    a larger action space) was not run.

### 4.2 DQN overestimation (`reports/dqn_overestimation.md`)

- **What happened:** DQN v1's greedy play peaked at 25 minutes (selection
  fitness 1.340) and then fell to 0.49-0.77 by 105-145 minutes.
- **Q-value bias:** offline estimates against Monte Carlo returns (12
  games) showed the bias moving from -65 at 25 minutes to +34 at 65 minutes
  and +42 at 135 minutes. In the last 20 pieces before a top-out it reached
  +63 to +66.
- **Decision margins:** the gap between the best and second-best placement's
  value shrank from 0.71 to 0.19.
- **The fix (v2):** a soft target update (Polyak 0.001), a cosine
  learning-rate schedule from 1e-4 to 1e-5, and a replay buffer grown from
  400k to 2M transitions.
- **Result of the fix:** it slowed the decline but did not raise the peak.
  v2's best was 1.253, so v1 stayed on the leaderboard under the strict
  version rule.
- **Selection caught it:** v1's held-out fitness (1.345) is the highest of
  any agent. Selecting checkpoints on held-apart seeds kept the decline out
  of the leaderboard.

### 4.3 The MPS gradient bug (`reports/ppo_mps_bug.md`)

- **The symptom:** the first afterstate PPO run never trained. Its selection
  fitness stayed at 0.061-0.064 (about 17 pieces per game, random play) for
  the whole run.
- **The cause:** on Apple MPS, the convolution backward pass returned wrong
  gradients when one call scored about 115,000 afterstates (2,048 states ×
  about 56 placements). The relative error against CPU gradients was about
  6 × 10^8. MPS gave correct gradients up to about 57,000 afterstates.
- **The fix:**
  - The scorer now runs in chunks of 4,096, which brings the relative error
    to about 10^-6.
  - Every GPU run now starts with a pre-flight check of MPS gradients
    against CPU gradients (tolerance 10^-2).
  - A watchdog stops runs whose learning has stalled.
- **Rerun:** PPO was rerun at the fixed commit and now ranks 2nd.
- **Audit:** results that depended on the broken runs were replaced
  (`reports/queue_pinning_audit.md`).
- **Lesson:** a silent numerical failure looked like a hard learning
  problem. Only a gradient check against a reference device separated the
  two.

### 4.4 DAgger added nothing at 95% agreement

- **Pure BC:** it matched the GA's choice on 95.1% of held-out GA states
  (top-3: 100%) and played as well as the GA: pure BC minus GA +0.007
  [-0.026, +0.040].
- **DAgger:** two rounds (the GA labeling the states BC visits) raised
  agreement to 96.3% but not play: pure BC minus DAgger +0.017 [-0.017,
  +0.049]. During the rounds, the learner agreed with the GA on 91.5% and
  94.0% of the states it visited itself.
- **Why:** DAgger fixes compounding errors on states the expert never
  visits. With so few disagreements, BC rarely drifted into such states,
  so there was little for DAgger to correct.

### 4.5 NEAT stayed nearly linear (`tetris/agents/neuroevolution/README.md`, `reports/neat_topology.png`)

- **What evolved:** starting from linear genomes over the 13 features, the
  selected network kept 2 hidden nodes. Its output is dominated by a linear
  term in four features.
- **How it played:** it ties the GA (-0.009 [-0.045, +0.025]) and is
  significantly behind CEM.
- **Takeaway:** on these features, nonlinearity bought nothing. This
  agrees with 4.1: the features and action space limit play more than the
  model class does.

### 4.6 Winner's curse in checkpoint selection (`reports/neat_selection.md`)

- **The size of it:** NEAT's selected genome scored 1.395 on the 30
  selection seeds and 1.280 held out, a drop of 0.115.
- **Why:** from generation 20 on, training fitness was flat. The 42
  candidates evaluated on the selection seeds had a mean of 1.269 and a
  standard deviation of 0.067.
  - The maximum of 42 noisy draws is expected about 2.2 SD above the mean.
  - The held-out score equals the plateau's median: the selected genome was
    a typical one, picked for a lucky 30-game draw.
- **The other agents** show the same bias, smaller with fewer candidates:
  GA 0.065, DAgger 0.063, pure BC 0.062, PPO 0.053, CEM 0.038.
  - DQN's held-out score was 0.005 above its selection score.
  - The DT dropped from 1.374 to 1.309.
- **Consequence:** selection scores are biased upward, so only held-out
  scores are compared. The selection rule was not changed after the
  held-out results were seen, since changing it would tune on them.

### 4.7 The Decision Transformer's target return (`reports/dt_calibration.md`, `reports/dt_v2.md`)

- **The plan:** the DT is conditioned on a target return, so asking for a
  lower return should give a weaker opponent: a difficulty knob learned
  from data.
- **The test:** each selected model played the 30 selection seeds at
  targets 0.2 to 1.4.

| Target | 0.20 | 0.40 | 0.60 | 0.80 | 1.00 | 1.20 | 1.40 | Rank correlation |
|---|---|---|---|---|---|---|---|---|
| v1 achieved fitness | 1.319 | 1.219 | 1.326 | 1.370 | 1.353 | 1.297 | 1.355 | +0.43 |
| v2 achieved fitness | 1.350 | 1.284 | 1.263 | 1.240 | 1.284 | 1.304 | 1.346 | +0.00 |

**v1** was trained on the trained agents' games, with decision noise
injected at three rates to create weak games.
- Every target played at the strong agents' level; the differences are
  about the size of 30-game noise.
- **Hypothesis:** injected mistakes are random, so the board does not
  predict them, and the model learned to imitate strong placements and
  ignore the return.

**v2** tested that hypothesis by changing only the data.
- **The data:** no injected mistakes. Instead, early, middle and late
  checkpoints of DQN, PPO, BC and NEAT, plus BC (you) and raw-action PPO,
  whose skill shows on the board. Mean source returns range from 0.10 to
  1.37.
- **Kept the same:** the model, the 2-hour budget and the selection protocol.
- **Result:** the curve became flat (rank correlation +0.00).
- **Held out:** v2 played worse than v1 at its default target: -0.090
  [-0.127, -0.054] fitness and +0.142 [+0.092, +0.192] top-out rate.
- **The version rule kept v1** (selection fitness 1.345 against 1.374).
- **Verdict:** the hypothesis was tested and rejected.

**A likely reason.** Weak agents top out fast, so their games are short.
- The five sources with mean return below 0.6 provide only 10% of v2's
  training states.
- Most of the loss is therefore minimized by imitating strong play, and the
  return token carries little signal.
- **Not run (future work):** a v3 that samples episodes, not states, evenly
  across return bins.

**In the app** the DT plays at its default target with no target control,
because the control would promise a weaker opponent and not deliver one.

### 4.8 BC (you): imitation from very little human data (`reports/bc_you.json`)

- **The data:** 7 recorded plain games of one person (1,397 placements the
  action space can express, plus 3 tucks it cannot).
- **Split by game:** 4 for training (751 states), 1 for validation and 2 for
  testing.
- **Training:** fine-tuned from pure BC (trained on the GA's games).
- **Agreement:** on the 2 test games, top-1 agreement with the human rose
  from 0.383 to 0.476 (+0.117 and +0.051 on the two games).
- **As an opponent it imitates a weaker player:**
  - garbage fitness 0.147, topping out in every garbage game;
  - in plain games it tops out in 15%, while every other afterstate agent
    survives them all.
- **Use:** this is the intended use, a personal sparring partner rather than
  a strong agent. With 2 test games, its agreement numbers are anecdotal
  (Section 5).

### 4.9 LLM agent (`reports/llm_agent.md`)

**What it is:** a novelty opponent, off by default.
- The model sees a text description of the board and a numbered list of
  legal placements, optionally with features (lines, holes, height after
  each placement), and answers with a number.
- On a timeout, an unusable answer or an error, the GA's move is played.

**The evaluation:**
- garbage setting, 100-piece cap, the first 20 held-out seeds;
- 15 s per move, so the evaluation measures the model rather than the
  fallback;
- the local models are 8-billion-parameter models quantized to 4 bits.

| Run | Pieces survived [95% CI] | Top-out rate | Lines/piece | Fallbacks | Seconds per piece |
|---|---|---|---|---|---|
| GA heuristic (reference) | 99.3 [97.9, 100.0] | 0.05 | 0.637 | | |
| qwen3:8b, features | 26.8 [25.1, 28.7] | 1.00 | 0.143 | 0% | 3.48 |
| llama3.1:8b, features | 23.9 [22.1, 25.8] | 1.00 | 0.092 | 0% | 3.54 |
| qwen3:8b, bare | 10.6 [10.2, 11.1] | 1.00 | 0.004 | 0% | 2.36 |
| gpt-4.1-mini, features and bare | Pending (no valid API key) | | | | |

- **Against the GA:** both models answer every move with a legal placement,
  and both top out in every game after about a quarter of the GA's pieces.
- **The features prompt matters:** qwen3 with features minus bare (both at
  15 s) is +16.2 [+14.4, +18.1] pieces. Without features, qwen3 almost never
  clears a line.
- **Between models:** qwen3 is slightly ahead of llama3.1 (+2.9 [+0.2,
  +5.8] pieces).
- **A correction:** an earlier bare run at 3 s looked better (13.2 pieces),
  but 22% of its moves were the GA's. Rerunning it at 15 s removed that
  confound.
- **Live play:** 8 s per move, with a matching speed cap of one piece per
  8 s. The bare run's per-move waits had a 90th percentile of 3.2 s and a
  maximum of 4.3 s.

## 5. Threats to validity

- **Single-seed main runs.** Each leaderboard agent comes from one training
  run.
  - The seed spread (3.3) shows 1-hour PPO runs ranging from 0.50 to 1.05,
    and DQN runs from 1.10 to 1.22.
  - Differences inside the 8-agent tie may reflect the seed rather than the
    method. The tie itself is the robust result; the order within it is not.
- **30 selection seeds.** Selection on 30 noisy games picks lucky
  candidates (4.6). The drop from selection to held-out was 0.04 to 0.12
  for most agents. Held-out comparisons are unbiased, but the chosen
  checkpoints are probably not the best ones each run produced.
- **Wall-clock budgets on one machine.**
  - Budgets are minutes on an M2 Max, not equal samples or FLOPs.
  - The GPU agents (DQN, PPO, BC, DT) and the CPU agents (GA, CEM, NEAT)
    convert a minute into very different amounts of experience.
  - The MPS bug cost the first PPO run all its learning, and the chunking
    fix slows the scorer.
- **Hard-drop-only action space.**
  - Tucks, spins and soft-drop slides are excluded, so no agent can use
    T-spins or fill an overhang.
  - The 8-way tie may be specific to this action space (4.1). Human players
    use moves the agents cannot make: 3 of the 1,400 recorded human
    placements were outside the action space.
- **Many comparisons.** The 55 pairwise bootstrap intervals are not
  corrected for multiple comparisons. Single marginal "significant" pairs,
  such as CEM over NEAT, should be read with that in mind.
- **2 test games for BC (you).** Its agreement gains (+0.117 and +0.051) come
  from 2 games of one person, with no confidence interval.
- **20 games for the LLMs**, with a 100-piece cap.
  - The intervals are wide, though the gap to the GA is far larger than any
    of them.
  - The local models are small (8B, 4-bit).
  - No frontier model was tested.
- **API runs pending.** The gpt-4.1-mini runs could not be made: the API key
  was rejected.
  - The spending ledger still counts $5.00 reserved for those rejected
    calls, which were not billed.
  - It waits for a working key and a ledger reset.
- **Seed overlaps outside evaluation.**
  - DT v1's last two trajectory blocks (720,000-720,044 and 721,000-721,049)
    reuse seeds from DT v2's first two blocks.
  - The DQN Q-bias diagnostic seeds (700,000-700,011) fall inside DT v1's
    first block.
  - No overlap touches the selection or held-out seeds, so no leaderboard
    number is affected.
- **Garbage that cannot be cancelled.** The evaluation garbage measures
  survival under a fixed load, not versus play. Battle-mode performance,
  where attacks cancel garbage, is not measured here.
- **Code provenance before pinning.** Queues before Phase 6b were not pinned
  to their code. An audit found one result that ran on uncommitted code, and
  it had already been replaced (`reports/queue_pinning_audit.md`). Some early
  runs did not record their commit (Section 6).

## 6. Reproduction

**Setup:**
- `pip install -e ".[web,dev]"`, Python 3.12, macOS with MPS for the deep
  agents.
- Training and evaluation commands write to `checkpoints/`, `data/` and
  `runs/` (not in git) and `reports/`.

**How queues run (from Phase 6b on):**
- Queue steps run through `scripts/pinned_step.py`. It refuses to run unless
  the `tetris` package and the commit are the pinned ones and the code is
  unedited.
- Each step logs the commit; the logs are in `logs/`.

**How to read the table:**
- Commit hashes are those on master after commit messages were rewritten
  on 2026-10-22, which changed every hash but no file. The queue logs in
  `logs/` print the hashes from before; `docs/commit_hash_map.txt` maps old
  to new. The tree hashes below are the same before and after.
- The last column gives the git tree hashes of `tetris/` and `scripts/` at
  the "Ran at" commit (`git rev-parse <commit>:tetris`, and the same for
  `scripts`). A tree hash depends only on file contents, not on commit
  messages, so it identifies the code even if commit hashes change when
  messages are rewritten. Equal hashes mean identical code: for example
  3cf7abd and e37ee78 have the same `tetris/` tree.
- "Ran at" is the commit whose code produced the result.
- "Recorded in" is the commit that added the report.
- "Not recorded" means the run predates commit logging. Its code is
  bracketed by the commits around it, but the exact commit is unknown.

| Result | Command | Ran at | Recorded in | Code trees at "Ran at" (`tetris/` / `scripts/`) |
|---|---|---|---|---|
| Leaderboard: garbage, plain, Marathon, bootstrap, sources (`leaderboard*`) | `python scripts/evaluate.py --games 500 --marathon-games 100` (Phase 6 queue, `scripts/run_phase6.py`) | 3ca1bde (the main copy, imported by the 14f166a queue; the audit found no relevant difference) | 9a1d9f9 | 3ca1bde: aa18e908697e / b2a1af5428e1 |
| GA, CEM checkpoints | `python -m tetris.agents.heuristic_ga.train`, `python -m tetris.agents.cem.train` | Not recorded (between 2cc4cea and 257ea76) | 257ea76 | n/a |
| DQN v1 checkpoint, `curve_dqn_seed0.png` | `python -m tetris.agents.dqn.train` | Not recorded (around 51c9115) | 543be7b | n/a |
| DQN v2, `curve_dqn_seed0_v2.png` | `python -m tetris.agents.dqn.train --run-name seed0_v2 --no-promote` (`scripts/run_phase4.py`) | 3cf7abd | 543be7b | 3cf7abd: b9d0d416b32b / 5fa3cefeaa53 |
| PPO (raw actions) and its tuning runs | `python -m tetris.agents.ppo.train --config tetris/agents/ppo/config_raw.yaml --ent-coef 0.03 --lr 0.0003` (`run_phase4.py`) | 3cf7abd | 543be7b | 3cf7abd: b9d0d416b32b / 5fa3cefeaa53 |
| PPO checkpoint, `curve_ppo_seed0.png` | `python -m tetris.agents.ppo.train` (`scripts/run_ppo_rerun.py`) | 9c4936e | 421d6b5 | 9c4936e: 056522eaedce / f09445abafc9 |
| Seed spread (`seed_spread.md/.json`, `curve_*_short*.png`) | DQN: `python -m tetris.agents.dqn.train --budget-minutes 60 --seed N --run-name shortN --no-promote` (`run_phase4.py`). PPO: the same in `run_ppo_rerun.py`. Then `python scripts/seed_spread.py` | DQN e37ee78; PPO and evaluation 9c4936e | 421d6b5 | e37ee78: b9d0d416b32b / f3aba7ee363f; 9c4936e: 056522eaedce / f09445abafc9 |
| `dqn_overestimation.md` | `python scripts/dqn_q_bias.py <run> <checkpoints>` and TensorBoard, written up by hand | Not recorded | f505488 (last edit) | n/a |
| `ppo_mps_bug.md`, `curve_ppo_fixsmoke.png` | Pre-flight check `python scripts/preflight.py`; 12-minute check run (command not recorded) | Fix in 9f3a432 | f505488 | n/a (fix: 9f3a432: 3c33379ef495 / f3aba7ee363f) |
| BC pure, BC + DAgger, BC (you) (`bc_you.json`, `curve_bc_*.png`) | `python -m tetris.agents.imitation.train pure`, `... dagger`, `... choose`, `... you` (`scripts/run_phase5.py`); GA logs from `python scripts/log_games.py` | 2c7cfca | 421d6b5 (`bc_you.json` reformatted in 8631ecf) | 2c7cfca: a094c6686162 / 2d5335945b65 |
| NEAT (`curve_neat_seed0.png`, `neat_topology.png`), `neat_selection.md` | `python -m tetris.agents.neuroevolution.train` (`run_phase5.py`); the selection analysis is by hand from `checkpoints/neat/selection.json` | 2c7cfca | 421d6b5, 8631ecf | 2c7cfca: a094c6686162 / 2d5335945b65 |
| DT v1 (`dt_calibration.md/.png`, `curve_dt_seed0.png`) | `python -m tetris.agents.decision_transformer.collect`, `... train train`, `... train calibrate` (`scripts/run_phase6.py`) | 14f166a | 9a1d9f9 | 14f166a: 5c300cb7361e / b2a1af5428e1 |
| DT v2 (`dt_calibration_v2.md/.png`, `curve_dt_seed0_v2.png`, `dt_v2.md`, `dt_versions.json`, `dt_v2_calibration.png`) | `scripts/run_phase6b.py`: collect, then `train --config tetris/agents/decision_transformer/config_v2.yaml --run-name seed0_v2 --no-promote`, `calibrate ... --report dt_calibration_v2`, `choose --run-name seed0_v2`, `python scripts/eval_dt_versions.py` | 68a5b2a (same tree as 471c9f9 on master; see note) | f07da18 | 68a5b2a = 471c9f9: f9f2166afc98 / 5a46a73f1ab6 |
| LLM: qwen3 and llama3.1 with features, API pending (`llm_agent.md/.json`) | `python scripts/eval_llm.py --mark-pending api-features,api-bare --reason "no valid API key"`, then `python scripts/eval_llm.py --runs qwen3-features,llama31-features` (`run_phase6b.py`) | 68a5b2a (same tree as 471c9f9) | f07da18 | 68a5b2a = 471c9f9: f9f2166afc98 / 5a46a73f1ab6 |
| LLM: qwen3 bare at 15 s, with the report regenerated | `python scripts/pinned_step.py scripts/eval_llm.py --runs qwen3-bare` | 21b0b65 | f8da418 | 21b0b65: f0b3a999c3b2 / 9f4e7f4415f8 |
| `difficulty.md` | `python scripts/eval_difficulty.py` | Not recorded (Phase 3) | 3f81d39 | n/a |
| `stress_test.md` | `python scripts/stress_test.py --mode race --seconds 60` (and `--mode battle`) | Not recorded (Phase 3) | 3f81d39 | n/a |
| `web_performance.md`, `ui/perf.json` | `LIVE=1 npx playwright test perf` in `web/`, and `python scripts/web_latency.py` | Not recorded (phase8-web branch) | 097fd81, fb62dbf | n/a |
| `queue_pinning_audit.md` | Manual audit of the queue logs and the session's edit history | n/a | 9f1a999 | n/a |

**Note on 68a5b2a.** The Phase 6b queue's worktree was created from a commit
whose message was then amended. 68a5b2a and 471c9f9 have identical trees
(`tetris/` f9f2166afc98, `scripts/` 5a46a73f1ab6, and the whole repository),
so they contain the same code. 68a5b2a is on no branch.
- The queue ran at 68a5b2a.
- Master holds 471c9f9, which has an identical tree (`git diff 68a5b2a
  471c9f9` is empty), so the code that produced those results is the code
  at 471c9f9.

**Determinism:**
- Evaluation is deterministic for a fixed checkpoint and seed.
- The Phase 6 leaderboard reproduced all 50 rows it shares with the Phase 5
  leaderboard exactly.
- DT v1's replay in `eval_dt_versions.py` reproduced its leaderboard rows.
- Training on MPS is not bit-for-bit reproducible, so rerunning a training
  command gives a different checkpoint; the seed spread (3.3) shows how
  different.
