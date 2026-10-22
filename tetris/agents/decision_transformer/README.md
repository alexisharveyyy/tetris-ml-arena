# Decision Transformer

Offline RL as sequence modeling. The model is trained on games played by the
other agents and is conditioned on a **target return**: ask for a high return
and it should play like the strong agents; ask for a low one and it should
play like the weak ones. The target return is the agent's difficulty knob.

## Why afterstate matching

The original Decision Transformer predicts each action with a classifier
over a fixed action set. Here that set would be the 80 abstract placement
actions, and raw-action PPO already shows that learning those 80 actions from
the board fails in this game: it reached a held-out fitness of only 0.103,
against 1.280 to 1.345 for the seven top agents, all of which score afterstates. The Decision
Transformer therefore keeps the standard sequence structure but chooses
actions by **afterstate matching**:

- every legal placement is represented by an embedding of its afterstate
  (board, the 13 features, next piece, hold, level), from a small network
  like the one BC and PPO use to score afterstates;
- an action's logit is the scaled dot product of the transformer's output at
  the state token with that placement's embedding;
- the action token in the sequence is the embedding of the afterstate that
  was actually chosen.

The policy is still a distribution over the same 80-action space with illegal
actions masked. There is no 80-way variant.

## Sequence and model

Three tokens per piece, `(return-to-go, state, action)`, over the last 20
pieces (60 tokens), in a causal transformer (4 layers, 128 dimensions, 4
heads). Each token also gets a learned embedding of the piece index within the
game and of its token type. The state token encodes the board (CNN) and the
PPO observation extras (current piece, hold, next 5, board features, level).
The output at each state token predicts that piece's action.

**Return:** each piece earns `(1 + 0.25 * points / 1000) / 500`. A game's
return is therefore its leaderboard fitness whenever it reaches the 500-piece
cap (a close lower bound otherwise), so targets are in fitness units, from 0
to about 1.4.

**Loss:** as in BC, the negative log of the total probability of the actions
equivalent to the logged one (same afterstate), on every piece of every
window.

## Data

`collect.py` plays garbage-setting games (CPU) with:

| Source | Mistake rates | Games each |
|---|---|---|
| Dellacherie, GA, CEM, DQN, PPO, pure BC, NEAT | 0, 0.1, 0.3 | 45 |
| Raw-action PPO | 0 | 50 |

That is 995 games and roughly 300k to 400k states. Mistakes come from the
difficulty system (a plausible next-best placement), so weaker games look like
an agent playing badly rather than randomly. The spread of returns matters:
the model can only reproduce returns it has seen. Seeds start at 700,000, a
block of 1,000 per source, apart from every other seed range.

## Training and selection

- 2 hours on MPS, batches of 32 windows x 20 pieces (about 44k afterstates).
- AdamW, learning rate warmed up over 500 steps, then cosine-decayed from
  3e-4 to 3e-5.
- The MPS gradient pre-flight check runs on a real batch first, and the
  parameter-change watchdog runs throughout (`tetris/rl/safeguards.py`).
- Every 10 minutes a checkpoint plays the 30 selection seeds at the
  selection target (the 90th percentile of episode returns in the data); the
  best becomes `checkpoints/dt/best.pt`.

## Calibration and the difficulty knob

`train.py calibrate` plays the selected model at targets 0.2, 0.4, ... 1.4 on
the selection seeds and records the fitness it actually achieves
(`reports/dt_calibration.md`, `checkpoints/dt/calibration.json`), with the
rank correlation between target and achieved fitness. The target with the
highest achieved fitness becomes the default, used on the leaderboard.

In the app, the opponent screen shows a **Target return** row for this agent.
Each stop is labeled with its calibrated fitness ("1.00 (plays at about
1.05)"), not just the raw target. The usual speed, reaction-delay and
mistake settings still apply on top.

## Playing live

The agent keeps the current game's last 20 pieces and lowers the
return-to-go by each move's reward (read from the score change). Live games
have no cap, so every 500 pieces it starts a new segment (history cleared,
piece index and return reset). A new game is detected when the piece count
does not advance. The return-to-go is floored at one piece's survival reward.

## Run

```bash
python -m tetris.agents.decision_transformer.collect            # CPU, about 30 min
python -m tetris.agents.decision_transformer.train train        # 2 hours, MPS
python -m tetris.agents.decision_transformer.train calibrate    # about 10 min
python scripts/run_phase6.py                                    # all three, then the leaderboard
```

## Results

**Data.** 995 games, 303,011 states (2.5 minutes on 11 CPU workers). Episode
returns span 0.073 to 1.439 (median 0.790); the selection target (90th
percentile) was 1.417.

**Training.** 2 hours on MPS, about 10,800 steps. The gradient pre-flight check
passed (relative difference 6e-7). Validation top-1 agreement with the
logged action rose from 0.743 at 10 minutes to 0.771. Selection fitness over
the 12 checkpoints ranged from 1.255 to 1.374 with no trend after the first
20 minutes; the 90-minute checkpoint was selected (1.374 on the 30 selection
seeds).

**Held-out leaderboard** (500 garbage games, `reports/leaderboard.md`):
fitness 1.309, 4th of 11, top-out rate 0.16 [0.13, 0.19]. It is
statistically tied with CEM above (+0.009 [-0.022, +0.038]) and pure BC
below, and with GA, NEAT and PPO; only DQN is significantly ahead (+0.035
[+0.006, +0.065]). The drop from selection (1.374) to held-out (1.309) is the
same winner's-curse bias seen for NEAT (`reports/neat_selection.md`). In
the plain setting it reaches the 1000-piece cap in every game (1.196).

**The target return is not an effective difficulty knob.** Calibration
(`reports/dt_calibration.md`, 30 selection seeds per target):

| Target | 0.20 | 0.40 | 0.60 | 0.80 | 1.00 | 1.20 | 1.40 |
|---|---|---|---|---|---|---|---|
| Achieved fitness | 1.319 | 1.219 | 1.326 | 1.370 | 1.353 | 1.297 | 1.355 |

Rank correlation +0.43. Every target plays at the level of the strong
agents; the differences are about the size of 30-game noise. A likely
reason: the weak games in the data come from injected mistakes, which the
state does not predict, so the model learns to imitate the strong agents'
placements and largely ignores the return-to-go. The app labels each stop
with its calibrated fitness, so the knob is honest but nearly flat; the
usual speed, reaction-delay and mistake settings remain the way to weaken
this agent.
