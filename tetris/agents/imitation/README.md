# Imitation learning

Behavior cloning (BC) over the masked 80-action placement space. There are
three trained variants and one playable alias:

| Agent | Key | Data | Role |
|---|---|---|---|
| Imitation (pure BC) | `bc_pure` | Logged GA games | Reported on the leaderboard |
| Imitation (BC + DAgger) | `bc_dagger` | GA games plus two DAgger rounds | Reported on the leaderboard |
| Imitation (BC) | `bc` | Copy of whichever variant above scores higher on the selection seeds | Playable |
| BC (you) | `bc_you` | Your recorded games, fine-tuned from `bc` | Playable, once you have 800 on-space placements |

`checkpoints/bc/version_choice.json` records which variant `bc` is.

## Policy

The network is the PPO afterstate policy (`AfterstateActorCritic` in
`tetris/rl/nets.py`):

- each legal action's logit is a shared network's score of that action's
  afterstate;
- the inputs are the board, the 13 features, the next piece, the hold slot
  after the placement and the level;
- illegal actions get -1e9.

Only the scorer is trained; the critic is unused. Two consequences:

- BC checkpoints load exactly like PPO checkpoints.
- A BC model can warm-start PPO.

## Loss and equivalent actions

Different actions can produce the same afterstate:

- the O piece in any rotation;
- the horizontal I piece in rotations 0 and 2;
- holding when the held piece equals the current one.

The loss is the negative log of the total probability of all actions whose
afterstate equals the teacher's. A choice the teacher could equally have made
is therefore not penalized.

Agreement metrics count distinct afterstates for the same reason. A state is
a top-k hit when fewer than k distinct non-teacher afterstates score higher
than the teacher's.

## Data

- **GA logs:** `data/logs/ga`, written by `scripts/log_games.py`. There are
  100 garbage-setting and 100 plain-setting games, about 146k states with
  about 69 legal actions each.
- **Splits:** 10% of the games are held out for agreement metrics.
- **Cache:** encoded datasets are cached in `data/imitation/`.

## Pure BC

Pure BC trains on the GA games for 30 minutes on MPS:

- Batches of 256 states, about 17.6k afterstates per batch.
- AdamW, learning rate cosine-decayed from 3e-4 to 3e-5.
- Every 3 minutes a checkpoint is evaluated on the 30 selection seeds (garbage
  setting). The best checkpoint becomes `checkpoints/bc_pure/best.pt`.

## DAgger

Plain BC only sees states the GA visits. Once the cloned policy makes a
mistake, it reaches boards the GA never produced, and errors compound.

DAgger starts from the pure BC model and runs 2 rounds. Each round:

1. The current policy plays 50 garbage and 50 plain games on new seeds.
2. The GA labels every state the policy visited
   (`tetris/agents/imitation/dagger.py`).
3. Training continues for 10 minutes on the GA games plus all DAgger data.

The best checkpoint across both rounds, chosen on the selection seeds,
becomes `checkpoints/bc_dagger/best.pt`.

## BC (you)

Record games with **Record Gameplay** in the main menu. Plain is the default
setting; garbage is also available.

- **Off-space steps:** each locked piece is matched to the hard-drop action
  with the same cells. Tucks, spins and slides have no equivalent in the
  80-action space; they are saved as off-space and not used.
- **When it trains:** once there are at least 800 on-space placements in at
  least 3 games.
- **Starting point:** the GA-trained BC model (`bc`).
- **Split by game, in recording order:**
  - the most recent 25% of games are the test set;
  - the 15% before them are the validation set;
  - the rest are training data.
- **Learning rate:** 3e-5, ten times below BC, cosine-decayed to 3e-6.
- **Early stopping:**
  - top-1 agreement on the validation games is checked every 30 seconds;
  - the first check is the GA-trained model itself, before any update;
  - training stops after 6 checks without improvement, or after 10 minutes;
  - the best validation checkpoint is kept, so BC (you) can be exactly the
    GA-trained model if fine-tuning never helps.
- **Report** (`reports/bc_you.json`): top-1 and top-3 agreement with your test
  games for:
  - the baseline (the GA-trained BC model);
  - the fine-tuned model;
  - their difference, overall and on each test game.

  There is no interval. Placements within a game are correlated, so only
  whole games are independent samples, and a bootstrap over placements would
  be falsely tight. With a handful of test games, no reliable interval is
  possible; the per-game differences show whether the change is consistent.

## Safeguards

Every run on MPS starts with the gradient pre-flight check: MPS and CPU
gradients are compared at the real batch size. Training runs under the
parameter-change watchdog (`tetris/rl/safeguards.py`).

## Train

```bash
python -m tetris.agents.imitation.train pure     # 30 minutes
python -m tetris.agents.imitation.train dagger   # 2 rounds, about 25 minutes
python -m tetris.agents.imitation.train choose   # sets the playable bc agent
python -m tetris.agents.imitation.train you      # after recording 800 placements (3+ games)
python -m tetris.agents.imitation.train pure --resume   # continue an interrupted run
tensorboard --logdir runs/bc_pure
```

`--resume` works for `pure`, `dagger` and `you`. The trainer saves
`state.pt` in the run directory at every check and at the end of each phase
(weights, optimizer, elapsed time, batch order, RNG states, selection
evaluations, and for BC (you) the early-stopping state and baseline). An
interrupted phase continues where it stopped; finished phases, including
DAgger rounds whose games were already collected, are skipped.

## Results

Held-out garbage setting, 500 seeds (`reports/leaderboard.md`):

| Agent | Selection fitness | Held-out fitness | Top-out rate | GA top-1 / top-3 agreement |
|---|---|---|---|---|
| Imitation (pure BC) | 1.358 | 1.296 | 0.20 | 0.951 / 1.000 |
| Imitation (BC + DAgger) | 1.343 | 1.280 | 0.21 | 0.963 / 1.000 |
| GA heuristic (teacher) | | 1.289 | 0.18 | |

- Agreement is measured on the held-out GA games.
- Pure BC minus DAgger: +0.017 [-0.017, +0.049], a tie. DAgger raised
  agreement with the GA slightly but did not improve play; pure BC already
  matches its teacher (pure BC minus GA +0.007 [-0.026, +0.040]).
- The playable `bc` agent is pure BC (higher selection fitness).

**BC (you)**, from 7 recorded plain games (1,397 on-space placements), split
4 train / 1 validation / 2 test games (`reports/bc_you.json`):

| Model | Top-1 on your test games | Top-3 |
|---|---|---|
| GA-trained BC (baseline) | 0.383 | 0.742 |
| BC (you), selected at 6 min | 0.476 | 0.785 |
| Difference | +0.093 | +0.043 |

| Test game | Placements | Top-1 difference | Top-3 difference |
|---|---|---|---|
| 6th recorded | 239 | +0.117 (0.356 to 0.473) | +0.059 (0.741 to 0.799) |
| 7th recorded | 137 | +0.051 (0.431 to 0.482) | +0.015 (0.745 to 0.759) |

Fine-tuning raised agreement on both test games. With 2 test games, no
reliable interval is possible, so this is a point estimate only.

As an opponent, BC (you) is much weaker than the GA-trained
model: held-out garbage fitness 0.147 (top-out rate 1.00), plain-setting
top-out rate 0.15. It was trained only on plain games.
