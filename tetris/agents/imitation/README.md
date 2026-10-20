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
  - their paired difference, with a 95% bootstrap interval that resamples
    whole test games.

  An interval that includes 0 means fine-tuning on your play made no
  measurable difference.

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
tensorboard --logdir runs/bc_pure
```

## Results

Filled in after training.
