# NEAT

NEAT evolves a small network that scores each afterstate. The agent plays the
placement with the highest score, like the GA and CEM agents.

## Same features, nonlinear function

GA and CEM score an afterstate as a weighted sum of 13 features
(`tetris/env/features.py`):

- aggregate height, max height;
- holes, rows with holes, hole depth;
- bumpiness, wells, max well depth;
- row transitions, column transitions;
- lines cleared, landing height, eroded cells.

NEAT receives exactly these 13 features and nothing else, so it is a
**nonlinear function of the same 13 features GA and CEM use**. The features
are divided by fixed scales so they fall roughly in [0, 1], the same scaling
the learned agents use.

Every initial genome connects all 13 inputs directly to one output with an
identity activation, so generation 0 is a population of linear evaluations.
Mutation can then add hidden nodes and connections, and change activations
to tanh, relu or sigmoid.

Whether NEAT beats CEM therefore measures what a nonlinear combination of
the same features adds, within the same evaluation budget. The leaderboard
reports it with a paired bootstrap against CEM.

## Fitness and seed protocol

Both are the same as for GA and CEM (`tetris/agents/linear_training.py`):

- **Setting:** the garbage setting (one direct garbage row every 3 pieces,
  500-piece cap).
- **Fitness:** mean pieces / cap + 0.25 x score per piece / 1000.
- **Seeds:** each generation, every genome plays the same 10 new training
  seeds (common random numbers).
- **Selection:** every 5 generations, that generation's best genome is
  evaluated on the 30 selection seeds. The best of those becomes
  `checkpoints/neat/best.json`.
- **Reporting:** held-out results (500 seeds) come from `scripts/evaluate.py`.

**Settings:**

- Population 150.
- `neat-python` 2.0, feed-forward, sum aggregation (see `neat_config.ini`).
- 3-hour wall-clock budget, CPU only, in parallel worker processes.

## Implementation notes

- **Evaluation:** `neat-python` evaluates a network one input row at a time.
  The agent scores about 35 to 70 afterstates per decision, so
  `network.py` evaluates the same network over all rows at once with NumPy.
  It reproduces neat-python's activation functions exactly; a test compares
  the two.
- **Storage:** genomes are stored as plain JSON (`best.json`), so playing
  needs neither neat-python nor a pickle. `state.pkl` in the run directory
  holds the population for `--resume`.
- **Topology:** `reports/neat_topology.png` is drawn with matplotlib, since
  graphviz is not installed. Inputs are labeled with feature names; hidden
  nodes are laid out by depth. Blue edges are positive weights, red are
  negative, and edge width shows |weight|.

## Train

```bash
python -m tetris.agents.neuroevolution.train                 # 3 hours, CPU
python -m tetris.agents.neuroevolution.train --resume
python -m tetris.agents.neuroevolution.train --budget-minutes 1 --pop-size 10 --games 2 \
    --run-name smoke --no-promote                              # quick check
```

## Results

Filled in after training.
