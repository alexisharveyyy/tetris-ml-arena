# Scripts

Command-line tools, grouped by purpose. Run them from the repository root, for
example `python scripts/eval/evaluate.py`. Each script's docstring gives its
options. Training itself is per agent: `python -m tetris.agents.<agent>.train`
(see each agent's README).

## `train/`: before and around training

| Script | What it does |
|---|---|
| `preflight.py` | Compares MPS and CPU gradients for the learned agents at their real batch sizes; every GPU run starts with this check |
| `bench_device.py` | Times MPS against CPU for the DQN and PPO workloads; the agent configs cite it for their device choice |
| `log_games.py` | Logs an agent's games, the training data for imitation learning and the Decision Transformer |

## `eval/`: results

| Script | What it does | Writes |
|---|---|---|
| `evaluate.py` | The leaderboard: every agent on the held-out seeds, Marathon, paired bootstrap, per-game results and sources | `reports/leaderboard/` |
| `check_leaderboard_rerun.py` | Compares a leaderboard rerun with the previous one, row by row | `reports/leaderboard/leaderboard_rerun_check.md` |
| `eval_difficulty.py` | Measures the difficulty presets in real-time play | `reports/analysis/difficulty.md` |
| `eval_dt_versions.py` | Decision Transformer v1 against v2 | `reports/analysis/dt_v2.md` and its figure and JSON |
| `eval_llm.py` | The LLM agent: backends and prompt variants on the same seeds | `reports/llm/` |
| `seed_spread.py` | Seed variance of the 1-hour DQN and PPO runs | `reports/training/seed_spread.md/.json` |

## `analysis/`: one-off diagnostics behind a report

| Script | What it does |
|---|---|
| `dqn_q_bias.py` | Offline Q calibration of DQN checkpoints (`reports/training/dqn_overestimation.md`) |
| `backfill_sources.py` | Infers `reports/leaderboard/leaderboard_sources.json` entries for leaderboards written before `evaluate.py` recorded them |

## `queue/`: the unattended runs

The long training and evaluation runs went through these queues, one step at
a time, from a git worktree of one commit. They are kept as the record of how
the results were produced. `reports/training/queue_pinning_audit.md` gives
which code each step ran, and `docs/architecture.md` explains the design.

| Script | What it ran |
|---|---|
| `run_phase4.py` | DQN v2, raw-action PPO and its tuning runs, the first afterstate PPO run (before the MPS fix), the leaderboard and the seed runs; also the shared queue helper |
| `run_ppo_rerun.py` | Afterstate PPO after the MPS gradient fix, and its seed runs |
| `run_phase5.py` | BC, DAgger, BC (you), NEAT and the full leaderboard |
| `run_phase6.py` | Decision Transformer v1 and the leaderboard |
| `run_phase6b.py` | Decision Transformer v2 and the local LLM runs |
| `pinned_step.py` | Starts each step, and refuses unless the step will run the pinned commit's unchanged code |

## `tools/`: checks of the running game

| Script | What it does |
|---|---|
| `play_path_check.py` | Every learned agent in live play: worker startup, decision latency, every mode, fallback |
| `stress_test.py` | Desktop frame times with a human and three Unlimited agents (`reports/ui/stress_test.md`) |
| `web_latency.py` | Web protocol message size and input-to-snapshot latency on a running server (`reports/ui/web_performance.md`) |

## Old paths

Until the reorganization every script sat directly in `scripts/`. The
commands in `reports/FINAL_REPORT.md` Section 6, the queue logs in `logs/`
and the checkpoint notes use those old paths, which are correct at the
commits they cite. In the current tree, `scripts/<name>.py` is:

| Old | Now |
|---|---|
| `scripts/preflight.py`, `bench_device.py`, `log_games.py` | `scripts/train/` |
| `scripts/evaluate.py`, `check_leaderboard_rerun.py`, `eval_difficulty.py`, `eval_dt_versions.py`, `eval_llm.py`, `seed_spread.py` | `scripts/eval/` |
| `scripts/dqn_q_bias.py`, `backfill_sources.py` | `scripts/analysis/` |
| `scripts/pinned_step.py`, `run_phase4.py`, `run_phase5.py`, `run_phase6.py`, `run_phase6b.py`, `run_ppo_rerun.py` | `scripts/queue/` |
| `scripts/play_path_check.py`, `stress_test.py`, `web_latency.py` | `scripts/tools/` |
| `scripts/bench_env.py` | Removed: a one-off environment benchmark that no report used |
