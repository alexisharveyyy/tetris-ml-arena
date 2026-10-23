# Reports

Every result in this project, grouped by what it is about. Start with
[FINAL_REPORT.md](FINAL_REPORT.md): it gives the setup, the evaluation
protocol, all results with confidence intervals, the findings and how to
reproduce each result (Section 6).

## Leaderboard (`leaderboard/`)

| File | What it shows |
|---|---|
| [leaderboard.md](leaderboard/leaderboard.md) | The main leaderboard: every agent on the 500 held-out seeds in the garbage and plain settings, Marathon by difficulty preset, and the paired bootstrap between adjacent ranks |
| [leaderboard_garbage.csv](leaderboard/leaderboard_garbage.csv), [leaderboard_plain.csv](leaderboard/leaderboard_plain.csv) | One row per agent and setting: fitness, pieces, lines, top-out rate with its Wilson interval |
| [leaderboard_marathon.csv](leaderboard/leaderboard_marathon.csv) | Marathon results per agent and difficulty preset |
| [leaderboard_bootstrap.csv](leaderboard/leaderboard_bootstrap.csv) | Paired bootstrap differences between agents, per metric |
| [leaderboard_sources.json](leaderboard/leaderboard_sources.json) | The checkpoint (path and SHA-256) behind each leaderboard result; the web leaderboard marks a result "pending rerun" when it changes |

The web leaderboard screen reads this folder.

## Training (`training/`)

| File | What it shows |
|---|---|
| [ppo_mps_bug.md](training/ppo_mps_bug.md) | Why the first afterstate PPO runs never trained (wrong gradients on the Apple GPU for large batches), the fix, and the rerun |
| [dqn_overestimation.md](training/dqn_overestimation.md) | DQN v1's decline after 25 minutes, diagnosed as value overestimation, and the v2 stability fix |
| [seed_spread.md](training/seed_spread.md) ([.json](training/seed_spread.json)) | Seed variance of DQN and PPO across three 1-hour runs each |
| [queue_pinning_audit.md](training/queue_pinning_audit.md) | Which code each unattended queue step actually ran, and the pinning check added afterwards |
| [bc_you.json](training/bc_you.json) | BC (you): fine-tuning on the human's recorded games, top-1 and top-3 accuracy against the GA-trained baseline |
| [neat_topology.png](training/neat_topology.png) | The selected NEAT network |
| [curves/](training/curves/) | Learning curves on the selection seeds, one per run (`curve_<agent>_<run>.png`), written by the trainers |

The curves:
- `curve_dqn_seed0.png` (v1), `curve_dqn_seed0_v2.png` (v2) and
  `curve_dqn_v1_v2.png` (both on one plot);
- `curve_ppo_seed0.png` (afterstate PPO, after the fix) and
  `curve_ppo_fixsmoke.png` (the check run that confirmed the fix);
- `curve_ppo_raw_seed0.png` (raw-action PPO) and its two tuning runs,
  `curve_ppo_raw_tuneA.png` and `curve_ppo_raw_tuneB.png`;
- `curve_dqn_short{0,1,2}.png` and `curve_ppo_short{0,1,2}.png` (the seed
  spread runs);
- `curve_bc_pure_seed0.png`, `curve_bc_dagger_seed0.png`,
  `curve_neat_seed0.png`, `curve_dt_seed0.png` (DT v1) and
  `curve_dt_seed0_v2.png` (DT v2).

## Analysis (`analysis/`)

| File | What it shows |
|---|---|
| [difficulty.md](analysis/difficulty.md) | What each difficulty preset does to an agent's play in real time |
| [neat_selection.md](analysis/neat_selection.md) | NEAT's drop from selection to held-out seeds, explained as winner's-curse bias |
| [dt_calibration.md](analysis/dt_calibration.md) ([.png](analysis/dt_calibration.png)) | DT v1: achieved fitness against the target return it was asked for |
| [dt_calibration_v2.md](analysis/dt_calibration_v2.md) ([.png](analysis/dt_calibration_v2.png)) | The same for DT v2 |
| [dt_v2.md](analysis/dt_v2.md) | DT v1 against v2 on held-out seeds: the hypothesis behind v2, and why it was rejected |
| [dt_v1_vs_v2_calibration.png](analysis/dt_v1_vs_v2_calibration.png) | Both calibration curves on one plot |
| [dt_versions.json](analysis/dt_versions.json) | The numbers behind `dt_v2.md` |

## LLM agent (`llm/`)

| File | What it shows |
|---|---|
| [llm_agent.md](llm/llm_agent.md) ([.json](llm/llm_agent.json)) | The LLM novelty opponent: local models with and without board features, fallback rates, per-move wait times |

## User interface (`ui/`)

| File | What it shows |
|---|---|
| [web_performance.md](ui/web_performance.md) ([perf.json](ui/perf.json)) | Web UI frame times, input-to-render latency and message size in every look |
| [stress_test.md](ui/stress_test.md) | Desktop frame times with a human and three Unlimited agents |

The UI screenshots are not kept here: `npx playwright test screens` in
`web/` writes every screen in every look to `web/screenshots/` (not in git).
The README uses five of them, in `docs/screenshots/`.
