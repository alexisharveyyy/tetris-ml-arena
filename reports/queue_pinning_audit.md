# Queue pinning audit (Phases 4 to 6)

The Phase 6 queue ran from a worktree pinned at 14f166a, but its two
`scripts/*.py` steps (leaderboard and LLM evaluation) imported the main
copy's code: the venv's editable install points at the main copy, and a
script's own directory, not the worktree root, is first on its import path.
`-m` steps were not affected (the working directory comes first). This audit
checks every `scripts/*.py` step of the earlier queues.

**Method.** Step times come from each queue's `logs/phase4_queue.log`; the main
copy's commit at each time from `git reflog master`; uncommitted edits from
the timestamps of every tool call in the session transcript that wrote to
`tetris/` or `scripts/` (this one session covers every queue; edits made
outside it would not show). How each queue was launched also comes from the
transcript.

| Queue | Step (local time) | Code imported | Main copy then | Differs from the queue's commit? | Current result from it? |
|---|---|---|---|---|---|
| Phase 4 (ran in the main copy, relaunched at e37ee78) | leaderboard, Oct 20 03:40-04:00 | main copy | e37ee78, no uncommitted edits | No | No: replaced by later full leaderboards |
| Phase 4 | seed_spread, Oct 20 11:39-11:47 | main copy | e37ee78 plus uncommitted edits from 11:00-11:02 (afterstate scorer chunking in `tetris/rl/nets.py`, `ppo/train.py`, `marathon_eval.py`, `controller.py`, `evaluate.py`; committed at 12:19 as 9f3a432, 543be7b) | **Yes**, and `nets.py` is loaded (DQN and PPO scoring) | No: rerun by the PPO rerun queue; `reports/seed_spread.md` is that rerun's output (byte-identical) |
| PPO rerun (worktree at 9c4936e, launched with `PYTHONPATH` = worktree) | leaderboard, Oct 20 22:04-22:45 | worktree | 2c7cfca (not imported) | No | No: replaced by later full leaderboards |
| PPO rerun | seed_spread, Oct 20 22:45-22:58 | worktree | 2c7cfca (not imported) | No | Yes: `reports/seed_spread.md` |
| Phase 5 (worktree at 2c7cfca, launched with `PYTHONPATH` = worktree) | leaderboard, Oct 21 03:03-04:36 | worktree | 2c7cfca, no uncommitted edits | No | No: replaced by the Phase 6 leaderboard |
| Phase 6 (worktree at 14f166a, launched without `PYTHONPATH`) | leaderboard, Oct 21 15:15-18:08 | **main copy** | 3ca1bde: only `tetris/agents/checkpoint_check.py` differs from 14f166a, and neither script loads it; no uncommitted edits | No (in effect) | Yes: `reports/leaderboard*` |
| Phase 6 | LLM evaluation, Oct 21 18:08-21:19 | **main copy** | 3ca1bde, as above | No (in effect) | No: invalid (rejected API key); being rerun |

Both worktrees are still clean under `tetris/` and `scripts/`.

**Cross-check.** The Phase 6 leaderboard reproduced every row the Phase 5
leaderboard had in common with it exactly: 10 garbage rows, 10 plain rows
and 30 Marathon rows, all columns identical.

**Conclusion.** Code different from the queue's commit was loaded once (Phase
4's first seed spread), and that result was replaced by a correctly pinned
rerun. No current result is affected; nothing needs rerunning.

**Fix.** Queue steps now run through `scripts/pinned_step.py` with the
worktree first on `PYTHONPATH`. Each step logs the path `tetris` was imported
from and the commit, and aborts unless that path is the pinned worktree, its
HEAD is the queue's commit and nothing under `tetris/` or `scripts/` has
changed (`tests/test_pinned_step.py`).
