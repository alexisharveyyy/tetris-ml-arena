# LLM agent (novelty opponent)

A language model chooses each placement from a text description of the game.
It is a novelty: slow (seconds per piece) and much weaker than the trained
agents. It is **off by default** and never on the main leaderboard.

## How it plays

Each decision sends the model:

- the board as a text grid (`#` filled, `.` empty), with row and column numbers;
- the current piece, the hold slot and the next 5 pieces;
- a numbered list of every distinct legal placement (equivalent placements
  that give the same board are listed once).

Two prompt variants (`prompt` in `config.yaml`):

- `features` (default): each placement also gives the lines it clears, the
  holes and the stack height after it;
- `bare`: rotation and leftmost column only (an ablation).

The model answers `{"choice": N}`; backends request JSON output where they
support it. Anything else falls back to the GA heuristic's move:

| Outcome | When |
|---|---|
| `timeout` | No reply within `timeout_seconds` (3 s in the app; `eval_timeout_seconds`, 15 s, in the evaluation) |
| `invalid` | The reply is not JSON with a listed number |
| `error` | Connection or HTTP error, missing API key |
| `budget` | The API spending cap would be crossed (nothing is sent) |

## Backends

| Backend | Config | Notes |
|---|---|---|
| Ollama (default) | `ollama.model: qwen3:8b` | Local, free, runs on the GPU. `brew install ollama`, `ollama pull qwen3:8b`. Thinking is turned off per request; models without a thinking mode (llama3.1) are retried without the flag. |
| OpenAI API | `openai.model: gpt-4.1-mini`, key in `OPENAI_API_KEY` | Charged to the spending ledger. |
| Anthropic API | `anthropic.model: claude-haiku-5-5`, key in `ANTHROPIC_API_KEY` | Charged to the spending ledger. |

All three use only the standard library (`urllib`).

## Spending cap

API calls are charged to `data/llm/spend.json`, shared by all processes under
a file lock, with a hard cap of $5.00 in total (`spending.cap_usd`). Before
each call its worst-case cost is reserved: one token per prompt byte (an
upper bound for ASCII text; real prompts use about a third) plus the maximum
output. A call that could cross the cap is not sent. In the app the agent
then falls back to the heuristic; in an evaluation the run stops cleanly.

After the call the reservation is settled at the real cost from the reply's
token counts. A request the API rejects with an HTTP error response (for
example 401 for a bad key) is settled at $0, since rejected requests are not
billed. The worst case is kept only when no reply arrives (timeout, dropped
connection) or a reply has no readable token usage.

Prices per million tokens are in `config.yaml` with the date they were
checked. `scripts/eval_llm.py` refuses to call an API when that date is more
than 7 days old.

## Enable in the app

Set `enabled: true` in `config.yaml`; "LLM (novelty)" then appears as an
opponent. Start `ollama serve` first for the local backend. The worker allows
5 seconds per decision before its own fallback.

## Evaluate

```bash
python scripts/eval_llm.py --dry-run            # plan and API spending so far
python scripts/eval_llm.py                      # all runs, 20 games each
python scripts/eval_llm.py --runs qwen3-features,llama31-features
```

Runs: `qwen3-features`, `qwen3-bare`, `llama31-features`, `api-features`,
`api-bare`. Each plays the first 20 held-out seeds in the garbage setting
with a 100-piece cap, one game at a time; the GA plays the same seeds with
the same cap as the paired reference.

- **Timeout:** 15 s per decision (`eval_timeout_seconds`), so the evaluation
  measures the model rather than the fallback; the app keeps 3 s. Results
  are saved per timeout (`data/llm/eval/<run>_cap100_t15.jsonl`), and the
  report shows each run's timeout and marks comparisons across timeouts.
- **API runs** make one test call first and are skipped if it fails; a bad
  or missing key shows as "Pending (no valid API key)".
  `--mark-pending RUNS --reason TEXT` records a reason without calling.
- **Any run** stops after 5 backend errors in a row (the game in progress
  is discarded), so a dead backend cannot run up thousands of fallbacks.

**Main metrics: pieces survived, top-out rate and lines per piece**, each
with a 95% bootstrap interval over games and as paired differences from the
GA and between runs. Leaderboard fitness is not used: with a 100-piece cap
the GA reaches the cap in almost every game. The report also gives fallback
rates by cause, seconds per piece, tokens per game and cost, and says that
20 games gives wide intervals. The API model, gpt-4.1-mini, is a small API
model, not a frontier model. Results: `reports/llm_agent.md`.

## Results

Filled in after the evaluation.
