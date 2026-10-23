# Web UI performance

Measured on 2026-10-21 on this laptop (Apple silicon Mac), with Playwright's
headless Chromium against a local server (`python -m tetris.web --demo`).
Agents ran in CPU worker processes. No GPU was used.

**Setup:** you and three agents (BC (you), CEM and PPO), all on Unlimited, in
a Marathon. Each run lasted 45 s, with a key press every 150 ms (moves and
rotations; gravity drops the pieces, so you never top out during the window).
The agents placed about 57 pieces per second each.

Raw numbers: `reports/ui/perf.json`. To rerun:
`LIVE=1 TETRIS_CHECKPOINT_DIR=$PWD/../checkpoints npx playwright test perf`
in `web/` (it starts its own server, with settings and history in a temporary
directory), or `LIVE_URL=<url> npx playwright test perf` for a server you started.
A rerun writes `web/screenshots/perf.json`, which is not in git.

## Browser

Frame time is the interval between animation frames, so 16.7 ms means 60 fps.

| Look | Window | Frames | Frame time p50 | p99 | Max | Frames over 20 ms | Long tasks |
|---|---|---:|---:|---:|---:|---:|---:|
| Tetromino | 1920x1080 | 2,709 | 16.7 ms | 16.8 ms | 16.8 ms | 0 | 0 |
| Terminal | 1920x1080 | 2,703 | 16.7 ms | 16.8 ms | 16.8 ms | 0 | 0 |
| Broadcast | 1920x1080 | 2,706 | 16.7 ms | 16.8 ms | 16.8 ms | 0 | 0 |
| Poster | 1920x1080 | 2,704 | 16.7 ms | 16.8 ms | 16.8 ms | 0 | 0 |
| Tetromino | 1280x720 | 2,706 | 16.7 ms | 16.8 ms | 16.8 ms | 0 | 0 |

The page held 60 fps for every look at both projector sizes. No frame was
dropped and no task blocked the main thread for 50 ms or more.

## Input to render

Input to render runs from the key press to the paint of the board that
acknowledges it. It covers the wait for the server's next 60 Hz tick, the
WebSocket, the React render and the canvas draw. Display scan-out is not
included.

| Look | Window | Inputs | p50 | p99 |
|---|---|---:|---:|---:|
| Tetromino | 1920x1080 | 295 | 26.7 ms | 35.5 ms |
| Terminal | 1920x1080 | 295 | 24.3 ms | 33.9 ms |
| Broadcast | 1920x1080 | 296 | 24.4 ms | 33.7 ms |
| Poster | 1920x1080 | 296 | 24.7 ms | 33.6 ms |
| Tetromino | 1280x720 | 296 | 24.7 ms | 34.1 ms |

For comparison, the server alone (`scripts/web_latency.py`, same opponents on
Unlimited) measured 12.1 ms p50 and 19.5 ms p99 from input to the snapshot
that acknowledges it. The browser adds about one frame: the snapshot waits
for the next animation frame and is drawn inside it.

8d took one frame off this path. Snapshots used to render after the frame's
paint, so boards appeared a frame late (42 ms p50). They now render and draw
inside the animation frame (`flushSync` in `web/src/state/match.ts`).

## Messages

| | |
|---|---|
| Snapshots | 60 per second |
| Mean size | 3.35 KB (four boards, candidates and the event feed) |
| Bandwidth | about 197 KB/s on localhost |
| Server-side p99 size | 3.65 KB, max 3.82 KB (`scripts/web_latency.py`) |

## Checking the demo machine

Open `http://127.0.0.1:8765/?perf#/` and start a match. A readout in the
bottom right shows frame time p50 and p99, input to render, long tasks and
message size, over the last 600 frames. Headless Chromium on this laptop is
the best case; a projector laptop with a slower GPU should still show 16.7 ms
at p50.
