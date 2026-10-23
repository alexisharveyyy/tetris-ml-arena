# Stress test

`python scripts/tools/stress_test.py --mode race|battle --seconds 60`

Setup: Apple M2 Max (12 cores), headless SDL. The human plus 3 agents (CEM,
GA, Dellacherie) all on Unlimited, one worker process per agent, real-time
mode, 60 fps target with the app's frame pacer. The human is driven by
synthetic key events (12 moves/s) through the normal keyboard path.

| Mode | Pool startup | fps | Frame p50 | Frame p99 | Frame max | Frames > 16.7 ms | Input applied after frame start (p99) | Agent speed |
|---|---|---|---|---|---|---|---|---|
| Race | 0.34 s | 60.0 | 5.21 ms | 10.06 ms | 16.43 ms | 0 / 3600 | 0.163 ms | about 56 pieces/s each |
| Battle | 0.35 s | 60.0 | 5.41 ms | 10.14 ms | 15.38 ms | 0 / 3600 | 0.106 ms | about 56 pieces/s each |

Human input is handled at the start of each frame, before any agent work.
In real-time mode the controller never waits on a worker: it only polls.

Notes:

- Rare wall-clock spikes (about 0.1% of frames, up to about 18 ms in earlier
  runs) showed only about 3 ms of CPU work in profiles; the rest was the OS
  descheduling the main thread.
- `pygame.time.Clock.tick` oversleeps on macOS (18.6 ms per frame at a 60 fps
  target), so the app uses its own `FramePacer`.
