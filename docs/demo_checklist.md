# Demo checklist

Steps for the class demo of the web UI. The demo runs entirely on this laptop:
the server, the agents (on the CPU) and the page. Nothing is fetched from the
internet, because the built frontend is committed in `tetris/web/static` and
every font is self-hosted.

## The day before (online)

1. Install, or update, the Python packages:

   ```bash
   pip install -e ".[web]"
   ```

   Node is not needed. The frontend build is committed.

2. Check that the committed build matches the source:

   ```bash
   python -m pytest tests/web/test_build.py
   ```

   If this fails, rebuild it (`cd web && npm install && npm run build`) and
   commit `tetris/web/static`. The launcher also checks this and prints a
   warning if it cannot rebuild.

3. Review the demo lineup in `configs/demo.yaml`: the opponents and their
   difficulty, the seed and the starting look. Defaults: BC (you) Medium, CEM
   Easy, PPO Medium, seed 48213, Tetromino look.

   - The Decision Transformer plays at its default target return; there is
     no target control, because its calibration showed the target does not
     change its strength (`reports/analysis/dt_v2.md`). Use the presets to change its
     strength, as for every other agent.
   - The LLM agent is off by default and is not needed for the demo. To show
     it, start Ollama, run `ollama pull qwen3:8b` while online, and set
     `enabled: true` in `tetris/agents/llm/config.yaml`. It takes up to 8 s
     per move and is capped at one piece per 8 s at every preset, so pick it
     for Watch AI or as a single slow opponent. Its board shows "GA
     fallbacks: n of m moves". Set `enabled: false` again afterwards.
   - Rehearse with the same `--checkpoint-dir` you will use in class, and
     check that every agent on the leaderboard screen shows its numbers. An
     agent listed without numbers has a checkpoint that changed or is missing
     since the leaderboard run; the screen says which.

4. Do a full rehearsal with Wi-Fi turned off:

   ```bash
   python -m tetris.web --demo
   ```

   - Home lists every lineup agent without "Not available yet". The server
     checks each checkpoint for a few seconds after it starts. The list
     updates by itself, so there is no need to reload.
   - Choose opponents shows the three demo opponents already in their slots.
   - Play a Marathon, press F9 to skip to the results, and press F10 to cycle
     through the four looks.

## At the projector

1. Set the display to 1920x1080 or 1280x720 (both are tested), and set browser
   zoom to 100% (Cmd+0).
2. Start the server with the command above. It opens the browser.
3. Go fullscreen: F11 on Windows and Linux. On a Mac, use Ctrl+Cmd+F or View >
   Enter Full Screen, because F11 shows the desktop there.
4. Function keys on a Mac: F9 and F10 need the fn key unless "Use F1, F2, etc.
   keys as standard function keys" is on, in System Settings > Keyboard.
5. Optional speed check: open `http://127.0.0.1:8765/?perf#/` and start a
   match. A readout in the bottom right shows the frame time (16.7 ms means
   60 fps) and the input-to-render latency. Reload without `?perf` before
   the class.
6. Close other heavy apps. Three agents on Unlimited keep three CPU worker
   processes busy. If the LLM agent is enabled, Ollama also uses the GPU and
   about 6 GB of memory while the model is loaded.

## Keys during the demo

| Key | Action |
|---|---|
| Arrows, X / Z, Space, C | Move, rotate, hard drop, hold (rebind in Settings) |
| Esc | Pause; the pause menu has Resume, End the match and Home |
| F9 | Demo mode: end the match and show the results |
| F10 | Demo mode: next look (Tetromino, Terminal, Broadcast, Poster) |
| Tab / Shift+Tab, Enter | Move between controls and choose one; the focus ring shows where you are |

## If the browser path fails

Use the Pygame app. It has the same agents, rules and checkpoints:

```bash
python -m tetris
```

The server log is the terminal it was started from. If the page shows "The
server is not answering", start the server again and reload the page. If a
match loses its connection, the server pauses it and the page shows
"Connection lost. Reconnecting...". The match resumes after a countdown once
the page reconnects.
