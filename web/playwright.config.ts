import { defineConfig } from "@playwright/test";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// Screenshot and audit runs use the production build (npm run build), with
// every API call and the WebSocket mocked from tests/screens/fixtures.ts.
//
// The live specs (keyboard, live, perf) need a real server. LIVE=1 starts one
// on port 8799 in demo mode, keeping settings, history and recordings in a new
// temporary directory (--state-dir), and passes it to the specs as LIVE_URL:
//   LIVE=1 npx playwright test keyboard
// (TETRIS_CHECKPOINT_DIR=<dir> uses another checkout's trained checkpoints.)
// LIVE_URL=<url> runs them against a server you started instead (for example
// the demo machine). Either way, tests/state-guard.ts fails the run if the
// real user state files change.
const LIVE_PORT = 8799;
const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const servers = [
  {
    command: "npx vite preview --host 127.0.0.1 --port 4173 --strictPort",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: true,
  },
];
if (process.env.LIVE && !process.env.LIVE_URL) {
  // Workers load this file again; they inherit the directory made here.
  process.env.ARENA_TEST_STATE_DIR ??= mkdtempSync(join(tmpdir(), "tetris-playwright-state-"));
  const state = process.env.ARENA_TEST_STATE_DIR;
  const venv = join(ROOT, ".venv", "bin", "python");
  const python = process.env.PYTHON ?? (existsSync(venv) ? venv : "python");
  servers.push({
    command: `${python} -m tetris.web --no-browser --demo --port ${LIVE_PORT} --state-dir ${state}`,
    url: `http://127.0.0.1:${LIVE_PORT}/api/agents`,
    // Never reuse a server already on the port: its state may be the real one.
    reuseExistingServer: false,
    cwd: ROOT,
    env: { ...process.env, TETRIS_STATE_DIR: state },
    timeout: 120_000,
  } as (typeof servers)[number]);
  process.env.LIVE_URL = `http://127.0.0.1:${LIVE_PORT}`;
}

export default defineConfig({
  testDir: "tests/screens",
  globalSetup: "./tests/state-guard.ts",
  workers: 1,
  timeout: 30_000,
  reporter: [["list"]],
  use: { browserName: "chromium", viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 },
  webServer: servers,
});
