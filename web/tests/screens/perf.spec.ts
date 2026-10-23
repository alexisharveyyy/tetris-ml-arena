// Performance run against a running server (no mocks): the human and three
// agents on Unlimited, with key presses throughout. Records browser frame
// times, input-to-render latency, long tasks and message size from the ?perf
// probe (src/lib/perf.ts) into web/screenshots/perf.json, which is not in git
// (the recorded run is reports/ui/perf.json).
//   LIVE=1 npx playwright test perf
//   (AGENTS=bc_you,cem,ppo SECONDS=45 RUNS=tetromino@1920x1080,...)
import { expect, test } from "@playwright/test";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const LIVE = process.env.LIVE_URL;
const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "screenshots");
const REPORT = join(OUT, "perf.json");
const AGENTS = (process.env.AGENTS ?? "bc_you,cem,ppo").split(",");
const SECONDS = Number(process.env.SECONDS ?? 45);
const RUNS = (
  process.env.RUNS ??
  "tetromino@1920x1080,terminal@1920x1080,broadcast@1920x1080,poster@1920x1080,tetromino@1280x720"
).split(",");

test.skip(!LIVE, "set LIVE=1 to start a server with temporary state, or LIVE_URL");
test.setTimeout((SECONDS + 60) * 1000);

const pct = (v: number[], q: number) => {
  if (!v.length) return null;
  const s = [...v].sort((a, b) => a - b);
  return Math.round(s[Math.min(s.length - 1, Math.floor(q * s.length))] * 100) / 100;
};

for (const run of RUNS) {
  const [look, size] = run.split("@");
  const [w, h] = size.split("x").map(Number);
  test(`perf ${look} ${size}`, async ({ page, request }) => {
    await page.setViewportSize({ width: w, height: h });
    const res = await request.post(`${LIVE}/api/matches`, {
      data: { mode: "marathon", opponents: AGENTS.map((agent) => ({ agent, preset: "Unlimited" })) },
    });
    expect(res.status(), await res.text()).toBe(201);
    const info = await res.json();
    await page.addInitScript((l) => {
      localStorage.setItem("arena.look", l);
      sessionStorage.setItem("arena.demo.started", "1");
    }, look);
    await page.goto(`${LIVE}/?perf#/match/${info.match}`);
    await expect(page.locator("canvas.board").first()).toBeVisible();
    // Countdown and model loading, then start counting from zero.
    await page.waitForTimeout(6000);
    await page.evaluate(() => (window as any).__arenaPerf.reset());

    // Taps and rotations every 150 ms; gravity does the drops, so the human
    // stays alive (a topped-out human ends the Marathon) for the whole window.
    const end = Date.now() + SECONDS * 1000;
    let i = 0;
    while (Date.now() < end) {
      await page.keyboard.press(i % 5 === 4 ? "ArrowUp" : i % 2 ? "ArrowLeft" : "ArrowRight");
      await page.waitForTimeout(150);
      i += 1;
    }
    await expect(page).toHaveURL(/#\/match\//);

    const stats = await page.evaluate(() => {
      const s = (window as any).__arenaPerf.stats;
      return { elapsed: (performance.now() - s.since) / 1000, players: s.players, frames: [...s.frames], inputToRender: [...s.inputToRender], longTasks: s.longTasks, bytes: s.bytes, messages: s.messages };
    });
    await request.delete(`${LIVE}/api/matches/${info.match}`);

    const frames = stats.frames;
    const mean = frames.reduce((a, b) => a + b, 0) / Math.max(1, frames.length);
    const result = {
      look,
      viewport: size,
      agents: AGENTS.map((a) => `${a} Unlimited`),
      seconds: Math.round(stats.elapsed * 10) / 10,
      frames: frames.length,
      fps: Math.round((1000 / mean) * 10) / 10,
      frame_ms: { p50: pct(frames, 0.5), p99: pct(frames, 0.99), max: pct(frames, 1) },
      frames_over_20ms: frames.filter((f) => f > 20).length,
      input_to_render_ms: { n: stats.inputToRender.length, p50: pct(stats.inputToRender, 0.5), p99: pct(stats.inputToRender, 0.99) },
      long_tasks: stats.longTasks,
      snapshots_per_s: Math.round((stats.messages / stats.elapsed) * 10) / 10,
      bytes_per_message: stats.messages ? Math.round(stats.bytes / stats.messages) : null,
      kb_per_s: Math.round(stats.bytes / stats.elapsed / 102.4) / 10,
      players: stats.players,
    };
    mkdirSync(OUT, { recursive: true });
    let all: Record<string, unknown> = {};
    try {
      all = JSON.parse(readFileSync(REPORT, "utf8"));
    } catch {
      /* first run */
    }
    all[`${look} ${size}`] = result;
    writeFileSync(REPORT, JSON.stringify(all, null, 1) + "\n");
    expect(stats.inputToRender.length).toBeGreaterThan(20);
    // The frame loop ran for the whole window (the match did not end early).
    expect(frames.length).toBeGreaterThan(stats.elapsed * 55);
  });
}
