// Checks of the built site in a real browser, at desktop and phone sizes.
// SITE_URL picks the target (the live site, or a local `npm run preview`):
//   SITE_URL=https://alexisalineharvey.com/tetris-ml-arena/ npx playwright test
// Screenshots and a JSON summary go to SHOTS_DIR (default test-results/shots).
import { expect, test, type Page } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const BASE = process.env.SITE_URL ?? "http://localhost:4180/tetris-ml-arena/";
const SHOTS = process.env.SHOTS_DIR ?? join("test-results", "shots");
mkdirSync(SHOTS, { recursive: true });

const ALL_AGENTS = ["dellacherie", "ga", "cem", "neat", "dqn", "ppo", "ppo_raw", "bc", "bc_dagger", "bc_you", "dt"];
const LOOKS = ["tetromino", "terminal", "broadcast", "poster"];

function watchConsole(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error" || m.type() === "warning") errors.push(`${m.type()}: ${m.text()}`);
  });
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("requestfailed", (r) => errors.push(`requestfailed: ${r.url()} ${r.failure()?.errorText}`));
  return errors;
}

async function setup(page: Page, look: string, slots: { agent: string; preset: string }[]) {
  await page.goto(BASE);
  await page.evaluate(
    ([look, slots]) => {
      localStorage.clear();
      localStorage.setItem("arena.look", look as string);
      localStorage.setItem("arena.settings", JSON.stringify({ look, last_match: { marathon: { slots } } }));
    },
    [look, slots] as const,
  );
}

async function startMarathon(page: Page, perf = false) {
  await page.goto(`${BASE}${perf ? "?perf" : ""}#/play/marathon`);
  await page.reload();
  await page.getByRole("button", { name: /start/i }).last().click();
  await page.waitForURL(/#\/match\//);
}

/** Pieces placed per player, from the perf probe (needs ?perf). */
async function pieces(page: Page): Promise<Record<string, number>> {
  return page.evaluate(() => {
    const s = (window as unknown as { __arenaPerf: { stats: { players: { name: string; pieces: number }[] } } }).__arenaPerf.stats;
    return Object.fromEntries(s.players.map((p) => [p.name, p.pieces]));
  });
}

async function noSideways(page: Page): Promise<void> {
  const r = await page.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: window.innerWidth }));
  expect(r.sw, "page scrolls sideways").toBeLessThanOrEqual(r.iw);
  expect(r.iw, "layout viewport widened (page zoomed out)").toBeLessThanOrEqual(page.viewportSize()!.width);
}

const summary: Record<string, unknown> = {};
test.afterAll(() => writeFileSync(join(SHOTS, `summary-${test.info().project.name}.json`), JSON.stringify(summary, null, 1)));

test.describe.configure({ mode: "serial" });

test("home, About and leaderboard load without console errors, in every look", async ({ page }, info) => {
  const errors = watchConsole(page);
  for (const look of LOOKS) {
    await setup(page, look, [{ agent: "ga", preset: "Medium" }]);
    await page.reload();
    await expect(page.locator("#root")).not.toBeEmpty();
    await page.waitForTimeout(500);
    await noSideways(page);
    await page.screenshot({ path: join(SHOTS, `${info.project.name}-home-${look}.png`), fullPage: true });
    await page.goto(`${BASE}#/leaderboard`);
    await page.waitForTimeout(400);
    await noSideways(page);
    if (look === "tetromino") await page.screenshot({ path: join(SHOTS, `${info.project.name}-leaderboard.png`), fullPage: true });
  }
  await page.goto(`${BASE}#/about`);
  await expect(page.getByRole("heading", { name: "About Tetris ML Arena" })).toBeVisible();
  await expect(page.getByText("The top eight agents form a chain of statistical ties")).toBeVisible();
  await noSideways(page);
  await page.screenshot({ path: join(SHOTS, `${info.project.name}-about.png`), fullPage: true });
  expect(errors).toEqual([]);
});

test("every agent plays (three per match, Hard)", async ({ page }) => {
  test.setTimeout(240_000);
  const errors = watchConsole(page);
  const played: Record<string, number> = {};
  for (let i = 0; i < ALL_AGENTS.length; i += 3) {
    const group = ALL_AGENTS.slice(i, i + 3);
    await setup(page, "tetromino", group.map((agent) => ({ agent, preset: "Hard" })));
    await startMarathon(page, true);
    await page.waitForTimeout(12_000);
    const counts = await pieces(page);
    const agents = Object.entries(counts).filter(([name]) => name !== "You");
    expect(agents.length).toBe(group.length);
    group.forEach((key, j) => (played[key] = agents[j][1]));
    for (const [name, n] of agents) expect(n, `${name} placed no pieces`).toBeGreaterThanOrEqual(5);
    await expect(page.getByText(/baseline is playing/i)).toHaveCount(0);
  }
  summary.piecesIn12s = played;
  expect(errors).toEqual([]);
});

test("frame rate with three neural agents, and screenshots of every look", async ({ page }, info) => {
  test.setTimeout(240_000);
  const errors = watchConsole(page);
  const frames: Record<string, unknown> = {};
  for (const look of LOOKS) {
    await setup(page, look, [
      { agent: "dt", preset: "Hard" },
      { agent: "ppo", preset: "Hard" },
      { agent: "dqn", preset: "Hard" },
    ]);
    await startMarathon(page, true);
    await page.waitForTimeout(6_000);
    await page.evaluate(() => (window as unknown as { __arenaPerf: { reset: () => void } }).__arenaPerf.reset());
    await page.waitForTimeout(look === "tetromino" ? 20_000 : 6_000);
    const f = await page.evaluate(() => {
      const s = (window as unknown as { __arenaPerf: { stats: { frames: number[]; longTasks: number } } }).__arenaPerf.stats;
      const v = [...s.frames].sort((a, b) => a - b);
      const q = (p: number) => v[Math.min(v.length - 1, Math.floor(v.length * p))];
      const mean = v.reduce((a, b) => a + b, 0) / v.length;
      return { frames: v.length, fps: Math.round(10000 / mean) / 10, p50: q(0.5), p95: q(0.95), p99: q(0.99), longTasks: s.longTasks };
    });
    frames[look] = f;
    await noSideways(page);
    await page.screenshot({ path: join(SHOTS, `${info.project.name}-match-${look}.png`) });
    if (look === "tetromino") expect(f.fps, "frame rate with three agents").toBeGreaterThan(55);
  }
  summary.frames = frames;
  expect(errors).toEqual([]);
});

test("transfer size: first load, and each agent's model", async ({ page }) => {
  test.setTimeout(120_000);
  const sizes = new Map<string, number>();
  const client = await page.context().newCDPSession(page);
  await client.send("Network.enable");
  await client.send("Network.setCacheDisabled", { cacheDisabled: true });
  const urls = new Map<string, string>();
  client.on("Network.responseReceived", (e) => urls.set(e.requestId, e.response.url));
  client.on("Network.loadingFinished", (e) => {
    const u = urls.get(e.requestId);
    if (u) sizes.set(u, (sizes.get(u) ?? 0) + e.encodedDataLength);
  });
  await page.goto(BASE);
  await expect(page.locator("#root")).not.toBeEmpty();
  await page.waitForLoadState("networkidle");
  const firstLoad = [...sizes.values()].reduce((a, b) => a + b, 0);
  const firstLoadFiles = [...sizes.entries()].map(([u, b]) => [u.replace(BASE, ""), b]);
  // Model files: fetch each (the browser's own compression, cache off).
  const models: Record<string, number> = {};
  for (const f of ["dqn", "ppo", "ppo_raw", "bc", "bc_dagger", "bc_you", "dt_embed", "dt_core"]) {
    const before = [...sizes.entries()].filter(([u]) => u.endsWith(`${f}.onnx`)).length;
    await page.evaluate(async (u) => {
      await (await fetch(u, { cache: "no-store" })).arrayBuffer();
    }, `${BASE}models/${f}.onnx`);
    await page.waitForTimeout(200);
    const hit = [...sizes.entries()].filter(([u]) => u.endsWith(`${f}.onnx`));
    expect(hit.length).toBeGreaterThan(before);
    models[f] = hit[hit.length - 1][1];
  }
  summary.transfer = { firstLoadBytes: firstLoad, firstLoadFiles, modelBytes: models };
  expect(firstLoad).toBeLessThan(1_500_000);
});

test("touch controls move the piece on a phone", async ({ page }, info) => {
  test.skip(info.project.name !== "phone", "phone only");
  await setup(page, "tetromino", [{ agent: "ga", preset: "Easy" }]);
  await startMarathon(page);
  await page.waitForTimeout(3_500); // countdown
  const controls = page.getByRole("group", { name: "Game controls" });
  await expect(controls).toBeVisible();
  const board = () => page.locator("canvas.board--human").getAttribute("aria-label");
  await controls.getByRole("button", { name: "Hard drop" }).tap();
  await page.waitForTimeout(300);
  expect(await board()).toMatch(/points/);
  const score = await page.evaluate(() => document.querySelector(".stats--human .stats__value")?.textContent);
  expect(Number(score)).toBeGreaterThan(0); // a hard drop scores 2 per row
  await noSideways(page);
});
