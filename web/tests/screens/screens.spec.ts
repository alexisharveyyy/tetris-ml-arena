// Screenshots of every screen in every look, saved to reports/ui/.
//   npx playwright test                     all looks and screens
//   LOOKS=tetromino SCREENS=home npx playwright test
import { expect, test, type Page } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { api, MATCH_ID, matchInfo, snapshot, tickerEvents } from "./fixtures";

const BASE = "http://127.0.0.1:4173/";
const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "reports", "ui");
const LETTER: Record<string, string> = { tetromino: "A", terminal: "B", broadcast: "C", poster: "D" };
const WIDTH = Number(process.env.WIDTH ?? 1440);
const HEIGHT = Number(process.env.HEIGHT ?? 900);
const LOOKS = (process.env.LOOKS ?? "tetromino,terminal,broadcast,poster").split(",");
const SCREENS = (process.env.SCREENS ?? "home,opponents,match,results,leaderboard,people,watch,solo,record,settings").split(",");

// Mocked screens: numbered like the mockups. Others: named.
const SHOTS: { name: string; file: string; hash: string; ready: string }[] = [
  { name: "home", file: "1-home", hash: "#/", ready: ".home, .screen" },
  { name: "opponents", file: "2-opponents", hash: "#/play/marathon", ready: "section[aria-label=Agents]" },
  { name: "match", file: "3-match", hash: `#/match/${MATCH_ID}`, ready: "canvas.board" },
  { name: "results", file: "4-results", hash: `#/results/${MATCH_ID}`, ready: "h1" },
  { name: "leaderboard", file: "5-leaderboard", hash: "#/leaderboard", ready: "table" },
  { name: "people", file: "leaderboard-people", hash: "#/leaderboard/people", ready: "table" },
  { name: "watch", file: "watch", hash: `#/watch/${MATCH_ID}`, ready: "canvas.board" },
  { name: "solo", file: "solo", hash: `#/solo/${MATCH_ID}`, ready: "canvas.board" },
  { name: "record", file: "record", hash: "#/record", ready: "h1" },
  { name: "settings", file: "settings", hash: "#/settings", ready: "table" },
];

async function mock(page: Page, look: string, screen: string) {
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.startsWith("/api/matches/")) {
      return route.fulfill({ json: matchInfo });
    }
    const body = api[url.pathname];
    if (body === undefined) return route.fulfill({ status: 404, json: { detail: "not mocked" } });
    if (url.pathname === "/api/settings") return route.fulfill({ json: { ...(body as object), look } });
    return route.fulfill({ json: body });
  });
  await page.routeWebSocket(/\/ws\?match=/, (ws) => {
    const watch = screen === "watch";
    const solo = screen === "solo";
    const players = watch ? [snapshot.players[1]] : solo ? [snapshot.players[0]] : snapshot.players;
    const standings = watch || solo ? players.map((p) => p.id) : snapshot.standings;
    const snap = { ...snapshot, mode: watch ? "watch" : solo ? "solo" : "marathon", players, standings };
    ws.send(JSON.stringify({ ...snap, ev: watch || solo ? [] : tickerEvents }));
    const timer = setInterval(() => ws.send(JSON.stringify({ ...snap, f: snap.f + 1 })), 100);
    ws.onClose(() => clearInterval(timer));
  });
  await page.addInitScript((l) => localStorage.setItem("arena.look", l), look);
}

mkdirSync(OUT, { recursive: true });

for (const look of LOOKS) {
  for (const shot of SHOTS.filter((s) => SCREENS.includes(s.name))) {
    test(`${look} ${shot.name}`, async ({ page }) => {
      await page.setViewportSize({ width: WIDTH, height: HEIGHT });
      await mock(page, look, shot.name);
      await page.goto(BASE + shot.hash);
      await expect(page.locator(shot.ready).first()).toBeVisible();
      await page.evaluate(() => document.fonts.ready);
      await page.waitForTimeout(250);
      const suffix = WIDTH === 1440 ? "" : `-${WIDTH}x${HEIGHT}`;
      await page.screenshot({ path: join(OUT, `${LETTER[look]}${shot.file}${suffix}.png`), fullPage: true });
    });
  }
}
