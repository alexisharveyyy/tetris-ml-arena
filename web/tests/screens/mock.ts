// Mocked API and WebSocket for screenshot, audit and keyboard runs.
import type { Page } from "@playwright/test";
import { api, MATCH_ID, matchInfo, snapshot, tickerEvents } from "./fixtures";

export const BASE = "http://127.0.0.1:4173/";
export const LOOK_IDS = ["tetromino", "terminal", "broadcast", "poster"];

// Mocked screens: numbered like the mockups. Others: named.
export const SHOTS: { name: string; file: string; hash: string; ready: string }[] = [
  { name: "home", file: "1-home", hash: "#/", ready: ".home, .screen" },
  { name: "opponents", file: "2-opponents", hash: "#/play/marathon", ready: "section[aria-label=Agents]" },
  { name: "match", file: "3-match", hash: `#/match/${MATCH_ID}`, ready: "canvas.board" },
  { name: "results", file: "4-results", hash: `#/results/${MATCH_ID}`, ready: "h1" },
  { name: "leaderboard", file: "5-leaderboard", hash: "#/leaderboard", ready: "table, [role=table], .po-bands" },
  { name: "people", file: "leaderboard-people", hash: "#/leaderboard/people", ready: "table" },
  { name: "comparisons", file: "leaderboard-comparisons", hash: "#/leaderboard/comparisons", ready: "table" },
  { name: "watch", file: "watch", hash: `#/watch/${MATCH_ID}`, ready: "canvas.board" },
  { name: "solo", file: "solo", hash: `#/solo/${MATCH_ID}`, ready: "canvas.board" },
  { name: "record", file: "record", hash: "#/record", ready: "h1" },
  { name: "settings", file: "settings", hash: "#/settings", ready: "table" },
];

/** Mock every /api call and the match WebSocket. ``demo`` turns on demo mode
 * (the "Demo: skip to results" link) without its starting-look switch.
 * ``players`` replaces the fixture's match players (the fixture is shared by
 * every spec in a worker, so specs must not change it). */
export async function mock(
  page: Page,
  look: string,
  screen: string,
  demo = screen === "match",
  players: typeof snapshot.players = snapshot.players,
) {
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname.startsWith("/api/matches/")) {
      return route.fulfill({ json: matchInfo });
    }
    if (url.pathname === "/api/demo") return route.fulfill({ json: demo ? { enabled: true, look, seed: 48213, opponents: [] } : { enabled: false } });
    const body = api[url.pathname];
    if (body === undefined) return route.fulfill({ status: 404, json: { detail: "not mocked" } });
    if (url.pathname === "/api/settings") return route.fulfill({ json: { ...(body as object), look } });
    return route.fulfill({ json: body });
  });
  await page.routeWebSocket(/\/ws\?match=/, (ws) => {
    const watch = screen === "watch";
    const solo = screen === "solo";
    const all = players;
    const shown = watch ? [all[1]] : solo ? [all[0]] : all;
    const standings = watch || solo ? shown.map((p) => p.id) : snapshot.standings;
    const snap = { ...snapshot, mode: watch ? "watch" : solo ? "solo" : "marathon", players: shown, standings };
    ws.send(JSON.stringify({ ...snap, ev: watch || solo ? [] : tickerEvents }));
    const timer = setInterval(() => ws.send(JSON.stringify({ ...snap, f: snap.f + 1 })), 100);
    ws.onClose(() => clearInterval(timer));
  });
  await page.addInitScript((l) => {
    localStorage.setItem("arena.look", l);
    sessionStorage.setItem("arena.demo.started", "1");
  }, look);
}

