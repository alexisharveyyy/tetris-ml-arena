// The REST API of tetris/web/api.py, answered in the page. The static build
// swaps this module in for web/src/api/client.ts, so every screen of the web
// client works unchanged.
import agentsData from "../data/agents.json";
import leaderboardData from "../data/leaderboard.json";
import presetsData from "../data/presets.json";
import type { AgentKey } from "../agents/agents";
import { PRESETS } from "../agents/base";
import { defaultConfig } from "../engine/game";
import { defaultRace, type PlayerSpec, type RaceRules } from "../match/controller";
import { WorkerRunner } from "../match/runner";
import { MatchSession } from "../match/session";
import { agentRecords, defaultSettings, loadSettings, recordMatch, saveSettings, type SettingsData } from "./store";
import type {
  AgentsResponse,
  DemoInfo,
  LeaderboardResponse,
  LobbyResponse,
  MatchInfo,
  Preset,
  PresetInfo,
  RecordingTotals,
  Settings,
} from "../../../web/src/api/types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export interface StartMatch {
  mode: MatchInfo["mode"];
  opponents?: { agent: string; preset: Preset }[];
  seed?: number;
  record_setting?: "plain" | "garbage";
}

const MAX_OPPONENTS = 3;
const START_COUNTDOWN = 3.0;
const LIMITS: Record<string, [number, number]> = { marathon: [1, MAX_OPPONENTS], race: [1, MAX_OPPONENTS], watch: [1, 1], solo: [0, 0] };

const sessions = new Map<string, MatchSession>();

/** Make an agent worker. Exposed for tests. */
export function makeWorker(): Worker {
  return new Worker(new URL("../worker/agent.worker.ts", import.meta.url), { type: "module" });
}

function agentList() {
  return agentsData.agents.map((a) => ({ ...a, record: agentRecords(a.key) }));
}

function playableKeys(): string[] {
  return agentsData.agents.filter((a) => a.playable && a.available).map((a) => a.key);
}

function lobbyFor(mode: "marathon" | "race", settings: SettingsData): LobbyResponse {
  const available = playableKeys();
  const saved = (settings.last_match[mode] ?? null) as { slots?: ({ agent: string; preset: string } | null)[]; race?: RaceRules } | null;
  let slots: ({ agent: string; preset: Preset } | null)[] = [null, null, null];
  if (saved?.slots) {
    slots = saved.slots
      .map((s) => (s && available.includes(s.agent) ? { agent: s.agent, preset: (s.preset in PRESETS ? s.preset : "Medium") as Preset } : null))
      .concat([null, null, null])
      .slice(0, MAX_OPPONENTS);
  }
  if (!slots.some(Boolean) && available.length) slots[0] = { agent: available[0], preset: "Medium" };
  return { mode, slots, race: { ...defaultRace(), ...(saved?.race ?? {}) } };
}

function info(s: MatchSession): MatchInfo {
  const m = s.match;
  return {
    match: s.id,
    mode: m.config.mode,
    recording: null,
    seed: m.config.seed,
    ws: "",
    finished: m.finished,
    connected: true,
    players: m.specs.map((p) => ({ id: p.playerId, kind: p.kind, name: p.name, agent: p.agentKey, preset: p.preset as Preset | null })),
  };
}

function session(id: string): MatchSession {
  const s = sessions.get(id);
  if (!s || s.closedFlag) throw new ApiError(404, `no match ${id}`);
  return s;
}

export function getSession(id: string): MatchSession | null {
  const s = sessions.get(id);
  return s && !s.closedFlag ? s : null;
}

function newId(): string {
  return Array.from(crypto.getRandomValues(new Uint8Array(6)), (b) => b.toString(16).padStart(2, "0")).join("");
}

const ok = <T>(v: T) => Promise.resolve(v);

export const api = {
  agents: () => ok({ ...agentsData, agents: agentList() } as unknown as AgentsResponse),
  presets: () => ok(presetsData as unknown as PresetInfo[]),
  settings: () => ok(loadSettings() as unknown as Settings),
  settingsDefaults: () => ok(defaultSettings() as unknown as Settings),
  saveSettings: (patch: Partial<Settings>) => {
    const s = loadSettings();
    const { bindings, ...rest } = patch;
    if (bindings) s.bindings = { ...s.bindings, ...bindings };
    Object.assign(s, rest);
    saveSettings(s);
    return ok(s as unknown as Settings);
  },
  lobby: (mode: "marathon" | "race") => ok(lobbyFor(mode, loadSettings())),
  demo: () => ok<DemoInfo>({ enabled: false }),
  leaderboard: () =>
    ok({
      ...leaderboardData,
      records: Object.fromEntries(agentsData.agents.filter((a) => a.on_leaderboard).map((a) => [a.key, agentRecords(a.key)])),
    } as unknown as LeaderboardResponse),
  recordings: () => ok({ settings: ["plain", "garbage"], totals: { games: 0, on_space: 0, off_space: 0 } as RecordingTotals }),

  startMatch: async (body: StartMatch): Promise<MatchInfo> => {
    const opponents = body.opponents ?? [];
    const limits = LIMITS[body.mode];
    if (!limits) throw new ApiError(422, `${body.mode} is not available in the browser version`);
    if (opponents.length < limits[0] || opponents.length > limits[1]) {
      throw new ApiError(422, `${body.mode} takes ${limits[0]} to ${limits[1]} opponents, got ${opponents.length}`);
    }
    const byKey = new Map(agentsData.agents.map((a) => [a.key, a]));
    for (const o of opponents) {
      const a = byKey.get(o.agent);
      if (!a) throw new ApiError(422, `unknown agent ${o.agent}`);
      if (!a.playable) throw new ApiError(422, `${o.agent} is not offered as an opponent`);
    }
    const settings = loadSettings();
    let race: RaceRules | null = null;
    if (body.mode === "marathon" || body.mode === "race") {
      const lobby = lobbyFor(body.mode, settings);
      settings.last_match[body.mode] = {
        slots: [...opponents.map((o) => ({ agent: o.agent, preset: o.preset, custom: null, target_return: null })), null, null, null].slice(0, MAX_OPPONENTS),
        race: lobby.race,
        battle: { targeting: "leader" },
      };
      saveSettings(settings);
      if (body.mode === "race") race = lobby.race;
    }
    for (const [id, s] of sessions) {
      s.close();
      sessions.delete(id);
    }
    const seed = body.seed ?? Math.floor(Math.random() * 100_000);
    const specs: PlayerSpec[] = body.mode === "watch" ? [] : [{ playerId: "human", name: agentsData.human.name, kind: "human", agentKey: null, difficulty: null, preset: null }];
    const totals = new Map<string, number>();
    opponents.forEach((o) => totals.set(o.agent, (totals.get(o.agent) ?? 0) + 1));
    const seen = new Map<string, number>();
    opponents.forEach((o, i) => {
      seen.set(o.agent, (seen.get(o.agent) ?? 0) + 1);
      let name = byKey.get(o.agent)!.name;
      if (totals.get(o.agent)! > 1) name = `${name} ${seen.get(o.agent)}`;
      specs.push({ playerId: `ai${i + 1}`, name, kind: "agent", agentKey: o.agent as AgentKey, difficulty: { ...PRESETS[o.preset] }, preset: o.preset });
    });
    const id = newId();
    const s = new MatchSession(
      id,
      {
        mode: body.mode as "solo" | "watch" | "race" | "marathon",
        seed,
        game: defaultConfig(),
        race,
        countdown: body.mode === "marathon" || body.mode === "race" ? START_COUNTDOWN : 0,
      },
      specs,
      opponents.length ? new WorkerRunner(makeWorker) : null,
      { das: settings.das_ms / 1000, arr: settings.arr_ms / 1000, softDropRate: settings.soft_drop_rate },
      (sess) =>
        sess.match.config.mode === "marathon" && sess.humanId !== null ? recordMatch(sess.match.standings, sess.match.config.seed) : null,
    );
    sessions.set(id, s);
    return info(s);
  },

  match: async (id: string): Promise<MatchInfo> => {
    const s = session(id);
    return { ...info(s), standings: s.match.finished ? (s.match.standings as unknown as MatchInfo["standings"]) : null, history_id: s.historyId };
  },
  endMatch: async (id: string) => {
    session(id).end();
    return api.match(id);
  },
  rematch: async (id: string, newSeed: boolean) => {
    const s = session(id);
    s.rematch(newSeed ? Math.floor(Math.random() * 100_000) : undefined);
    return info(s);
  },
  keepWatching: async (id: string) => {
    const s = session(id);
    if (!s.match.finished) throw new ApiError(409, "the match is still running");
    s.keepWatching();
    return info(s);
  },
  setDifficulty: async (id: string, player: string, preset: Preset) => {
    const s = session(id);
    const p = s.match.players.get(player);
    if (!p || p.spec.kind !== "agent") throw new ApiError(422, `${player} is not an agent player`);
    s.match.setDifficulty(player, { ...PRESETS[preset] });
    p.spec.preset = preset;
    return info(s);
  },
  saveRecording: async (): Promise<MatchInfo> => {
    throw new ApiError(409, "Recording games needs the local app");
  },
  closeMatch: async (id: string) => {
    const s = sessions.get(id);
    s?.close();
    sessions.delete(id);
    return null;
  },
};
