// Settings and the Marathon record, kept in localStorage (the server keeps
// them in configs/ files). Same shapes as tetris/settings.py and
// tetris/match/history.py.
import defaults from "../data/settings_defaults.json";

const SETTINGS_KEY = "arena.settings";
const HISTORY_KEY = "arena.history";
const PRESET_NAMES = ["Easy", "Medium", "Hard", "Unlimited"] as const;

export type SettingsData = typeof defaults & { last_match: Record<string, unknown> };

function read<T>(key: string): T | null {
  try {
    const v = localStorage.getItem(key);
    return v ? (JSON.parse(v) as T) : null;
  } catch {
    return null;
  }
}

function write(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* private mode or full storage: settings last for this visit only */
  }
}

let memSettings: SettingsData | null = null;

export function defaultSettings(): SettingsData {
  return JSON.parse(JSON.stringify(defaults)) as SettingsData;
}

export function loadSettings(): SettingsData {
  if (memSettings) return memSettings;
  const s = defaultSettings();
  const saved = read<Partial<SettingsData>>(SETTINGS_KEY) ?? {};
  for (const [k, v] of Object.entries(saved)) if (k in s) (s as Record<string, unknown>)[k] = v;
  for (const [action, keys] of Object.entries(defaults.bindings)) {
    if (!(action in s.bindings)) (s.bindings as Record<string, string[]>)[action] = keys;
  }
  memSettings = s;
  return s;
}

export function saveSettings(s: SettingsData): void {
  memSettings = s;
  write(SETTINGS_KEY, s);
}

// ------------------------------------------------------------------ history

export interface Record3 {
  wins: number;
  losses: number;
  ties: number;
}

interface History {
  version: 2;
  marathon: Record<string, Record3>;
  matches: unknown[];
}

const MAX_MATCHES = 200;

function loadHistory(): History {
  return read<History>(HISTORY_KEY) ?? { version: 2, marathon: {}, matches: [] };
}

export function recordMatch(standings: { kind: string; agent_key: string | null; preset: string | null; score: number }[], seed: number): string | null {
  const human = standings.find((r) => r.kind === "human");
  if (!human) return null;
  const h = loadHistory();
  for (const row of standings) {
    if (row.kind !== "agent" || !row.agent_key) continue;
    const key = `${row.agent_key}|${row.preset ?? "Custom"}`;
    const rec = (h.marathon[key] ??= { wins: 0, losses: 0, ties: 0 });
    if (human.score > row.score) rec.wins += 1;
    else if (human.score < row.score) rec.losses += 1;
    else rec.ties += 1;
  }
  const id = Math.random().toString(16).slice(2, 14).padEnd(12, "0");
  h.matches.push({ id, finished_at: new Date().toISOString(), mode: "marathon", seed, players: standings });
  h.matches = h.matches.slice(-MAX_MATCHES);
  write(HISTORY_KEY, h);
  return id;
}

function describe(r: Record3): string {
  if (!r.wins && !r.losses && !r.ties) return "no games yet";
  return `${r.wins}W ${r.losses}L` + (r.ties ? ` ${r.ties}T` : "");
}

export function agentRecords(agentKey: string) {
  const h = loadHistory();
  const out: Record<string, Record3 & { text: string }> = {};
  for (const p of PRESET_NAMES) {
    const r = h.marathon[`${agentKey}|${p}`] ?? { wins: 0, losses: 0, ties: 0 };
    out[p] = { ...r, text: describe(r) };
  }
  return out;
}
