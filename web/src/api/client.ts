// REST client for /api.
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
} from "./types";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { headers: { "content-type": "application/json" }, ...init });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep the status text */
    }
    throw new ApiError(res.status, detail);
  }
  return (res.status === 204 ? null : await res.json()) as T;
}

const post = <T>(path: string, body: unknown = {}) => call<T>(path, { method: "POST", body: JSON.stringify(body) });

export interface StartMatch {
  mode: MatchInfo["mode"];
  opponents?: { agent: string; preset: Preset }[];
  seed?: number;
  record_setting?: "plain" | "garbage";
}

export const api = {
  agents: () => call<AgentsResponse>("/api/agents"),
  presets: () => call<PresetInfo[]>("/api/presets"),
  settings: () => call<Settings>("/api/settings"),
  settingsDefaults: () => call<Settings>("/api/settings/defaults"),
  saveSettings: (patch: Partial<Settings>) => call<Settings>("/api/settings", { method: "PUT", body: JSON.stringify(patch) }),
  lobby: (mode: "marathon" | "race") => call<LobbyResponse>(`/api/lobby/${mode}`),
  demo: () => call<DemoInfo>("/api/demo").catch((): DemoInfo => ({ enabled: false })),
  leaderboard: () => call<LeaderboardResponse>("/api/leaderboard"),
  recordings: () => call<{ settings: string[]; totals: RecordingTotals }>("/api/recordings"),
  startMatch: (body: StartMatch) => post<MatchInfo>("/api/matches", body),
  match: (id: string) => call<MatchInfo>(`/api/matches/${id}`),
  endMatch: (id: string) => post<MatchInfo>(`/api/matches/${id}/end`),
  rematch: (id: string, newSeed: boolean) => post<MatchInfo>(`/api/matches/${id}/rematch`, { new_seed: newSeed }),
  keepWatching: (id: string) => post<MatchInfo>(`/api/matches/${id}/keep_watching`),
  setDifficulty: (id: string, player: string, preset: Preset) =>
    post<MatchInfo>(`/api/matches/${id}/difficulty`, { player, preset }),
  saveRecording: (id: string) => post<MatchInfo>(`/api/matches/${id}/recording/save`),
  closeMatch: (id: string) => call<null>(`/api/matches/${id}`, { method: "DELETE" }),
};
