// App-wide data: agents, presets, settings and the current look.
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type { Agent, AgentsResponse, LookId, PresetInfo, Settings } from "../api/types";
import { getTheme, isLook } from "../themes/registry";
import type { Theme } from "../themes/types";

const LOOK_KEY = "arena.look";

export interface AppState {
  agents: Agent[];
  human: Agent | null;
  presets: PresetInfo[];
  settings: Settings | null;
  look: LookId;
  theme: Theme;
  error: string | null;
  setLook: (look: LookId) => void;
  updateSettings: (patch: Partial<Settings>) => Promise<void>;
  refreshAgents: () => Promise<void>;
  agent: (key: string | null | undefined) => Agent | null;
}

const Ctx = createContext<AppState | null>(null);

function storedLook(): LookId | null {
  try {
    const v = localStorage.getItem(LOOK_KEY);
    return isLook(v) ? v : null;
  } catch {
    return null;
  }
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [agentsResp, setAgents] = useState<AgentsResponse | null>(null);
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [look, setLookState] = useState<LookId>(() => storedLook() ?? "tetromino");
  const [error, setError] = useState<string | null>(null);

  const refreshAgents = useCallback(async () => {
    setAgents(await api.agents());
  }, []);

  useEffect(() => {
    Promise.all([api.agents(), api.presets(), api.settings()])
      .then(([a, p, s]) => {
        setAgents(a);
        setPresets(p);
        setSettings(s);
        if (!storedLook() && isLook(s.look)) setLookState(s.look);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    document.documentElement.dataset.look = look;
  }, [look]);

  const setLook = useCallback((next: LookId) => {
    setLookState(next);
    try {
      localStorage.setItem(LOOK_KEY, next);
    } catch {
      /* private mode: the server setting still persists it */
    }
    api.saveSettings({ look: next }).then(setSettings).catch(() => undefined);
  }, []);

  const updateSettings = useCallback(async (patch: Partial<Settings>) => {
    setSettings((s) => (s ? { ...s, ...patch } : s));
    setSettings(await api.saveSettings(patch));
  }, []);

  const value = useMemo<AppState>(() => {
    const byKey = new Map((agentsResp?.agents ?? []).map((a) => [a.key, a]));
    return {
      agents: agentsResp?.agents ?? [],
      human: agentsResp?.human ?? null,
      presets,
      settings,
      look,
      theme: getTheme(look),
      error,
      setLook,
      updateSettings,
      refreshAgents,
      agent: (key) => (key ? byKey.get(key) ?? null : null),
    };
  }, [agentsResp, presets, settings, look, error, setLook, updateSettings, refreshAgents]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp outside AppProvider");
  return v;
}
