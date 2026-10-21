// App-wide data: agents, presets, settings and the current look.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api } from "../api/client";
import type { Agent, AgentsResponse, DemoInfo, LookId, PresetInfo, Settings } from "../api/types";
import { allThemes, getTheme, isLook } from "../themes/registry";
import type { Theme } from "../themes/types";

const LOOK_KEY = "arena.look";
/** Set once demo mode has applied its starting look in this tab. */
const DEMO_KEY = "arena.demo.started";

/** How often to re-read /api/agents while the server is still checking checkpoints. */
export const CHECK_POLL_MS = [500, 1000, 2000];

/** True while any agent's checkpoint load check has not finished on the server. */
export function stillChecking(resp: AgentsResponse | null): boolean {
  return !!resp?.agents.some((a) => a.status?.state === "checking");
}

export interface AppState {
  agents: Agent[];
  human: Agent | null;
  presets: PresetInfo[];
  settings: Settings | null;
  look: LookId;
  theme: Theme;
  error: string | null;
  /** Demo mode (--demo), or null for a normal run. */
  demo: DemoInfo | null;
  setLook: (look: LookId) => void;
  /** Switch to the next look (F10 in demo mode). */
  cycleLook: () => void;
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

function demoStarted(): boolean {
  try {
    return sessionStorage.getItem(DEMO_KEY) === "1";
  } catch {
    return false;
  }
}

function markDemoStarted(): void {
  try {
    sessionStorage.setItem(DEMO_KEY, "1");
  } catch {
    /* private mode: the demo look is applied on every load */
  }
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [agentsResp, setAgents] = useState<AgentsResponse | null>(null);
  const [presets, setPresets] = useState<PresetInfo[]>([]);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [look, setLookState] = useState<LookId>(() => storedLook() ?? "tetromino");
  const [error, setError] = useState<string | null>(null);
  const [demo, setDemo] = useState<DemoInfo | null>(null);

  const refreshAgents = useCallback(async () => {
    setAgents(await api.agents());
  }, []);

  useEffect(() => {
    Promise.all([api.agents(), api.presets(), api.settings(), api.demo()])
      .then(([a, p, s, d]) => {
        setAgents(a);
        setPresets(p);
        setSettings(s);
        setDemo(d.enabled ? d : null);
        if (d.enabled && isLook(d.look) && !demoStarted()) {
          // Demo mode starts on its look once per tab; F10 changes it after that.
          setLookState(d.look);
          markDemoStarted();
        } else if (!storedLook() && isLook(s.look)) {
          setLookState(s.look);
        }
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  // The server checks checkpoints in the background after startup, so the first
  // response can list agents as "checking". Keep re-reading until every check
  // has finished, so the lineup updates without a reload. Also re-read when the
  // tab regains focus, in case a checkpoint was promoted in the meantime.
  const checking = stillChecking(agentsResp);
  const polls = useRef(0);
  useEffect(() => {
    if (!checking) {
      polls.current = 0;
      return;
    }
    const delay = CHECK_POLL_MS[Math.min(polls.current, CHECK_POLL_MS.length - 1)];
    const timer = setTimeout(() => {
      polls.current += 1;
      api.agents().then(setAgents).catch(() => setAgents((a) => (a ? { ...a } : a)));
    }, delay);
    return () => clearTimeout(timer);
  }, [checking, agentsResp]);

  useEffect(() => {
    const onFocus = () => {
      if (document.visibilityState === "visible") api.agents().then(setAgents).catch(() => undefined);
    };
    document.addEventListener("visibilitychange", onFocus);
    return () => document.removeEventListener("visibilitychange", onFocus);
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

  const cycleLook = useCallback(() => {
    const ids = allThemes().map((t) => t.id);
    setLookState((cur) => {
      const next = ids[(ids.indexOf(cur) + 1) % ids.length];
      try {
        localStorage.setItem(LOOK_KEY, next);
      } catch {
        /* private mode */
      }
      return next;
    });
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
      demo,
      setLook,
      cycleLook,
      updateSettings,
      refreshAgents,
      agent: (key) => (key ? byKey.get(key) ?? null : null),
    };
  }, [agentsResp, presets, settings, look, error, demo, setLook, cycleLook, updateSettings, refreshAgents]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp(): AppState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp outside AppProvider");
  return v;
}
