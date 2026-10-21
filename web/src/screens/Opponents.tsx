import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { Agent, LobbyResponse, Preset, Slot } from "../api/types";
import { PRESETS } from "../api/types";
import { raceGoal } from "../lib/race";
import { go } from "../router";
import { useApp } from "../state/app";
import { layout } from "../themes/registry";
import type { AgentRow, OpponentsModel } from "./models";

const MAX = 3;

export function recordText(agent: Agent): string {
  const parts = PRESETS.flatMap((p) => {
    const r = agent.record?.[p];
    if (!r || r.wins + r.losses + r.ties === 0) return [];
    return [`${p} ${r.text}`];
  });
  return parts.length ? parts.join(", ") : "no games yet";
}

export function modeText(mode: "marathon" | "race", lobby: LobbyResponse | null): string {
  if (mode === "marathon") return "Marathon is a normal game. It ends when you top out, and the highest score wins.";
  const goal = lobby ? raceGoal(lobby.race) : "most lines in 2 minutes";
  return `Race: ${goal} wins. Everyone gets the same pieces, and topping out ends your game.`;
}

export function OpponentsScreen({ mode }: { mode: "marathon" | "race" }) {
  const app = useApp();
  const [lobby, setLobby] = useState<LobbyResponse | null>(null);
  const [slots, setSlots] = useState<(Slot | null)[]>([null, null, null]);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api
      .lobby(mode)
      .then((l) => {
        if (!live) return;
        setLobby(l);
        setSlots([...l.slots, null, null, null].slice(0, MAX));
      })
      .catch((e: Error) => setError(e.message));
    return () => {
      live = false;
    };
  }, [mode]);

  const model: OpponentsModel = useMemo(() => {
    const filled = slots.filter(Boolean) as Slot[];
    const count = (key: string) => filled.filter((s) => s.agent === key).length;
    const row = (agent: Agent): AgentRow => ({
      agent,
      inSlots: count(agent.key),
      available: !!agent.available && !!agent.playable,
      record: recordText(agent),
    });
    const nextIndex = slots.findIndex((s) => s === null);
    const presetText = (p: Preset) => app.presets.find((x) => x.name === p)?.description ?? "";
    const n = filled.length;
    const label = mode === "marathon" ? "Marathon" : "Race";
    return {
      mode,
      setMode: (m) => go(`/play/${m}`),
      groups: [
        { title: null, rows: app.agents.filter((a) => a.group === "main").map(row) },
        { title: "Variants", rows: app.agents.filter((a) => a.group === "variant").map(row) },
      ].filter((g) => g.rows.length),
      slots: slots.map((s) => {
        const agent = s ? app.agent(s.agent) : null;
        return s && agent ? { agent, preset: s.preset, presetText: presetText(s.preset) } : null;
      }),
      nextSlot: nextIndex >= 0 ? nextIndex + 1 : null,
      add: (key) =>
        setSlots((cur) => {
          const i = cur.findIndex((s) => s === null);
          if (i < 0) return cur;
          const next = [...cur];
          next[i] = { agent: key, preset: "Medium" };
          return next;
        }),
      remove: (i) =>
        setSlots((cur) => {
          const kept = cur.filter((_, j) => j !== i);
          return [...kept, null];
        }),
      setPreset: (i, preset) =>
        setSlots((cur) => cur.map((s, j) => (j === i && s ? { ...s, preset } : s))),
      presets: app.presets,
      instruction:
        nextIndex >= 0
          ? `Choose an agent to put it in slot ${nextIndex + 1}. The same agent can fill more than one slot.`
          : "All three slots are full. Remove one to make a change.",
      modeText: modeText(mode, lobby),
      startLabel: n ? `Start ${label} against ${n} agent${n === 1 ? "" : "s"}` : "Pick at least one opponent",
      canStart: n > 0 && !starting,
      start: () => {
        if (!n) return;
        setStarting(true);
        setError(null);
        api
          .startMatch({ mode, opponents: filled.map((s) => ({ agent: s.agent, preset: s.preset })) })
          .then((info) => go(`/match/${info.match}`))
          .catch((e: Error) => {
            setError(e.message);
            setStarting(false);
          });
      },
      starting,
      error,
      backHref: "#/",
    };
  }, [app, slots, mode, lobby, starting, error]);

  const Layout = layout(app.theme, "Opponents");
  return <Layout {...model} />;
}
