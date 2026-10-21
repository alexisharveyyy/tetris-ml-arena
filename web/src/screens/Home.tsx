import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { LeaderboardResponse } from "../api/types";
import { pct } from "../lib/format";
import { go } from "../router";
import { useApp } from "../state/app";
import { THEMES, layout } from "../themes/registry";
import type { ControlHint, HomeModel } from "./models";

function pick(keys: string[] | undefined): string[] {
  if (!keys?.length) return [];
  return [keys.find((k) => k.length === 1) ?? keys[0]];
}

export function controlHints(bindings: Record<string, string[]> | undefined): ControlHint[] {
  const b = bindings ?? {};
  return [
    { keys: [...(b.left ?? []).slice(0, 1), ...(b.right ?? []).slice(0, 1)], label: "move" },
    { keys: [...pick(b.cw), ...pick(b.ccw)], label: "rotate" },
    { keys: (b.hard ?? []).slice(0, 1), label: "hard drop" },
    { keys: pick(b.hold), label: "hold" },
    { keys: (b.pause ?? []).slice(0, 1), label: "pause" },
  ].filter((h) => h.keys.length);
}

export function evalTickerLines(data: LeaderboardResponse | null, name: (key: string) => string): string[] {
  if (!data) return [];
  const rows = data.survival.rows;
  const valid = rows.filter((r) => r.status === "valid" && r.top_out_rate != null);
  const lines: string[] = [];
  valid.forEach((r, i) => {
    const label = r.key === "dellacherie" ? "Hand-tuned baseline" : name(r.key);
    lines.push(i === 0 ? `${label} leads with a ${pct(r.top_out_rate!)} top-out rate` : `${label} ${pct(r.top_out_rate!)}`);
  });
  const pending = rows.filter((r) => r.status === "pending_rerun").length;
  const waiting = rows.filter((r) => r.status === "not_evaluated").length;
  if (pending) lines.push(`${pending} result${pending === 1 ? "" : "s"} pending a rerun`);
  if (waiting) lines.push(`${waiting} agent${waiting === 1 ? "" : "s"} still to be evaluated`);
  return lines;
}

export function HomeScreen() {
  const app = useApp();
  const [board, setBoard] = useState<LeaderboardResponse | null>(null);
  useEffect(() => {
    api.leaderboard().then(setBoard).catch(() => setBoard(null));
  }, []);
  const model: HomeModel = useMemo(
    () => ({
      lineup: app.agents.filter((a) => a.group === "main"),
      looks: THEMES.map((t) => ({ id: t.id, label: t.label })),
      look: app.look,
      setLook: app.setLook,
      controls: controlHints(app.settings?.bindings),
      evalTicker: evalTickerLines(board, (k) => app.agent(k)?.name ?? k),
      raceMinutes: 2,
      startMarathon: () => go("/play/marathon"),
      links: {
        marathon: "#/play/marathon",
        race: "#/play/race",
        watch: "#/watch",
        solo: "#/solo",
        record: "#/record",
        leaderboard: "#/leaderboard",
        settings: "#/settings",
      },
    }),
    [app, board],
  );
  const Layout = layout(app.theme, "Home");
  return <Layout {...model} />;
}
