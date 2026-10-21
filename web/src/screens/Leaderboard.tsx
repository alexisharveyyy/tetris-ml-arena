import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { LeaderboardResponse, Preset } from "../api/types";
import { PRESETS } from "../api/types";
import { fmtInt, numberWord } from "../lib/format";
import { go } from "../router";
import { useApp } from "../state/app";
import { layout } from "../themes/registry";
import type { LeaderboardModel } from "./models";

export function survivalIntro(data: LeaderboardResponse | null): string {
  const s = data?.survival;
  const games = s?.rows.find((r) => r.games)?.games;
  const every = s?.setting.garbage_every;
  const rows = s?.setting.garbage_rows ?? 1;
  const garbage = every
    ? `when ${rows === 1 ? "a garbage row arrives" : `${numberWord(rows)} garbage rows arrive`} every ${numberWord(every)} pieces`
    : "under garbage";
  const over = games ? `, over ${fmtInt(games)} games it never trained or was tuned on` : "";
  return `How often each agent tops out ${garbage}${over}. Lower is better. Bars show 95% confidence intervals.`;
}

export function LeaderboardScreen({ tab }: { tab: "survival" | "people" }) {
  const app = useApp();
  const [data, setData] = useState<LeaderboardResponse | null>(null);
  const [preset, setPreset] = useState<Preset>("Medium");
  useEffect(() => {
    api.leaderboard().then(setData).catch(() => setData(null));
  }, []);
  const model: LeaderboardModel = useMemo(
    () => ({
      tab,
      setTab: (t) => go(t === "survival" ? "/leaderboard" : "/leaderboard/people"),
      data,
      agent: app.agent,
      intro: survivalIntro(data),
      preset,
      setPreset,
      presets: PRESETS,
      homeHref: "#/",
    }),
    [tab, data, app, preset],
  );
  const Layout = layout(app.theme, "Leaderboard");
  return <Layout {...model} />;
}
