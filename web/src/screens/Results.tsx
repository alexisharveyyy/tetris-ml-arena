import { useEffect, useMemo, useState } from "react";
import { api, ApiError } from "../api/client";
import type { MatchInfo } from "../api/types";
import { resultsHeadline, resultsSummary } from "../lib/copy";
import { go } from "../router";
import { useApp } from "../state/app";
import { useShortcuts } from "../state/match";
import { layout } from "../themes/registry";
import type { ResultsModel } from "./models";

export function resultsCaption(info: MatchInfo): string {
  const mode = info.mode === "race" ? "Race" : "Marathon";
  const rows = info.standings ?? [];
  const human = rows.find((r) => r.kind === "human");
  const frozen = rows.some((r) => r.kind === "agent" && !r.topped_out);
  const tail =
    frozen && human?.topped_out ? " Agents still playing were frozen when you topped out." : frozen ? " Agents still playing were frozen when the match ended." : "";
  return `${mode}, seed ${info.seed}.${tail}`;
}

export function ResultsScreen({ matchId }: { matchId: string }) {
  const app = useApp();
  const [info, setInfo] = useState<MatchInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api
      .match(matchId)
      .then(setInfo)
      .catch((e: Error) => setError(e instanceof ApiError && e.status === 404 ? "This match is no longer on the server." : e.message));
    void app.refreshAgents(); // records changed
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [matchId]);

  const act = (fn: () => Promise<unknown>, then: string) => {
    setBusy(true);
    fn()
      .then(() => go(then))
      .catch((e: Error) => setError(e.message))
      .finally(() => setBusy(false));
  };
  const mode: "marathon" | "race" = info?.mode === "race" ? "race" : "marathon";
  const model: ResultsModel | null = useMemo(() => {
    if (!info?.standings) return null;
    const metric = mode === "race" ? "lines" : "score";
    return {
      rows: info.standings,
      mode,
      seed: info.seed,
      headline: resultsHeadline(info.standings),
      headlineNumerals: resultsHeadline(info.standings, "numerals"),
      summary: resultsSummary(info.standings, metric),
      caption: resultsCaption(info),
      agent: app.agent,
      rematch: () => act(() => api.rematch(matchId, false), `/match/${matchId}`),
      newSeed: () => act(() => api.rematch(matchId, true), `/match/${matchId}`),
      keepWatching: () => act(() => api.keepWatching(matchId), `/match/${matchId}`),
      opponentsHref: `#/play/${mode}`,
      homeHref: "#/",
      busy,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [info, app, busy, matchId, mode]);

  useShortcuts(
    {
      r: () => model?.rematch(),
      n: () => model?.newSeed(),
      w: () => model?.keepWatching(),
      o: () => go(`/play/${mode}`),
      h: () => go("/"),
    },
    !!model && !busy,
  );

  const Layout = layout(app.theme, "Results");
  if (!model) {
    return (
      <main className="screen screen--message">
        <p>{error ?? (info ? "The match is still running." : "Loading results...")}</p>
        <p>
          <a className="link" href={info ? `#/match/${matchId}` : "#/"}>
            {info ? "Back to the match" : "Home"}
          </a>
        </p>
      </main>
    );
  }
  return <Layout {...model} />;
}
