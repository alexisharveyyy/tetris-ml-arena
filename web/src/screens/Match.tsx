import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { FeedEvent, PlayerSnap, Snapshot } from "../api/types";
import { ConnectionBanner } from "../components/Overlays";
import { tickerLine } from "../lib/copy";
import { clock } from "../lib/format";
import { raceGoal } from "../lib/race";
import { go } from "../router";
import { useApp } from "../state/app";
import { useGameKeys, useMatchSession } from "../state/match";
import { layout } from "../themes/registry";
import type { MatchModel } from "./models";

const TICKER_LINES = 6;

export function MatchScreen({ matchId }: { matchId: string }) {
  const app = useApp();
  const [ticker, setTicker] = useState<string[]>([]);
  const onEvent = useCallback((ev: FeedEvent, s: Snapshot) => {
    const players = new Map<string, PlayerSnap>(s.players.map((p) => [p.id, p]));
    const human = s.players.find((p) => p.kind === "human")?.id ?? null;
    const line = tickerLine(ev, players, human);
    if (line) setTicker((t) => [...t, line].slice(-TICKER_LINES));
  }, []);
  const { socket, snap, status } = useMatchSession(matchId, onEvent);
  const showThinking = app.settings?.show_thinking ?? true;

  const togglePause = useCallback(() => {
    if (!socket?.latest || socket.latest.finished || socket.latest.disconnected) return;
    socket.pause(!socket.latest.paused);
  }, [socket]);
  const live = !!snap && !snap.paused && !snap.finished && snap.countdown <= 0 && !snap.loading;
  useGameKeys(socket, app.settings?.bindings, { enabled: live, onPause: togglePause });

  // A finished Marathon or Race goes to the results screen (unless spectating).
  const finished = !!snap?.finished && !snap.spectating;
  const isMatch = snap?.mode === "marathon" || snap?.mode === "race";
  useEffect(() => {
    if (!finished || !isMatch) return;
    const t = setTimeout(() => go(`/results/${matchId}`), 1200);
    return () => clearTimeout(t);
  }, [finished, isMatch, matchId]);

  const model: MatchModel | null = useMemo(() => {
    if (!snap) return null;
    const human = snap.players.find((p) => p.kind === "human") ?? null;
    const byId = new Map(snap.players.map((p) => [p.id, p]));
    const timed = snap.race?.time_remaining;
    const modeLine =
      snap.mode === "race" && snap.race
        ? `Race: ${raceGoal(snap.race)}, seed ${snap.seed}`
        : `${snap.mode === "marathon" ? "Marathon" : snap.mode === "watch" ? "Watch AI" : "Solo"}, seed ${snap.seed}`;
    return {
      snap,
      status,
      human,
      agents: snap.players.filter((p) => p.kind === "agent"),
      ranked: snap.standings.map((id) => byId.get(id)!).filter(Boolean),
      agent: app.agent,
      timer: clock(timed != null ? timed : snap.time),
      modeLine,
      showThinking,
      toggleThinking: () => void app.updateSettings({ show_thinking: !showThinking }),
      showGhost: app.settings?.show_ghost ?? true,
      togglePause,
      endMatch: () => void api.endMatch(matchId).catch(() => undefined),
      ticker,
      resultsHref: `#/results/${matchId}`,
    };
  }, [snap, status, app, showThinking, togglePause, ticker, matchId]);

  const Layout = layout(app.theme, "Match");
  if (!model) {
    return (
      <main className="screen screen--loading">
        <ConnectionBanner status={status === "gone" ? "gone" : "connecting"} />
      </main>
    );
  }
  return <Layout {...model} />;
}
