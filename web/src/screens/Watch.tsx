// Watch AI: one agent with its top three candidates always available.
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Preset } from "../api/types";
import { PRESETS } from "../api/types";
import { Board } from "../components/Board";
import { Button } from "../components/Button";
import { ConnectionBanner } from "../components/Overlays";
import { agentStats, StatRow } from "../components/Panels";
import { PieceIcon } from "../components/PieceIcon";
import { Segmented } from "../components/Segmented";
import { agentStatusLine, unavailableText } from "../lib/copy";
import { ModelFallbacks } from "../components/ModelFallbacks";
import { pct } from "../lib/format";
import { go } from "../router";
import { useApp } from "../state/app";
import { useMatchSession, useShortcuts } from "../state/match";
import { PageHead } from "./PageHead";

export function WatchScreen({ matchId }: { matchId: string | null }) {
  return matchId ? <WatchLive matchId={matchId} /> : <WatchSetup />;
}

function WatchSetup() {
  const app = useApp();
  const options = app.agents.filter((a) => (a.group === "main" || a.group === "variant") && a.available && a.playable);
  const [key, setKey] = useState<string | null>(null);
  const [preset, setPreset] = useState<Preset>("Hard");
  const [error, setError] = useState<string | null>(null);
  const chosen = key ?? options[0]?.key ?? null;
  const start = () => {
    if (!chosen) return;
    api
      .startMatch({ mode: "watch", opponents: [{ agent: chosen, preset }] })
      .then((info) => go(`/watch/${info.match}`))
      .catch((e: Error) => setError(e.message));
  };
  return (
    <main className="screen watch">
      <PageHead title="Watch AI" />
      <p className="lead">Pick an agent and a speed. Its top three candidate placements are drawn on the board as numbered outlines.</p>
      <fieldset className="choice-list">
        <legend>Agent</legend>
        {app.agents
          .filter((a) => a.group === "main" || a.group === "variant")
          .map((a) => {
            const on = !!a.available && !!a.playable;
            return (
              <label key={a.key} className={`choice${on ? "" : " choice--off"}`}>
                <input type="radio" name="agent" value={a.key} checked={chosen === a.key} disabled={!on} onChange={() => setKey(a.key)} />
                <PieceIcon agent={a} cell={12} />
                <span className="choice__name">{a.name}</span>
                <span className="choice__meta">{on ? a.family : unavailableText(a)}</span>
              </label>
            );
          })}
      </fieldset>
      <div className="form-row">
        <span className="form-row__label" id="watch-speed">
          Speed
        </span>
        <Segmented label="Speed" value={preset} onChange={setPreset} options={PRESETS.map((p) => ({ value: p, label: p }))} />
      </div>
      <p className="hint">{app.presets.find((p) => p.name === preset)?.description}</p>
      {error ? (
        <p className="form-error" role="alert">
          {error}
        </p>
      ) : null}
      <Button onClick={start} disabled={!chosen}>
        Watch {chosen ? app.agent(chosen)?.name : "an agent"}
      </Button>
    </main>
  );
}

function WatchLive({ matchId }: { matchId: string }) {
  const app = useApp();
  const { snap, status } = useMatchSession(matchId);
  const p = snap?.players[0];
  const agent = app.agent(p?.agent);
  const thinking = app.settings?.show_thinking ?? true;
  const toggle = () => void app.updateSettings({ show_thinking: !thinking });
  const newGame = () => void api.rematch(matchId, true);
  useShortcuts({ t: toggle, r: newGame, escape: () => go("/watch") });
  useEffect(() => () => void api.closeMatch(matchId).catch(() => undefined), [matchId]);
  if (!snap || !p) return <main className="screen"><ConnectionBanner status={status} /></main>;
  const probability = p.agent && ["ppo", "ppo_raw", "bc", "bc_you", "bc_pure", "bc_dagger"].includes(p.agent);
  return (
    <main className="screen watch">
      <ConnectionBanner status={status} />
      <PageHead title={`Watch AI: ${p.name}`} back="#/watch" backLabel="Change agent">
        <Button onClick={toggle} pressed={thinking} shortcut="t">
          {thinking ? "Hide model thinking" : "Show model thinking"}
        </Button>
        <Button variant="secondary" onClick={newGame} shortcut="r">
          New game
        </Button>
      </PageHead>
      <div className="watch__body">
        <div className="watch__board">
          <Board player={p} agent={agent} cell={30} showThinking={thinking} />
        </div>
        <section className="watch__side">
          <p className="watch__who">
            <PieceIcon agent={agent} cell={16} /> {agent?.name}, {agent?.family}
          </p>
          <div className="form-row">
            <span className="form-row__label">Speed</span>
            <Segmented
              label="Speed"
              value={(p.preset ?? "Hard") as Preset}
              onChange={(preset) => void api.setDifficulty(matchId, p.id, preset)}
              options={PRESETS.map((x) => ({ value: x, label: x }))}
            />
          </div>
          <StatRow stats={[...agentStats(p), { label: "Pieces/s", value: p.stats.pps.toFixed(2) }]} />
          <p className="watch__status">{agentStatusLine(p, thinking)}</p>
          <ModelFallbacks player={p} className="watch__status" />
          {thinking && p.cand.length ? (
            <table className="cand-table">
              <caption>Top candidates {p.cand_current ? "for this piece" : "for the piece it just placed"}</caption>
              <thead>
                <tr>
                  <th scope="col">Rank</th>
                  <th scope="col">{probability ? "Policy probability" : "Value"}</th>
                  <th scope="col">Uses hold</th>
                </tr>
              </thead>
              <tbody>
                {p.cand.map((c) => (
                  <tr key={c.rank}>
                    <td>{c.rank}</td>
                    <td className="num">{probability ? pct(c.score, 1) : c.score.toFixed(2)}</td>
                    <td>{c.hold ? "Yes" : "No"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
          {p.status.state === "topped_out" ? (
            <p className="watch__over" role="status">
              {p.name} topped out. <Button onClick={newGame}>New game</Button>
            </p>
          ) : null}
        </section>
      </div>
    </main>
  );
}
