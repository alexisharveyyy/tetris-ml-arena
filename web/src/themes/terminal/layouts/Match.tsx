import { Board } from "../../../components/Board";
import { ConnectionBanner, MatchState } from "../../../components/Overlays";
import { PiecePreview } from "../../../components/PiecePreview";
import { agentStatusLine } from "../../../lib/copy";
import { clock2 } from "../../../lib/format";
import { keyLabel } from "../../../lib/keys";
import type { MatchModel } from "../../../screens/models";
import { useApp } from "../../../state/app";
import { useShortcuts } from "../../../state/match";

export function TerminalMatch(m: MatchModel) {
  const { settings } = useApp();
  useShortcuts({ t: m.toggleThinking });
  const h = m.human;
  const time = m.snap.race?.time_remaining != null ? m.snap.race.time_remaining : m.snap.time;
  const b = settings?.bindings ?? {};
  return (
    <div className="screen term match">
      <ConnectionBanner status={m.status} />
      <header className="term-head">
        <span className="term-title">Tetris ML Arena</span>
        <span className="term-status">
          {m.modeLine.replace(",", "")} <span role="timer">Time {clock2(time)}</span>
        </span>
        <span className="term-actions">
          <button type="button" className="term-link" onClick={m.toggleThinking} aria-pressed={m.showThinking}>
            [T] Thinking {m.showThinking ? "on" : "off"}
          </button>
          <button type="button" className="term-link" onClick={m.togglePause}>
            [Esc] Pause
          </button>
        </span>
      </header>
      {m.snap.spectating ? (
        <p className="term-spectating" role="status">
          Final standings frozen. Agents still playing. <a href={m.resultsHref}>[Results]</a>
        </p>
      ) : null}
      <div className="term-match">
        <aside className="term-side">
          {h ? (
            <dl className="term-stats">
              <dt>Lines:</dt>
              <dd>{h.stats.lines}</dd>
              <dt>Level:</dt>
              <dd>{h.stats.level}</dd>
              <dt>Score:</dt>
              <dd>{h.stats.score}</dd>
              <dt>Speed:</dt>
              <dd>{h.stats.pps.toFixed(2)}</dd>
            </dl>
          ) : null}
          <h2 className="term-h2">Standings</h2>
          <ol className="term-standings">
            {m.ranked.map((p) => (
              <li key={p.id} className={p.kind === "human" ? "is-you" : undefined}>
                <span>{p.place}</span>
                <span>{p.kind === "human" ? "You" : p.name}</span>
                <span>{p.stats.score}</span>
                <span aria-hidden="true">{p.kind === "human" ? "<" : ""}</span>
              </li>
            ))}
          </ol>
          <dl className="term-legend">
            <dt>{[...(b.left ?? []).slice(0, 1), ...(b.right ?? []).slice(0, 1)].map(keyLabel).join(" ") || "Arrows"}</dt>
            <dd>Move</dd>
            <dt>{[...(b.cw ?? []).slice(-1), ...(b.ccw ?? []).slice(0, 1)].map(keyLabel).join(" ")}</dt>
            <dd>Rotate</dd>
            <dt>{(b.hard ?? []).slice(0, 1).map(keyLabel).join(" ")}</dt>
            <dd>Drop</dd>
            <dt>{(b.hold ?? []).slice(0, 1).map(keyLabel).join(" ")}</dt>
            <dd>Hold</dd>
          </dl>
        </aside>
        {h ? (
          <section className="term-you" aria-label="Your board">
            <Board player={h} agent={null} cell={29} human showThinking={false} />
            <div className="term-queue">
              <h2 className="term-h2">Next</h2>
              <PiecePreview letter={h.next[0] ?? null} />
              <p className="term-dim">Then {h.next.slice(1).join(" ")}</p>
              <h2 className="term-h2">Hold</h2>
              <PiecePreview letter={h.hold} />
            </div>
          </section>
        ) : null}
        <section className="term-agents-grid" aria-label="Agents">
          {m.agents.map((p) => (
            <div key={p.id} className="term-agent-board">
              <h2 className="term-h2">
                {p.place} {p.name}
              </h2>
              <p className="term-dim">
                {m.agent(p.agent)?.family} / {p.preset}
              </p>
              <Board player={p} agent={m.agent(p.agent)} cell={17} showThinking={m.showThinking} />
              <p>
                Score {p.stats.score} L{p.stats.level}
              </p>
              <p className="term-agent-status">{agentStatusLine(p, m.showThinking).replace("Outlines show its top three landing spots for this", "Top 3 spots for next")}</p>
            </div>
          ))}
        </section>
      </div>
      <MatchState snap={m.snap} onResume={m.togglePause} onEnd={m.endMatch} />
    </div>
  );
}
