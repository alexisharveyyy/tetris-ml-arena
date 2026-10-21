import { Board } from "../../../components/Board";
import { ConnectionBanner, MatchState } from "../../../components/Overlays";
import { PiecePreview } from "../../../components/PiecePreview";
import type { PlayerSnap } from "../../../api/types";
import { fmtInt } from "../../../lib/format";
import type { MatchModel } from "../../../screens/models";
import { Plate, Ticker } from "./parts";

/** Short fallback badge text for the board banner. */
export function fallbackBadge(p: PlayerSnap): string | null {
  if (p.status.state !== "fallback") return null;
  const reason = p.status.fallback;
  if (reason === "timeout") return "Timed out. Baseline playing.";
  if (reason === "load_failed" || reason === "load_timeout") return "Model did not load. Baseline playing.";
  return "Model stopped. Baseline playing.";
}

export function BroadcastMatch(m: MatchModel) {
  const leader = m.ranked[0]?.id;
  const h = m.human;
  return (
    <div className="bc-page bc-page--match">
      <ConnectionBanner status={m.status} />
      <header className="bc-scorebar">
        <Plate className="plate--white bc-clock">
          <span role="timer" aria-label={`Time ${m.timer}`}>
            {m.timer}
          </span>
        </Plate>
        <ol className="bc-scores" aria-label="Live standings">
          {m.ranked.map((p) => (
            <li key={p.id}>
              <Plate className={`bc-score${p.id === leader ? " is-leader" : p.kind === "human" ? " is-you" : ""}`}>
                <span className="bc-score__place">{p.place}</span>
                <span className="bc-score__name">{p.kind === "human" ? "You" : p.name}</span>
                <span className="bc-score__pts">{fmtInt(p.stats.score)}</span>
              </Plate>
            </li>
          ))}
        </ol>
        <div className="bc-scorebar__actions">
          <button type="button" className={`bc-btn${m.showThinking ? " bc-btn--on" : ""}`} aria-pressed={m.showThinking} onClick={m.toggleThinking}>
            Model thinking {m.showThinking ? "on" : "off"}
          </button>
          <button type="button" className="bc-btn bc-btn--outline" onClick={m.togglePause}>
            Pause
          </button>
        </div>
      </header>
      <p className="visually-hidden">{m.modeLine}</p>
      {m.snap.spectating ? (
        <p className="bc-spectating" role="status">
          Final standings are frozen. The agents keep playing. <a className="link" href={m.resultsHref}>Back to results</a>
        </p>
      ) : null}
      <div className="bc-arena">
        {h ? (
          <section className="bc-you" aria-label="Your board">
            <div className="bc-well bc-well--you">
              <Board player={h} agent={null} cell={30} human showThinking={false} />
              <Plate className="plate--coral bc-banner bc-banner--you">
                <span className="bc-banner__name">You</span>
                <span className="bc-banner__pts">{fmtInt(h.stats.score)}</span>
                <span className="bc-banner__meta">
                  {h.stats.lines} lines, level {h.stats.level}
                </span>
              </Plate>
            </div>
            <div className="bc-side">
              <div className="bc-box">
                <h2 className="bc-box__title">Next</h2>
                <ol className="bc-queue" aria-label={`Next pieces: ${h.next.join(", ")}`}>
                  {h.next.map((l, i) => (
                    <li key={i}>
                      <PiecePreview letter={l} cell={i === 0 ? 15 : 11} />
                    </li>
                  ))}
                </ol>
              </div>
              <div className="bc-box">
                <h2 className="bc-box__title">Hold</h2>
                <PiecePreview letter={h.hold} cell={13} dim={!h.can_hold} />
              </div>
            </div>
          </section>
        ) : null}
        {m.agents.map((p) => {
          const a = m.agent(p.agent);
          const badge = fallbackBadge(p);
          return (
            <section key={p.id} className="bc-agent-board" aria-label={`${p.name} board`}>
              <div className="bc-well">
                <Board player={p} agent={a} cell={17} showThinking={m.showThinking} />
                <Plate className="plate--white bc-banner">
                  <span className="bc-banner__name">{p.name}</span>
                  <span className="bc-banner__pts">{fmtInt(p.stats.score)}</span>
                </Plate>
              </div>
              <p className="bc-agent-board__meta">
                {a?.family}, {p.preset}
              </p>
              {badge ? <p className="bc-badge">{badge}</p> : null}
              {p.status.state === "topped_out" ? <p className="bc-badge bc-badge--out">{p.status.text}</p> : null}
            </section>
          );
        })}
      </div>
      <Ticker label="Live" lines={m.ticker.slice(-4)} />
      <MatchState snap={m.snap} onResume={m.togglePause} onEnd={m.endMatch} />
    </div>
  );
}
