import { Board } from "../../../components/Board";
import { ConnectionBanner, MatchState } from "../../../components/Overlays";
import type { PlayerSnap } from "../../../api/types";
import { agentStatusLine } from "../../../lib/copy";
import { fmtInt } from "../../../lib/format";
import type { MatchModel } from "../../../screens/models";
import { posterColors } from "../board";

function queueSentence(p: PlayerSnap): string {
  const [first, ...rest] = p.next;
  const next = first ? `Next: ${first}${rest.length ? `, then ${rest.join(", ")}` : ""}.` : "";
  return `${next} ${p.hold ? `Holding ${["I", "O", "S", "L"].includes(p.hold) ? "an" : "a"} ${p.hold}.` : "Nothing held."}`;
}

export function PosterMatch(m: MatchModel) {
  const players = [...(m.human ? [m.human] : []), ...m.agents];
  return (
    <div className="po match">
      <ConnectionBanner status={m.status} />
      <header className="po-matchbar">
        <span className="po-matchbar__timer" role="timer" aria-label={`Time ${m.timer}`}>
          {m.timer}
        </span>
        <span className="po-matchbar__mode">{m.modeLine.replace(", seed", " on seed")}</span>
        <span className="po-matchbar__actions">
          <button type="button" className={`po-btn po-btn--small ${m.showThinking ? "po-btn--yellow" : "po-btn--outline"}`} aria-pressed={m.showThinking} onClick={m.toggleThinking}>
            {m.showThinking ? "Hide model thinking" : "Show model thinking"}
          </button>
          <button type="button" className="po-btn po-btn--small po-btn--outline" onClick={m.togglePause}>
            Pause
          </button>
          {m.demoSkip ? (
            <button type="button" className="demo-link" onClick={m.demoSkip} aria-keyshortcuts="F9">
              Demo: results
            </button>
          ) : null}
        </span>
      </header>
      {m.snap.spectating ? (
        <p className="po-spectating" role="status">
          The final standings are frozen. The agents keep playing. <a className="link" href={m.resultsHref}>Back to results</a>
        </p>
      ) : null}
      <div className="po-columns">
        {players.map((p) => {
          const human = p.kind === "human";
          const a = m.agent(p.agent);
          const c = posterColors(a, human);
          return (
            <section key={p.id} className={`po-col${human ? " po-col--you" : ""}`} style={{ background: c.field, color: c.text }} aria-label={human ? "Your board" : `${p.name} board`}>
              <div className="po-col__top">
                <span className="po-col__place" aria-label={`Place ${p.place}`}>
                  {p.place}
                </span>
                <span className="po-col__score">
                  <span className="po-col__pts">{fmtInt(p.stats.score)}</span>
                  <span className="po-col__lines">
                    {p.stats.lines} lines, level {p.stats.level}
                  </span>
                </span>
              </div>
              <h2 className="po-col__name">{human ? "You" : p.name}</h2>
              <p className="po-col__meta">{human ? "Human player" : `${a?.family}, ${p.preset}`}</p>
              <div className="po-col__board">
                <Board player={p} agent={a} cell={human ? 27 : 17} human={human} showThinking={!human && m.showThinking} />
              </div>
              <p className="po-col__status">
                {human ? queueSentence(p) : agentStatusLine(p, m.showThinking).replace("Outlines show its", "Dashed outlines are its")}
              </p>
            </section>
          );
        })}
      </div>
      <MatchState snap={m.snap} onResume={m.togglePause} onEnd={m.endMatch} />
    </div>
  );
}
