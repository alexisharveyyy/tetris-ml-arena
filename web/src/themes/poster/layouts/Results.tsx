import { endedText } from "../../../lib/copy";
import { fmtInt } from "../../../lib/format";
import type { ResultsModel } from "../../../screens/models";
import { posterColors } from "../board";

export function PosterResults(m: ResultsModel) {
  return (
    <main className="po results">
      <header className="po-results__head">
        <h1 className="po-title po-title--results">{m.headline}</h1>
        <p className="po-lead">{m.summary}</p>
        <p className="visually-hidden">{m.caption}</p>
        <div className="po-actions">
          <button type="button" className="po-btn po-btn--yellow" onClick={m.rematch} disabled={m.busy}>
            Rematch, same seed (R)
          </button>
          <button type="button" className="po-btn po-btn--outline" onClick={m.newSeed} disabled={m.busy}>
            New seed (N)
          </button>
          <button type="button" className="po-btn po-btn--outline" onClick={m.keepWatching} disabled={m.busy}>
            Keep watching (W)
          </button>
          <a className="link" href={m.opponentsHref}>
            Opponents
          </a>
          <a className="link" href={m.homeHref}>
            Home
          </a>
        </div>
      </header>
      <ol className="po-columns po-columns--results">
        {m.rows.map((r) => {
          const human = r.kind === "human";
          const a = m.agent(r.agent_key);
          const c = posterColors(a, human);
          return (
            <li key={r.id} className={`po-col${human ? " po-col--you" : ""}`} style={{ background: c.field, color: c.text }}>
              <span className="po-col__place po-col__place--big">{r.place}</span>
              <h2 className="po-col__name">{human ? "You" : r.name}</h2>
              <p className="po-col__meta">{human ? "Human" : `${a?.family ?? "Agent"}, ${r.preset}`}</p>
              <p className="po-col__pts po-col__pts--big">{fmtInt(m.mode === "race" ? r.lines : r.score)}</p>
              <dl className="po-col__stats">
                {m.mode === "race" ? (
                  <>
                    <dt>Score</dt>
                    <dd>{fmtInt(r.score)}</dd>
                  </>
                ) : (
                  <>
                    <dt>Lines</dt>
                    <dd>{r.lines}</dd>
                  </>
                )}
                <dt>Tetrises</dt>
                <dd>{r.tetrises}</dd>
                <dt>Pieces/s</dt>
                <dd>{r.pps.toFixed(2)}</dd>
                <dt>Level</dt>
                <dd>{r.level}</dd>
              </dl>
              <p className="po-col__ended">{endedText(r)}</p>
            </li>
          );
        })}
      </ol>
    </main>
  );
}
