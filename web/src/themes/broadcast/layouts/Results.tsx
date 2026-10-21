import { fmtInt } from "../../../lib/format";
import { endedText } from "../../../lib/copy";
import type { ResultsModel } from "../../../screens/models";
import { Plate } from "./parts";

export function BroadcastResults(m: ResultsModel) {
  const metric = m.mode === "race" ? "lines" : "score";
  const max = Math.max(1, ...m.rows.map((r) => r[metric]));
  const leader = m.rows[0]?.id;
  return (
    <main className="screen bc results">
      <Plate className="plate--yellow bc-tag">Final</Plate>
      <h1 className="bc-hero bc-hero--results">{m.headlineNumerals}</h1>
      <p className="bc-lead">{m.summary}</p>
      <p className="visually-hidden">{m.caption}</p>
      <ol className="bc-results">
        {m.rows.map((r) => {
          const a = m.agent(r.agent_key);
          const you = r.kind === "human";
          const color = you ? "#FF5A3C" : a?.looks.broadcast.fill ?? "#AEB4C8";
          const stats = `${r.lines} lines, ${r.tetrises ? `${r.tetrises} Tetris${r.tetrises === 1 ? "" : "es"}` : "no Tetrises"}, ${r.pps.toFixed(2)} pieces/s, level ${r.level}. ${endedText(r)}.`;
          return (
            <li key={r.id} className="bc-result">
              <Plate className={`bc-result__place${r.id === leader ? " is-leader" : you ? " is-you" : ""}`}>{r.place}</Plate>
              <Plate className="plate--deep bc-result__plate">
                <span className="bc-result__top">
                  <span className="bc-result__name">{you ? "You" : r.name}</span>
                  <span className="bc-result__meta">{you ? "Human" : `${a?.family ?? "Agent"}, ${r.preset}`}</span>
                  <span className="bc-result__score">{fmtInt(r[metric])}</span>
                </span>
                <span className="bc-result__bar" aria-hidden="true">
                  <span style={{ width: `${(r[metric] / max) * 100}%`, background: color }} />
                </span>
                <span className="bc-result__stats">{stats}</span>
              </Plate>
            </li>
          );
        })}
      </ol>
      <div className="bc-actions">
        <button type="button" className="plate plate--coral plate--button" onClick={m.rematch} disabled={m.busy}>
          <span className="plate__in">Rematch on the same seed (R)</span>
        </button>
        <button type="button" className="plate plate--outline plate--button" onClick={m.newSeed} disabled={m.busy}>
          <span className="plate__in">New seed (N)</span>
        </button>
        <button type="button" className="plate plate--outline plate--button" onClick={m.keepWatching} disabled={m.busy}>
          <span className="plate__in">Keep watching (W)</span>
        </button>
        <a className="link" href={m.opponentsHref}>
          Change opponents
        </a>
        <a className="link" href={m.homeHref}>
          Home
        </a>
      </div>
    </main>
  );
}
