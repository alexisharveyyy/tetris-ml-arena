import { clock2 } from "../../../lib/format";
import type { ResultsModel } from "../../../screens/models";

const pad = (s: string | number, n: number, right = false) => (right ? String(s).padStart(n) : String(s).padEnd(n));

export function TerminalResults(m: ResultsModel) {
  const human = m.rows.find((r) => r.kind === "human");
  const over = human?.topped_out && human.finish_time != null ? `Game over at ${clock2(human.finish_time)}` : "Match over";
  const ended = (r: (typeof m.rows)[number]) => (r.topped_out && r.finish_time != null ? `Topped out ${clock2(r.finish_time)}` : "Still playing");
  return (
    <main className="screen term results">
      <p className="term-dim">
        {over} / Seed {m.seed}
      </p>
      <h1 className="term-hero">{m.headlineNumerals}</h1>
      <p className="term-summary">{m.summary}</p>
      <table className="term-table term-results">
        <caption className="visually-hidden">{m.caption}</caption>
        <thead>
          <tr>
            <th scope="col">Pl</th>
            <th scope="col">ID</th>
            <th scope="col">Player</th>
            <th scope="col" className="num">Score</th>
            <th scope="col" className="num">Lines</th>
            <th scope="col" className="num">Tetris</th>
            <th scope="col" className="num">PPS</th>
            <th scope="col" className="num">Lvl</th>
            <th scope="col">Ended</th>
          </tr>
        </thead>
        <tbody>
          {m.rows.map((r) => (
            <tr key={r.id} className={r.kind === "human" ? "is-you" : undefined}>
              <td>{r.place}</td>
              <td>{r.kind === "human" ? "@" : m.agent(r.agent_key)?.looks.terminal.letter}</td>
              <th scope="row">{pad(r.kind === "human" ? "You" : r.name, 12)}</th>
              <td className="num">{r.score}</td>
              <td className="num">{r.lines}</td>
              <td className="num">{r.tetrises}</td>
              <td className="num">{r.pps.toFixed(2)}</td>
              <td className="num">{r.level}</td>
              <td>{ended(r)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="term-dim">{m.caption.split(". ").slice(1).join(". ") || m.caption}</p>
      <div className="term-actions-row">
        <button type="button" className="term-cta term-cta--inline" onClick={m.rematch} disabled={m.busy}>
          [R] Rematch same seed
        </button>
        <button type="button" className="term-link" onClick={m.newSeed} disabled={m.busy}>
          [N] New seed
        </button>
        <button type="button" className="term-link" onClick={m.keepWatching} disabled={m.busy}>
          [W] Keep watching
        </button>
        <a className="term-link" href={m.opponentsHref}>
          [O] Opponents
        </a>
        <a className="term-link" href={m.homeHref}>
          [H] Home
        </a>
      </div>
    </main>
  );
}
