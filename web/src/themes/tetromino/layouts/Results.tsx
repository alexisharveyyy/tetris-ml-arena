import { Button } from "../../../components/Button";
import { PieceIcon } from "../../../components/PieceIcon";
import { endedText } from "../../../lib/copy";
import { fmtInt } from "../../../lib/format";
import type { ResultsModel } from "../../../screens/models";

export function ResultsActions({ m }: { m: ResultsModel }) {
  return (
    <div className="results__actions">
      <Button onClick={m.rematch} shortcut="r" disabled={m.busy}>
        Rematch on the same seed
      </Button>
      <Button variant="secondary" onClick={m.newSeed} shortcut="n" disabled={m.busy}>
        Play a new seed
      </Button>
      <Button variant="secondary" onClick={m.keepWatching} shortcut="w" disabled={m.busy}>
        Keep watching the agents
      </Button>
      <a className="link link--strong" href={m.opponentsHref}>
        Change opponents
      </a>
      <a className="link link--strong" href={m.homeHref}>
        Home
      </a>
    </div>
  );
}

export function TetrominoResults(m: ResultsModel) {
  return (
    <main className="screen results">
      <h1 className="results__title">{m.headline}</h1>
      <p className="results__summary">{m.summary}</p>
      <div className="results__table-wrap">
        <table className="results-table">
          <caption>{m.caption}</caption>
          <thead>
            <tr>
              <th scope="col">Place</th>
              <th scope="col">Player</th>
              <th scope="col" className="num">Score</th>
              <th scope="col" className="num">Lines</th>
              <th scope="col" className="num">Tetrises</th>
              <th scope="col" className="num">Pieces/s</th>
              <th scope="col" className="num">Level</th>
              <th scope="col">Ended</th>
            </tr>
          </thead>
          <tbody>
            {m.rows.map((r) => {
              const agent = m.agent(r.agent_key);
              return (
                <tr key={r.id} className={r.kind === "human" ? "is-you" : undefined}>
                  <td className="results-table__place">{r.place}</td>
                  <th scope="row">
                    <div className="results-table__player">
                    <span className="results-table__icon">
                      {agent ? <PieceIcon agent={agent} cell={14} /> : <span className="you-mark" aria-hidden="true" />}
                    </span>
                    <span>
                      <span className="results-table__name">{r.kind === "human" ? "You" : r.name}</span>
                      <span className="results-table__meta">{r.kind === "human" ? "Human" : `${agent?.family ?? "Agent"}, ${r.preset}`}</span>
                    </span>
                    </div>
                  </th>
                  <td className="num results-table__score">{fmtInt(r.score)}</td>
                  <td className="num">{r.lines}</td>
                  <td className="num">{r.tetrises}</td>
                  <td className="num">{r.pps.toFixed(2)}</td>
                  <td className="num">{r.level}</td>
                  <td>{endedText(r)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <ResultsActions m={m} />
    </main>
  );
}
