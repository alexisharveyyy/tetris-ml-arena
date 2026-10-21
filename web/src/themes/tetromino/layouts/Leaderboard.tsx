import { SignificanceNote } from "../../../components/SignificanceNote";
import { PeopleTable, pendingNotes } from "../../../components/PeopleTable";
import { PieceIcon } from "../../../components/PieceIcon";
import { Tabs } from "../../../components/Tabs";
import type { SurvivalRow } from "../../../api/types";
import { fmtInt, pct } from "../../../lib/format";
import type { LeaderboardModel } from "../../../screens/models";

export const LEADERBOARD_TABS = [
  { id: "survival" as const, label: "Survival under garbage" },
  { id: "people" as const, label: "Against people" },
];

/** Position on the 0 to scale chart, as a percentage of its width (clipped). */
export function pos(x: number, scale: number): number {
  return Math.max(0, Math.min(100, (x / scale) * 100));
}

export function intervalText(r: SurvivalRow): string {
  return `${pct(r.top_out_rate!)} top-out rate, 95% interval ${pct(r.lo!, 1)} to ${pct(r.hi!, 1)}`;
}

function Interval({ r, scale, color }: { r: SurvivalRow; scale: number; color: string }) {
  const lo = pos(r.lo!, scale);
  const hi = pos(r.hi!, scale);
  const est = pos(r.top_out_rate!, scale);
  const off = r.top_out_rate! > scale;
  return (
    <div className="chart-track ci">
      <span className="visually-hidden">{intervalText(r)}</span>
      <span className="ci__bar" style={{ left: `${lo}%`, width: `${Math.max(hi - lo, 0.4)}%`, background: color }} aria-hidden="true" />
      {off ? (
        <>
          <span className="ci__off" aria-hidden="true" style={{ borderLeftColor: color }} />
          <span className="ci__label ci__label--off" aria-hidden="true">
            {pct(r.top_out_rate!)}, off scale
          </span>
        </>
      ) : (
        <>
          <span className="ci__dot" style={{ left: `${est}%`, background: color }} aria-hidden="true" />
          <span className="ci__label" style={{ left: `calc(${hi}% + 12px)` }} aria-hidden="true">
            {pct(r.top_out_rate!)}
          </span>
        </>
      )}
    </div>
  );
}

export function TetrominoLeaderboard(m: LeaderboardModel) {
  const s = m.data?.survival;
  const scale = s?.scale_max ?? 0.5;
  const ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5].filter((t) => t <= scale + 1e-9);
  return (
    <main className="screen leaderboard">
      <header className="page-head">
        <a className="link link--strong" href={m.homeHref}>
          Home
        </a>
        <h1 className="page-title">Leaderboard</h1>
        <Tabs label="Leaderboard" idPrefix="lb" tabs={LEADERBOARD_TABS} value={m.tab} onChange={m.setTab} className="tabs--boxed" />
      </header>
      <div id="lb-panel" role="tabpanel" aria-labelledby={`lb-tab-${m.tab}`}>
        {!m.data ? (
          <p>Loading the leaderboard...</p>
        ) : m.tab === "people" ? (
          <PeopleTable data={m.data} agent={m.agent} preset={m.preset} setPreset={m.setPreset} presets={m.presets} />
        ) : (
          <>
            <p className="leaderboard__intro">{m.intro}</p>
            <table className="survival-table">
              <thead>
                <tr>
                  <th scope="col">Agent</th>
                  <th scope="col" className="survival-table__axis">
                    <span className="visually-hidden">Top-out rate, 0 to {pct(scale)}</span>
                    <span className="chart-track" aria-hidden="true">
                      {ticks.map((t) => (
                        <span key={t} className="axis-tick" style={{ left: `${pos(t, scale)}%` }}>
                          {pct(t)}
                        </span>
                      ))}
                    </span>
                  </th>
                  <th scope="col" className="num">
                    Pieces survived
                  </th>
                  <th scope="col" className="num">
                    Score per piece
                  </th>
                </tr>
              </thead>
              <tbody>
                {s!.rows.map((r) => {
                  const a = m.agent(r.key);
                  const color = a?.looks.tetromino.fill ?? "#6B7080";
                  return (
                    <tr key={r.key}>
                      <th scope="row">
                        <div className="survival-table__agent">
                        <span className="survival-table__icon">
                          <PieceIcon agent={a} cell={12} />
                        </span>
                        <span>
                          <span className="survival-table__name">{a?.name ?? r.key}</span>
                          <span className="survival-table__family">{a?.family}</span>
                        </span>
                        </div>
                      </th>
                      <td className="survival-table__chart">
                        <div className="chart-track chart-grid" aria-hidden="true">
                          {ticks.map((t) => (
                            <span key={t} style={{ left: `${pos(t, scale)}%` }} />
                          ))}
                        </div>
                        {r.status === "valid" ? (
                          <Interval r={r} scale={scale} color={color} />
                        ) : (
                          <span className="survival-table__none">{r.status === "pending_rerun" ? "Pending rerun" : "Not evaluated yet"}</span>
                        )}
                      </td>
                      <td className="num">{r.status === "valid" ? r.pieces_survived!.toFixed(1) : "-"}</td>
                      <td className="num">{r.status === "valid" ? fmtInt(r.score_per_piece!) : "-"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className="leaderboard__note">
              <SignificanceNote survival={s!} />
              {pendingNotes(m.data, m.agent).map((t) => (
                <p key={t}>{t}</p>
              ))}
            </div>
          </>
        )}
      </div>
    </main>
  );
}
