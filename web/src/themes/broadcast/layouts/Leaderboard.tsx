import { PeopleTable, pendingNotes } from "../../../components/PeopleTable";
import { PieceIcon } from "../../../components/PieceIcon";
import { pct } from "../../../lib/format";
import type { LeaderboardModel } from "../../../screens/models";
import { intervalText, LEADERBOARD_TABS, pos } from "../../tetromino/layouts/Leaderboard";

export function BroadcastLeaderboard(m: LeaderboardModel) {
  const s = m.data?.survival;
  const scale = s?.scale_max ?? 0.5;
  const ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
  return (
    <main className="screen bc leaderboard">
      <header className="bc-pagehead">
        <a className="link" href={m.homeHref}>
          Home
        </a>
        <h1 className="bc-title">Standings</h1>
        <div role="tablist" aria-label="Leaderboard" className="bc-tabs">
          {LEADERBOARD_TABS.map((t) => (
            <button key={t.id} id={`lb-tab-${t.id}`} role="tab" type="button" aria-selected={m.tab === t.id} aria-controls="lb-panel" className="plate bc-tab" onClick={() => m.setTab(t.id)}>
              <span className="plate__in">{t.label}</span>
            </button>
          ))}
        </div>
      </header>
      <div id="lb-panel" role="tabpanel" aria-labelledby={`lb-tab-${m.tab}`}>
        {!m.data ? (
          <p>Loading...</p>
        ) : m.tab === "people" ? (
          <PeopleTable data={m.data} agent={m.agent} preset={m.preset} setPreset={m.setPreset} presets={m.presets} />
        ) : (
          <>
            <p className="bc-lead">{m.intro.replace("Bars show", "Yellow bars are")}</p>
            <div className="bc-board" role="table" aria-label="Top-out rate under garbage">
              <div className="bc-board__head" role="row">
                <span role="columnheader">Agent</span>
                <span role="columnheader" className="bc-board__axis">
                  <span className="visually-hidden">Top-out rate with interval</span>
                  {ticks.map((t) => (
                    <span key={t} className="bc-tick" style={{ left: `${pos(t, scale)}%` }} aria-hidden="true">
                      {pct(t)}
                    </span>
                  ))}
                </span>
                <span role="columnheader" className="num">
                  Rate
                </span>
              </div>
              {s!.rows.map((r) => {
                const a = m.agent(r.key);
                return (
                  <div key={r.key} className="bc-board__row" role="row">
                    <span role="rowheader" className="bc-board__agent">
                      <PieceIcon agent={a} cell={10} />
                      <span>{a?.name ?? r.key}</span>
                    </span>
                    <span role="cell" className="bc-board__chart">
                      <span className="bc-grid" aria-hidden="true">
                        {ticks.map((t) => (
                          <span key={t} style={{ left: `${pos(t, scale)}%` }} />
                        ))}
                      </span>
                      {r.status === "valid" ? (
                        <>
                          <span className="visually-hidden">{intervalText(r)}</span>
                          <span className="bc-ci" style={{ left: `${pos(r.lo!, scale)}%`, width: `${Math.max(pos(r.hi!, scale) - pos(r.lo!, scale), 0.5)}%` }} aria-hidden="true" />
                          {r.off_scale ? (
                            <span className="bc-off" aria-hidden="true">
                              Off scale
                            </span>
                          ) : (
                            <span className="bc-dot" style={{ left: `${pos(r.top_out_rate!, scale)}%` }} aria-hidden="true" />
                          )}
                        </>
                      ) : (
                        <span className="bc-none">{r.status === "pending_rerun" ? "Pending rerun" : "Not evaluated yet"}</span>
                      )}
                    </span>
                    <span role="cell" className="num bc-board__rate">
                      {r.status === "valid" ? pct(r.top_out_rate!) : "-"}
                    </span>
                  </div>
                );
              })}
            </div>
            <p className="bc-note">
              {s!.method} {s!.significance}
            </p>
            {pendingNotes(m.data, m.agent).map((t) => (
              <p key={t} className="bc-note">
                {t}
              </p>
            ))}
          </>
        )}
      </div>
    </main>
  );
}
