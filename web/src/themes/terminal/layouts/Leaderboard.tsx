import { SignificanceNote } from "../../../components/SignificanceNote";
import { PeopleTable, pendingNotes } from "../../../components/PeopleTable";
import { pct } from "../../../lib/format";
import { go } from "../../../router";
import type { LeaderboardModel } from "../../../screens/models";
import { useShortcuts } from "../../../state/match";
import { asciiInterval } from "../board";
import { intervalText } from "../../tetromino/layouts/Leaderboard";

const WIDTH = 52; // characters for 0 to 50%

export function TerminalLeaderboard(m: LeaderboardModel) {
  useShortcuts({ escape: () => go("/"), "1": () => m.setTab("survival"), "2": () => m.setTab("people") });
  const s = m.data?.survival;
  const scale = s?.scale_max ?? 0.5;
  const axis = Array.from({ length: WIDTH }, () => " ");
  [0, 0.1, 0.2, 0.3, 0.4, 0.5].forEach((t) => {
    const label = pct(t);
    const at = Math.round((t / scale) * (WIDTH - 1));
    const start = Math.min(WIDTH - label.length, Math.max(0, at - 1));
    for (let i = 0; i < label.length; i++) axis[start + i] = label[i];
  });
  return (
    <main className="screen term leaderboard">
      <header className="term-head term-head--page">
        <a className="term-link" href={m.homeHref}>
          [Esc] Home
        </a>
        <h1 className="term-h1">Leaderboard</h1>
        <div role="tablist" aria-label="Leaderboard" className="term-tabs">
          {(
            [
              ["survival", "[1] Survival"],
              ["people", "[2] VS people"],
            ] as const
          ).map(([id, label]) => (
            <button key={id} id={`lb-tab-${id}`} role="tab" type="button" aria-selected={m.tab === id} aria-controls="lb-panel" className="term-opt" onClick={() => m.setTab(id)}>
              {label}
            </button>
          ))}
        </div>
      </header>
      <div id="lb-panel" role="tabpanel" aria-labelledby={`lb-tab-${m.tab}`}>
        {!m.data ? (
          <p>Loading...</p>
        ) : m.tab === "people" ? (
          <PeopleTable data={m.data} agent={m.agent} preset={m.preset} setPreset={m.setPreset} presets={m.presets} icons={false} />
        ) : (
          <>
            <p className="term-lead">
              {m.intro.replace("Bars show 95% confidence intervals.", "")} [===*===] is the 95% interval.
            </p>
            <table className="term-table term-chart">
              <thead>
                <tr>
                  <th scope="col">
                    <span className="visually-hidden">ID</span>
                  </th>
                  <th scope="col">
                    <span className="visually-hidden">Agent</span>
                  </th>
                  <th scope="col" className="num">
                    <span className="visually-hidden">Top-out rate</span>
                  </th>
                  <th scope="col" className="term-dim term-axis">
                    <span aria-hidden="true">{axis.join("")}</span>
                    <span className="visually-hidden">Interval chart, 0 to 50 percent</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {s!.rows.map((r) => {
                  const a = m.agent(r.key);
                  return (
                    <tr key={r.key}>
                      <td>{a?.looks.terminal.letter}</td>
                      <th scope="row">{a?.name ?? r.key}</th>
                      <td className="num">{r.status === "valid" ? pct(r.top_out_rate!) : ""}</td>
                      <td>
                        {r.status === "valid" ? (
                          <>
                            <span aria-hidden="true">{asciiInterval(r.lo!, r.hi!, r.top_out_rate!, scale, WIDTH)}</span>
                            <span className="visually-hidden">{intervalText(r)}</span>
                            {r.off_scale ? <span className="term-dim"> off scale</span> : null}
                          </>
                        ) : r.status === "pending_rerun" ? (
                          "Pending rerun"
                        ) : (
                          "Not evaluated yet"
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <SignificanceNote survival={s!} className="term-dim term-note" />
            {pendingNotes(m.data, m.agent).map((t) => (
              <p key={t} className="term-dim term-note">
                {t}
              </p>
            ))}
          </>
        )}
      </div>
    </main>
  );
}
