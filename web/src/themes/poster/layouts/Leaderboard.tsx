import { PeopleTable, pendingNotes } from "../../../components/PeopleTable";
import { pct } from "../../../lib/format";
import type { LeaderboardModel } from "../../../screens/models";
import { intervalText, LEADERBOARD_TABS, pos } from "../../tetromino/layouts/Leaderboard";

export function PosterLeaderboard(m: LeaderboardModel) {
  const s = m.data?.survival;
  const scale = s?.scale_max ?? 0.5;
  return (
    <main className="po leaderboard">
      <div className="po-lb-head">
        <a className="link" href={m.homeHref}>
          Home
        </a>
        <header className="po-pagehead">
          <h1 className="po-title">Leaderboard</h1>
          <div role="tablist" aria-label="Leaderboard" className="po-toggle">
            {LEADERBOARD_TABS.map((t) => (
              <button key={t.id} id={`lb-tab-${t.id}`} role="tab" type="button" aria-selected={m.tab === t.id} aria-controls="lb-panel" onClick={() => m.setTab(t.id)}>
                {t.label}
              </button>
            ))}
          </div>
        </header>
        {m.tab === "survival" ? (
          <p className="po-lead">
            {m.intro.replace("Bars show 95% confidence intervals.", `The bar shows the 95% interval on a 0 to ${pct(scale)} scale.`)}
          </p>
        ) : null}
      </div>
      <div id="lb-panel" role="tabpanel" aria-labelledby={`lb-tab-${m.tab}`}>
        {!m.data ? (
          <p className="po-pad">Loading...</p>
        ) : m.tab === "people" ? (
          <div className="po-pad">
            <PeopleTable data={m.data} agent={m.agent} preset={m.preset} setPreset={m.setPreset} presets={m.presets} />
          </div>
        ) : (
          <>
            <ul className="po-bands">
              {s!.rows.map((r) => {
                const a = m.agent(r.key);
                const c = a?.looks.poster ?? { field: "#6B7080", text: "#FFFFFF" };
                const valid = r.status === "valid";
                return (
                  <li
                    key={r.key}
                    className={`po-band${valid ? "" : " po-band--grey"}`}
                    style={valid ? { background: c.field, color: c.text } : { boxShadow: `inset 8px 0 0 ${c.field}` }}
                  >
                    <span className="po-band__who">
                      <span className="po-band__name">{a?.name ?? r.key}</span>
                      <span className="po-band__family">{a?.family}</span>
                    </span>
                    {valid ? (
                      <>
                        <span className="po-band__chart">
                          <span className="visually-hidden">{intervalText(r)}</span>
                          <span className="po-band__track" aria-hidden="true" style={{ background: c.text }} />
                          <span
                            className="po-band__ci"
                            aria-hidden="true"
                            style={{ left: `${pos(r.lo!, scale)}%`, width: `${Math.max(pos(r.hi!, scale) - pos(r.lo!, scale), 0.6)}%`, background: c.text }}
                          />
                          {r.off_scale ? (
                            <span className="po-band__off" aria-hidden="true">
                              Off scale
                            </span>
                          ) : (
                            <span className="po-band__mark" aria-hidden="true" style={{ left: `${pos(r.top_out_rate!, scale)}%`, borderColor: c.text, background: c.field }} />
                          )}
                        </span>
                        <span className="po-band__rate">{pct(r.top_out_rate!)}</span>
                      </>
                    ) : (
                      <span className="po-band__none">{r.status === "pending_rerun" ? "Pending rerun" : "Not evaluated yet"}</span>
                    )}
                  </li>
                );
              })}
            </ul>
            <div className="po-pad po-note">
              <p>
                {s!.method} {s!.significance}
              </p>
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
