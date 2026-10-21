import { unavailableText } from "../../../lib/copy";
import { PieceIcon } from "../../../components/PieceIcon";
import type { OpponentsModel } from "../../../screens/models";

export function PosterOpponents(m: OpponentsModel) {
  const n = m.slots.filter(Boolean).length;
  return (
    <div className="screen po opponents">
      <a className="link" href={m.backHref}>
        Back
      </a>
      <header className="po-pagehead">
        <h1 className="po-title">Pick up to three</h1>
        <div className="po-toggle" role="group" aria-label="Mode">
          {(["marathon", "race"] as const).map((mode) => (
            <button key={mode} type="button" aria-pressed={m.mode === mode} onClick={() => m.setMode(mode)}>
              {mode === "marathon" ? "Marathon" : "Race"}
            </button>
          ))}
        </div>
      </header>
      <p className="po-hint">{m.instruction}</p>
      <div className="po-cols">
        <section aria-label="Agents">
          {m.groups.map((g) => (
            <div key={g.title ?? "main"}>
              {g.title ? <h2 className="po-h2">{g.title}</h2> : null}
              <ul className="po-tiles">
                {g.rows.map((r) => {
                  const c = r.agent.looks.poster;
                  return (
                    <li
                      key={r.agent.key}
                      className={`po-tile${r.available ? "" : " is-off"}`}
                      style={r.available ? { background: c.field, color: c.text } : { boxShadow: `inset 8px 0 0 ${c.field}` }}
                    >
                      <div className="po-tile__top">
                        <PieceIcon agent={r.agent} cell={14} color={r.available ? c.text : "#E6E8EF"} />
                        {r.inSlots ? <span className="po-tile__in">In {r.inSlots} slot{r.inSlots > 1 ? "s" : ""}</span> : null}
                      </div>
                      <h3 className="po-tile__name">{r.agent.name}</h3>
                      <p className="po-tile__blurb">{r.agent.blurb}</p>
                      <p className="po-tile__record">Your record: {r.record}</p>
                      {r.available ? (
                        <button
                          type="button"
                          className="po-tile__add"
                          onClick={() => m.add(r.agent.key)}
                          disabled={m.nextSlot === null}
                          aria-label={m.nextSlot === null ? `${r.agent.name}: all slots are full` : `Add ${r.agent.name} to slot ${m.nextSlot}`}
                        >
                          {m.nextSlot === null ? "Slots full" : `Add to slot ${m.nextSlot}`}
                        </button>
                      ) : (
                        <p className="po-tile__na">{unavailableText(r.agent)}</p>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </section>
        <aside className="po-panel" aria-label="Your match">
          {m.slots.map((s, i) => {
            if (!s) {
              return (
                <div key={i} className={`po-slot po-slot--open${m.nextSlot === i + 1 ? " is-next" : ""}`}>
                  <span className="po-slot__num">{i + 1}</span>
                  <span className="po-slot__name">Open slot</span>
                </div>
              );
            }
            const c = s.agent.looks.poster;
            return (
              <div key={i} className="po-slot" style={{ background: c.field, color: c.text, ["--po-text" as string]: c.text, ["--po-field" as string]: c.field }}>
                <div className="po-slot__head">
                  <span className="po-slot__num">{i + 1}</span>
                  <span className="po-slot__name">{s.agent.name}</span>
                  <button type="button" className="link" onClick={() => m.remove(i)} aria-label={`Remove ${s.agent.name} from slot ${i + 1}`}>
                    Remove
                  </button>
                </div>
                <div className="po-diff" role="group" aria-label={`Slot ${i + 1} difficulty`}>
                  {m.presets.map((p) => (
                    <button key={p.name} type="button" aria-pressed={s.preset === p.name} onClick={() => m.setPreset(i, p.name)}>
                      {p.name}
                    </button>
                  ))}
                </div>
                <p className="po-slot__desc">{s.presetText}</p>
              </div>
            );
          })}
          <p className="po-mode">{m.modeText}</p>
          {m.error ? (
            <p className="form-error" role="alert">
              {m.error}
            </p>
          ) : null}
          <button type="button" className="po-btn po-btn--yellow po-start" onClick={m.start} disabled={!m.canStart}>
            {m.starting ? "Starting..." : n ? `Start ${m.mode === "marathon" ? "Marathon" : "Race"} against ${n}` : "Pick an opponent"}
          </button>
        </aside>
      </div>
    </div>
  );
}
