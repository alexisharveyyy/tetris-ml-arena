import { unavailableText } from "../../../lib/copy";
import { PieceIcon } from "../../../components/PieceIcon";
import type { OpponentsModel } from "../../../screens/models";
import { Plate } from "./parts";

export function BroadcastOpponents(m: OpponentsModel) {
  const n = m.slots.filter(Boolean).length;
  return (
    <div className="screen bc opponents">
      <header className="bc-pagehead">
        <a className="link" href={m.backHref}>
          Back
        </a>
        <h1 className="bc-title">Pick your lineup</h1>
        <div className="bc-tabs" role="group" aria-label="Mode">
          {(["marathon", "race"] as const).map((mode) => (
            <button key={mode} type="button" className="plate bc-tab" aria-pressed={m.mode === mode} onClick={() => m.setMode(mode)}>
              <span className="plate__in">{mode === "marathon" ? "Marathon" : "Race"}</span>
            </button>
          ))}
        </div>
      </header>
      <div className="bc-cols">
        <section aria-label="Agents">
          <p className="bc-hint">{m.instruction}</p>
          {m.groups.map((g) => (
            <div key={g.title ?? "main"}>
              {g.title ? <h2 className="bc-h2 bc-group">{g.title}</h2> : null}
              <ul className="bc-agents">
                {g.rows.map((r) => (
                  <li key={r.agent.key}>
                    <Plate className={`plate--deep bc-agent${r.inSlots ? " is-in" : ""}${r.available ? "" : " is-off"}`}>
                      <span className="bc-agent__icon">
                        <PieceIcon agent={r.agent} cell={11} />
                      </span>
                      <span className="bc-agent__text">
                        <span className="bc-agent__name">{r.agent.name}</span>
                        <span className="bc-agent__blurb">{r.agent.blurb}</span>
                        <span className="bc-agent__record">Your record: {r.record}</span>
                      </span>
                      {r.available ? (
                        <button
                          type="button"
                          className={`bc-add${r.inSlots ? " bc-add--added" : ""}`}
                          onClick={() => m.add(r.agent.key)}
                          disabled={m.nextSlot === null}
                          aria-label={m.nextSlot === null ? `${r.agent.name}: all slots are full` : `Add ${r.agent.name} to slot ${m.nextSlot}`}
                        >
                          {r.inSlots ? "Added" : "Add"}
                        </button>
                      ) : (
                        <span className="bc-agent__na">{unavailableText(r.agent)}</span>
                      )}
                    </Plate>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </section>
        <aside className="bc-panel" aria-label="Your match">
          {m.slots.map((s, i) =>
            s ? (
              <div key={i} className="bc-slot">
                <div className="bc-slot__head">
                  <div>
                    <p className="bc-slot__label">Slot {i + 1}</p>
                    <p className="bc-slot__name">{s.agent.name}</p>
                  </div>
                  <button type="button" className="link" onClick={() => m.remove(i)} aria-label={`Remove ${s.agent.name} from slot ${i + 1}`}>
                    Remove
                  </button>
                </div>
                <div className="bc-diff" role="group" aria-label={`Slot ${i + 1} difficulty`}>
                  {m.presets.map((p) => (
                    <button key={p.name} type="button" className="plate bc-diff__opt" aria-pressed={s.preset === p.name} onClick={() => m.setPreset(i, p.name)}>
                      <span className="plate__in">{p.name}</span>
                    </button>
                  ))}
                </div>
                <p className="bc-slot__desc">{s.presetText}</p>
              </div>
            ) : (
              <div key={i} className={`bc-slot bc-slot--open${m.nextSlot === i + 1 ? " is-next" : ""}`}>
                <p className="bc-slot__label">Slot {i + 1}</p>
                <p className="bc-slot__name">Open slot</p>
              </div>
            ),
          )}
          <p className="bc-mode">{m.modeText}</p>
          {m.error ? (
            <p className="form-error" role="alert">
              {m.error}
            </p>
          ) : null}
          <button type="button" className="plate plate--coral plate--button bc-start" onClick={m.start} disabled={!m.canStart}>
            <span className="plate__in">{m.starting ? "Starting..." : n ? `Start ${m.mode === "marathon" ? "Marathon" : "Race"} vs ${n}` : "Pick an opponent"}</span>
          </button>
        </aside>
      </div>
    </div>
  );
}
