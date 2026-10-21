import { unavailableText } from "../../../lib/copy";
import { Button } from "../../../components/Button";
import { PieceIcon } from "../../../components/PieceIcon";
import { Segmented } from "../../../components/Segmented";
import type { OpponentsModel } from "../../../screens/models";

export function TetrominoOpponents(m: OpponentsModel) {
  return (
    <div className="screen opponents">
      <header className="page-head">
        <a className="link link--strong" href={m.backHref}>
          Back
        </a>
        <h1 className="page-title">Pick up to three opponents</h1>
        <Segmented
          label="Mode"
          className="seg--mode"
          value={m.mode}
          onChange={m.setMode}
          options={[
            { value: "marathon", label: "Marathon" },
            { value: "race", label: "Race" },
          ]}
        />
      </header>
      <div className="opponents__body">
        <section className="opponents__list" aria-label="Agents">
          <p className="opponents__hint">{m.instruction}</p>
          {m.groups.map((g) => (
            <div key={g.title ?? "main"} className="agent-group">
              {g.title ? <h2 className="agent-group__title">{g.title}</h2> : null}
              <ul className="agent-rows">
                {g.rows.map((r) => (
                  <li key={r.agent.key} className={`agent-row${r.available ? "" : " agent-row--off"}`}>
                    <span className="agent-row__icon">
                      <PieceIcon agent={r.agent} cell={18} />
                    </span>
                    <div className="agent-row__text">
                      <p className="agent-row__name">
                        <strong>{r.agent.name}</strong> <span className="agent-row__family">{r.agent.family}</span>
                      </p>
                      <p className="agent-row__blurb">{r.agent.blurb}</p>
                      <p className="agent-row__record">Your record: {r.record}</p>
                    </div>
                    <div className="agent-row__action">
                      {r.available ? (
                        <Button
                          variant={r.inSlots ? "secondary" : "primary"}
                          onClick={() => m.add(r.agent.key)}
                          disabled={m.nextSlot === null}
                          label={
                            m.nextSlot === null
                              ? `${r.agent.name}: all slots are full`
                              : `Add ${r.agent.name} to slot ${m.nextSlot}${r.inSlots ? ` (already in ${r.inSlots} slot${r.inSlots > 1 ? "s" : ""})` : ""}`
                          }
                        >
                          {r.inSlots ? "Added" : "Add"}
                        </Button>
                      ) : (
                        <span className="agent-row__na">{unavailableText(r.agent)}</span>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </section>
        <aside className="match-panel" aria-labelledby="your-match">
          <h2 id="your-match" className="match-panel__title">
            Your match
          </h2>
          <ol className="slots">
            {m.slots.map((s, i) =>
              s ? (
                <li key={i} className="slot">
                  <div className="slot__head">
                    <PieceIcon agent={s.agent} cell={14} />
                    <div>
                      <p className="slot__label">Slot {i + 1}</p>
                      <p className="slot__name">{s.agent.name}</p>
                    </div>
                    <button type="button" className="link slot__remove" onClick={() => m.remove(i)} aria-label={`Remove ${s.agent.name} from slot ${i + 1}`}>
                      Remove
                    </button>
                  </div>
                  <Segmented
                    label={`Slot ${i + 1} difficulty`}
                    className="seg--difficulty"
                    value={s.preset}
                    onChange={(p) => m.setPreset(i, p)}
                    options={m.presets.map((p) => ({ value: p.name, label: p.name }))}
                  />
                  <p className="slot__desc">{s.presetText}</p>
                </li>
              ) : (
                <li key={i} className={`slot slot--empty${m.nextSlot === i + 1 ? " slot--next" : ""}`}>
                  <div className="slot__head">
                    <span className="slot__placeholder" aria-hidden="true" />
                    <div>
                      <p className="slot__label">Slot {i + 1}</p>
                      <p className="slot__name">Empty</p>
                    </div>
                  </div>
                </li>
              ),
            )}
          </ol>
          <p className="match-panel__mode">{m.modeText}</p>
          {m.error ? (
            <p className="form-error" role="alert">
              {m.error}
            </p>
          ) : null}
          <Button className="btn--block" onClick={m.start} disabled={!m.canStart}>
            {m.starting ? "Starting..." : m.startLabel}
          </Button>
        </aside>
      </div>
    </div>
  );
}
