import { unavailableText } from "../../../lib/copy";
import { go } from "../../../router";
import type { OpponentsModel } from "../../../screens/models";
import { useShortcuts } from "../../../state/match";

export function TerminalOpponents(m: OpponentsModel) {
  useShortcuts({ escape: () => go("/"), return: () => m.canStart && m.start() });
  const n = m.slots.filter(Boolean).length;
  return (
    <div className="screen term opponents">
      <header className="term-head term-head--page">
        <a className="term-link" href={m.backHref}>
          [Esc] Back
        </a>
        <h1 className="term-h1">Select up to 3 opponents</h1>
        <div className="term-mode" role="group" aria-label="Mode">
          <span className="term-dim" aria-hidden="true">
            Mode:
          </span>
          {(["marathon", "race"] as const).map((mode) => (
            <button key={mode} type="button" className="term-opt" aria-pressed={m.mode === mode} onClick={() => m.setMode(mode)}>
              {mode}
            </button>
          ))}
        </div>
      </header>
      <div className="term-cols">
        <section aria-label="Agents">
          <p className="term-dim term-hint">{m.instruction}</p>
          {m.groups.map((g) => (
            <div key={g.title ?? "main"}>
              {g.title ? <h2 className="term-h2 term-group">{g.title}</h2> : null}
              <ul className="term-agents">
                {g.rows.map((r) => (
                  <li key={r.agent.key} className={`term-agent${r.available ? "" : " is-dim"}`}>
                    <span className="term-agent__id">{r.agent.looks.terminal.letter}</span>
                    <span className="term-agent__name">{r.agent.name}</span>
                    <span className="term-agent__family">{r.agent.family}</span>
                    <span className="term-agent__slots term-dim">{r.inSlots ? `In ${r.inSlots} slot${r.inSlots > 1 ? "s" : ""}` : ""}</span>
                    {r.available ? (
                      <button
                        type="button"
                        className="term-btn"
                        onClick={() => m.add(r.agent.key)}
                        disabled={m.nextSlot === null}
                        aria-label={m.nextSlot === null ? `${r.agent.name}: all slots are full` : `Add ${r.agent.name} to slot ${m.nextSlot}`}
                      >
                        [ Add ]
                      </button>
                    ) : (
                      <span className="term-agent__na">{unavailableText(r.agent)}</span>
                    )}
                    <span className="term-agent__record term-dim">Record: {r.record}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </section>
        <aside className="term-box" aria-labelledby="your-match">
          <h2 id="your-match" className="term-h2">
            Your match
          </h2>
          {m.slots.map((s, i) =>
            s ? (
              <div key={i} className="term-slot">
                <p className="term-slot__head">
                  <span>
                    Slot {i + 1}: {s.agent.looks.terminal.letter} {s.agent.name}
                  </span>
                  <button type="button" className="term-link term-dim" onClick={() => m.remove(i)} aria-label={`Remove ${s.agent.name} from slot ${i + 1}`}>
                    [X] Remove
                  </button>
                </p>
                <div className="term-diff" role="group" aria-label={`Slot ${i + 1} difficulty`}>
                  {m.presets.map((p) => (
                    <button key={p.name} type="button" className="term-opt" aria-pressed={s.preset === p.name} onClick={() => m.setPreset(i, p.name)}>
                      {s.preset === p.name ? `[${p.name}]` : p.name}
                    </button>
                  ))}
                </div>
                <p className="term-dim">{s.presetText}</p>
              </div>
            ) : (
              <p key={i} className={`term-slot term-slot--empty${m.nextSlot === i + 1 ? " is-next" : ""}`}>
                {m.nextSlot === i + 1 ? "> " : "  "}Slot {i + 1}: -- empty --
              </p>
            ),
          )}
          <p className="term-dim term-mode-text">{m.modeText}</p>
          {m.error ? (
            <p className="form-error" role="alert">
              {m.error}
            </p>
          ) : null}
          <button type="button" className="term-cta" onClick={m.start} disabled={!m.canStart}>
            {m.starting ? "Starting..." : n ? `Start ${m.mode} vs ${n} [Enter]` : "Pick an opponent"}
          </button>
        </aside>
      </div>
    </div>
  );
}
