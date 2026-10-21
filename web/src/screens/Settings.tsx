import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Settings } from "../api/types";
import { Button } from "../components/Button";
import { KeyCap } from "../components/KeyCap";
import { LookSwitcher } from "../components/LookSwitcher";
import { keyName } from "../lib/keys";
import { useApp } from "../state/app";
import { allThemes } from "../themes/registry";
import { PageHead } from "./PageHead";

const ACTIONS: [string, string][] = [
  ["left", "Move left"],
  ["right", "Move right"],
  ["soft", "Soft drop"],
  ["hard", "Hard drop"],
  ["cw", "Rotate clockwise"],
  ["ccw", "Rotate counterclockwise"],
  ["hold", "Hold"],
  ["pause", "Pause"],
];

function NumberSetting({ id, label, help, value, min, max, step, onChange }: {
  id: string; label: string; help: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void;
}) {
  const [text, setText] = useState(String(value));
  useEffect(() => setText(String(value)), [value]);
  const commit = () => {
    const v = Number(text);
    if (Number.isFinite(v) && v >= min && v <= max) onChange(v);
    else setText(String(value));
  };
  return (
    <div className="setting">
      <label htmlFor={id} className="setting__label">
        {label}
      </label>
      <input id={id} className="setting__input" type="number" min={min} max={max} step={step} value={text}
        onChange={(e) => setText(e.target.value)} onBlur={commit} onKeyDown={(e) => e.key === "Enter" && commit()}
        aria-describedby={`${id}-help`} />
      <p id={`${id}-help`} className="setting__help">
        {help}
      </p>
    </div>
  );
}

export function SettingsScreen() {
  const app = useApp();
  const s = app.settings;
  const [capturing, setCapturing] = useState<string | null>(null);
  const [defaults, setDefaults] = useState<Settings | null>(null);
  useEffect(() => {
    api.settingsDefaults().then(setDefaults).catch(() => undefined);
  }, []);
  useEffect(() => {
    if (!capturing || !s) return;
    const onKey = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const name = keyName(e);
      if (name === "escape" && capturing !== "pause") {
        setCapturing(null);
        return;
      }
      // The new key becomes the action's first key and is removed from every other action.
      const bindings = Object.fromEntries(Object.entries(s.bindings).map(([a, keys]) => [a, keys.filter((k) => k !== name)]));
      bindings[capturing] = [name, ...(bindings[capturing] ?? [])];
      setCapturing(null);
      void app.updateSettings({ bindings });
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [capturing, s, app]);
  if (!s) return <main className="screen"><p>Loading settings...</p></main>;
  return (
    <main className="screen settings">
      <PageHead title="Settings" />
      <section className="settings__group" aria-labelledby="set-look">
        <h2 id="set-look" className="section-title">
          Look
        </h2>
        <LookSwitcher looks={allThemes().map((t) => ({ id: t.id, label: t.label }))} look={app.look} setLook={app.setLook} />
        <p className="setting__help">Switching takes effect immediately, including during a match.</p>
      </section>
      <section className="settings__group" aria-labelledby="set-display">
        <h2 id="set-display" className="section-title">
          Display
        </h2>
        <label className="check">
          <input type="checkbox" checked={s.show_thinking} onChange={(e) => void app.updateSettings({ show_thinking: e.target.checked })} />
          Show model thinking (agents' top three candidate placements)
        </label>
        <label className="check">
          <input type="checkbox" checked={s.show_ghost} onChange={(e) => void app.updateSettings({ show_ghost: e.target.checked })} />
          Show the ghost piece
        </label>
      </section>
      <section className="settings__group" aria-labelledby="set-timing">
        <h2 id="set-timing" className="section-title">
          Input timing
        </h2>
        <NumberSetting id="das" label="DAS (ms)" help="How long you hold left or right before the piece starts sliding." value={s.das_ms} min={0} max={1000} step={1} onChange={(v) => void app.updateSettings({ das_ms: v })} />
        <NumberSetting id="arr" label="ARR (ms)" help="Time between slides once it starts. 0 moves straight to the wall." value={s.arr_ms} min={0} max={500} step={1} onChange={(v) => void app.updateSettings({ arr_ms: v })} />
        <NumberSetting id="soft" label="Soft drop (rows per second)" help="How fast a held soft drop moves the piece down." value={s.soft_drop_rate} min={1} max={1000} step={1} onChange={(v) => void app.updateSettings({ soft_drop_rate: v })} />
      </section>
      <section className="settings__group" aria-labelledby="set-keys">
        <h2 id="set-keys" className="section-title">
          Keys
        </h2>
        <table className="keys-table">
          <thead>
            <tr>
              <th scope="col">Action</th>
              <th scope="col">Keys</th>
              <th scope="col">
                <span className="visually-hidden">Change</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {ACTIONS.map(([action, label]) => (
              <tr key={action}>
                <th scope="row">{label}</th>
                <td>
                  {(s.bindings[action] ?? []).map((k) => (
                    <KeyCap key={k} name={k} />
                  ))}
                </td>
                <td>
                  <Button variant="secondary" onClick={() => setCapturing(action)} pressed={capturing === action} label={`Change the key for ${label}`}>
                    {capturing === action ? "Press a key (Esc cancels)" : "Change"}
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {defaults ? (
          <Button variant="link" onClick={() => void app.updateSettings({ bindings: defaults.bindings, das_ms: defaults.das_ms, arr_ms: defaults.arr_ms, soft_drop_rate: defaults.soft_drop_rate })}>
            Reset keys and timing to defaults
          </Button>
        ) : null}
      </section>
    </main>
  );
}
