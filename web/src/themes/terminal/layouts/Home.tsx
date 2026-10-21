import { isChecking } from "../../../lib/copy";
import { useRef, useState } from "react";
import { LookSwitcher } from "../../../components/LookSwitcher";
import { keyLabel } from "../../../lib/keys";
import { go } from "../../../router";
import type { HomeModel } from "../../../screens/models";
import { useShortcuts } from "../../../state/match";

export const LETTERS = ["A", "B", "C", "D", "E", "F"];

export function TermHeader({ m }: { m: Pick<HomeModel, "looks" | "look" | "setLook"> }) {
  return (
    <header className="term-head">
      <a className="term-title" href="#/">
        Tetris ML Arena
      </a>
      <LookSwitcher looks={m.looks} look={m.look} setLook={m.setLook} format={(o, i) => `[${LETTERS[i]}] ${o.label}`} />
    </header>
  );
}

export function TerminalHome(m: HomeModel) {
  const items = [
    { label: "Marathon", desc: "Side by side, ends when you top out", href: m.links.marathon },
    { label: "Race", desc: `Most lines in ${m.raceMinutes} minutes`, href: m.links.race },
    { label: "Watch AI", desc: "One agent, top 3 spots shown", href: m.links.watch },
    { label: "Solo practice", desc: "Warm up alone", href: m.links.solo },
    { label: "Record games", desc: "Training data for cloning", href: m.links.record },
    { label: "Leaderboard", desc: "Research eval and records", href: m.links.leaderboard },
    { label: "Settings", desc: "Keys, DAS, ARR", href: m.links.settings },
  ];
  const [sel, setSel] = useState(0);
  const refs = useRef<(HTMLAnchorElement | null)[]>([]);
  const select = (i: number) => {
    setSel(i);
    refs.current[i]?.focus();
  };
  const keys: Record<string, () => void> = {
    return: () => go(items[sel].href),
    up: () => select((sel + items.length - 1) % items.length),
    down: () => select((sel + 1) % items.length),
  };
  items.forEach((_, i) => (keys[String(i + 1)] = () => select(i)));
  useShortcuts(keys);
  return (
    <div className="screen term home">
      <TermHeader m={m} />
      <div className="term-home">
        <main className="term-home__main">
          <pre className="term-logo" aria-hidden="true">{"[][][]\n  []"}</pre>
          <h1 className="term-hero">Can you outscore the models?</h1>
          <p className="term-lead">
            Same pieces, separate boards. Play next to up to three learning agents. When you top out, the highest score wins.
          </p>
          <nav aria-label="Menu">
            <ol className="term-menu">
              {items.map((it, i) => (
                <li key={it.label}>
                  <a
                    ref={(el) => {
                      refs.current[i] = el;
                    }}
                    href={it.href}
                    className={`term-menu__row${i === sel ? " is-selected" : ""}`}
                    aria-current={i === sel ? "true" : undefined}
                    onFocus={() => setSel(i)}
                    onMouseEnter={() => setSel(i)}
                  >
                    <span className="term-menu__caret" aria-hidden="true">
                      {i === sel ? ">" : " "}
                    </span>
                    <span className="term-menu__num">{i + 1}</span>
                    <span className="term-menu__label">{it.label}</span>
                    <span className="term-menu__desc">{it.desc}</span>
                  </a>
                </li>
              ))}
            </ol>
          </nav>
          <p className="term-prompt">
            Press 1-7 or Enter <span className="cursor" aria-hidden="true">_</span>
          </p>
        </main>
        <aside className="term-roster" aria-labelledby="roster">
          <h2 id="roster" className="term-h2">
            Agent roster
          </h2>
          <table className="term-table">
            <thead>
              <tr>
                <th scope="col">ID</th>
                <th scope="col">Agent</th>
                <th scope="col">Method</th>
                <th scope="col">
                  <span aria-hidden="true">Up</span>
                  <span className="visually-hidden">Available</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {m.lineup.map((a) => (
                <tr key={a.key} className={a.available ? undefined : "is-dim"}>
                  <td>{a.looks.terminal.letter}</td>
                  <td>{a.name}</td>
                  <td>{a.family}</td>
                  <td>{a.available ? "OK" : isChecking(a) ? <abbr title="Checking...">...</abbr> : <abbr title="Not available yet">N/A</abbr>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="term-note">
            Each agent's ID is its piece. The two cloning agents share the J. The baseline is one block. The LLM is a five-block P.
            N/A means not available yet.
          </p>
        </aside>
      </div>
      <footer className="term-footer" aria-label="Controls">
        {m.controls.map((c) => (
          <span key={c.label}>
            {c.keys.map(keyLabel).join(" ")} {c.label}
          </span>
        ))}
      </footer>
    </div>
  );
}
