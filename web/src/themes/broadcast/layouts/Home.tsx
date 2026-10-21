import { unavailableText } from "../../../lib/copy";
import { LookSwitcher } from "../../../components/LookSwitcher";
import { PieceIcon } from "../../../components/PieceIcon";
import type { HomeModel } from "../../../screens/models";
import { useShortcuts } from "../../../state/match";
import { Plate, Ticker } from "./parts";

export function BroadcastTopbar({ m }: { m: Pick<HomeModel, "looks" | "look" | "setLook" | "links"> }) {
  return (
    <header className="bc-topbar">
      <a className="bc-brand" href="#/">
        Tetris ML Arena
      </a>
      <LookSwitcher looks={m.looks} look={m.look} setLook={m.setLook} />
      <nav className="bc-topbar__links" aria-label="Main">
        <a className="link" href={m.links.leaderboard}>
          Leaderboard
        </a>
        <a className="link" href={m.links.settings}>
          Settings
        </a>
      </nav>
    </header>
  );
}

export function BroadcastHome(m: HomeModel) {
  useShortcuts({ return: m.startMarathon, enter: m.startMarathon });
  const tiles = [
    { title: "Watch AI", desc: "One agent, top 3 spots shown", href: m.links.watch },
    { title: "Solo practice", desc: "Warm up alone", href: m.links.solo },
    { title: "Record your games", desc: "Training data for cloning", href: m.links.record },
    { title: "Leaderboard", desc: "Research eval and records", href: m.links.leaderboard },
  ];
  return (
    <div className="bc-page">
      <BroadcastTopbar m={m} />
      <div className="screen bc home">
        <div className="bc-home">
          <main className="bc-home__main">
            <Plate className="plate--coral bc-tag">Tonight: human vs. machine</Plate>
            <h1 className="bc-hero">Can you outscore the models?</h1>
            <p className="bc-lead">
              Play normal Tetris next to up to three learning agents. Same pieces, separate boards. When you top out, the highest score
              wins.
            </p>
            <div className="bc-cta">
              <Plate as="a" href={m.links.marathon} className="plate--coral plate--button">
                Start a Marathon
              </Plate>
              <Plate as="a" href={m.links.race} className="plate--outline plate--button">
                Race the clock
              </Plate>
            </div>
          </main>
          <section className="bc-lineup" aria-labelledby="lineup-title">
            <h2 id="lineup-title" className="bc-h2">
              The lineup
            </h2>
            <ul>
              {m.lineup.map((a, i) => (
                <li key={a.key} style={{ marginLeft: i * 3 }}>
                  <Plate className={`plate--deep bc-lineup__row${a.available ? "" : " is-off"}`}>
                    <PieceIcon agent={a} cell={10} />
                    <span className="bc-lineup__name">{a.name}</span>
                    <span className="bc-lineup__family">{a.available ? a.family : unavailableText(a)}</span>
                  </Plate>
                </li>
              ))}
            </ul>
          </section>
        </div>
        <ul className="bc-tiles">
          {tiles.map((t) => (
            <li key={t.title}>
              <Plate as="a" href={t.href} className="plate--white bc-tile">
                <span className="bc-tile__title">{t.title}</span>
                <span className="bc-tile__desc">{t.desc}</span>
              </Plate>
            </li>
          ))}
        </ul>
      </div>
      <Ticker label="Survival eval" lines={m.evalTicker} />
    </div>
  );
}
