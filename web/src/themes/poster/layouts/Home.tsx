import { LookSwitcher } from "../../../components/LookSwitcher";
import { PieceIcon } from "../../../components/PieceIcon";
import type { HomeModel } from "../../../screens/models";
import { useShortcuts } from "../../../state/match";

export function PosterHome(m: HomeModel) {
  useShortcuts({ return: m.startMarathon, enter: m.startMarathon });
  const more = [
    { title: "Watch AI", desc: "One agent, its top three spots drawn on the board", href: m.links.watch },
    { title: "Solo practice", desc: "Warm up alone before a match", href: m.links.solo },
    { title: "Record your games", desc: "Training data for behavior cloning", href: m.links.record },
    { title: "Leaderboard", desc: "Research results and your records", href: m.links.leaderboard },
  ];
  return (
    <div className="screen po home">
      <header className="po-top">
        <a className="po-brand" href="#/">
          Tetris ML Arena
        </a>
        <LookSwitcher looks={m.looks} look={m.look} setLook={m.setLook} />
        <nav className="po-top__links" aria-label="Main">
          <a className="link" href={m.links.leaderboard}>
            Leaderboard
          </a>
          <a className="link" href={m.links.settings}>
            Settings
          </a>
        </nav>
      </header>
      <div className="po-home">
        <main className="po-home__main">
          <h1 className="po-hero">Can you outscore the models?</h1>
          <div className="po-home__bottom">
            <p className="po-lead">Play normal Tetris next to up to three learning agents. Same pieces, separate boards. When you top out, the highest score wins.</p>
            <div className="po-cta">
              <a className="po-btn po-btn--yellow" href={m.links.marathon}>
                Start a Marathon
              </a>
              <a className="po-btn po-btn--outline" href={m.links.race}>
                Race the clock
              </a>
            </div>
          </div>
        </main>
        <ul className="po-stripes" aria-label="Who you can play">
          {m.lineup.map((a) => (
            <li key={a.key} className={`po-stripe${a.available ? "" : " is-off"}`} style={{ background: a.looks.poster.field, color: a.looks.poster.text }}>
              <span className="po-stripe__name">{a.name}</span>
              <span className="po-stripe__foot">
                <PieceIcon agent={a} cell={11} color={a.looks.poster.text} />
                <span className="po-stripe__family">{a.available ? a.family : "Not available yet"}</span>
              </span>
            </li>
          ))}
        </ul>
      </div>
      <ul className="po-more">
        {more.map((x) => (
          <li key={x.title}>
            <a href={x.href}>
              <span className="po-more__title">{x.title}</span>
              <span className="po-more__desc">{x.desc}</span>
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
