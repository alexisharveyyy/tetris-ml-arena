import { Button } from "../../../components/Button";
import { KeyCap } from "../../../components/KeyCap";
import { LookSwitcher } from "../../../components/LookSwitcher";
import { PieceIcon } from "../../../components/PieceIcon";
import type { HomeModel } from "../../../screens/models";
import { useShortcuts } from "../../../state/match";

export function Logo() {
  return (
    <span className="logo" aria-hidden="true">
      <span />
      <span />
      <span />
      <span />
    </span>
  );
}

export function TetrominoHome(m: HomeModel) {
  useShortcuts({ return: m.startMarathon, enter: m.startMarathon });
  return (
    <div className="screen home">
      <header className="topbar">
        <a className="brand" href="#/">
          <Logo />
          <span>Tetris ML Arena</span>
        </a>
        <nav className="topbar__nav" aria-label="Main">
          <LookSwitcher looks={m.looks} look={m.look} setLook={m.setLook} />
          <a className="link link--strong" href={m.links.leaderboard}>
            Leaderboard
          </a>
          <a className="link link--strong" href={m.links.settings}>
            Settings
          </a>
        </nav>
      </header>
      <main className="home__main">
        <section className="home__intro">
          <h1 className="home__title">Can you outscore the models?</h1>
          <p className="home__lead">
            Play normal Tetris next to up to three learning agents. Everyone gets the same pieces on their own board. When you
            top out, the highest score wins.
          </p>
          <div className="home__cta">
            <Button href={m.links.marathon} shortcut="return">
              Start a Marathon
            </Button>
            <Button variant="secondary" href={m.links.race}>
              Race the clock
            </Button>
          </div>
          <p className="home__note">
            Each agent plays as its own piece. The two behavior cloning agents share the J. The hand-tuned baseline is a single
            block, and the LLM, the odd one out, is a five-block piece.
          </p>
        </section>
        <section className="home__lineup" aria-labelledby="lineup-title">
          <h2 id="lineup-title" className="section-title">
            Who you can play
          </h2>
          <ul className="lineup-grid">
            {m.lineup.map((a) => (
              <li key={a.key} className={`lineup-card${a.available ? "" : " lineup-card--off"}`}>
                <PieceIcon agent={a} cell={20} />
                <span className="lineup-card__name">{a.name}</span>
                <span className="lineup-card__family">{a.family}</span>
                {a.available ? null : <span className="lineup-card__status">Not available yet</span>}
              </li>
            ))}
          </ul>
        </section>
      </main>
      <section className="home__more" aria-labelledby="more-title">
        <h2 id="more-title" className="section-title">
          Other ways to play
        </h2>
        <ul className="more-list">
          <li>
            <a href={m.links.watch}>
              <span className="more-list__title">Watch AI</span>
              <span className="more-list__desc">Watch one agent play, with its top three candidate placements drawn on the board.</span>
            </a>
          </li>
          <li>
            <a href={m.links.solo}>
              <span className="more-list__title">Solo practice</span>
              <span className="more-list__desc">Warm up alone before a match.</span>
            </a>
          </li>
          <li>
            <a href={m.links.record}>
              <span className="more-list__title">Record your games</span>
              <span className="more-list__desc">Save your play as training data for the behavior cloning agent.</span>
            </a>
          </li>
          <li>
            <a href={m.links.leaderboard}>
              <span className="more-list__title">Leaderboard</span>
              <span className="more-list__desc">How each agent ranks in the research evaluation and against people.</span>
            </a>
          </li>
        </ul>
      </section>
      <footer className="controls" aria-label="Controls">
        {m.controls.map((c) => (
          <span className="controls__item" key={c.label}>
            {c.keys.map((k) => (
              <KeyCap key={k} name={k} />
            ))}
            <span className="controls__label">{c.label}</span>
          </span>
        ))}
      </footer>
    </div>
  );
}
