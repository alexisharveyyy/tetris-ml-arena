// Board panels shared by the match, Watch AI, Solo and Record screens. Looks
// restyle them through CSS; Terminal, Broadcast and Poster also have their own
// match layouts built from the same Board and PiecePreview parts.
import type { Agent, PlayerSnap } from "../api/types";
import { agentStatusLine } from "../lib/copy";
import { ModelFallbacks } from "./ModelFallbacks";
import { fmtInt, ordinal } from "../lib/format";
import { Board } from "./Board";
import { PieceIcon } from "./PieceIcon";
import { PiecePreview } from "./PiecePreview";

interface Stat {
  label: string;
  value: string;
}

export function StatRow({ stats, className }: { stats: Stat[]; className?: string }) {
  return (
    <dl className={`stats${className ? ` ${className}` : ""}`}>
      {stats.map((s) => (
        <div className="stats__item" key={s.label}>
          <dt className="stats__label">{s.label}</dt>
          <dd className="stats__value">{s.value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function humanStats(p: PlayerSnap): Stat[] {
  return [
    { label: "Score", value: fmtInt(p.stats.score) },
    { label: "Lines", value: String(p.stats.lines) },
    { label: "Level", value: String(p.stats.level) },
    { label: "Pieces/s", value: p.stats.pps.toFixed(2) },
  ];
}

export function agentStats(p: PlayerSnap): Stat[] {
  return [
    { label: "Score", value: fmtInt(p.stats.score) },
    { label: "Lines", value: String(p.stats.lines) },
    { label: "Level", value: String(p.stats.level) },
  ];
}

interface HumanProps {
  player: PlayerSnap;
  cell: number;
  /** "3rd of 4" (omitted in Solo and Record). */
  place?: string;
  title?: string;
}

export function HumanPanel({ player, cell, place, title = "You" }: HumanProps) {
  return (
    <section className="human-panel" aria-label="Your board">
      <header className="human-panel__head">
        <h2 className="human-panel__title">{title}</h2>
        {place ? <span className="human-panel__place">{place}</span> : null}
      </header>
      <div className="human-panel__play">
        <div className="side side--hold">
          <h3 className="side__label">Hold</h3>
          <div className="side__box">
            <PiecePreview letter={player.hold} cell={Math.round(cell * 0.55)} dim={!player.can_hold} />
          </div>
        </div>
        <Board player={player} agent={null} cell={cell} human showThinking={false} />
        <div className="side side--next">
          <h3 className="side__label">Next</h3>
          <ol className="side__box side__queue" aria-label={`Next pieces: ${player.next.join(", ")}`}>
            {player.next.map((letter, i) => (
              <li key={i}>
                <PiecePreview letter={letter} cell={Math.round(cell * 0.45)} />
              </li>
            ))}
          </ol>
        </div>
      </div>
      <StatRow stats={humanStats(player)} className="stats--human" />
    </section>
  );
}

interface AgentProps {
  player: PlayerSnap;
  agent: Agent | null;
  cell: number;
  showThinking: boolean;
  /** Total players, for "1st". */
  of?: number;
}

export function AgentCard({ player, agent, cell, showThinking, of }: AgentProps) {
  const fill = agent?.looks.tetromino.fill;
  const text = agent?.looks.tetromino.text;
  return (
    <section className={`agent-card agent-card--${player.status.state}`} aria-label={`${player.name} board`}>
      <header className="agent-card__head" style={{ background: fill, color: text }}>
        <PieceIcon agent={agent} cell={14} color={text} className="agent-card__icon" />
        <div className="agent-card__who">
          <h2 className="agent-card__name">{player.name}</h2>
          <p className="agent-card__meta">
            {agent?.family}, {player.preset}
          </p>
        </div>
        {of ? <span className="agent-card__place">{ordinal(player.place)}</span> : null}
      </header>
      <div className="agent-card__board">
        <Board player={player} agent={agent} cell={cell} showThinking={showThinking} />
      </div>
      <StatRow stats={agentStats(player)} className="stats--agent" />
      <p className="agent-card__status">
        {agentStatusLine(player, showThinking)}
      </p>
      <ModelFallbacks player={player} className="agent-card__status" />
    </section>
  );
}
