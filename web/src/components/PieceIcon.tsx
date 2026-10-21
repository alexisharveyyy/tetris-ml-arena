// An agent's identity piece, in the current look's colors.
import type { Agent } from "../api/types";
import { useApp } from "../state/app";

interface Props {
  agent: Agent | null;
  cell?: number;
  /** Override the fill (Poster draws pieces in the column's text color). */
  color?: string;
  className?: string;
}

export function agentFill(agent: Agent, look: string): string {
  if (look === "broadcast") return agent.looks.broadcast.fill;
  if (look === "poster") return agent.looks.poster.field;
  return agent.looks.tetromino.fill;
}

export function PieceIcon({ agent, cell = 12, color, className }: Props) {
  const { look } = useApp();
  if (!agent || !agent.cells.length) {
    return <span className={`piece-icon piece-icon--empty${className ? ` ${className}` : ""}`} aria-hidden="true" />;
  }
  if (look === "terminal") {
    return (
      <span className={`piece-letter${className ? ` ${className}` : ""}`} aria-hidden="true">
        {agent.looks.terminal.letter}
      </span>
    );
  }
  const rows = Math.max(...agent.cells.map(([r]) => r)) + 1;
  const cols = Math.max(...agent.cells.map(([, c]) => c)) + 1;
  const gap = Math.max(1, Math.round(cell / 8));
  const fill = color ?? agentFill(agent, look);
  return (
    <span
      className={`piece-icon${className ? ` ${className}` : ""}`}
      aria-hidden="true"
      style={{
        display: "inline-grid",
        gridTemplateColumns: `repeat(${cols}, ${cell}px)`,
        gridTemplateRows: `repeat(${rows}, ${cell}px)`,
        gap,
      }}
    >
      {agent.cells.map(([r, c]) => (
        <span key={`${r}-${c}`} className="piece-icon__cell" style={{ gridRow: r + 1, gridColumn: c + 1, background: fill }} />
      ))}
    </span>
  );
}
