// A board on <canvas>, drawn by the current look's renderer. role="img" with a
// plain label so screen readers get the state without the drawing.
import { useEffect, useRef } from "react";
import type { Agent, PlayerSnap } from "../api/types";
import { fmtInt } from "../lib/format";
import { useApp } from "../state/app";

interface Props {
  player: PlayerSnap;
  agent: Agent | null;
  cell: number;
  human?: boolean;
  showThinking: boolean;
  className?: string;
}

export function boardLabel(p: PlayerSnap): string {
  const who = p.kind === "human" ? "Your board" : `${p.name} board`;
  return `${who}: ${fmtInt(p.stats.score)} points, ${p.stats.lines} lines, level ${p.stats.level}. ${p.status.text}.`;
}

export function Board({ player, agent, cell, human = false, showThinking, className }: Props) {
  const { theme, settings } = useApp();
  const ref = useRef<HTMLCanvasElement>(null);
  const { width, height } = theme.board.size(cell);
  const showGhost = settings?.show_ghost ?? true;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * dpr)) canvas.width = Math.round(width * dpr);
    if (canvas.height !== Math.round(height * dpr)) canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    theme.board.draw(ctx, { player, agent, showThinking, showGhost, human }, cell);
  });

  return (
    <canvas
      ref={ref}
      className={`board${human ? " board--human" : ""}${className ? ` ${className}` : ""}`}
      role="img"
      aria-label={boardLabel(player)}
      style={{ width, height }}
    />
  );
}
