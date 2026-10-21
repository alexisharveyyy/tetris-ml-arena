// Board renderer for looks that draw colored cells (Tetromino, Broadcast).
import type { Agent } from "../../api/types";
import { firstCell } from "../../lib/pieces";
import type { BoardRenderer, BoardView } from "../types";

export interface CellBoardStyle {
  pieceColors: Record<string, string>;
  /** Color behind the cells (shows as grid lines between them). */
  grid: string;
  empty: string;
  gap: number;
  /** Inner border on filled cells. */
  cellBorder: { color: string; width: number } | null;
  ghost: { fill: string; stroke: string };
  /** Candidate outline color for an agent. */
  candidate: (agent: Agent | null) => string;
  candidateFill: number; // fill opacity
  badge: { fill: string; text: string; font: (px: number) => string };
  toppedOut: string; // wash over a finished board
}

function hexAlpha(hex: string, alpha: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

export function cellBoard(style: CellBoardStyle): BoardRenderer {
  const fillCell = (ctx: CanvasRenderingContext2D, r: number, c: number, cell: number, color: string) => {
    const g = style.gap;
    const x = c * cell + g;
    const y = r * cell + g;
    const s = cell - g;
    ctx.fillStyle = color;
    ctx.fillRect(x, y, s, s);
    if (style.cellBorder) {
      const w = style.cellBorder.width;
      ctx.strokeStyle = style.cellBorder.color;
      ctx.lineWidth = w;
      ctx.strokeRect(x + w / 2, y + w / 2, s - w, s - w);
    }
  };

  return {
    size: (cell) => ({ width: 10 * cell + style.gap, height: 20 * cell + style.gap }),
    draw(ctx, view: BoardView, cell) {
      const { player } = view;
      const g = style.gap;
      ctx.fillStyle = style.grid;
      ctx.fillRect(0, 0, 10 * cell + g, 20 * cell + g);
      for (let i = 0; i < 200; i++) {
        const ch = player.board[i];
        const r = Math.floor(i / 10);
        const c = i % 10;
        if (ch === ".") {
          ctx.fillStyle = style.empty;
          ctx.fillRect(c * cell + g, r * cell + g, cell - g, cell - g);
        } else {
          fillCell(ctx, r, c, cell, style.pieceColors[ch] ?? style.pieceColors.G);
        }
      }
      if (view.showThinking && player.cand.length) {
        ctx.save();
        ctx.globalAlpha = player.cand_current ? 1 : 0.55;
        const color = style.candidate(view.agent);
        for (const cand of [...player.cand].reverse()) {
          for (const [r, c] of cand.c) {
            if (r < 0) continue;
            const x = c * cell + g;
            const y = r * cell + g;
            const s = cell - g;
            ctx.fillStyle = hexAlpha(color, style.candidateFill);
            ctx.fillRect(x, y, s, s);
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x + 1, y + 1, s - 2, s - 2);
          }
          const [fr, fc] = firstCell(cand.c.filter(([r]) => r >= 0).length ? cand.c.filter(([r]) => r >= 0) : cand.c);
          const bx = fc * cell + g + 2;
          const by = Math.max(fr, 0) * cell + g + 2;
          const bs = cell - g - 4;
          ctx.fillStyle = style.badge.fill;
          ctx.fillRect(bx, by, bs, bs);
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(bx + 0.75, by + 0.75, bs - 1.5, bs - 1.5);
          ctx.fillStyle = style.badge.text;
          ctx.font = style.badge.font(Math.max(9, Math.round(bs * 0.8)));
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(String(cand.rank), bx + bs / 2, by + bs / 2 + 1);
        }
        ctx.restore();
      }
      if (view.showGhost && player.ghost && player.active) {
        for (const [r, c] of player.ghost) {
          if (r < 0) continue;
          const x = c * cell + g;
          const y = r * cell + g;
          const s = cell - g;
          ctx.fillStyle = style.ghost.fill;
          ctx.fillRect(x, y, s, s);
          ctx.strokeStyle = style.ghost.stroke;
          ctx.lineWidth = 2;
          ctx.strokeRect(x + 1, y + 1, s - 2, s - 2);
        }
      }
      if (player.active) {
        for (const [r, c] of player.active.c) if (r >= 0) fillCell(ctx, r, c, cell, style.pieceColors[player.active.p]);
      }
      if (player.status.state === "topped_out") {
        ctx.fillStyle = style.toppedOut;
        ctx.fillRect(0, 0, 10 * cell + g, 20 * cell + g);
      }
    },
  };
}
