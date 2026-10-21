// Poster: one-color boards on the column's field. Filled cells in the column's
// text color, empty cells as faint grid lines (14%), the ghost as an outline
// (55%), candidates as dashed outlines with their rank.
import type { Agent } from "../../api/types";
import { firstCell } from "../../lib/pieces";
import type { BoardRenderer, BoardView } from "../types";

export const HUMAN_FIELD = "#FFE14D";
export const HUMAN_TEXT = "#14161F";
const PAD = 8; // frame and gap around the cells

export function posterColors(agent: Agent | null, human: boolean): { field: string; text: string } {
  if (human || !agent) return { field: HUMAN_FIELD, text: HUMAN_TEXT };
  return agent.looks.poster;
}

function rgba(hex: string, alpha: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

export const posterBoard: BoardRenderer = {
  size: (cell) => ({ width: 10 * cell + 2 * PAD, height: 20 * cell + 2 * PAD }),
  draw(ctx, view: BoardView, cell) {
    const { field, text } = posterColors(view.agent, view.human);
    const p = view.player;
    const w = 10 * cell + 2 * PAD;
    const h = 20 * cell + 2 * PAD;
    ctx.fillStyle = field;
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = text;
    ctx.lineWidth = 3;
    ctx.strokeRect(1.5, 1.5, w - 3, h - 3);
    const x0 = PAD;
    const y0 = PAD;
    const inset = Math.max(1, Math.round(cell * 0.08));
    const fill = (r: number, c: number) => ctx.fillRect(x0 + c * cell + inset, y0 + r * cell + inset, cell - 2 * inset, cell - 2 * inset);
    const outline = (r: number, c: number) => ctx.strokeRect(x0 + c * cell + inset + 1, y0 + r * cell + inset + 1, cell - 2 * inset - 2, cell - 2 * inset - 2);
    ctx.lineWidth = 1;
    ctx.strokeStyle = rgba(text, 0.14);
    for (let r = 0; r < 20; r++) for (let c = 0; c < 10; c++) outline(r, c);
    ctx.fillStyle = text;
    for (let i = 0; i < 200; i++) if (p.board[i] !== ".") fill(Math.floor(i / 10), i % 10);
    if (view.showGhost && p.ghost && p.active) {
      ctx.strokeStyle = rgba(text, 0.55);
      ctx.lineWidth = 2;
      for (const [r, c] of p.ghost) if (r >= 0) outline(r, c);
    }
    if (view.showThinking && p.cand.length) {
      ctx.save();
      ctx.globalAlpha = p.cand_current ? 1 : 0.6;
      ctx.strokeStyle = text;
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      for (const cand of p.cand) {
        const shown = cand.c.filter(([r]) => r >= 0);
        for (const [r, c] of shown) outline(r, c);
        if (!shown.length) continue;
        const [fr, fc] = firstCell(shown);
        ctx.fillStyle = text;
        ctx.font = `700 ${Math.round(cell * 0.62)}px "Archivo", sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(cand.rank), x0 + fc * cell + cell / 2, y0 + fr * cell + cell / 2 + 1);
      }
      ctx.restore();
    }
    if (p.active) {
      ctx.fillStyle = text;
      for (const [r, c] of p.active.c) if (r >= 0) fill(r, c);
    }
    if (p.status.state === "topped_out") {
      ctx.fillStyle = rgba(field, 0.55);
      ctx.fillRect(0, 0, w, h);
    }
  },
};
