// Terminal 1984: the board is text. "[]" filled, " ." empty, "::" ghost,
// "()" candidate with "1)" in its first cell, walls "<!" and "!>", a "<!===!>"
// floor and a "\/\/" base row. Characters are drawn, never colored cells.
import { firstCell } from "../../lib/pieces";
import type { BoardRenderer, BoardView } from "../types";

export const GREEN = "#39FF7A";
export const BRIGHT = "#B8FFD0";
export const DIM = "#28A855";
const GLOW = "rgba(57, 255, 122, 0.45)";
const FONT = '"VT323", monospace';
const COLS = 24; // "<!" + 10 cells x 2 + "!>"
const ROWS = 22; // 20 rows + floor + base

let ratio: number | null = null;

/** Advance width of one VT323 character as a fraction of the font size. */
export function charRatio(): number {
  if (ratio !== null) return ratio;
  const loaded = typeof document !== "undefined" && document.fonts?.check?.(`100px ${FONT}`);
  const ctx = typeof document !== "undefined" ? document.createElement("canvas").getContext("2d") : null;
  if (!ctx) return 0.5;
  ctx.font = `100px ${FONT}`;
  const measured = ctx.measureText("[]").width / 200;
  if (loaded) ratio = measured;
  return measured || 0.5;
}

/** The 22 lines of a board as [text, color] runs (two characters per cell). */
export function boardLines(view: BoardView): [string, string][][] {
  const p = view.player;
  const grid: [string, string][] = Array.from({ length: 200 }, (_, i) => (p.board[i] === "." ? [" .", GREEN] : ["[]", GREEN]));
  const at = (r: number, c: number) => r * 10 + c;
  if (view.showGhost && p.ghost && p.active) for (const [r, c] of p.ghost) if (r >= 0 && grid[at(r, c)][0] === " .") grid[at(r, c)] = ["::", DIM];
  if (view.showThinking) {
    for (const cand of p.cand) {
      const shown = cand.c.filter(([r]) => r >= 0);
      if (!shown.length) continue;
      const [fr, fc] = firstCell(shown);
      for (const [r, c] of shown) grid[at(r, c)] = r === fr && c === fc ? [`${cand.rank})`, BRIGHT] : ["()", p.cand_current ? BRIGHT : GREEN];
    }
  }
  if (p.active) for (const [r, c] of p.active.c) if (r >= 0) grid[at(r, c)] = ["[]", BRIGHT];
  const lines: [string, string][][] = [];
  for (let r = 0; r < 20; r++) lines.push([["<!", GREEN], ...grid.slice(r * 10, r * 10 + 10), ["!>", GREEN]]);
  // Two-character runs, like the rows above, so rounding never shifts the last character.
  lines.push([["<!", GREEN], ...Array.from({ length: 10 }, (): [string, string] => ["==", GREEN]), ["!>", GREEN]]);
  lines.push([["  ", GREEN], ...Array.from({ length: 10 }, (): [string, string] => ["\\/", GREEN]), ["  ", GREEN]]);
  return lines;
}

export const terminalBoard: BoardRenderer = {
  size(cell) {
    const font = Math.round(cell * 0.95);
    return { width: Math.ceil(COLS * font * charRatio()), height: ROWS * cell };
  },
  draw(ctx, view: BoardView, cell) {
    const font = Math.round(cell * 0.95);
    const cw = font * charRatio();
    ctx.font = `${font}px ${FONT}`;
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.shadowColor = GLOW;
    ctx.shadowBlur = 6;
    const dimAll = view.player.status.state === "topped_out";
    boardLines(view).forEach((runs, r) => {
      let x = 0;
      for (const [text, color] of runs) {
        ctx.fillStyle = dimAll ? DIM : color;
        ctx.fillText(text, x, r * cell + cell / 2 + 1);
        x += text.length * cw;
      }
    });
    ctx.shadowBlur = 0;
  },
};

/** ASCII interval chart: "." track, "[" and "]" ends, "=" inside, "*" estimate. */
export function asciiInterval(lo: number, hi: number, est: number, scale: number, width: number): string {
  const at = (x: number) => Math.round((Math.min(x, scale) / scale) * (width - 1));
  const chars = Array.from({ length: width }, () => ".");
  const a = at(lo);
  const off = hi > scale;
  const b = off ? width - 1 : at(hi);
  for (let i = a; i <= b; i++) chars[i] = "=";
  chars[a] = "[";
  chars[b] = off ? ">" : "]";
  if (est <= scale) chars[Math.min(Math.max(at(est), a + 1), Math.max(a + 1, b - 1))] = "*";
  return chars.join("");
}
