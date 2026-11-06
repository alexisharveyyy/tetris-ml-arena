// Board geometry (tetris/engine/board.py). The board is an Int8Array of
// HEIGHT x WIDTH, row-major, row 0 at the top (4 hidden spawn rows).
import { BOX_SIZE, CELL_CHARS, CELLS, GARBAGE, type Piece } from "./pieces";

export const WIDTH = 10;
export const VISIBLE_HEIGHT = 20;
export const HIDDEN_ROWS = 4;
export const HEIGHT = VISIBLE_HEIGHT + HIDDEN_ROWS;
export const SPAWN_ROW = 2;

export type Board = Int8Array;

export function spawnCol(piece: Piece): number {
  return BOX_SIZE[piece] === 2 ? 4 : 3;
}

export function emptyBoard(): Board {
  return new Int8Array(HEIGHT * WIDTH);
}

export function fits(board: Board, piece: Piece, rot: number, row: number, col: number): boolean {
  for (const [dr, dc] of CELLS[piece][rot]) {
    const r = row + dr;
    const c = col + dc;
    if (c < 0 || c >= WIDTH || r < 0 || r >= HEIGHT || board[r * WIDTH + c]) return false;
  }
  return true;
}

export function pieceCells(piece: Piece, rot: number, row: number, col: number): [number, number][] {
  return CELLS[piece][rot].map(([dr, dc]) => [row + dr, col + dc]);
}

export function dropRow(board: Board, piece: Piece, rot: number, row: number, col: number): number {
  while (fits(board, piece, rot, row + 1, col)) row += 1;
  return row;
}

/** Remove full rows. Returns the new board and the cleared row indices. */
export function clearFullRows(board: Board): [Board, number[]] {
  const full: number[] = [];
  for (let r = 0; r < HEIGHT; r++) {
    let all = true;
    for (let c = 0; c < WIDTH; c++) {
      if (!board[r * WIDTH + c]) {
        all = false;
        break;
      }
    }
    if (all) full.push(r);
  }
  if (!full.length) return [board, []];
  const out = emptyBoard();
  let dst = HEIGHT - 1;
  for (let r = HEIGHT - 1; r >= 0; r--) {
    if (full.includes(r)) continue;
    out.set(board.subarray(r * WIDTH, (r + 1) * WIDTH), dst * WIDTH);
    dst -= 1;
  }
  return [out, full];
}

/** Push the stack up and add garbage rows with a hole. Returns [board, overflow]. */
export function insertGarbageRows(board: Board, rows: number, hole: number): [Board, boolean] {
  if (rows <= 0) return [board, false];
  rows = Math.min(rows, HEIGHT);
  let overflow = false;
  for (let i = 0; i < rows * WIDTH; i++) if (board[i]) overflow = true;
  const out = new Int8Array(HEIGHT * WIDTH);
  out.set(board.subarray(rows * WIDTH), 0);
  for (let r = HEIGHT - rows; r < HEIGHT; r++) {
    for (let c = 0; c < WIDTH; c++) out[r * WIDTH + c] = c === hole ? 0 : GARBAGE;
  }
  return [out, overflow];
}

export function boardToStrings(board: Board): string[] {
  const out: string[] = [];
  for (let r = 0; r < HEIGHT; r++) {
    let s = "";
    for (let c = 0; c < WIDTH; c++) s += CELL_CHARS[board[r * WIDTH + c]];
    out.push(s);
  }
  return out;
}

export function boardFromStrings(rows: string[]): Board {
  if (rows.length !== HEIGHT) throw new Error(`board must be ${HEIGHT} rows`);
  const b = emptyBoard();
  rows.forEach((row, r) => {
    if (row.length !== WIDTH) throw new Error(`board row must be ${WIDTH} wide`);
    for (let c = 0; c < WIDTH; c++) b[r * WIDTH + c] = CELL_CHARS.indexOf(row[c]);
  });
  return b;
}
