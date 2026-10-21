// Piece shapes for next and hold previews (spawn orientation) and agent icons.
import type { Cell } from "../api/types";

export const PREVIEW_CELLS: Record<string, Cell[]> = {
  I: [[0, 0], [0, 1], [0, 2], [0, 3]],
  O: [[0, 0], [0, 1], [1, 0], [1, 1]],
  T: [[0, 1], [1, 0], [1, 1], [1, 2]],
  S: [[0, 1], [0, 2], [1, 0], [1, 1]],
  Z: [[0, 0], [0, 1], [1, 1], [1, 2]],
  J: [[0, 0], [1, 0], [1, 1], [1, 2]],
  L: [[0, 2], [1, 0], [1, 1], [1, 2]],
};

export function extent(cells: Cell[]): { rows: number; cols: number } {
  return {
    rows: Math.max(...cells.map(([r]) => r)) + 1,
    cols: Math.max(...cells.map(([, c]) => c)) + 1,
  };
}

/** The cell that carries a candidate's rank: top-most, then left-most. */
export function firstCell(cells: Cell[]): Cell {
  return [...cells].sort((a, b) => a[0] - b[0] || a[1] - b[1])[0];
}

export function boardRows(board: string): string[] {
  const rows: string[] = [];
  for (let r = 0; r < 20; r++) rows.push(board.slice(r * 10, r * 10 + 10));
  return rows;
}
