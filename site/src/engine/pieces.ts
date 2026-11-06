// Tetrominoes, SRS rotation states and kick tables (tetris/engine/pieces.py).
// Coordinates are [row, col] with rows increasing downward.

export const P = { I: 1, O: 2, T: 3, S: 4, Z: 5, J: 6, L: 7 } as const;
export type Piece = 1 | 2 | 3 | 4 | 5 | 6 | 7;

export const GARBAGE = 8;
export const CELL_CHARS = ".IOTSZJLG";
export const ALL_PIECES: readonly Piece[] = [1, 2, 3, 4, 5, 6, 7];
export const LETTERS = "IOTSZJL";

export function letter(p: Piece): string {
  return LETTERS[p - 1];
}

export function fromLetter(ch: string): Piece {
  const i = LETTERS.indexOf(ch);
  if (i < 0) throw new Error(`unknown piece ${ch}`);
  return (i + 1) as Piece;
}

const SPAWN: Record<Piece, string[]> = {
  1: ["....", "####", "....", "...."],
  2: ["##", "##"],
  3: [".#.", "###", "..."],
  4: [".##", "##.", "..."],
  5: ["##.", ".##", "..."],
  6: ["#..", "###", "..."],
  7: ["..#", "###", "..."],
};

export type Cell = readonly [number, number];

function rotateCw(m: string[]): string[] {
  const n = m.length;
  const out: string[] = [];
  for (let r = 0; r < n; r++) {
    let row = "";
    for (let c = 0; c < n; c++) row += m[n - 1 - c][r];
    out.push(row);
  }
  return out;
}

function cellsOf(m: string[]): Cell[] {
  const out: Cell[] = [];
  m.forEach((row, r) => {
    for (let c = 0; c < row.length; c++) if (row[c] === "#") out.push([r, c]);
  });
  return out;
}

/** CELLS[piece][rot]: [row, col] offsets in the bounding box. */
export const CELLS: Cell[][][] = [[]];
export const BOX_SIZE: number[] = [0];
export const MIN_DC: number[][] = [[]];
export const MAX_DC: number[][] = [[]];
for (const p of ALL_PIECES) {
  let m = SPAWN[p];
  const rots: Cell[][] = [];
  for (let i = 0; i < 4; i++) {
    rots.push(cellsOf(m));
    m = rotateCw(m);
  }
  CELLS[p] = rots;
  BOX_SIZE[p] = SPAWN[p].length;
  MIN_DC[p] = rots.map((cs) => Math.min(...cs.map(([, c]) => c)));
  MAX_DC[p] = rots.map((cs) => Math.max(...cs.map(([, c]) => c)));
}

type Kicks = Record<string, [number, number][]>;
// Guideline notation (dx, dy) with y up.
const JLSTZ_UP: Kicks = {
  "0,1": [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
  "1,0": [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
  "1,2": [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
  "2,1": [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
  "2,3": [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
  "3,2": [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
  "3,0": [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
  "0,3": [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
};
const I_UP: Kicks = {
  "0,1": [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]],
  "1,0": [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]],
  "1,2": [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]],
  "2,1": [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]],
  "2,3": [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]],
  "3,2": [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]],
  "3,0": [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]],
  "0,3": [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]],
};

function toRowCol(t: Kicks): Record<string, Cell[]> {
  const out: Record<string, Cell[]> = {};
  for (const [k, v] of Object.entries(t)) out[k] = v.map(([dx, dy]) => [-dy, dx] as const);
  return out;
}

const KICKS_JLSTZ = toRowCol(JLSTZ_UP);
const KICKS_I = toRowCol(I_UP);
const NO_KICK: Cell[] = [[0, 0]];

/** Ordered [drow, dcol] kick offsets to try. */
export function kicks(piece: Piece, from: number, to: number): Cell[] {
  if (piece === P.O) return NO_KICK;
  return (piece === P.I ? KICKS_I : KICKS_JLSTZ)[`${from},${to}`];
}
