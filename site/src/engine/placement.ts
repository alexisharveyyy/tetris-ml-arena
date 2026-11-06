// Placement-level actions and fast collision queries (tetris/engine/placement.py).
import { type Board, HEIGHT, SPAWN_ROW, WIDTH, spawnCol } from "./board";
import { ALL_PIECES, CELLS, MAX_DC, MIN_DC, type Piece } from "./pieces";

export const NUM_ROTATIONS = 4;
export const NUM_COLUMNS = WIDTH;
export const NUM_ACTIONS = NUM_ROTATIONS * NUM_COLUMNS * 2;
export const FULL_ROW = (1 << WIDTH) - 1;

/** ROW_MASKS[piece][rot]: [row offset, bitmask at box column 0], sorted by row. */
const ROW_MASKS: [number, number][][][] = [[]];
/** BOTTOMS[piece][rot]: [col offset, lowest row offset in that column], sorted by col. */
const BOTTOMS: [number, number][][][] = [[]];
for (const p of ALL_PIECES) {
  ROW_MASKS[p] = [];
  BOTTOMS[p] = [];
  for (let r = 0; r < 4; r++) {
    const rows = new Map<number, number>();
    const low = new Map<number, number>();
    for (const [dr, dc] of CELLS[p][r]) {
      rows.set(dr, (rows.get(dr) ?? 0) | (1 << dc));
      low.set(dc, Math.max(low.get(dc) ?? -1, dr));
    }
    ROW_MASKS[p].push([...rows.entries()].sort((a, b) => a[0] - b[0]));
    BOTTOMS[p].push([...low.entries()].sort((a, b) => a[0] - b[0]));
  }
}

export function pieceRowMasks(piece: Piece, rot: number): [number, number][] {
  return ROW_MASKS[piece][rot];
}

function rotationPaths(startRot: number, rot: number): number[][] {
  const d = (((rot - startRot) % 4) + 4) % 4;
  const cw1 = (startRot + 1) % 4;
  const cw2 = (startRot + 2) % 4;
  const ccw1 = (startRot + 3) % 4;
  if (d === 0) return [[]];
  if (d === 1) return [[cw1]];
  if (d === 3) return [[ccw1]];
  return [
    [cw1, cw2],
    [ccw1, cw2],
  ];
}

export interface Placement {
  rotation: number;
  column: number;
  useHold: boolean;
}

export function placementIndex(p: Placement): number {
  return (p.useHold ? NUM_ROTATIONS * NUM_COLUMNS : 0) + p.rotation * NUM_COLUMNS + p.column;
}

export function placementFromIndex(index: number): Placement {
  if (!(index >= 0 && index < NUM_ACTIONS)) throw new Error(`action index out of range: ${index}`);
  const useHold = index >= NUM_ROTATIONS * NUM_COLUMNS;
  const rest = index % (NUM_ROTATIONS * NUM_COLUMNS);
  return { rotation: Math.floor(rest / NUM_COLUMNS), column: rest % NUM_COLUMNS, useHold };
}

export interface Target {
  piece: Piece;
  rotation: number;
  column: number; // leftmost occupied column
  boxCol: number;
  row: number; // box row after hard drop
}

export function boardRows(board: Board): number[] {
  const out: number[] = new Array(HEIGHT);
  for (let r = 0; r < HEIGHT; r++) {
    let bits = 0;
    for (let c = 0; c < WIDTH; c++) if (board[r * WIDTH + c]) bits |= 1 << c;
    out[r] = bits;
  }
  return out;
}

export type Start = [number, number, number]; // rot, row, box col

export class BoardIndex {
  rows: number[];
  private nf: number[][] | null = null;

  constructor(board: Board | null, rows?: number[]) {
    this.rows = rows ?? boardRows(board!);
  }

  get nextFilled(): number[][] {
    if (this.nf === null) {
      const nf: number[][] = new Array(HEIGHT + 1);
      nf[HEIGHT] = new Array(WIDTH).fill(HEIGHT);
      let below = new Array(WIDTH).fill(HEIGHT);
      for (let r = HEIGHT - 1; r >= 0; r--) {
        const bits = this.rows[r];
        if (bits) below = below.map((v, c) => ((bits >> c) & 1 ? r : v));
        nf[r] = below;
      }
      this.nf = nf;
    }
    return this.nf;
  }

  fits(piece: Piece, rot: number, row: number, col: number): boolean {
    if (col + MIN_DC[piece][rot] < 0 || col + MAX_DC[piece][rot] >= WIDTH) return false;
    const rows = this.rows;
    for (const [dr, mask] of ROW_MASKS[piece][rot]) {
      const r = row + dr;
      if (r < 0 || r >= HEIGHT) return false;
      if (rows[r] & (col >= 0 ? mask << col : mask >> -col)) return false;
    }
    return true;
  }

  landingRow(piece: Piece, rot: number, row: number, col: number): number {
    const nf = this.nextFilled;
    let best = HEIGHT;
    for (const [dc, low] of BOTTOMS[piece][rot]) {
      const land = nf[row + low + 1][col + dc] - 1 - low;
      if (land < best) best = land;
    }
    return best;
  }

  spawn(piece: Piece): [number, number] | null {
    const col = spawnCol(piece);
    if (!this.fits(piece, 0, SPAWN_ROW, col)) return null;
    if (this.fits(piece, 0, SPAWN_ROW + 1, col)) return [SPAWN_ROW + 1, col];
    return [SPAWN_ROW, col];
  }

  rotationReachable(piece: Piece, startRot: number, rot: number, row: number, col: number): boolean {
    for (const path of rotationPaths(startRot, rot)) {
      if (path.every((r) => this.fits(piece, r, row, col))) return true;
    }
    return false;
  }

  private start(piece: Piece, start: Start | null): Start | null {
    if (start !== null) return start;
    const s = this.spawn(piece);
    return s === null ? null : [0, s[0], s[1]];
  }

  targets(piece: Piece, start: Start | null = null): Target[] {
    const origin = this.start(piece, start);
    if (origin === null) return [];
    const [startRot, row, startCol] = origin;
    const out: Target[] = [];
    for (let rot = 0; rot < NUM_ROTATIONS; rot++) {
      if (!this.rotationReachable(piece, startRot, rot, row, startCol)) continue;
      let lo = startCol;
      while (this.fits(piece, rot, row, lo - 1)) lo -= 1;
      let hi = startCol;
      while (this.fits(piece, rot, row, hi + 1)) hi += 1;
      const minDc = MIN_DC[piece][rot];
      for (let boxCol = lo; boxCol <= hi; boxCol++) {
        out.push({ piece, rotation: rot, column: boxCol + minDc, boxCol, row: this.landingRow(piece, rot, row, boxCol) });
      }
    }
    return out;
  }

  resolve(piece: Piece, rotation: number, column: number, start: Start | null = null): Target | null {
    const origin = this.start(piece, start);
    if (origin === null || !(rotation >= 0 && rotation < NUM_ROTATIONS)) return null;
    const [startRot, row, startCol] = origin;
    if (!this.rotationReachable(piece, startRot, rotation, row, startCol)) return null;
    const boxCol = column - MIN_DC[piece][rotation];
    const step = boxCol >= startCol ? 1 : -1;
    for (let c = startCol; c !== boxCol + step; c += step) {
      if (!this.fits(piece, rotation, row, c)) return null;
    }
    return { piece, rotation, column, boxCol, row: this.landingRow(piece, rotation, row, boxCol) };
  }
}

export function spawnPosition(board: Board, piece: Piece): [number, number] | null {
  return new BoardIndex(board).spawn(piece);
}
