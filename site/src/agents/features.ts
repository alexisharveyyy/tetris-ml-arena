// Board features (tetris/env/features.py), on a filled mask of HEIGHT x WIDTH.
// Every feature is an integer (or a half, for landing height), so the values
// are exact and the order of summation does not matter.
import { HEIGHT, WIDTH } from "../engine/board";

export const BOARD_FEATURES = [
  "aggregate_height",
  "max_height",
  "holes",
  "rows_with_holes",
  "hole_depth",
  "bumpiness",
  "wells",
  "max_well_depth",
  "row_transitions",
  "col_transitions",
] as const;
export const PLACEMENT_FEATURES = ["lines_cleared", "landing_height", "eroded_cells"] as const;
export const FEATURE_NAMES = [...BOARD_FEATURES, ...PLACEMENT_FEATURES];
export const NUM_FEATURES = FEATURE_NAMES.length;

/** Ten board features of a board (any nonzero cell is filled). */
export function boardFeatures(board: ArrayLike<number>): number[] {
  const H = HEIGHT;
  const W = WIDTH;
  const filled = (r: number, c: number) => board[r * W + c] !== 0;
  const heights = new Array(W).fill(0);
  let holes = 0;
  let holeDepth = 0;
  const rowHasHole = new Array(H).fill(false);
  for (let c = 0; c < W; c++) {
    let covered = false;
    let above = 0;
    for (let r = 0; r < H; r++) {
      const f = filled(r, c);
      if (f && !covered) heights[c] = H - r;
      if (!f && covered) {
        holes += 1;
        holeDepth += above;
        rowHasHole[r] = true;
      }
      if (f) {
        covered = true;
        above += 1;
      }
    }
  }
  let bump = 0;
  for (let c = 0; c + 1 < W; c++) bump += Math.abs(heights[c + 1] - heights[c]);
  // Wells: empty cells with filled (or wall) neighbours on both sides.
  let wells = 0;
  let maxWell = 0;
  for (let c = 0; c < W; c++) {
    let run = 0;
    for (let r = 0; r < H; r++) {
      const left = c === 0 || filled(r, c - 1);
      const right = c === W - 1 || filled(r, c + 1);
      const well = !filled(r, c) && left && right;
      run = well ? run + 1 : 0;
      wells += run;
      if (run > maxWell) maxWell = run;
    }
  }
  let rowT = 0;
  for (let r = 0; r < H; r++) {
    let prev = true; // left wall
    for (let c = 0; c < W; c++) {
      const f = filled(r, c);
      if (f !== prev) rowT += 1;
      prev = f;
    }
    if (prev !== true) rowT += 1; // right wall
  }
  let colT = 0;
  for (let c = 0; c < W; c++) {
    let prev = false; // above the top
    for (let r = 0; r < H; r++) {
      const f = filled(r, c);
      if (f !== prev) colT += 1;
      prev = f;
    }
    if (prev !== true) colT += 1; // floor
  }
  const aggregate = heights.reduce((a, b) => a + b, 0);
  const rowsWithHoles = rowHasHole.filter(Boolean).length;
  return [aggregate, Math.max(...heights), holes, rowsWithHoles, holeDepth, bump, wells, maxWell, rowT, colT];
}
