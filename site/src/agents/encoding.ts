// Network inputs for the learned agents (tetris/rl/encoding.py), in float32
// exactly as NumPy computes them.
import { HEIGHT, WIDTH } from "../engine/board";
import type { Game } from "../engine/game";
import type { Piece } from "../engine/pieces";
import { NUM_ACTIONS } from "../engine/placement";
import { type Afterstate, enumerateAfterstates } from "./afterstates";
import { boardFeatures, FEATURE_NAMES } from "./features";

const f32 = Math.fround;
export const CELLS = HEIGHT * WIDTH;
export const REWARD_SURVIVAL = 1.0;
export const REWARD_SCORE_SCALE = 0.25 / 1000.0;

const SCALES: Record<string, number> = {
  aggregate_height: 200,
  max_height: 24,
  holes: 40,
  rows_with_holes: 20,
  hole_depth: 100,
  bumpiness: 60,
  wells: 60,
  max_well_depth: 20,
  row_transitions: 120,
  col_transitions: 120,
  lines_cleared: 4,
  landing_height: 24,
  eroded_cells: 16,
};
export const FEATURE_SCALE = FEATURE_NAMES.map((n) => SCALES[n]);
const NF = FEATURE_NAMES.length;
export const AFTERSTATE_EXTRA_DIM = NF + 7 + 8 + 1;
export const OBS_EXTRA_DIM = 7 + 8 + 1 + 35 + 10 + 1;
const QUEUE_LEN = 5;

/** float32(feature) / float32(scale), as NumPy does. */
export function scaled(features: number[]): number[] {
  return features.map((v, i) => f32(f32(v) / FEATURE_SCALE[i]));
}

function levelAfter(game: Game, lines: number): number {
  const c = game.config;
  const lvl = c.start_level + Math.floor((game.lines + lines) / c.lines_per_level);
  return Math.min(c.max_level, Math.max(game.level, lvl));
}

export function rewardForScore(score: number): number {
  // float32 arithmetic (NumPy 2 keeps float32 for a Python scalar times a float32 array).
  return f32(f32(REWARD_SURVIVAL) + f32(f32(REWARD_SCORE_SCALE) * f32(score)));
}

export interface AfterstateArrays {
  boards: Float32Array; // (n, 240) occupancy
  extras: Float32Array; // (n, 29)
  rewards: Float32Array;
  tops: boolean[];
  actions: number[];
}

export function afterstateArrays(game: Game, afs: Afterstate[]): AfterstateArrays {
  const n = afs.length;
  const boards = new Float32Array(n * CELLS);
  const extras = new Float32Array(n * AFTERSTATE_EXTRA_DIM);
  const rewards = new Float32Array(n);
  const current = game.active!.piece;
  afs.forEach((a, i) => {
    for (let c = 0; c < CELLS; c++) boards[i * CELLS + c] = a.board[c] !== 0 ? 1 : 0;
    const o = i * AFTERSTATE_EXTRA_DIM;
    scaled(a.features).forEach((v, j) => (extras[o + j] = v));
    let next: Piece;
    let holdAfter: Piece | null;
    if (a.placement.useHold) {
      next = game.hold === null ? game.queue[1] : game.queue[0];
      holdAfter = current;
    } else {
      next = game.queue[0];
      holdAfter = game.hold;
    }
    extras[o + NF + next - 1] = 1;
    extras[o + NF + 7 + (holdAfter ?? 0)] = 1;
    extras[o + AFTERSTATE_EXTRA_DIM - 1] = f32(levelAfter(game, a.linesCleared) / 20);
    rewards[i] = rewardForScore(a.score);
  });
  return { boards, extras, rewards, tops: afs.map((a) => a.topOut), actions: afs.map((a) => a.index) };
}

/** PPO observation: board occupancy (240) and extras (62). */
export function observationArrays(game: Game): { board: Float32Array; extras: Float32Array } {
  const board = new Float32Array(CELLS);
  for (let c = 0; c < CELLS; c++) board[c] = game.board[c] !== 0 ? 1 : 0;
  const extras = new Float32Array(OBS_EXTRA_DIM);
  let o = 0;
  if (game.active !== null) extras[o + game.active.piece - 1] = 1;
  o += 7;
  extras[o + (game.hold ?? 0)] = 1;
  o += 8;
  extras[o] = game.canHold ? 1 : 0;
  o += 1;
  game.queue.slice(0, QUEUE_LEN).forEach((p, k) => (extras[o + 7 * k + p - 1] = 1));
  o += 7 * QUEUE_LEN;
  const bf = boardFeatures(game.board);
  for (let j = 0; j < 10; j++) extras[o + j] = f32(f32(bf[j]) / FEATURE_SCALE[j]);
  o += 10;
  extras[o] = f32(game.level / 20);
  return { board, extras };
}

/** Afterstate inputs of every legal action (no dedupe), in action order. */
export function legalActionArrays(game: Game): { arrays: AfterstateArrays; afs: Afterstate[] } {
  const afs = enumerateAfterstates(game, true, false);
  const order = afs.map((_, i) => i).sort((a, b) => afs[a].index - afs[b].index);
  const sorted = order.map((i) => afs[i]);
  return { arrays: afterstateArrays(game, sorted), afs: sorted };
}

export { NUM_ACTIONS };
