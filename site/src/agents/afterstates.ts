// Legal placements and their afterstates (tetris/env/afterstates.py).
import { HEIGHT, HIDDEN_ROWS, WIDTH } from "../engine/board";
import type { Game } from "../engine/game";
import { CELLS, type Piece } from "../engine/pieces";
import {
  boardRows,
  BoardIndex,
  FULL_ROW,
  NUM_ACTIONS,
  type Placement,
  placementIndex,
  pieceRowMasks,
  type Target,
} from "../engine/placement";
import { attackFor, scoreFor } from "../engine/scoring";
import { boardFeatures } from "./features";

export interface Afterstate {
  placement: Placement;
  index: number;
  target: Target;
  board: Int8Array; // HEIGHT x WIDTH after line clears
  linesCleared: number;
  landingHeight: number;
  erodedCells: number;
  attack: number;
  score: number;
  topOut: boolean;
  features: number[]; // 13
}

/** Piece placed when holding, and the piece that spawns after it. */
export function holdPieceAndNext(game: Game): [Piece, Piece] {
  if (game.hold !== null) return [game.hold, game.queue[0]];
  return [game.queue[0], game.queue[1]];
}

export function legalPlacements(game: Game, includeHold = true): [Placement, Target][] {
  if (game.gameOver || game.active === null) return [];
  const a = game.active;
  const key = `${game.board.join("")}|${a.piece},${a.rot},${a.row},${a.col}|${game.hold}|${game.canHold}|${game.queue[0]}|${includeHold}`;
  if (game.placementCache && game.placementCache[0] === key) return game.placementCache[1] as [Placement, Target][];
  const index = new BoardIndex(game.board);
  const out: [Placement, Target][] = index
    .targets(a.piece, [a.rot, a.row, a.col])
    .map((t) => [{ rotation: t.rotation, column: t.column, useHold: false }, t]);
  if (includeHold && game.canHold) {
    const [piece] = holdPieceAndNext(game);
    for (const t of index.targets(piece)) out.push([{ rotation: t.rotation, column: t.column, useHold: true }, t]);
  }
  game.placementCache = [key, out];
  return out;
}

export function actionMask(game: Game, includeHold = true): boolean[] {
  const mask = new Array(NUM_ACTIONS).fill(false);
  for (const [p] of legalPlacements(game, includeHold)) mask[placementIndex(p)] = true;
  return mask;
}

export function enumerateAfterstates(game: Game, includeHold = true, dedupe = false): Afterstate[] {
  const candidates = legalPlacements(game, includeHold);
  if (!candidates.length) return [];
  const index = new BoardIndex(game.board);
  const startRows = new Map<string, number>();
  const rows0 = boardRows(game.board);
  const nextPlain = game.queue[0];
  const [, nextHold] = holdPieceAndNext(game);
  const result: Afterstate[] = [];
  for (const [placement, t] of candidates) {
    let rows = rows0.slice();
    const touched: number[] = [];
    for (const [dr, mask] of pieceRowMasks(t.piece, t.rotation)) {
      const r = t.row + dr;
      rows[r] |= t.boxCol >= 0 ? mask << t.boxCol : mask >> -t.boxCol;
      touched.push(r);
    }
    const minR = touched[0];
    const maxR = touched[touched.length - 1];
    const landing = HEIGHT - (minR + maxR) / 2;
    const lockOut = maxR < HIDDEN_ROWS;

    let board = game.board.slice();
    for (const [dr, dc] of CELLS[t.piece][t.rotation]) board[(t.row + dr) * WIDTH + t.boxCol + dc] = t.piece;
    const full = touched.filter((r) => rows[r] === FULL_ROW);
    let lines = 0;
    let eroded = 0;
    let attack = 0;
    if (full.length) {
      const n = full.length;
      const kept: number[] = [];
      for (let r = 0; r < HEIGHT; r++) if (!full.includes(r)) kept.push(r);
      const nb = new Int8Array(HEIGHT * WIDTH);
      kept.forEach((r, i) => nb.set(board.subarray(r * WIDTH, (r + 1) * WIDTH), (n + i) * WIDTH));
      board = nb;
      rows = [...new Array(n).fill(0), ...kept.map((r) => rows[r])];
      lines = n;
      eroded = n * CELLS[t.piece][t.rotation].filter(([dr]) => full.includes(t.row + dr)).length;
      attack = attackFor(n, game.combo + 1, n === 4 && game.b2b);
    }
    const key = `${placement.useHold}|${t.piece}`;
    if (!startRows.has(key)) {
      if (placement.useHold) {
        const spawn = index.spawn(t.piece);
        startRows.set(key, spawn ? spawn[0] : t.row);
      } else startRows.set(key, game.active!.row);
    }
    const score =
      2 * Math.max(0, t.row - startRows.get(key)!) +
      scoreFor(lines, game.level, lines ? game.combo + 1 : 0, lines === 4 && game.b2b);
    const nxt = placement.useHold ? nextHold : nextPlain;
    const topOut = lockOut || new BoardIndex(null, rows).spawn(nxt) === null;
    result.push({
      placement,
      index: placementIndex(placement),
      target: t,
      board,
      linesCleared: lines,
      landingHeight: landing,
      erodedCells: eroded,
      attack,
      score,
      topOut,
      features: [...boardFeatures(board), lines, landing, eroded],
    });
  }
  if (!dedupe) return result;
  const seen = new Set<string>();
  return result.filter((a) => {
    const k = `${a.placement.useHold}|${a.board.join(",")}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}
