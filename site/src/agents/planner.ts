// Path planner (tetris/agents/planner.py): turns a placement into real moves.
import type { Game, Move } from "../engine/game";
import { kicks, MIN_DC, type Piece } from "../engine/pieces";
import { BoardIndex, type Placement, type Target } from "../engine/placement";

type State = [number, number, number]; // rot, row, col
export const MAX_REPLANS = 8;
const LATERAL: Move[] = ["cw", "ccw", "left", "right"];

function successor(index: BoardIndex, piece: Piece, s: State, move: Move): State | null {
  const [rot, row, col] = s;
  if (move === "left" || move === "right") {
    const c = move === "left" ? col - 1 : col + 1;
    return index.fits(piece, rot, row, c) ? [rot, row, c] : null;
  }
  if (move === "soft") return index.fits(piece, rot, row + 1, col) ? [rot, row + 1, col] : null;
  const newRot = (((rot + (move === "cw" ? 1 : -1)) % 4) + 4) % 4;
  for (const [dr, dc] of kicks(piece, rot, newRot)) {
    if (index.fits(piece, newRot, row + dr, col + dc)) return [newRot, row + dr, col + dc];
  }
  return null;
}

function bfs(index: BoardIndex, piece: Piece, start: State, goal: Target, moves: Move[]): Move[] | null {
  const isGoal = (s: State) => s[0] === goal.rotation && s[2] === goal.boxCol && index.landingRow(piece, s[0], s[1], s[2]) === goal.row;
  if (isGoal(start)) return [];
  const key = (s: State) => `${s[0]},${s[1]},${s[2]}`;
  const parents = new Map<string, [State, Move]>();
  const seen = new Set([key(start)]);
  const queue: State[] = [start];
  for (let qi = 0; qi < queue.length; qi++) {
    const state = queue[qi];
    for (const move of moves) {
      const nxt = successor(index, piece, state, move);
      if (nxt === null || seen.has(key(nxt))) continue;
      seen.add(key(nxt));
      parents.set(key(nxt), [state, move]);
      if (isGoal(nxt)) {
        const path: Move[] = [];
        let s = nxt;
        while (key(s) !== key(start)) {
          const [p, m] = parents.get(key(s))!;
          path.push(m);
          s = p;
        }
        return path.reverse();
      }
      queue.push(nxt);
    }
  }
  return null;
}

export function findPath(index: BoardIndex, piece: Piece, start: State, goal: Target): Move[] | null {
  return bfs(index, piece, start, goal, LATERAL) ?? bfs(index, piece, start, goal, [...LATERAL, "soft"]);
}

/** Moves from the current state to the placement, ending with a hard drop. */
export function planPath(game: Game, placement: Placement): Move[] | null {
  if (game.gameOver || game.active === null) return null;
  const prefix: Move[] = [];
  let sim = game;
  if (placement.useHold) {
    if (!game.canHold) return null;
    sim = game.copy();
    if (!sim.apply("hold") || sim.active === null) return null;
    prefix.push("hold");
  }
  const a = sim.active!;
  const index = new BoardIndex(sim.board);
  let goal = index.resolve(a.piece, placement.rotation, placement.column, [a.rot, a.row, a.col]);
  if (goal === null) {
    const rot = placement.rotation % 4;
    const boxCol = placement.column - MIN_DC[a.piece][rot];
    if (!index.fits(a.piece, rot, a.row, boxCol)) return null;
    goal = { piece: a.piece, rotation: rot, column: placement.column, boxCol, row: index.landingRow(a.piece, rot, a.row, boxCol) };
  }
  const path = findPath(index, a.piece, [a.rot, a.row, a.col], goal);
  return path === null ? null : [...prefix, ...path, "hard"];
}

/** Feeds a placement's moves to a live game, re-planning when gravity interferes. */
export class PlanExecutor {
  placement: Placement;
  done = false;
  replans = 0;
  unreachable = false;
  moves: Move[] = [];
  private pieceCount: number;
  private expected: State | null = null;

  constructor(game: Game, placement: Placement) {
    this.placement = placement;
    this.pieceCount = game.piecesPlaced;
    this.plan(game);
  }

  private plan(game: Game): void {
    const moves = planPath(game, this.placement);
    this.unreachable = moves === null;
    this.moves = moves ?? ["hard"];
    this.expected = null;
  }

  nextMove(game: Game): Move | null {
    if (this.done || game.gameOver || game.active === null) {
      this.done = true;
      return null;
    }
    if (game.piecesPlaced !== this.pieceCount) {
      this.done = true;
      return null;
    }
    const a = game.active;
    if (this.expected !== null && (a.rot !== this.expected[0] || a.row !== this.expected[1] || a.col !== this.expected[2])) {
      this.replans += 1;
      this.placement = { ...this.placement, useHold: false };
      this.plan(game);
      if (this.replans > MAX_REPLANS) {
        this.unreachable = true;
        this.moves = ["hard"];
      }
      if (this.unreachable) return null;
    }
    if (!this.moves.length) this.moves.push("hard");
    const move = this.moves.shift()!;
    const ok = game.apply(move);
    if (move === "hard" || game.piecesPlaced !== this.pieceCount) this.done = true;
    else if (!ok) {
      this.replans += 1;
      this.plan(game);
    } else {
      if (move === "hold") this.placement = { ...this.placement, useHold: false };
      const b = game.active;
      this.expected = b ? [b.rot, b.row, b.col] : null;
    }
    return move;
  }
}
