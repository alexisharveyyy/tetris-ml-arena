// One Tetris board with its own piece sequence (tetris/engine/game.py).
// Real-time play through apply() and tick(), instant placement through place().
// The parity test replays Python games move for move and compares every field.
import {
  type Board,
  boardFromStrings,
  boardToStrings,
  clearFullRows,
  dropRow,
  emptyBoard,
  fits,
  HIDDEN_ROWS,
  insertGarbageRows,
  pieceCells,
  WIDTH,
} from "./board";
import { ALL_PIECES, CELLS, fromLetter, kicks, letter, type Piece } from "./pieces";
import { BoardIndex, type Placement, spawnPosition } from "./placement";
import { Rng, deriveSeed } from "./rng";
import { attackFor, gravityRowsPerSecond, scoreFor } from "./scoring";

const BAG_STREAM = 1;
const GARBAGE_STREAM = 2;

export type Move = "left" | "right" | "cw" | "ccw" | "soft" | "hard" | "hold";
export const MOVES: readonly Move[] = ["left", "right", "cw", "ccw", "soft", "hard", "hold"];

export class IllegalPlacementError extends Error {}

export interface GameConfig {
  start_level: number;
  max_level: number;
  lines_per_level: number;
  queue_size: number;
  lock_delay: number;
  max_lock_resets: number;
  gravity: number | null;
  garbage_cap: number | null;
  hole_change_prob: number;
}

export function defaultConfig(over: Partial<GameConfig> = {}): GameConfig {
  return {
    start_level: 1,
    max_level: 20,
    lines_per_level: 10,
    queue_size: 5,
    lock_delay: 0.5,
    max_lock_resets: 15,
    gravity: null,
    garbage_cap: null,
    hole_change_prob: 0.3,
    ...over,
  };
}

export interface ActivePiece {
  piece: Piece;
  rot: number;
  row: number;
  col: number;
}

export interface LockResult {
  piece: Piece;
  lines: number;
  combo: number;
  b2b: boolean;
  attack: number;
  sent: number;
  score: number;
  cells: [number, number][];
}

export interface GameEvent {
  type: string;
  data: Record<string, unknown>;
}

/** JSON-shaped state, the same fields as Game.to_dict() in Python. */
export interface GameDict {
  seed: number | bigint;
  config: GameConfig;
  board: string[];
  bag_rng: bigint;
  garbage_rng: bigint;
  queue: string;
  active: { piece: string; rot: number; row: number; col: number } | null;
  hold: string | null;
  can_hold: boolean;
  score: number;
  lines: number;
  level: number;
  combo: number;
  b2b: boolean;
  pieces_placed: number;
  attack_sent: number;
  clears: number[];
  game_over: boolean;
  pending_garbage: number[];
  garbage_hole: number | null;
  time: number;
  gravity_acc: number;
  lock_timer: number;
  lock_resets: number;
  lowest_row: number;
}

/** JSON.parse that keeps the 64-bit RNG states exact (as BigInt). */
export function parseGameJson<T = unknown>(text: string): T {
  return JSON.parse(text, function (this: unknown, key: string, value: unknown, ctx?: { source?: string }) {
    if ((key === "bag_rng" || key === "garbage_rng") && typeof value === "number") {
      if (!ctx?.source) throw new Error("JSON.parse source text access is required");
      return BigInt(ctx.source);
    }
    return value;
  } as (key: string, value: unknown) => unknown) as T;
}

export class Game {
  seed: number | bigint = 0;
  config: GameConfig = defaultConfig();
  board: Board = emptyBoard();
  bagRng: Rng = new Rng(0);
  garbageRng: Rng = new Rng(0);
  queue: Piece[] = [];
  active: ActivePiece | null = null;
  hold: Piece | null = null;
  canHold = true;
  score = 0;
  lines = 0;
  level = 1;
  combo = -1;
  b2b = false;
  piecesPlaced = 0;
  attackSent = 0;
  clears = [0, 0, 0, 0, 0];
  gameOver = false;
  pendingGarbage: number[] = [];
  garbageHole: number | null = null;
  time = 0;
  gravityAcc = 0;
  lockTimer = 0;
  lockResets = 0;
  lowestRow = 0;
  private events: GameEvent[] = [];
  /** Afterstate enumeration cache (see env/afterstates). */
  placementCache: [string, unknown] | null = null;

  static create(seed: number | bigint = 0, config?: GameConfig, garbageSeed?: number | bigint | null): Game {
    const g = new Game();
    g.seed = seed;
    g.config = config ?? defaultConfig();
    g.bagRng = new Rng(deriveSeed(seed, BAG_STREAM));
    g.garbageRng = new Rng(deriveSeed(garbageSeed ?? seed, GARBAGE_STREAM));
    g.level = g.config.start_level;
    g.fillQueue();
    g.spawn(g.nextPiece());
    return g;
  }

  // ---------------------------------------------------------------- queue

  private fillQueue(): void {
    while (this.queue.length < Math.max(this.config.queue_size + 1, 7)) {
      const bag = [...ALL_PIECES];
      this.bagRng.shuffle(bag);
      this.queue.push(...bag);
    }
  }

  private nextPiece(): Piece {
    const p = this.queue.shift()!;
    this.fillQueue();
    return p;
  }

  get nextQueue(): Piece[] {
    return this.queue.slice(0, this.config.queue_size);
  }

  // ---------------------------------------------------------------- spawn

  private spawn(piece: Piece): boolean {
    const pos = spawnPosition(this.board, piece);
    if (pos === null) {
      this.active = null;
      this.topOut("block_out");
      return false;
    }
    const [row, col] = pos;
    this.active = { piece, rot: 0, row, col };
    this.gravityAcc = 0;
    this.lockTimer = 0;
    this.lockResets = 0;
    this.lowestRow = row;
    this.emit("spawn", { piece: letter(piece) });
    return true;
  }

  private topOut(reason: string): void {
    if (!this.gameOver) {
      this.gameOver = true;
      this.emit("game_over", { reason });
    }
  }

  private emit(type: string, data: Record<string, unknown>): void {
    this.events.push({ type, data });
  }

  popEvents(): GameEvent[] {
    const e = this.events;
    this.events = [];
    return e;
  }

  // --------------------------------------------------------- live control

  gravity(): number {
    if (this.config.gravity !== null) return this.config.gravity;
    return gravityRowsPerSecond(this.level);
  }

  private grounded(): boolean {
    const a = this.active;
    return a !== null && !fits(this.board, a.piece, a.rot, a.row + 1, a.col);
  }

  private onSuccessfulMove(wasGrounded: boolean): void {
    const a = this.active!;
    if (a.row > this.lowestRow) {
      this.lowestRow = a.row;
      this.lockResets = 0;
      this.lockTimer = 0;
    } else if ((wasGrounded || this.grounded()) && this.lockResets < this.config.max_lock_resets) {
      this.lockResets += 1;
      this.lockTimer = 0;
    }
  }

  private shift(dcol: number): boolean {
    const a = this.active;
    if (a === null || !fits(this.board, a.piece, a.rot, a.row, a.col + dcol)) return false;
    const wasGrounded = this.grounded();
    a.col += dcol;
    this.onSuccessfulMove(wasGrounded);
    return true;
  }

  private rotate(direction: number): boolean {
    const a = this.active;
    if (a === null) return false;
    const newRot = (((a.rot + direction) % 4) + 4) % 4;
    for (const [drow, dcol] of kicks(a.piece, a.rot, newRot)) {
      if (fits(this.board, a.piece, newRot, a.row + drow, a.col + dcol)) {
        const wasGrounded = this.grounded();
        a.rot = newRot;
        a.row += drow;
        a.col += dcol;
        this.onSuccessfulMove(wasGrounded);
        return true;
      }
    }
    return false;
  }

  private softDrop(): boolean {
    const a = this.active;
    if (a === null || !fits(this.board, a.piece, a.rot, a.row + 1, a.col)) return false;
    a.row += 1;
    this.score += 1;
    this.onSuccessfulMove(false);
    return true;
  }

  private hardDrop(): LockResult | null {
    const a = this.active;
    if (a === null) return null;
    const target = dropRow(this.board, a.piece, a.rot, a.row, a.col);
    this.score += 2 * (target - a.row);
    a.row = target;
    return this.lock();
  }

  private doHold(): boolean {
    const a = this.active;
    if (a === null || !this.canHold) return false;
    const current = a.piece;
    const incoming = this.hold !== null ? this.hold : this.nextPiece();
    this.hold = current;
    this.canHold = false;
    this.emit("hold", { piece: letter(current) });
    this.spawn(incoming);
    return true;
  }

  /** Apply a player move. True if it changed the state. */
  apply(move: Move): boolean {
    if (this.gameOver || this.active === null) return false;
    switch (move) {
      case "left":
        return this.shift(-1);
      case "right":
        return this.shift(1);
      case "cw":
        return this.rotate(1);
      case "ccw":
        return this.rotate(-1);
      case "soft":
        return this.softDrop();
      case "hard":
        return this.hardDrop() !== null;
      case "hold":
        return this.doHold();
    }
    throw new Error(`unknown move ${move}`);
  }

  /** Advance gravity and lock delay by dt seconds. */
  tick(dt: number): void {
    if (this.gameOver) return;
    this.time += dt;
    const a = this.active;
    if (a === null) return;
    this.gravityAcc += dt * this.gravity();
    while (this.gravityAcc >= 1.0) {
      if (!fits(this.board, a.piece, a.rot, a.row + 1, a.col)) {
        this.gravityAcc = 0;
        break;
      }
      this.gravityAcc -= 1.0;
      a.row += 1;
      if (a.row > this.lowestRow) {
        this.lowestRow = a.row;
        this.lockResets = 0;
        this.lockTimer = 0;
      }
    }
    if (this.grounded()) {
      this.lockTimer += dt;
      if (this.lockTimer >= this.config.lock_delay || this.lockResets >= this.config.max_lock_resets) {
        this.lock();
      }
    } else {
      this.lockTimer = 0;
    }
  }

  ghostRow(): number | null {
    const a = this.active;
    return a === null ? null : dropRow(this.board, a.piece, a.rot, a.row, a.col);
  }

  // ------------------------------------------------------ placement level

  /** Instantly place the current piece (or the hold piece) and lock it. */
  place(placement: Placement): LockResult {
    if (this.gameOver || this.active === null) throw new IllegalPlacementError("game is over");
    const index = new BoardIndex(this.board);
    let piece: Piece;
    let target;
    if (placement.useHold) {
      if (!this.canHold) throw new IllegalPlacementError("hold already used for this piece");
      piece = this.hold !== null ? this.hold : this.queue[0];
      target = index.resolve(piece, placement.rotation, placement.column);
    } else {
      const a = this.active;
      piece = a.piece;
      target = index.resolve(piece, placement.rotation, placement.column, [a.rot, a.row, a.col]);
    }
    if (target === null) throw new IllegalPlacementError(`illegal placement for ${letter(piece)}`);
    if (placement.useHold) {
      this.doHold();
      if (this.active === null) throw new IllegalPlacementError("hold piece could not spawn");
    }
    const a = this.active!;
    const startRow = a.row;
    a.rot = target.rotation;
    a.col = target.boxCol;
    a.row = target.row;
    this.score += 2 * Math.max(0, target.row - startRow);
    return this.lock()!;
  }

  // ----------------------------------------------------------------- lock

  private lock(): LockResult | null {
    const a = this.active;
    if (a === null) return null;
    const cells = pieceCells(a.piece, a.rot, a.row, a.col);
    for (const [r, c] of cells) this.board[r * WIDTH + c] = a.piece;
    this.active = null;
    this.piecesPlaced += 1;
    const lockOut = cells.every(([r]) => r < HIDDEN_ROWS);

    const [board, cleared] = clearFullRows(this.board);
    this.board = board;
    const lines = cleared.length;
    let b2b = false;
    if (lines > 0) {
      this.combo += 1;
      if (lines === 4) {
        b2b = this.b2b;
        this.b2b = true;
      } else {
        this.b2b = false;
      }
    } else {
      this.combo = -1;
    }

    const combo = Math.max(this.combo, 0);
    const score = scoreFor(lines, this.level, combo, b2b);
    const attack = attackFor(lines, combo, b2b);
    let sent = attack;
    while (sent > 0 && this.pendingGarbage.length) {
      const cancel = Math.min(sent, this.pendingGarbage[0]);
      sent -= cancel;
      this.pendingGarbage[0] -= cancel;
      if (this.pendingGarbage[0] === 0) this.pendingGarbage.shift();
    }

    this.score += score;
    this.lines += lines;
    this.clears[lines] += 1;
    this.attackSent += sent;
    this.level = Math.min(
      this.config.max_level,
      Math.max(this.level, this.config.start_level + Math.floor(this.lines / this.config.lines_per_level)),
    );
    this.canHold = true;
    const result: LockResult = { piece: a.piece, lines, combo: lines ? combo : -1, b2b, attack, sent, score, cells };
    this.emit("lock", { ...result, piece: letter(a.piece) });

    if (lockOut) {
      this.topOut("lock_out");
      return result;
    }
    if (lines === 0 && this.pendingGarbage.length) {
      this.insertPendingGarbage();
      if (this.gameOver) return result;
    }
    this.spawn(this.nextPiece());
    return result;
  }

  // -------------------------------------------------------------- garbage

  queueGarbage(rows: number): void {
    if (rows > 0) {
      this.pendingGarbage.push(rows);
      this.emit("garbage_queued", { rows, pending: this.pendingTotal });
    }
  }

  get tetrises(): number {
    return this.clears[4];
  }

  get pendingTotal(): number {
    return this.pendingGarbage.reduce((a, b) => a + b, 0);
  }

  private insertPendingGarbage(): void {
    let budget = this.config.garbage_cap;
    while (this.pendingGarbage.length && !this.gameOver) {
      let rows = this.pendingGarbage[0];
      if (budget !== null) {
        if (budget <= 0) break;
        rows = Math.min(rows, budget);
        budget -= rows;
      }
      this.pendingGarbage[0] -= rows;
      if (this.pendingGarbage[0] === 0) this.pendingGarbage.shift();
      this.receiveGarbage(rows);
    }
  }

  receiveGarbage(rows: number): void {
    if (rows <= 0 || this.gameOver) return;
    let hole = this.garbageHole;
    if (hole === null) hole = this.garbageRng.randbelow(WIDTH);
    else if (this.garbageRng.random() < this.config.hole_change_prob) {
      hole = (hole + 1 + this.garbageRng.randbelow(WIDTH - 1)) % WIDTH;
    }
    this.garbageHole = hole;
    let overflow: boolean;
    [this.board, overflow] = insertGarbageRows(this.board, rows, hole);
    this.emit("garbage", { rows, hole });
    const a = this.active;
    if (a !== null) {
      while (!fits(this.board, a.piece, a.rot, a.row, a.col)) {
        a.row -= 1;
        if (a.row + Math.min(...CELLS[a.piece][a.rot].map(([dr]) => dr)) < 0) {
          overflow = true;
          break;
        }
      }
      this.lowestRow = Math.min(this.lowestRow, a.row);
    }
    if (overflow) this.topOut("garbage_out");
  }

  // -------------------------------------------------------- serialization

  toDict(): GameDict {
    const a = this.active;
    return {
      seed: this.seed,
      config: { ...this.config },
      board: boardToStrings(this.board),
      bag_rng: this.bagRng.state,
      garbage_rng: this.garbageRng.state,
      queue: this.queue.map(letter).join(""),
      active: a ? { piece: letter(a.piece), rot: a.rot, row: a.row, col: a.col } : null,
      hold: this.hold !== null ? letter(this.hold) : null,
      can_hold: this.canHold,
      score: this.score,
      lines: this.lines,
      level: this.level,
      combo: this.combo,
      b2b: this.b2b,
      pieces_placed: this.piecesPlaced,
      attack_sent: this.attackSent,
      clears: [...this.clears],
      game_over: this.gameOver,
      pending_garbage: [...this.pendingGarbage],
      garbage_hole: this.garbageHole,
      time: this.time,
      gravity_acc: this.gravityAcc,
      lock_timer: this.lockTimer,
      lock_resets: this.lockResets,
      lowest_row: this.lowestRow,
    };
  }

  static fromDict(d: GameDict): Game {
    const g = new Game();
    g.seed = d.seed;
    g.config = { ...defaultConfig(), ...d.config };
    g.board = boardFromStrings(d.board);
    g.bagRng = new Rng(0);
    g.bagRng.state = BigInt(d.bag_rng);
    g.garbageRng = new Rng(0);
    g.garbageRng.state = BigInt(d.garbage_rng);
    g.queue = [...d.queue].map(fromLetter);
    g.active = d.active ? { piece: fromLetter(d.active.piece), rot: d.active.rot, row: d.active.row, col: d.active.col } : null;
    g.hold = d.hold ? fromLetter(d.hold) : null;
    g.canHold = d.can_hold;
    g.score = d.score;
    g.lines = d.lines;
    g.level = d.level;
    g.combo = d.combo;
    g.b2b = d.b2b;
    g.piecesPlaced = d.pieces_placed;
    g.attackSent = d.attack_sent;
    g.clears = [...(d.clears ?? [0, 0, 0, 0, 0])];
    g.gameOver = d.game_over;
    g.pendingGarbage = [...d.pending_garbage];
    g.garbageHole = d.garbage_hole ?? null;
    g.time = d.time;
    g.gravityAcc = d.gravity_acc;
    g.lockTimer = d.lock_timer;
    g.lockResets = d.lock_resets;
    g.lowestRow = d.lowest_row;
    return g;
  }

  copy(): Game {
    const g = Object.assign(new Game(), this) as Game;
    g.config = { ...this.config };
    g.board = this.board.slice();
    g.bagRng = new Rng(0);
    g.bagRng.state = this.bagRng.state;
    g.garbageRng = new Rng(0);
    g.garbageRng.state = this.garbageRng.state;
    g.queue = [...this.queue];
    g.active = this.active ? { ...this.active } : null;
    g.pendingGarbage = [...this.pendingGarbage];
    g.clears = [...this.clears];
    g.events = [];
    g.placementCache = null;
    return g;
  }
}
