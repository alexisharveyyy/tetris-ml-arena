// Drives one agent board in real time (tetris/match/agent_driver.py): ask for
// a decision, wait the reaction delay, then play the planned moves at the
// preset's move rate, holding the hard drop back to respect the speed cap.
import type { Difficulty, Scored } from "../agents/base";
import { PlanExecutor } from "../agents/planner";
import { pieceCells } from "../engine/board";
import type { Game } from "../engine/game";
import { letter } from "../engine/pieces";
import { BoardIndex, type Placement, placementFromIndex } from "../engine/placement";
import type { Decision, WorkerRunner } from "./runner";

const MAX_REDECISIONS = 3;
type Phase = "idle" | "thinking" | "waiting" | "moving";

export interface CandidateCells {
  rank: number;
  score: number;
  piece: string;
  hold: boolean;
  cells: [number, number][];
}

export class AgentDriver {
  clock = 0;
  phase: Phase = "idle";
  requestId = 0;
  decisions = 0;
  totalThinkTime = 0;
  candidateCells: CandidateCells[] = [];
  private token = -1;
  private spawnClock = 0;
  private lastLockClock = -1e9;
  private decision: Decision | null = null;
  private executor: PlanExecutor | null = null;
  private moveAcc = 0;
  private redecisions = 0;
  private lastCells: CandidateCells[] = [];
  private lastToken = -1;
  private lastLines = -1;
  private candLines = -1;

  constructor(
    readonly playerId: string,
    private readonly runner: WorkerRunner,
    public difficulty: Difficulty,
    public topK: number,
  ) {}

  get thinking(): boolean {
    return this.phase === "thinking";
  }

  get idle(): boolean {
    return this.phase === "idle";
  }

  setDifficulty(d: Difficulty): void {
    this.difficulty = d;
    this.runner.setDifficulty(this.playerId, d);
  }

  update(game: Game, dt: number): void {
    this.clock += dt;
    if (game.gameOver || game.active === null) return;
    if (this.phase !== "idle" && game.piecesPlaced !== this.token) this.finishPiece(false);
    if (this.phase === "idle") this.request(game);
    if (this.phase === "thinking") {
      const d = this.runner.poll(this.playerId);
      if (d !== null && d.requestId === this.requestId) this.accept(game, d);
    }
    if (this.phase === "waiting" && this.clock - this.spawnClock >= this.difficulty.reaction_delay) {
      const placement = this.decision?.placement ?? null;
      this.executor = new PlanExecutor(game, placement ?? placementFromIndex(0));
      if (placement === null) this.executor.moves = ["hard"];
      this.phase = "moving";
      this.moveAcc = 1;
    }
    if (this.phase === "moving") this.runMoves(game, dt);
  }

  private request(game: Game, keepClock = false): void {
    this.requestId += 1;
    this.token = game.piecesPlaced;
    if (!keepClock) {
      this.spawnClock = this.clock;
      this.redecisions = 0;
    }
    this.phase = "thinking";
    this.runner.submit(this.playerId, this.requestId, game.toDict(), this.topK);
  }

  private accept(game: Game, d: Decision): void {
    this.decision = d;
    this.decisions += 1;
    this.totalThinkTime += d.thinkTime;
    this.candidateCells = d.fallback ? [] : cellsFor(game, d.candidates);
    this.candLines = game.lines;
    if (d.fallback) this.lastCells = [];
    this.phase = "waiting";
  }

  private dropAllowed(): boolean {
    const pps = this.difficulty.pps;
    return pps === null || this.clock - this.lastLockClock >= 1 / pps;
  }

  private runMoves(game: Game, dt: number): void {
    const ex = this.executor!;
    const rate = this.difficulty.move_rate;
    if (rate === null) this.moveAcc = Infinity;
    else this.moveAcc += dt * rate;
    while (this.moveAcc >= 1 && !ex.done) {
      if (ex.unreachable && this.redecisions < MAX_REDECISIONS) {
        this.redecisions += 1;
        this.request(game, true);
        return;
      }
      const upcoming = ex.moves.length ? ex.moves[0] : "hard";
      if (upcoming === "hard" && !this.dropAllowed()) {
        this.moveAcc = Math.min(this.moveAcc, 1);
        return;
      }
      ex.nextMove(game);
      this.moveAcc -= 1;
    }
    if (ex.done) this.finishPiece(true);
  }

  private finishPiece(lockedByDriver: boolean): void {
    if (lockedByDriver) this.lastLockClock = this.clock;
    this.phase = "idle";
    this.decision = null;
    this.executor = null;
    if (this.candidateCells.length) {
      this.lastCells = this.candidateCells;
      this.lastToken = this.token;
      this.lastLines = this.candLines;
    }
    this.candidateCells = [];
  }

  /** Candidates to draw, and whether they are for the current piece. */
  overlay(game: Game): [CandidateCells[], boolean] {
    if (this.candidateCells.length) return [this.candidateCells, true];
    if (this.lastCells.length && game.piecesPlaced === this.lastToken + 1 && game.lines === this.lastLines) {
      return [this.lastCells, false];
    }
    return [[], false];
  }
}

function cellsFor(game: Game, candidates: Scored[]): CandidateCells[] {
  const out: CandidateCells[] = [];
  const index = new BoardIndex(game.board);
  candidates.forEach((c, i) => {
    const p: Placement = placementFromIndex(c.index);
    const a = game.active!;
    let piece;
    let target;
    if (p.useHold) {
      piece = game.hold !== null ? game.hold : game.queue[0];
      target = index.resolve(piece, p.rotation, p.column);
    } else {
      piece = a.piece;
      target = index.resolve(piece, p.rotation, p.column, [a.rot, a.row, a.col]);
    }
    if (target === null) return;
    out.push({
      rank: i + 1,
      score: c.score,
      piece: letter(piece),
      hold: p.useHold,
      cells: pieceCells(piece, target.rotation, target.row, target.boxCol),
    });
  });
  return out;
}
