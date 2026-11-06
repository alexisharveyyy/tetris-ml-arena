// Decision Transformer agent (tetris/agents/decision_transformer/agent.py).
// Keeps the history of the game it is playing: return-to-go, state and the
// chosen afterstate's embedding for the last context - 1 pieces.
import type { Game } from "../engine/game";
import { type Placement, placementFromIndex, placementIndex } from "../engine/placement";
import type { DTCore, DTEmbedder, DTStepInput } from "../nn/models";
import { Agent, rankBy, type Scored } from "./base";
import { afterstateArrays, legalActionArrays, observationArrays, REWARD_SCORE_SCALE, REWARD_SURVIVAL } from "./encoding";

interface Step {
  rtg: number;
  board: Float32Array;
  extras: Float32Array;
  timestep: number;
  actions: number[]; // legal actions, best first
  emb: Float32Array[]; // their embeddings, same order
  chosen: Float32Array | null;
}

export class DTAgent extends Agent {
  private history: Step[] = [];
  private current: Step | null = null;
  private lastPieces = -1;
  private lastScore = 0;
  private segmentStart = 0;
  private earned = 0;
  private readonly maxHistory: number;
  private readonly zero: Float32Array;

  constructor(
    private readonly embedder: DTEmbedder,
    private readonly core: DTCore,
    context: number,
    readonly defaultTarget: number,
    private readonly cap: number,
    seed?: number | bigint,
  ) {
    super(seed);
    this.maxHistory = Math.max(1, context - 1);
    this.zero = new Float32Array(core.d);
  }

  get target(): number {
    const t = this.difficulty.target_return;
    return t !== null && t !== undefined ? t : this.defaultTarget;
  }

  reset(): void {
    this.history = [];
    this.current = null;
    this.lastPieces = -1;
    this.lastScore = 0;
    this.segmentStart = 0;
    this.earned = 0;
  }

  private sync(game: Game): void {
    const pieces = game.piecesPlaced;
    if (this.current === null || pieces <= this.lastPieces) {
      this.reset();
      this.segmentStart = pieces;
    } else {
      const moved = pieces - this.lastPieces;
      const reward = (REWARD_SURVIVAL * moved + REWARD_SCORE_SCALE * (game.score - this.lastScore)) / this.cap;
      const done = this.current;
      if (done.chosen === null) done.chosen = done.emb[0];
      this.history.push(done);
      if (this.history.length > this.maxHistory) this.history.shift();
      this.earned += reward;
      if (pieces - this.segmentStart >= this.cap) {
        this.history = [];
        this.segmentStart = pieces;
        this.earned = 0;
      }
    }
    this.lastPieces = pieces;
    this.lastScore = game.score;
  }

  rank(game: Game): Scored[] {
    this.sync(game);
    const { arrays, afs } = legalActionArrays(game);
    if (!afs.length) {
      this.current = null;
      return [];
    }
    const n = afs.length;
    const d = this.core.d;
    const flat = this.embedder.forward(arrays.boards, arrays.extras, n);
    const emb = Array.from({ length: n }, (_, i) => flat.subarray(i * d, (i + 1) * d));
    const { board, extras } = observationArrays(game);
    const rtg = Math.max(this.target - this.earned, REWARD_SURVIVAL / this.cap);
    const step: Step = {
      rtg,
      board,
      extras,
      timestep: game.piecesPlaced - this.segmentStart,
      actions: arrays.actions,
      emb,
      chosen: null,
    };
    const steps: DTStepInput[] = [...this.history, step].map((s) => ({
      rtg: s.rtg,
      board: s.board,
      extras: s.extras,
      action: s.chosen ?? this.zero,
      timestep: s.timestep,
    }));
    const q = this.core.forward(steps);
    const scale = Math.sqrt(d);
    const scores = emb.map((e) => {
      let s = 0;
      for (let i = 0; i < d; i++) s += q[i] * e[i];
      return Math.fround(s) / scale;
    });
    const ranked = rankBy(
      arrays.actions.map((a, i) => ({ a, e: emb[i] })),
      scores,
    );
    step.actions = ranked.map((r) => r.item.a);
    step.emb = ranked.map((r) => r.item.e);
    this.current = step;
    return ranked.map((r) => ({ index: r.item.a, score: r.score }));
  }

  /** Record the placement actually played (the history uses its afterstate). */
  commit(game: Game, placement: Placement): void {
    const cur = this.current;
    if (cur === null) return;
    const index = placementIndex(placement);
    const hit = cur.actions.indexOf(index);
    if (hit >= 0) cur.chosen = cur.emb[hit];
    else {
      // A random (epsilon) placement: embed its afterstate.
      const { afs } = legalActionArrays(game);
      const a = afs.find((x) => x.index === index);
      if (!a) return;
      const arr = afterstateArrays(game, [a]);
      cur.chosen = this.embedder.forward(arr.boards, arr.extras, 1);
    }
  }

  override choose(game: Game, ranked: Scored[]): Placement {
    const placement = super.choose(game, ranked);
    this.commit(game, placement);
    return placement;
  }
}

export { placementFromIndex };
