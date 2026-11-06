// Common agent interface and difficulty noise (tetris/agents/base.py).
import type { Game } from "../engine/game";
import { type Placement, placementFromIndex } from "../engine/placement";
import { Rng } from "../engine/rng";
import { legalPlacements } from "./afterstates";

export const MISTAKE_POOL = 5;
export const MISTAKE_DECAY = 0.5;

export interface Difficulty {
  pps: number | null;
  move_rate: number | null;
  reaction_delay: number;
  noise: number;
  epsilon: number;
  target_return: number | null;
}

export const PRESETS: Record<string, Difficulty> = {
  Easy: { pps: 0.5, move_rate: 5.0, reaction_delay: 0.9, noise: 0.4, epsilon: 0.03, target_return: null },
  Medium: { pps: 1.0, move_rate: 10.0, reaction_delay: 0.45, noise: 0.15, epsilon: 0, target_return: null },
  Hard: { pps: 2.0, move_rate: 25.0, reaction_delay: 0.2, noise: 0.03, epsilon: 0, target_return: null },
  Unlimited: { pps: null, move_rate: null, reaction_delay: 0, noise: 0, epsilon: 0, target_return: null },
};

export interface Scored {
  index: number; // 80-action index
  score: number;
}

/** Stable sort, best first; ties keep the original order (np.argsort(-s, kind="stable")). */
export function rankBy<T>(items: T[], scores: ArrayLike<number>): { item: T; score: number }[] {
  return items
    .map((item, i) => ({ item, score: scores[i], i }))
    .sort((a, b) => (b.score === a.score ? a.i - b.i : b.score - a.score))
    .map(({ item, score }) => ({ item, score }));
}

export abstract class Agent {
  difficulty: Difficulty = { pps: 2.0, move_rate: 15.0, reaction_delay: 0.3, noise: 0, epsilon: 0, target_return: null };
  rng: Rng;

  constructor(seed: number | bigint = 0) {
    this.rng = new Rng(seed);
  }

  setDifficulty(d: Difficulty): void {
    this.difficulty = d;
  }

  /** Legal placements with scores, best first. */
  abstract rank(game: Game): Scored[];

  /** Difficulty noise over a ranking. */
  choose(game: Game, ranked: Scored[]): Placement {
    const d = this.difficulty;
    if (d.epsilon > 0 && this.rng.random() < d.epsilon) {
      const options = legalPlacements(game);
      if (options.length) return options[this.rng.randbelow(options.length)][0];
    }
    if (!ranked.length) throw new Error("no legal placement");
    if (d.noise > 0 && ranked.length > 1 && this.rng.random() < d.noise) {
      const pool = ranked.slice(1, 1 + MISTAKE_POOL);
      const weights = pool.map((_, i) => MISTAKE_DECAY ** i);
      const total = weights.reduce((a, b) => a + b, 0);
      let x = this.rng.random() * total;
      for (let i = 0; i < pool.length; i++) {
        x -= weights[i];
        if (x < 0) return placementFromIndex(pool[i].index);
      }
      return placementFromIndex(pool[pool.length - 1].index);
    }
    return placementFromIndex(ranked[0].index);
  }

  /** Placement plus the top-k ranked candidates. */
  decide(game: Game, topK: number): { placement: Placement; candidates: Scored[] } {
    const ranked = this.rank(game);
    return { placement: this.choose(game, ranked), candidates: ranked.slice(0, topK) };
  }
}
