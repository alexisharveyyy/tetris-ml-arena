// Match controller (tetris/match/controller.py) for the modes the web app
// offers: solo, watch, race and marathon. Every board shares the piece seed.
import type { AgentKey } from "../agents/agents";
import type { Difficulty } from "../agents/base";
import { defaultConfig, Game, type GameConfig, type Move } from "../engine/game";
import { deriveSeed } from "../engine/rng";
import { AgentDriver } from "./driver";
import { FeedTracker } from "./feed";
import type { WorkerRunner } from "./runner";

const GARBAGE_SEED_STREAM = 1000;
export const LINES_IN_TIME = "lines_in_time";
export const FIRST_TO_LINES = "first_to_lines";
export const SCORE_IN_TIME = "score_in_time";

export interface RaceRules {
  condition: string;
  time_limit: number;
  target_lines: number;
}

export function defaultRace(): RaceRules {
  return { condition: LINES_IN_TIME, time_limit: 120, target_lines: 40 };
}

export function describeRace(r: RaceRules): string {
  if (r.condition === FIRST_TO_LINES) return `First to ${r.target_lines} lines`;
  const minutes = r.time_limit / 60;
  return `Most ${r.condition === LINES_IN_TIME ? "lines" : "score"} in ${Number(minutes.toPrecision(6))} min`;
}

export interface PlayerSpec {
  playerId: string;
  name: string;
  kind: "human" | "agent";
  agentKey: AgentKey | null;
  difficulty: Difficulty | null;
  preset: string | null;
}

export interface MatchConfig {
  mode: "solo" | "watch" | "race" | "marathon";
  seed: number;
  game: GameConfig;
  race: RaceRules | null;
  countdown: number;
}

export interface PlayerState {
  spec: PlayerSpec;
  game: Game;
  driver: AgentDriver | null;
  finishTime: number | null;
  done: boolean;
  goalTime: number | null;
}

const alive = (p: PlayerState) => !p.game.gameOver && !p.done;

export interface ResultRow {
  place?: number;
  id: string;
  name: string;
  kind: "human" | "agent";
  agent_key: string | null;
  preset: string | null;
  lines: number;
  score: number;
  pieces: number;
  tetrises: number;
  level: number;
  attack_sent: number;
  attack_received: number;
  pps: number;
  topped_out: boolean;
  finish_time: number | null;
  survival_time: number;
  model_moves: null;
}

export class MatchController {
  time = 0;
  countdown = 0;
  paused = false;
  finished = false;
  spectating = false;
  winner: string | null = null;
  standings: ResultRow[] = [];
  players = new Map<string, PlayerState>();
  feed = new FeedTracker();
  private standingsDirty = false;

  constructor(
    readonly config: MatchConfig,
    readonly specs: PlayerSpec[],
    readonly runner: WorkerRunner | null,
    readonly topK: number,
  ) {
    let i = 0;
    for (const spec of specs) {
      if (spec.kind !== "agent") continue;
      runner!.addAgent(spec.playerId, spec.agentKey!, spec.difficulty!, deriveSeed(config.seed, 3000 + i));
      i += 1;
    }
    this.reset();
  }

  reset(seed?: number): void {
    if (seed !== undefined) this.config.seed = seed;
    const s = this.config.seed;
    this.time = 0;
    this.countdown = this.config.countdown;
    this.paused = false;
    this.finished = false;
    this.spectating = false;
    this.winner = null;
    this.standings = [];
    this.players = new Map();
    this.specs.forEach((spec, i) => {
      const driver = spec.kind === "agent" ? new AgentDriver(spec.playerId, this.runner!, spec.difficulty!, this.topK) : null;
      const game = Game.create(s, this.config.game ?? defaultConfig(), deriveSeed(s, GARBAGE_SEED_STREAM + i));
      this.players.set(spec.playerId, { spec, game, driver, finishTime: null, done: false, goalTime: null });
    });
    this.feed.reset(this.liveStandings());
    this.standingsDirty = false;
  }

  close(): void {
    this.runner?.close();
  }

  get ready(): boolean {
    return this.runner === null || this.runner.ready();
  }

  setPaused(paused: boolean): void {
    if (!this.finished) this.paused = paused;
  }

  setDifficulty(pid: string, d: Difficulty): void {
    const p = this.players.get(pid)!;
    p.spec.difficulty = d;
    p.driver?.setDifficulty(d);
  }

  keepWatching(): void {
    if (this.finished) this.spectating = true;
  }

  endMatch(): void {
    if (!this.finished) this.finish();
  }

  private running(): boolean {
    return !(this.paused || this.finished || this.countdown > 0);
  }

  applyInput(pid: string, move: Move): boolean {
    const p = this.players.get(pid)!;
    if (!this.running() || p.driver !== null || !alive(p)) return false;
    const changed = p.game.apply(move);
    this.collect(p);
    this.checkEnd();
    this.updateFeed();
    return changed;
  }

  step(dt: number): void {
    if (this.paused || (this.finished && !this.spectating)) return;
    if (!this.ready) return;
    if (this.countdown > 0) {
      this.countdown = Math.max(0, this.countdown - dt);
      return;
    }
    this.time += dt;
    for (const p of this.players.values()) {
      if (p.driver !== null && alive(p)) {
        const before = p.game.piecesPlaced;
        p.driver.update(p.game, dt);
        this.collect(p);
        if (p.driver.idle && alive(p) && p.game.piecesPlaced !== before) {
          p.driver.update(p.game, 0);
          this.collect(p);
        }
      }
      if (alive(p)) p.game.tick(dt);
      this.collect(p);
    }
    this.checkEnd();
    this.updateFeed();
  }

  popFeed() {
    return this.feed.pop();
  }

  liveStandings(): string[] {
    return (this.finished ? this.standings : this.rank()).map((r) => r.id);
  }

  humanId(): string | null {
    return this.specs.find((s) => s.kind === "human")?.playerId ?? null;
  }

  private updateFeed(): void {
    if (this.runner !== null) {
      for (const p of this.players.values()) {
        if (p.driver === null) continue;
        const reason = this.runner.fallbackReason(p.spec.playerId);
        if (reason !== null) this.feed.onFallback(p.spec.playerId, reason, this.time);
      }
    }
    if (this.standingsDirty && !this.finished && (this.config.mode === "marathon" || this.config.mode === "race") && this.players.size > 1) {
      this.feed.onStandings(this.rank(), this.time, this.humanId());
    }
    this.standingsDirty = false;
  }

  private collect(p: PlayerState): void {
    const pid = p.spec.playerId;
    for (const ev of p.game.popEvents()) {
      this.feed.onEngineEvent(pid, ev, this.time, p.game.lines, p.spec.kind === "human");
      if (ev.type === "lock" || ev.type === "game_over") this.standingsDirty = true;
      if (ev.type === "lock") this.onLock(p);
      else if (ev.type === "game_over" && p.finishTime === null) p.finishTime = this.time;
    }
  }

  private onLock(p: PlayerState): void {
    const race = this.config.race;
    if (this.config.mode === "race" && race && race.condition === FIRST_TO_LINES && p.game.lines >= race.target_lines && p.goalTime === null) {
      p.goalTime = this.time;
      p.done = true;
      p.finishTime = this.time;
    }
  }

  private checkEnd(): void {
    if (this.finished) return;
    const living = [...this.players.values()].filter(alive);
    const mode = this.config.mode;
    if (mode === "marathon") {
      const humans = [...this.players.values()].filter((p) => p.spec.kind === "human");
      if ((humans.length && humans.some((h) => !alive(h))) || !living.length) this.finish();
    } else if (mode === "race") {
      const race = this.config.race!;
      if (!living.length) this.finish();
      else if (race.condition === FIRST_TO_LINES) {
        if ([...this.players.values()].some((p) => p.goalTime !== null)) this.finish();
      } else if (this.time >= race.time_limit) this.finish();
    } else if (!living.length) this.finish();
  }

  private finish(): void {
    if (this.finished) return;
    this.finished = true;
    this.paused = false;
    this.standings = this.rank();
    this.winner = this.standings[0]?.id ?? null;
    this.feed.onMatchEnd(this.winner, this.standings, this.time);
  }

  private rank(): ResultRow[] {
    const seat = new Map(this.specs.map((s, i) => [s.playerId, i]));
    const mode = this.config.mode;
    const key = (p: PlayerState): number[] => {
      const g = p.game;
      if (mode === "race") {
        const cond = this.config.race!.condition;
        if (cond === FIRST_TO_LINES) {
          const reached = p.goalTime !== null;
          return [reached ? 0 : 1, p.goalTime ?? 0, -g.lines, -g.score];
        }
        if (cond === SCORE_IN_TIME) return [-g.score, -g.lines];
        return [-g.lines, -g.score];
      }
      return [-g.score, -g.lines, seat.get(p.spec.playerId)!];
    };
    const players = [...this.players.values()];
    // Stable sort by the key tuple, like Python's sorted().
    const ranked = players
      .map((p, i) => ({ p, k: key(p), i }))
      .sort((a, b) => {
        for (let j = 0; j < Math.max(a.k.length, b.k.length); j++) {
          if (a.k[j] !== b.k[j]) return a.k[j] - b.k[j];
        }
        return a.i - b.i;
      })
      .map((x) => x.p);
    return ranked.map((p, i) => ({ place: i + 1, ...this.resultRow(p) }));
  }

  playerRow(pid: string): ResultRow {
    return this.resultRow(this.players.get(pid)!);
  }

  private resultRow(p: PlayerState): ResultRow {
    const g = p.game;
    const active = p.finishTime !== null ? p.finishTime : this.time;
    return {
      id: p.spec.playerId,
      name: p.spec.name,
      kind: p.spec.kind,
      agent_key: p.spec.agentKey,
      preset: p.spec.preset,
      lines: g.lines,
      score: g.score,
      pieces: g.piecesPlaced,
      tetrises: g.tetrises,
      level: g.level,
      attack_sent: g.attackSent,
      attack_received: 0,
      pps: active > 0 ? g.piecesPlaced / active : 0,
      topped_out: g.gameOver,
      finish_time: p.finishTime,
      survival_time: active,
      model_moves: null,
    };
  }
}

export { alive };
