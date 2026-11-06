// One live match in the page (tetris/web/session.py): controller, human
// input through DAS/ARR, and a loop on animation frames. When the tab is
// hidden the match pauses, and it resumes after a short countdown, like the
// server does when the client disconnects.
import type { Move } from "../engine/game";
import { MatchController, type MatchConfig, type PlayerSpec } from "./controller";
import { InputRepeater, type RepeatSettings } from "./das";
import type { FeedEventOut } from "./feed";
import { encodeSnapshot } from "./protocol";
import type { WorkerRunner } from "./runner";

const MAX_DT = 0.1;
const RESUME_COUNTDOWN = 3.0;
export const TOP_K = 3;

export type Snapshot = ReturnType<typeof encodeSnapshot>;

export class MatchSession {
  readonly match: MatchController;
  readonly humanId: string | null;
  historyId: string | null = null;
  private repeater: InputRepeater;
  private frame = 0;
  private ack: { seq: number; ts: number } | null = null;
  private userPaused = false;
  private disconnected = false;
  private started = false;
  private events: FeedEventOut[] = [];
  private listeners = new Set<(s: Snapshot) => void>();
  private raf = 0;
  private last = 0;
  private closed = false;
  private onHidden = () => this.visibility();

  constructor(
    readonly id: string,
    config: MatchConfig,
    players: PlayerSpec[],
    runner: WorkerRunner | null,
    repeat: RepeatSettings,
    private readonly onFinished: (s: MatchSession) => string | null,
  ) {
    this.match = new MatchController(config, players, runner, TOP_K);
    this.match.setPaused(true); // the clock starts when the first client attaches
    this.humanId = players.find((p) => p.kind === "human")?.playerId ?? null;
    this.repeater = new InputRepeater(repeat);
    document.addEventListener("visibilitychange", this.onHidden);
    this.last = performance.now();
    this.raf = requestAnimationFrame(this.loop);
  }

  get closedFlag(): boolean {
    return this.closed;
  }

  attach(listener: (s: Snapshot) => void): () => void {
    this.listeners.add(listener);
    if (!this.started) {
      this.started = true;
      this.match.setPaused(this.userPaused);
    }
    return () => this.listeners.delete(listener);
  }

  private visibility(): void {
    if (document.visibilityState === "hidden") {
      this.repeater.reset();
      if (!this.match.finished && this.started) {
        this.disconnected = true;
        this.match.setPaused(true);
      }
    } else if (this.disconnected) {
      this.disconnected = false;
      this.last = performance.now();
      if (!this.match.finished && !this.userPaused) {
        this.match.countdown = Math.max(this.match.countdown, RESUME_COUNTDOWN);
        this.match.setPaused(false);
      }
    }
  }

  handleInput(action: Move, down: boolean, seq: number, ts: number): void {
    if (down) for (const m of this.repeater.press(action)) this.apply(m);
    else this.repeater.release(action);
    this.ack = { seq, ts };
  }

  setPaused(paused: boolean): void {
    this.userPaused = paused;
    if (paused) this.repeater.reset();
    this.match.setPaused(paused || this.disconnected);
  }

  private apply(m: Move): void {
    if (this.humanId !== null) this.match.applyInput(this.humanId, m);
  }

  rematch(seed?: number): void {
    this.match.reset(seed);
    this.match.countdown = this.match.config.countdown;
    this.historyId = null;
    this.userPaused = false;
    this.repeater.reset();
    this.events = [];
  }

  keepWatching(): void {
    this.match.keepWatching();
  }

  end(): void {
    this.match.endMatch();
    this.afterStep();
  }

  step(dt: number): void {
    if (this.humanId !== null && !this.match.paused) for (const m of this.repeater.update(dt)) this.apply(m);
    this.match.step(dt);
    this.afterStep();
  }

  private afterStep(): void {
    this.events.push(...this.match.popFeed());
    if (this.match.finished && this.historyId === null) this.historyId = this.onFinished(this);
  }

  snapshot(): Snapshot {
    this.frame += 1;
    const events = this.events;
    this.events = [];
    return encodeSnapshot(this.match, this.frame, { ack: this.ack, events, disconnected: this.disconnected, historyId: this.historyId });
  }

  private loop = (now: number) => {
    if (this.closed) return;
    const dt = Math.min(Math.max(0, (now - this.last) / 1000), MAX_DT);
    this.last = now;
    this.step(dt);
    if (this.listeners.size) {
      const snap = this.snapshot();
      this.listeners.forEach((fn) => fn(snap));
    }
    this.raf = requestAnimationFrame(this.loop);
  };

  close(): void {
    this.closed = true;
    cancelAnimationFrame(this.raf);
    document.removeEventListener("visibilitychange", this.onHidden);
    this.listeners.clear();
    this.match.close();
  }
}
