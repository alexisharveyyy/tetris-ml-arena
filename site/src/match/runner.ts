// Agent runner: one Web Worker per agent board, so decisions never block the
// 60 fps game loop. Mirrors tetris/match/agent_runtime.py: a decision that
// misses its time limit, or a worker that fails, hands the board to the
// Dellacherie baseline (run in the page; it decides in a few milliseconds).
import { LinearAgent, linearWeights, type AgentKey } from "../agents/agents";
import type { Difficulty, Scored } from "../agents/base";
import { Game, type GameDict } from "../engine/game";
import { type Placement, placementFromIndex } from "../engine/placement";
import type { FromWorker, ToWorker } from "../worker/protocol";

/** Seconds per decision before the baseline takes over (play.decision_timeout). */
export const DECISION_TIMEOUT = 2.0;
/** Seconds to load a model (play.startup_timeout). */
export const STARTUP_TIMEOUT = 60.0;

export interface Decision {
  requestId: number;
  placement: Placement | null;
  candidates: Scored[];
  thinkTime: number;
  fallback: boolean;
}

interface Slot {
  key: AgentKey;
  worker: Worker | null;
  ready: boolean;
  difficulty: Difficulty;
  fallbackReason: string | null;
  baseline: LinearAgent;
  pending: { id: number; at: number; state: GameDict; topK: number } | null;
  result: Decision | null;
  startedAt: number;
}

export type WorkerFactory = () => Worker;

export class WorkerRunner {
  private slots = new Map<string, Slot>();
  private closed = false;

  constructor(private readonly makeWorker: WorkerFactory) {}

  addAgent(pid: string, key: AgentKey, difficulty: Difficulty, seed: bigint): void {
    const baseline = new LinearAgent(linearWeights("dellacherie"), seed);
    baseline.setDifficulty(difficulty);
    const slot: Slot = {
      key,
      worker: null,
      ready: false,
      difficulty,
      fallbackReason: null,
      baseline,
      pending: null,
      result: null,
      startedAt: performance.now(),
    };
    this.slots.set(pid, slot);
    let worker: Worker;
    try {
      worker = this.makeWorker();
    } catch {
      this.fail(slot, "load_failed");
      return;
    }
    slot.worker = worker;
    worker.onmessage = (e: MessageEvent<FromWorker>) => this.onMessage(slot, e.data);
    worker.onerror = () => this.fail(slot, slot.ready ? "crash" : "load_failed");
    const msg: ToWorker = { t: "init", key, seed, difficulty, base: import.meta.env.BASE_URL };
    worker.postMessage(msg);
  }

  private onMessage(slot: Slot, m: FromWorker): void {
    if (slot.fallbackReason) return;
    if (m.t === "ready") {
      slot.ready = true;
      if (slot.pending) this.send(slot);
    } else if (m.t === "error") this.fail(slot, slot.ready ? "crash" : "load_failed");
    else if (m.t === "decision" && slot.pending && m.id === slot.pending.id) {
      slot.result = {
        requestId: m.id,
        placement: m.placement === null ? null : placementFromIndex(m.placement),
        candidates: m.candidates.map(([index, score]) => ({ index, score })),
        thinkTime: m.thinkTime,
        fallback: false,
      };
      slot.pending = null;
    }
  }

  private fail(slot: Slot, reason: string): void {
    if (slot.fallbackReason) return;
    slot.fallbackReason = reason;
    slot.ready = true;
    slot.worker?.terminate();
    slot.worker = null;
    if (slot.pending) this.decideBaseline(slot);
  }

  private decideBaseline(slot: Slot): void {
    const p = slot.pending!;
    const t0 = performance.now();
    const game = Game.fromDict(p.state);
    const { placement, candidates } = slot.baseline.decide(game, p.topK);
    slot.result = { requestId: p.id, placement, candidates, thinkTime: (performance.now() - t0) / 1000, fallback: true };
    slot.pending = null;
  }

  private send(slot: Slot): void {
    const p = slot.pending!;
    p.at = performance.now();
    const msg: ToWorker = { t: "decide", id: p.id, state: p.state, topK: p.topK };
    slot.worker!.postMessage(msg);
  }

  ready(): boolean {
    const now = performance.now();
    for (const s of this.slots.values()) {
      if (!s.ready && (now - s.startedAt) / 1000 > STARTUP_TIMEOUT) this.fail(s, "load_timeout");
    }
    return [...this.slots.values()].every((s) => s.ready);
  }

  submit(pid: string, requestId: number, state: GameDict, topK: number): void {
    const slot = this.slots.get(pid)!;
    slot.result = null;
    slot.pending = { id: requestId, at: performance.now(), state, topK };
    if (slot.fallbackReason) this.decideBaseline(slot);
    else if (slot.ready) this.send(slot);
  }

  poll(pid: string): Decision | null {
    const slot = this.slots.get(pid)!;
    if (!slot.result && slot.pending && slot.ready && !slot.fallbackReason) {
      if ((performance.now() - slot.pending.at) / 1000 > DECISION_TIMEOUT) this.fail(slot, "timeout");
    }
    const r = slot.result;
    slot.result = null;
    return r;
  }

  setDifficulty(pid: string, d: Difficulty): void {
    const slot = this.slots.get(pid);
    if (!slot) return;
    slot.difficulty = d;
    slot.baseline.setDifficulty(d);
    const msg: ToWorker = { t: "difficulty", difficulty: d };
    slot.worker?.postMessage(msg);
  }

  fallbackReason(pid: string): string | null {
    return this.slots.get(pid)?.fallbackReason ?? null;
  }

  /** Warning text for a board, or null. */
  status(pid: string): string | null {
    const r = this.fallbackReason(pid);
    return r === null ? null : r;
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    for (const s of this.slots.values()) s.worker?.terminate();
    this.slots.clear();
  }
}
