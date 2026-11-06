// Stands in for web/src/api/socket.ts: the same MatchSocket interface, fed by
// the match running in the page instead of a WebSocket.
import { perfAcked, perfMessage } from "../../../web/src/lib/perf";
import type { Action, FeedEvent, Snapshot } from "../../../web/src/api/types";
import { getSession } from "./client";

export type SocketStatus = "connecting" | "open" | "lost" | "gone";

export interface SocketStats {
  messages: number;
  bytes: number;
  latencies: number[];
}

let active: MatchSocket | null = null;

/** The socket of the match on screen (for the touch controls). */
export function activeSocket(): MatchSocket | null {
  return active;
}

export class MatchSocket {
  latest: Snapshot | null = null;
  status: SocketStatus = "connecting";
  readonly stats: SocketStats = { messages: 0, bytes: 0, latencies: [] };
  private seq = 0;
  private pending = new Map<number, number>();
  private detach: (() => void) | null = null;
  private snapListeners = new Set<(s: Snapshot) => void>();
  private eventListeners = new Set<(e: FeedEvent, snap: Snapshot) => void>();
  private statusListeners = new Set<(s: SocketStatus) => void>();

  constructor(private readonly matchId: string) {
    // Attach on the next task, after the screen has registered its listeners.
    setTimeout(() => this.connect(), 0);
    active = this;
  }

  get id(): string {
    return this.matchId;
  }

  onSnapshot(fn: (s: Snapshot) => void): () => void {
    this.snapListeners.add(fn);
    return () => this.snapListeners.delete(fn);
  }

  onEvent(fn: (e: FeedEvent, snap: Snapshot) => void): () => void {
    this.eventListeners.add(fn);
    return () => this.eventListeners.delete(fn);
  }

  onStatus(fn: (s: SocketStatus) => void): () => void {
    this.statusListeners.add(fn);
    return () => this.statusListeners.delete(fn);
  }

  input(action: Action, down: boolean): void {
    const s = getSession(this.matchId);
    if (!s) return;
    this.seq += 1;
    const ts = performance.now();
    if (down) this.pending.set(this.seq, ts);
    s.handleInput(action, down, this.seq, ts);
  }

  pause(paused: boolean): void {
    getSession(this.matchId)?.setPaused(paused);
  }

  close(): void {
    if (active === this) active = null;
    this.detach?.();
    this.detach = null;
  }

  private setStatus(s: SocketStatus): void {
    if (this.status === s) return;
    this.status = s;
    this.statusListeners.forEach((fn) => fn(s));
  }

  private connect(): void {
    const session = getSession(this.matchId);
    if (!session) {
      this.setStatus("gone");
      return;
    }
    this.setStatus("open");
    this.detach = session.attach((raw) => {
      const snap = raw as unknown as Snapshot;
      this.stats.messages += 1;
      perfMessage(0);
      this.latest = snap;
      if (snap.ack) {
        const now = performance.now();
        const done: number[] = [];
        for (const [s, ts] of this.pending) {
          if (s <= snap.ack.seq) {
            this.stats.latencies.push(now - ts);
            done.push(ts);
            this.pending.delete(s);
          }
        }
        perfAcked(snap.f, done);
        if (this.stats.latencies.length > 2000) this.stats.latencies.splice(0, 1000);
      }
      for (const ev of snap.ev) this.eventListeners.forEach((fn) => fn(ev, snap));
      this.snapListeners.forEach((fn) => fn(snap));
    });
  }
}
