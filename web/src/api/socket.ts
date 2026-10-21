// One WebSocket per match. Reconnects on its own; the server pauses the match
// while no client is attached and resumes it after a countdown.
import type { Action, FeedEvent, Snapshot } from "./types";

export type SocketStatus = "connecting" | "open" | "lost" | "gone";

export interface SocketStats {
  messages: number;
  bytes: number;
  latencies: number[]; // input to first acknowledging snapshot, ms
}

export class MatchSocket {
  latest: Snapshot | null = null;
  status: SocketStatus = "connecting";
  readonly stats: SocketStats = { messages: 0, bytes: 0, latencies: [] };
  private ws: WebSocket | null = null;
  private seq = 0;
  private pending = new Map<number, number>();
  private closed = false;
  private retry = 0;
  private snapListeners = new Set<(s: Snapshot) => void>();
  private eventListeners = new Set<(e: FeedEvent) => void>();
  private statusListeners = new Set<(s: SocketStatus) => void>();

  constructor(private readonly matchId: string, private readonly url = defaultUrl(matchId)) {
    this.connect();
  }

  onSnapshot(fn: (s: Snapshot) => void): () => void {
    this.snapListeners.add(fn);
    return () => this.snapListeners.delete(fn);
  }

  onEvent(fn: (e: FeedEvent) => void): () => void {
    this.eventListeners.add(fn);
    return () => this.eventListeners.delete(fn);
  }

  onStatus(fn: (s: SocketStatus) => void): () => void {
    this.statusListeners.add(fn);
    return () => this.statusListeners.delete(fn);
  }

  get id(): string {
    return this.matchId;
  }

  input(action: Action, down: boolean): void {
    this.seq += 1;
    const ts = performance.now();
    if (down) this.pending.set(this.seq, ts);
    this.send({ t: "input", seq: this.seq, action, down, ts });
  }

  pause(paused: boolean): void {
    this.send({ t: "pause", paused });
  }

  close(): void {
    this.closed = true;
    this.ws?.close();
  }

  private setStatus(s: SocketStatus): void {
    if (this.status === s) return;
    this.status = s;
    this.statusListeners.forEach((fn) => fn(s));
  }

  private send(msg: unknown): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(msg));
  }

  private connect(): void {
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onopen = () => {
      this.retry = 0;
      this.setStatus("open");
      this.send({ t: "hello", last_frame: this.latest?.f ?? null });
    };
    ws.onmessage = (m) => {
      const text = String(m.data);
      this.stats.messages += 1;
      this.stats.bytes += text.length;
      const msg = JSON.parse(text);
      if (msg.t !== "snap") return;
      const snap = msg as Snapshot;
      this.latest = snap;
      if (snap.ack) {
        const now = performance.now();
        for (const [s, ts] of this.pending) {
          if (s <= snap.ack.seq) {
            this.stats.latencies.push(now - ts);
            this.pending.delete(s);
          }
        }
        if (this.stats.latencies.length > 2000) this.stats.latencies.splice(0, 1000);
      }
      for (const ev of snap.ev) this.eventListeners.forEach((fn) => fn(ev));
      this.snapListeners.forEach((fn) => fn(snap));
    };
    ws.onclose = (e) => {
      if (ws !== this.ws || this.closed) return;
      if (e.code === 4404) {
        this.setStatus("gone");
        return;
      }
      this.setStatus("lost");
      this.retry += 1;
      setTimeout(() => !this.closed && this.connect(), Math.min(2000, 250 * this.retry));
    };
  }
}

function defaultUrl(matchId: string): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws?match=${matchId}`;
}
