// Typed event feed for tickers and results (tetris/match/feed.py).

export type FeedEventOut = { type: string; player: string | null; time: number; [k: string]: unknown };

export const FALLBACK_TEXT: Record<string, string> = {
  timeout: "The model missed its time limit, so the baseline is playing for it until this match ends.",
  crash: "The model stopped responding, so the baseline is playing for it until this match ends.",
  load_failed: "The model could not be loaded, so the baseline is playing for it.",
  load_timeout: "The model took too long to load, so the baseline is playing for it.",
};

export interface RankRow {
  id: string;
  score: number;
}

export class FeedTracker {
  events: FeedEventOut[] = [];
  private places = new Map<string, number>();
  private leader: string | null = null;
  private fallbacks = new Set<string>();
  private tetrises = new Map<string, number>();

  reset(order: string[]): void {
    this.events = [];
    this.places = new Map(order.map((pid, i) => [pid, i + 1]));
    this.leader = order[0] ?? null;
    this.fallbacks.clear();
    this.tetrises.clear();
  }

  pop(): FeedEventOut[] {
    const e = this.events;
    this.events = [];
    return e;
  }

  private emit(type: string, player: string | null, time: number, data: Record<string, unknown> = {}): void {
    this.events.push({ type, player, time, ...data });
  }

  onEngineEvent(player: string, ev: { type: string; data: Record<string, unknown> }, time: number, lines: number, human: boolean): void {
    const d = ev.data;
    if (ev.type === "lock" && (d.lines as number) > 0) {
      this.emit("line_clear", player, time, { count: d.lines, lines });
      if (d.lines === 4) {
        const nth = (this.tetrises.get(player) ?? 0) + 1;
        this.tetrises.set(player, nth);
        this.emit("tetris", player, time, { nth });
        if (d.b2b) this.emit("b2b", player, time);
      }
    } else if (ev.type === "game_over") this.emit("top_out", player, time, { reason: d.reason ?? null });
    else if (ev.type === "hold" && human) this.emit("hold", player, time, { piece: d.piece });
  }

  onStandings(rows: RankRow[], time: number, humanId: string | null): void {
    const scores = new Map(rows.map((r) => [r.id, r.score]));
    const order = rows.map((r) => r.id);
    const humanScore = humanId !== null ? scores.get(humanId) : undefined;
    order.forEach((pid, i) => {
      const place = i + 1;
      const before = this.places.get(pid) ?? place;
      if (place < before) {
        this.emit("rank_change", pid, time, {
          from: before,
          to: place,
          passed: order[i + 1] ?? null,
          gap_to_human: humanScore === undefined ? null : scores.get(pid)! - humanScore,
        });
      }
    });
    if (order.length && order[0] !== this.leader) {
      const gap = order.length > 1 ? scores.get(order[0])! - scores.get(order[1])! : 0;
      this.emit("lead_change", order[0], time, { previous: this.leader, gap });
      this.leader = order[0];
    }
    this.places = new Map(order.map((pid, i) => [pid, i + 1]));
  }

  onFallback(player: string, reason: string, time: number): void {
    if (this.fallbacks.has(player)) return;
    this.fallbacks.add(player);
    this.emit("fallback", player, time, { reason, text: FALLBACK_TEXT[reason] ?? FALLBACK_TEXT.crash });
  }

  onMatchEnd(winner: string | null, standings: unknown[], time: number): void {
    this.emit("match_end", null, time, { winner, standings });
  }
}
