// Opt-in performance probe: add ?perf to the URL (or set localStorage
// "arena.perf" to "1"). It records
//   - frame time: the interval between animation frames on a match screen;
//   - input to render: key press to the paint of the board that acknowledges
//     it (server tick, network, React render and canvas draw; display scan-out
//     is not included). Boards drawn inside an animation frame are painted at
//     its end, so the time is taken when drawing finishes; boards drawn outside
//     one are timed at the next frame.
//   - long tasks (main thread busy for 50 ms or more).
// window.__arenaPerf exposes the numbers to the Playwright performance run, and
// a small readout shows them on screen for checking the demo machine.

export interface PerfStats {
  /** performance.now() at the last reset. */
  since: number;
  /** Per-player pieces and pieces per second from the latest snapshot. */
  players: { name: string; preset: string | null; pieces: number; pps: number }[];
  frames: number[];
  inputToRender: number[];
  longTasks: number;
  bytes: number;
  messages: number;
}

function enabledFromUrl(): boolean {
  try {
    return /[?&#]perf\b/.test(location.search + location.hash) || localStorage.getItem("arena.perf") === "1";
  } catch {
    return false;
  }
}

export const PERF_ENABLED = typeof window !== "undefined" && enabledFromUrl();

const stats: PerfStats = { since: 0, players: [], frames: [], inputToRender: [], longTasks: 0, bytes: 0, messages: 0 };
/** Inputs acknowledged by snapshot frame f, waiting for that snapshot to be drawn. */
const acked: { f: number; ts: number[] }[] = [];
let running = false;
let inFrame = false;

/** Run ``fn`` as the snapshot render of an animation frame. */
export function inAnimationFrame(fn: () => void): void {
  inFrame = true;
  try {
    fn();
  } finally {
    inFrame = false;
  }
}

export function resetPerf(): void {
  stats.since = performance.now();
  stats.frames.length = 0;
  stats.inputToRender.length = 0;
  stats.longTasks = 0;
  stats.bytes = 0;
  stats.messages = 0;
}

/** Start counting frame intervals (match screens call this while mounted). */
export function startFrameLoop(): () => void {
  if (!PERF_ENABLED || running) return () => undefined;
  running = true;
  let last = 0;
  let id = 0;
  const tick = (t: number) => {
    if (last) stats.frames.push(t - last);
    if (stats.frames.length > 20_000) stats.frames.splice(0, 10_000);
    last = t;
    id = requestAnimationFrame(tick);
  };
  id = requestAnimationFrame(tick);
  return () => {
    running = false;
    cancelAnimationFrame(id);
  };
}

export function perfMessage(bytes: number): void {
  if (!PERF_ENABLED) return;
  stats.bytes += bytes;
  stats.messages += 1;
}

/** Inputs sent at ``ts`` (performance.now) were acknowledged by snapshot ``f``. */
export function perfAcked(f: number, ts: number[]): void {
  if (PERF_ENABLED && ts.length) acked.push({ f, ts });
}

/** The boards for this snapshot have been drawn. */
export function perfDrawn(snap: { f: number; players: { name: string; preset: string | null; stats: { pieces: number; pps: number } }[] }): void {
  if (!PERF_ENABLED) return;
  const f = snap.f;
  stats.players = snap.players.map((p) => ({ name: p.name, preset: p.preset, pieces: p.stats.pieces, pps: p.stats.pps }));
  if (!acked.length) return;
  const done: number[] = [];
  while (acked.length && acked[0].f <= f) done.push(...acked.shift()!.ts);
  if (!done.length) return;
  if (inFrame) {
    const now = performance.now();
    for (const ts of done) stats.inputToRender.push(now - ts);
  } else {
    requestAnimationFrame((t) => {
      for (const ts of done) stats.inputToRender.push(t - ts);
    });
  }
}

export function pct(values: number[], q: number): number | null {
  if (!values.length) return null;
  const s = [...values].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
}

if (PERF_ENABLED) {
  (window as unknown as { __arenaPerf: unknown }).__arenaPerf = { stats, reset: resetPerf };
  try {
    new PerformanceObserver((list) => {
      stats.longTasks += list.getEntries().length;
    }).observe({ type: "longtask", buffered: false });
  } catch {
    /* not supported */
  }
}

export function perfStats(): PerfStats {
  return stats;
}
