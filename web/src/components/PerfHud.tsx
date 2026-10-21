// On-screen readout for ?perf (see lib/perf.ts). Diagnostic only.
import { useEffect, useState } from "react";
import { PERF_ENABLED, pct, perfStats } from "../lib/perf";

const ms = (v: number | null) => (v == null ? "-" : v.toFixed(1));

export function PerfHud() {
  const [, tick] = useState(0);
  useEffect(() => {
    if (!PERF_ENABLED) return;
    const id = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);
  if (!PERF_ENABLED) return null;
  const s = perfStats();
  const recent = s.frames.slice(-600);
  const lat = s.inputToRender.slice(-200);
  return (
    <div className="perf-hud" aria-hidden="true">
      frame p50 {ms(pct(recent, 0.5))} p99 {ms(pct(recent, 0.99))} ms | input to render p50 {ms(pct(lat, 0.5))} p99{" "}
      {ms(pct(lat, 0.99))} ms | long tasks {s.longTasks} | {s.messages ? Math.round(s.bytes / s.messages) : 0} B/msg
    </div>
  );
}
