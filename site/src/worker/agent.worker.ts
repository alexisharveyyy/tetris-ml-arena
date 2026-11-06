// One agent in a Web Worker: loads its model files (only for this agent, when
// it is first chosen; the browser caches them) and answers decisions.
import { type AgentKey, createAgent, modelFiles } from "../agents/agents";
import type { Agent } from "../agents/base";
import { Game } from "../engine/game";
import { placementIndex } from "../engine/placement";
import type { FromWorker, ToWorker } from "./protocol";

let agent: Agent | null = null;
const queue: ToWorker[] = [];

function post(m: FromWorker): void {
  (self as unknown as Worker).postMessage(m);
}

async function init(key: AgentKey, seed: bigint, base: string): Promise<Agent> {
  const files = await Promise.all(
    modelFiles(key).map(async (f) => {
      const res = await fetch(`${base}models/${f}`);
      if (!res.ok) throw new Error(`${f}: HTTP ${res.status}`);
      return res.arrayBuffer();
    }),
  );
  return createAgent(key, files, seed);
}

function handle(m: ToWorker): void {
  if (!agent) {
    queue.push(m);
    return;
  }
  if (m.t === "difficulty") agent.setDifficulty(m.difficulty);
  else if (m.t === "decide") {
    const t0 = performance.now();
    const game = Game.fromDict(m.state);
    let placement: number | null = null;
    let candidates: [number, number][] = [];
    try {
      const d = agent.decide(game, m.topK);
      placement = placementIndex(d.placement);
      candidates = d.candidates.map((c) => [c.index, c.score]);
    } catch {
      placement = null; // no legal placement: the driver hard drops
    }
    post({ t: "decision", id: m.id, placement, candidates, thinkTime: (performance.now() - t0) / 1000 });
  }
}

self.onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  if (m.t !== "init") return handle(m);
  const t0 = performance.now();
  init(m.key, m.seed, m.base)
    .then((a) => {
      agent = a;
      agent.setDifficulty(m.difficulty);
      post({ t: "ready", loadMs: performance.now() - t0 });
      queue.splice(0).forEach(handle);
    })
    .catch((err: Error) => post({ t: "error", message: err.message }));
};
