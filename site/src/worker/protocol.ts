// Messages between the page and an agent worker.
import type { AgentKey } from "../agents/agents";
import type { Difficulty } from "../agents/base";
import type { GameDict } from "../engine/game";

export type ToWorker =
  | { t: "init"; key: AgentKey; seed: bigint; difficulty: Difficulty; base: string }
  | { t: "difficulty"; difficulty: Difficulty }
  | { t: "decide"; id: number; state: GameDict; topK: number };

export type FromWorker =
  | { t: "ready"; loadMs: number }
  | { t: "error"; message: string }
  | { t: "decision"; id: number; placement: number | null; candidates: [number, number][]; thinkTime: number };
