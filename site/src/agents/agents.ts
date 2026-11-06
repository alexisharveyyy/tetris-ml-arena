// Every agent of the web version. The heuristic agents are plain TypeScript;
// the neural agents run their networks with the inference code in ../nn.
import weightsData from "../data/weights.json";
import agentMeta from "../data/agent_meta.json";
import type { Game } from "../engine/game";
import { NUM_ACTIONS } from "../engine/placement";
import { fromBits } from "../engine/scoring";
import { AfterstateNet, DTCore, DTEmbedder, loadWeights, RawPolicy } from "../nn/models";
import { actionMask, enumerateAfterstates } from "./afterstates";
import { Agent, rankBy, type Scored } from "./base";
import { DTAgent } from "./dt";
import { afterstateArrays, legalActionArrays, observationArrays, scaled } from "./encoding";

const f32 = Math.fround;
export const TOP_OUT_PENALTY = -1e9;

// --------------------------------------------------------------- linear

export class LinearAgent extends Agent {
  constructor(
    readonly weights: number[],
    seed?: number | bigint,
  ) {
    super(seed);
  }

  rank(game: Game): Scored[] {
    const afs = enumerateAfterstates(game, true, true);
    const scores = afs.map((a) => {
      let s = 0;
      for (let i = 0; i < a.features.length; i++) s += a.features[i] * this.weights[i];
      return a.topOut ? TOP_OUT_PENALTY + s : s;
    });
    return rankBy(afs, scores).map(({ item, score }) => ({ index: item.index, score }));
  }
}

export function linearWeights(key: "dellacherie" | "ga" | "cem"): number[] {
  return (weightsData.linear_bits as Record<string, string[]>)[key].map(fromBits);
}

// ----------------------------------------------------------------- NEAT

interface NeatNode {
  key: number;
  activation: string;
  bias: number;
  response: number;
  links: [number, number][];
}

const ACTIVATIONS: Record<string, (z: number) => number> = {
  identity: (z) => z,
  tanh: (z) => Math.tanh(Math.max(-60, Math.min(60, 2.5 * z))),
  sigmoid: (z) => 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, 5 * z)))),
  relu: (z) => Math.max(z, 0),
  clamped: (z) => Math.max(-1, Math.min(1, z)),
};

export class NeatAgent extends Agent {
  private readonly inputs: number[];
  private readonly outputs: number[];
  private readonly nodes: NeatNode[];

  constructor(seed?: number | bigint) {
    super(seed);
    const net = weightsData.neat as { inputs: number[]; outputs: number[]; nodes: NeatNode[] };
    this.inputs = net.inputs;
    this.outputs = net.outputs;
    this.nodes = net.nodes;
    for (const n of this.nodes) if (!ACTIVATIONS[n.activation]) throw new Error(`unknown activation ${n.activation}`);
    if (this.outputs.length !== 1) throw new Error("NEAT network must have one output");
  }

  evaluate(x: number[]): number {
    const values = new Map<number, number>();
    this.inputs.forEach((k, i) => values.set(k, x[i]));
    for (const node of this.nodes) {
      let s = 0;
      for (const [i, w] of node.links) s += (values.get(i) ?? 0) * w;
      values.set(node.key, ACTIVATIONS[node.activation](node.bias + node.response * s));
    }
    return values.get(this.outputs[0]) ?? 0;
  }

  rank(game: Game): Scored[] {
    const afs = enumerateAfterstates(game, true, true);
    const scores = afs.map((a) => {
      const s = this.evaluate(scaled(a.features));
      return a.topOut ? TOP_OUT_PENALTY + s : s;
    });
    return rankBy(afs, scores).map(({ item, score }) => ({ index: item.index, score }));
  }
}

// --------------------------------------------------------------- neural

export class DQNAgent extends Agent {
  private readonly gamma: number;
  constructor(
    private readonly net: AfterstateNet,
    seed?: number | bigint,
  ) {
    super(seed);
    this.gamma = (agentMeta as Record<string, { gamma?: number }>).dqn.gamma ?? 0.99;
  }

  rank(game: Game): Scored[] {
    const afs = enumerateAfterstates(game, true, true);
    if (!afs.length) return [];
    const arr = afterstateArrays(game, afs);
    const v = this.net.forward(arr.boards, arr.extras, afs.length);
    const g = f32(this.gamma);
    // rewards + gamma * v * (~tops), in float32 like NumPy.
    const q = afs.map((_, i) => f32(arr.rewards[i] + f32(f32(g * v[i]) * (arr.tops[i] ? 0 : 1))));
    return rankBy(afs, q).map(({ item, score }) => ({ index: item.index, score }));
  }
}

function softmaxRank(indices: number[], logits: ArrayLike<number>): Scored[] {
  let max = -Infinity;
  for (let i = 0; i < indices.length; i++) max = Math.max(max, logits[i]);
  const e = indices.map((_, i) => Math.exp(logits[i] - max));
  const sum = e.reduce((a, b) => a + b, 0);
  const probs = e.map((x) => x / sum);
  return rankBy(indices, probs).map(({ item, score }) => ({ index: item, score }));
}

/** PPO and the BC agents: each legal action's logit scores its afterstate. */
export class PPOAgent extends Agent {
  constructor(
    private readonly net: AfterstateNet,
    seed?: number | bigint,
  ) {
    super(seed);
  }

  rank(game: Game): Scored[] {
    const { arrays, afs } = legalActionArrays(game);
    if (!afs.length) return [];
    const logits = this.net.forward(arrays.boards, arrays.extras, afs.length);
    return softmaxRank(arrays.actions, logits);
  }
}

/** PPO over the 80 actions from the observation alone. */
export class RawPPOAgent extends Agent {
  constructor(
    private readonly net: RawPolicy,
    seed?: number | bigint,
  ) {
    super(seed);
  }

  rank(game: Game): Scored[] {
    const mask = actionMask(game);
    const legal: number[] = [];
    for (let a = 0; a < NUM_ACTIONS; a++) if (mask[a]) legal.push(a);
    if (!legal.length) return [];
    const { board, extras } = observationArrays(game);
    const logits = this.net.forward(board, extras);
    return softmaxRank(
      legal,
      legal.map((a) => logits[a]),
    );
  }
}

// ---------------------------------------------------------------- factory

export const WEB_AGENT_KEYS = [
  "dellacherie",
  "ga",
  "cem",
  "neat",
  "dqn",
  "ppo",
  "ppo_raw",
  "bc",
  "bc_dagger",
  "bc_you",
  "dt",
] as const;
export type AgentKey = (typeof WEB_AGENT_KEYS)[number];

/** ONNX files each agent needs (none for the heuristic agents). */
export function modelFiles(key: AgentKey): string[] {
  if (key === "dt") return ["dt_embed.onnx", "dt_core.onnx"];
  if (key === "dellacherie" || key === "ga" || key === "cem" || key === "neat") return [];
  return [`${key}.onnx`];
}

/** Build an agent from its model files (already fetched). */
export function createAgent(key: AgentKey, files: ArrayBuffer[], seed: number | bigint = 0): Agent {
  switch (key) {
    case "dellacherie":
    case "ga":
    case "cem":
      return new LinearAgent(linearWeights(key), seed);
    case "neat":
      return new NeatAgent(seed);
    case "dqn":
      return new DQNAgent(new AfterstateNet(loadWeights(files[0])), seed);
    case "ppo":
    case "bc":
    case "bc_dagger":
    case "bc_you":
      return new PPOAgent(new AfterstateNet(loadWeights(files[0])), seed);
    case "ppo_raw":
      return new RawPPOAgent(new RawPolicy(loadWeights(files[0])), seed);
    case "dt": {
      const meta = (agentMeta as unknown as { dt: { default_target: number; return_cap: number; config: { n_heads: number; context: number } } }).dt;
      const embed = new DTEmbedder(loadWeights(files[0]));
      const core = new DTCore(loadWeights(files[1], "m."), meta.config.n_heads);
      return new DTAgent(embed, core, meta.config.context, meta.default_target, meta.return_cap, seed);
    }
  }
}
