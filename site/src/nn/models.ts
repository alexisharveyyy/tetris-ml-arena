// The agents' networks, built from ONNX initializers.
//   AfterstateNet  nets.AfterstateValueNet (DQN; the PPO and BC scorers)
//   RawPolicy      nets.ActorCritic policy head (PPO raw actions)
//   DTEmbedder     DecisionTransformer.scorer (AfterstateEmbedder)
//   DTCore         DecisionTransformer.forward, the query at the last state token
import { BoardEncoder, LayerNorm, Linear } from "./layers";
import { readInitializers, Weights } from "./onnx";

const CELLS = 240;

export function loadWeights(onnx: ArrayBuffer | Uint8Array, prefix = ""): Weights {
  return new Weights(readInitializers(onnx), prefix);
}

/** CNN over the board plus an MLP over extras, concatenated, then a head. */
class BoardPlusExtras {
  readonly board: BoardEncoder;
  readonly extra: Linear;
  readonly hidden: Float32Array;

  constructor(w: Weights, extraName: string) {
    this.board = new BoardEncoder(w.scope("board."));
    this.extra = new Linear(w, extraName);
    this.hidden = new Float32Array(this.board.outDim + this.extra.out);
  }

  /** Fills this.hidden for row r. */
  embed(boards: Float32Array, extras: Float32Array, r: number): Float32Array {
    this.board.encode(boards, r * CELLS, this.hidden, 0);
    this.extra.apply(extras, r * this.extra.in, this.hidden, this.board.outDim, true);
    return this.hidden;
  }
}

export class AfterstateNet {
  private readonly base: BoardPlusExtras;
  private readonly h0: Linear;
  private readonly h2: Linear;
  private readonly mid: Float32Array;
  private readonly out = new Float32Array(1);
  readonly extraDim: number;

  constructor(w: Weights) {
    this.base = new BoardPlusExtras(w, "extra.0");
    this.h0 = new Linear(w, "head.0");
    this.h2 = new Linear(w, "head.2");
    this.mid = new Float32Array(this.h0.out);
    this.extraDim = this.base.extra.in;
  }

  /** Scores (n,) for boards (n, 240) and extras (n, extraDim). */
  forward(boards: Float32Array, extras: Float32Array, n: number): Float32Array {
    const out = new Float32Array(n);
    for (let r = 0; r < n; r++) {
      const h = this.base.embed(boards, extras, r);
      this.h0.apply(h, 0, this.mid, 0, true);
      this.h2.apply(this.mid, 0, this.out, 0);
      out[r] = this.out[0];
    }
    return out;
  }
}

export class RawPolicy {
  private readonly base: BoardPlusExtras;
  private readonly trunk: Linear;
  private readonly policy: Linear;

  constructor(w: Weights) {
    this.base = new BoardPlusExtras(w, "extra.0");
    this.trunk = new Linear(w, "trunk.0");
    this.policy = new Linear(w, "policy");
  }

  /** Logits (80,) for one observation board (240) and extras (62). */
  forward(board: Float32Array, extras: Float32Array): Float32Array {
    const h = this.base.embed(board, extras, 0);
    const t = new Float32Array(this.trunk.out);
    this.trunk.apply(h, 0, t, 0, true);
    const logits = new Float32Array(this.policy.out);
    this.policy.apply(t, 0, logits, 0);
    return logits;
  }
}

export class DTEmbedder {
  private readonly base: BoardPlusExtras;
  private readonly o0: Linear;
  private readonly o2: Linear;
  private readonly mid: Float32Array;
  readonly dim: number;

  constructor(w: Weights) {
    this.base = new BoardPlusExtras(w, "extra.0");
    this.o0 = new Linear(w, "out.0");
    this.o2 = new Linear(w, "out.2");
    this.mid = new Float32Array(this.o0.out);
    this.dim = this.o2.out;
  }

  /** Embeddings (n, dim). */
  forward(boards: Float32Array, extras: Float32Array, n: number): Float32Array {
    const out = new Float32Array(n * this.dim);
    for (let r = 0; r < n; r++) {
      const h = this.base.embed(boards, extras, r);
      this.o0.apply(h, 0, this.mid, 0, true);
      this.o2.apply(this.mid, 0, out, r * this.dim);
    }
    return out;
  }
}

interface Block {
  inW: Float32Array; // (3d, d)
  inB: Float32Array;
  out: Linear;
  l1: Linear;
  l2: Linear;
  n1: LayerNorm;
  n2: LayerNorm;
}

export interface DTStepInput {
  rtg: number;
  board: Float32Array; // 240 occupancy values
  extras: Float32Array; // 62
  action: Float32Array; // d (zeros when not yet chosen)
  timestep: number;
}

export class DTCore {
  readonly d: number;
  readonly heads: number;
  readonly maxTimestep: number;
  private readonly stateBoard: BoardEncoder;
  private readonly stateExtra: Linear;
  private readonly stateProj: Linear;
  private readonly rtgW: Float32Array;
  private readonly rtgB: Float32Array;
  private readonly actionProj: Linear;
  private readonly timestep: Float32Array;
  private readonly tokenType: Float32Array;
  private readonly embedNorm: LayerNorm;
  private readonly blocks: Block[] = [];
  private readonly finalNorm: LayerNorm;
  private readonly query: Linear;

  constructor(w: Weights, heads: number) {
    this.heads = heads;
    this.stateBoard = new BoardEncoder(w.scope("state_board."));
    this.stateExtra = new Linear(w, "state_extra.0");
    this.stateProj = new Linear(w, "state_proj");
    this.d = this.stateProj.out;
    this.rtgW = w.get("rtg_proj.weight", [this.d, 1]).data;
    this.rtgB = w.get("rtg_proj.bias").data;
    this.actionProj = new Linear(w, "action_proj");
    const ts = w.get("timestep.weight");
    this.maxTimestep = ts.dims[0];
    this.timestep = ts.data;
    this.tokenType = w.get("token_type.weight", [3, this.d]).data;
    this.embedNorm = new LayerNorm(w, "embed_norm");
    for (let i = 0; ; i++) {
      const p = `blocks.layers.${i}.`;
      let inW;
      try {
        inW = w.get(`${p}self_attn.in_proj_weight`, [3 * this.d, this.d]).data;
      } catch {
        break;
      }
      this.blocks.push({
        inW,
        inB: w.get(`${p}self_attn.in_proj_bias`).data,
        out: new Linear(w, `${p}self_attn.out_proj`),
        l1: new Linear(w, `${p}linear1`),
        l2: new Linear(w, `${p}linear2`),
        n1: new LayerNorm(w, `${p}norm1`),
        n2: new LayerNorm(w, `${p}norm2`),
      });
    }
    if (!this.blocks.length) throw new Error("no transformer blocks in the model");
    this.finalNorm = new LayerNorm(w, "final_norm");
    this.query = new Linear(w, "query");
  }

  /**
   * The action query (d,) at the last state token. ``steps`` are the valid
   * steps only, oldest first: left padding is never attended to by a valid
   * token (DecisionTransformer.attention_mask), so leaving it out gives the
   * same result.
   */
  forward(steps: DTStepInput[]): Float32Array {
    const d = this.d;
    const k = steps.length;
    const n = 3 * k;
    const x = new Float32Array(n * d);
    const hidden = new Float32Array(this.stateBoard.outDim + this.stateExtra.out);
    const state = new Float32Array(d);
    const act = new Float32Array(d);
    for (let j = 0; j < k; j++) {
      const s = steps[j];
      const t = Math.max(0, Math.min(this.maxTimestep - 1, s.timestep)) * d;
      this.stateBoard.encode(s.board, 0, hidden, 0);
      this.stateExtra.apply(s.extras, 0, hidden, this.stateBoard.outDim, true);
      this.stateProj.apply(hidden, 0, state, 0);
      this.actionProj.apply(s.action, 0, act, 0);
      const rtg = Math.fround(s.rtg);
      const r0 = 3 * j * d;
      for (let i = 0; i < d; i++) {
        const time = this.timestep[t + i];
        x[r0 + i] = Math.fround(this.rtgW[i] * rtg + this.rtgB[i]) + time + this.tokenType[i];
        x[r0 + d + i] = state[i] + time + this.tokenType[d + i];
        x[r0 + 2 * d + i] = act[i] + time + this.tokenType[2 * d + i];
      }
    }
    let h = this.embedNorm.forward(x, n);
    const a = new Float32Array(n * d);
    const qkv = new Float32Array(n * 3 * d);
    const att = new Float32Array(n * d);
    const proj = new Float32Array(d);
    const hd = d / this.heads;
    const scale = 1 / Math.sqrt(hd);
    const probs = new Float64Array(n);
    for (const b of this.blocks) {
      b.n1.forward(h, n, a);
      for (let r = 0; r < n; r++) {
        for (let o = 0; o < 3 * d; o++) {
          let s = b.inB[o];
          const row = o * d;
          for (let i = 0; i < d; i++) s += b.inW[row + i] * a[r * d + i];
          qkv[r * 3 * d + o] = s;
        }
      }
      for (let head = 0; head < this.heads; head++) {
        const off = head * hd;
        for (let i = 0; i < n; i++) {
          // Causal: token i sees tokens 0..i.
          let max = -Infinity;
          for (let j = 0; j <= i; j++) {
            let s = 0;
            for (let c = 0; c < hd; c++) s += qkv[i * 3 * d + off + c] * qkv[j * 3 * d + d + off + c];
            s *= scale;
            probs[j] = s;
            if (s > max) max = s;
          }
          let sum = 0;
          for (let j = 0; j <= i; j++) {
            probs[j] = Math.exp(probs[j] - max);
            sum += probs[j];
          }
          for (let c = 0; c < hd; c++) {
            let v = 0;
            for (let j = 0; j <= i; j++) v += probs[j] * qkv[j * 3 * d + 2 * d + off + c];
            att[i * d + off + c] = v / sum;
          }
        }
      }
      const next = new Float32Array(n * d);
      for (let r = 0; r < n; r++) {
        b.out.apply(att, r * d, proj, 0);
        for (let i = 0; i < d; i++) next[r * d + i] = h[r * d + i] + proj[i];
      }
      b.n2.forward(next, n, a);
      const ff = new Float32Array(b.l1.out);
      for (let r = 0; r < n; r++) {
        b.l1.apply(a, r * d, ff, 0, true);
        b.l2.apply(ff, 0, proj, 0);
        for (let i = 0; i < d; i++) next[r * d + i] += proj[i];
      }
      h = next;
    }
    const last = (3 * (k - 1) + 1) * d;
    const row = this.finalNorm.forward(h.subarray(last, last + d), 1);
    const q = new Float32Array(d);
    this.query.apply(row, 0, q, 0);
    return q;
  }
}
