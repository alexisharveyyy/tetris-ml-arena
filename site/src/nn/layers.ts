// The layers the agents use, in float32 storage with float64 accumulation.
// Architectures mirror tetris/rl/nets.py and the Decision Transformer model;
// weights come from the ONNX exports (see onnx.ts).
import type { Weights } from "./onnx";

const f32 = Math.fround;

export class Linear {
  readonly out: number;
  readonly in: number;
  private readonly w: Float32Array;
  private readonly b: Float32Array | null;

  constructor(weights: Weights, name: string, bias = true) {
    const w = weights.get(`${name}.weight`);
    [this.out, this.in] = w.dims;
    this.w = w.data;
    this.b = bias ? weights.get(`${name}.bias`, [this.out]).data : null;
  }

  /** y = W x + b for one vector x (length in) at offset xo; writes y at yo. */
  apply(x: Float32Array, xo: number, y: Float32Array, yo: number, relu = false): void {
    const { w, b } = this;
    const n = this.in;
    for (let o = 0; o < this.out; o++) {
      let s = b ? b[o] : 0;
      const row = o * n;
      for (let i = 0; i < n; i++) s += w[row + i] * x[xo + i];
      y[yo + o] = relu && s < 0 ? 0 : s;
    }
  }

  /** Batched: x is (rows, in), returns (rows, out). */
  forward(x: Float32Array, rows: number, relu = false): Float32Array {
    const y = new Float32Array(rows * this.out);
    for (let r = 0; r < rows; r++) this.apply(x, r * this.in, y, r * this.out, relu);
    return y;
  }
}

/** 3x3 convolution with padding 1 (nets.Conv3x3), on one (C, H, W) map. */
export class Conv3x3 {
  readonly cout: number;
  readonly cin: number;
  private readonly w: Float32Array;
  private readonly b: Float32Array;

  constructor(
    weights: Weights,
    name: string,
    readonly stride: number,
  ) {
    const w = weights.get(`${name}.weight`);
    [this.cout, this.cin] = w.dims;
    if (w.dims[2] !== 3 || w.dims[3] !== 3) throw new Error(`${name} is not 3x3`);
    this.w = w.data;
    this.b = weights.get(`${name}.bias`, [this.cout]).data;
  }

  outSize(h: number, w: number): [number, number] {
    const s = this.stride;
    return [Math.floor((h - 1) / s) + 1, Math.floor((w - 1) / s) + 1];
  }

  /** ReLU applied (every conv in BoardEncoder is followed by one). */
  forwardRelu(x: Float32Array, h: number, w: number, y: Float32Array): void {
    const [ho, wo] = this.outSize(h, w);
    const s = this.stride;
    const { cin, cout } = this;
    const wt = this.w;
    for (let co = 0; co < cout; co++) {
      const bias = this.b[co];
      for (let oy = 0; oy < ho; oy++) {
        for (let ox = 0; ox < wo; ox++) {
          let sum = bias;
          for (let ci = 0; ci < cin; ci++) {
            const wbase = (co * cin + ci) * 9;
            const xbase = ci * h * w;
            for (let ky = 0; ky < 3; ky++) {
              const iy = oy * s + ky - 1;
              if (iy < 0 || iy >= h) continue;
              const xrow = xbase + iy * w;
              for (let kx = 0; kx < 3; kx++) {
                const ix = ox * s + kx - 1;
                if (ix < 0 || ix >= w) continue;
                sum += wt[wbase + ky * 3 + kx] * x[xrow + ix];
              }
            }
          }
          y[(co * ho + oy) * wo + ox] = sum < 0 ? 0 : sum;
        }
      }
    }
  }
}

export const BOARD_H = 24;
export const BOARD_W = 10;

/** nets.BoardEncoder: three convs (strides 1, 2, 2), flatten, linear, ReLU. */
export class BoardEncoder {
  private readonly convs: Conv3x3[];
  private readonly fc: Linear;
  private readonly shapes: [number, number, number][] = [];
  private readonly bufs: Float32Array[];
  readonly outDim: number;

  constructor(weights: Weights) {
    this.convs = [new Conv3x3(weights, "conv.0", 1), new Conv3x3(weights, "conv.2", 2), new Conv3x3(weights, "conv.4", 2)];
    let h = BOARD_H;
    let w = BOARD_W;
    for (const c of this.convs) {
      [h, w] = c.outSize(h, w);
      this.shapes.push([c.cout, h, w]);
    }
    this.bufs = this.shapes.map(([c, hh, ww]) => new Float32Array(c * hh * ww));
    this.fc = new Linear(weights, "fc.0");
    const flat = this.shapes[2][0] * this.shapes[2][1] * this.shapes[2][2];
    if (this.fc.in !== flat) throw new Error(`fc expects ${this.fc.in} inputs, convs give ${flat}`);
    this.outDim = this.fc.out;
  }

  /** One board (240 floats, 0 or 1) at offset bo; writes outDim values to y at yo. */
  encode(boards: Float32Array, bo: number, y: Float32Array, yo: number): void {
    const x0 = boards.subarray(bo, bo + BOARD_H * BOARD_W);
    this.convs[0].forwardRelu(x0, BOARD_H, BOARD_W, this.bufs[0]);
    this.convs[1].forwardRelu(this.bufs[0], this.shapes[0][1], this.shapes[0][2], this.bufs[1]);
    this.convs[2].forwardRelu(this.bufs[1], this.shapes[1][1], this.shapes[1][2], this.bufs[2]);
    this.fc.apply(this.bufs[2], 0, y, yo, true);
  }
}

export class LayerNorm {
  private readonly g: Float32Array;
  private readonly b: Float32Array;
  readonly dim: number;

  constructor(
    weights: Weights,
    name: string,
    private readonly eps = 1e-5,
  ) {
    this.g = weights.get(`${name}.weight`).data;
    this.b = weights.get(`${name}.bias`).data;
    this.dim = this.g.length;
  }

  /** In place over rows of x (rows, dim), or into y. */
  forward(x: Float32Array, rows: number, y: Float32Array = new Float32Array(x.length)): Float32Array {
    const d = this.dim;
    for (let r = 0; r < rows; r++) {
      const o = r * d;
      let mean = 0;
      for (let i = 0; i < d; i++) mean += x[o + i];
      mean /= d;
      let v = 0;
      for (let i = 0; i < d; i++) {
        const t = x[o + i] - mean;
        v += t * t;
      }
      const inv = 1 / Math.sqrt(v / d + this.eps);
      for (let i = 0; i < d; i++) y[o + i] = (x[o + i] - mean) * inv * this.g[i] + this.b[i];
    }
    return y;
  }
}

export { f32 };
