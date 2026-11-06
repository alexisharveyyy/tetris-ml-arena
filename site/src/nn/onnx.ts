// Minimal ONNX reader: the float32 initializers of a model, by name.
// ONNX files are protobuf (ModelProto). Only the fields needed to find the
// initializers are decoded: ModelProto.graph (7), GraphProto.initializer (5)
// and TensorProto dims (1), data_type (2), float_data (4), name (8), raw_data (9).

export interface Tensor {
  dims: number[];
  data: Float32Array;
}

const FLOAT = 1;

class Reader {
  pos: number;
  constructor(
    readonly buf: Uint8Array,
    start = 0,
    readonly end = buf.length,
  ) {
    this.pos = start;
  }

  get done(): boolean {
    return this.pos >= this.end;
  }

  varint(): number {
    let result = 0;
    let mul = 1;
    for (;;) {
      const b = this.buf[this.pos++];
      result += (b & 0x7f) * mul;
      if (b < 0x80) return result;
      mul *= 128;
    }
  }

  /** Returns [field, wire type]. */
  tag(): [number, number] {
    const t = this.varint();
    return [Math.floor(t / 8), t & 7];
  }

  bytes(): [number, number] {
    const len = this.varint();
    const start = this.pos;
    this.pos += len;
    return [start, this.pos];
  }

  skip(wire: number): void {
    if (wire === 0) this.varint();
    else if (wire === 1) this.pos += 8;
    else if (wire === 2) this.bytes();
    else if (wire === 5) this.pos += 4;
    else throw new Error(`unsupported protobuf wire type ${wire}`);
  }
}

function readTensor(buf: Uint8Array, start: number, end: number): [string, Tensor | null] {
  const r = new Reader(buf, start, end);
  const dims: number[] = [];
  let type = 0;
  let name = "";
  let raw: [number, number] | null = null;
  const floats: number[] = [];
  while (!r.done) {
    const [field, wire] = r.tag();
    if (field === 1 && wire === 0) dims.push(r.varint());
    else if (field === 1 && wire === 2) {
      const [s, e] = r.bytes();
      const sub = new Reader(buf, s, e);
      while (!sub.done) dims.push(sub.varint());
    } else if (field === 2 && wire === 0) type = r.varint();
    else if (field === 4 && wire === 2) {
      const [s, e] = r.bytes();
      const view = new DataView(buf.buffer, buf.byteOffset + s, e - s);
      for (let o = 0; o < e - s; o += 4) floats.push(view.getFloat32(o, true));
    } else if (field === 8 && wire === 2) {
      const [s, e] = r.bytes();
      name = new TextDecoder().decode(buf.subarray(s, e));
    } else if (field === 9 && wire === 2) raw = r.bytes();
    else r.skip(wire);
  }
  if (type !== FLOAT) return [name, null];
  let data: Float32Array;
  if (raw) {
    const [s, e] = raw;
    // Copy: raw_data is not 4-byte aligned in the file. ONNX stores it little-endian.
    const bytes = buf.slice(s, e);
    data = new Float32Array(bytes.buffer, 0, (e - s) / 4);
    if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) throw new Error("big-endian host");
  } else data = Float32Array.from(floats);
  const size = dims.reduce((a, b) => a * b, 1);
  if (data.length !== size) throw new Error(`initializer ${name}: ${data.length} values for dims ${dims}`);
  return [name, { dims, data }];
}

/** Every float32 initializer of an ONNX model, keyed by name. */
export function readInitializers(model: ArrayBuffer | Uint8Array): Map<string, Tensor> {
  const buf = model instanceof Uint8Array ? model : new Uint8Array(model);
  const out = new Map<string, Tensor>();
  const r = new Reader(buf);
  while (!r.done) {
    const [field, wire] = r.tag();
    if (field !== 7 || wire !== 2) {
      r.skip(wire);
      continue;
    }
    const [gs, ge] = r.bytes();
    const g = new Reader(buf, gs, ge);
    while (!g.done) {
      const [f, w] = g.tag();
      if (f === 5 && w === 2) {
        const [s, e] = g.bytes();
        const [name, t] = readTensor(buf, s, e);
        if (t) out.set(name, t);
      } else g.skip(w);
    }
  }
  if (!out.size) throw new Error("no float32 initializers found in the ONNX model");
  return out;
}

export class Weights {
  constructor(
    private readonly map: Map<string, Tensor>,
    private readonly prefix = "",
  ) {}

  get(name: string, dims?: number[]): Tensor {
    const t = this.map.get(this.prefix + name);
    if (!t) throw new Error(`missing weight ${this.prefix + name}`);
    if (dims && (dims.length !== t.dims.length || dims.some((d, i) => d >= 0 && d !== t.dims[i]))) {
      throw new Error(`weight ${this.prefix + name} has dims ${t.dims}, expected ${dims}`);
    }
    return t;
  }

  scope(prefix: string): Weights {
    return new Weights(this.map, this.prefix + prefix);
  }
}
