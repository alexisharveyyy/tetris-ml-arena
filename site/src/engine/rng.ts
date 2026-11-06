// SplitMix64, bit for bit the same as tetris/engine/rng.py (state is a BigInt).

const MASK = (1n << 64n) - 1n;

export class Rng {
  state: bigint;

  constructor(seed: bigint | number) {
    this.state = BigInt(seed) & MASK;
  }

  nextU64(): bigint {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK;
    return z ^ (z >> 31n);
  }

  /** Integer in [0, n). */
  randbelow(n: number): number {
    if (n <= 0) throw new Error("n must be positive");
    return Number(this.nextU64() % BigInt(n));
  }

  /** Float in [0, 1) with 53 bits of precision. */
  random(): number {
    return Number(this.nextU64() >> 11n) / 2 ** 53;
  }

  /** Fisher-Yates shuffle in place. */
  shuffle<T>(seq: T[]): void {
    for (let i = seq.length - 1; i > 0; i--) {
      const j = this.randbelow(i + 1);
      const t = seq[i];
      seq[i] = seq[j];
      seq[j] = t;
    }
  }
}

/** Independent seed for a named sub-stream. */
export function deriveSeed(seed: bigint | number, stream: number): bigint {
  const s = BigInt(seed) & MASK;
  return new Rng(s ^ ((BigInt(stream) * 0xd1b54a32d192ed03n) & MASK)).nextU64();
}
