// Score, attack and gravity tables (tetris/engine/scoring.py).
// Gravity uses the exact float64 values exported from Python, because pow()
// can differ in the last bit between Python and JavaScript.
import gravity from "../data/gravity.json";

const LINE_ATTACK = [0, 0, 1, 2, 4];
const LINE_SCORE = [0, 100, 300, 500, 800];
const COMBO_SCORE = 50;
const B2B_SCORE_MULTIPLIER = 1.5;
const B2B_ATTACK_BONUS = 1;

export function fromBits(hex: string): number {
  const view = new DataView(new ArrayBuffer(8));
  view.setBigUint64(0, BigInt(`0x${hex}`));
  return view.getFloat64(0);
}

export function toBits(x: number): string {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, x);
  return view.getBigUint64(0).toString(16).padStart(16, "0");
}

const GRAVITY = gravity.bits.map(fromBits);

export function comboAttack(combo: number): number {
  if (combo <= 1) return 0;
  if (combo <= 3) return 1;
  if (combo <= 5) return 2;
  if (combo <= 7) return 3;
  if (combo <= 10) return 4;
  return 5;
}

export function attackFor(lines: number, combo: number, b2b: boolean): number {
  if (lines <= 0) return 0;
  return LINE_ATTACK[lines] + comboAttack(combo) + (b2b ? B2B_ATTACK_BONUS : 0);
}

export function scoreFor(lines: number, level: number, combo: number, b2b: boolean): number {
  if (lines <= 0) return 0;
  let base = LINE_SCORE[lines] * level;
  if (b2b) base = Math.trunc(base * B2B_SCORE_MULTIPLIER);
  return base + (combo > 0 ? COMBO_SCORE * combo * level : 0);
}

/** Guideline gravity in rows per second. */
export function gravityRowsPerSecond(level: number): number {
  const lv = Math.max(1, Math.min(level, 20));
  return GRAVITY[lv - 1];
}
