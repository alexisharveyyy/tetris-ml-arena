// Engine parity: the TypeScript engine replays Python games move for move.
// The traces come from the Python engine (npm run fixtures:engine; CI runs it
// before the tests), so this compares against the real Python code each run.
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import gravity from "../../src/data/gravity.json";
import { defaultConfig, Game, type Move } from "../../src/engine/game";
import { placementFromIndex } from "../../src/engine/placement";
import { fromBits } from "../../src/engine/scoring";
import { canon, fnv1a } from "./canon";

interface Trace {
  kind: string;
  seed: number;
  start_level: number;
  initial: string;
  initial_hash: string;
  ops: [string, string | number][];
  hashes: string[];
  full: Record<string, string>;
}

const PATH = join(__dirname, "..", "fixtures", "engine_traces.json");
const data: { gravity_bits: string[]; games: Trace[] } | null = existsSync(PATH)
  ? JSON.parse(readFileSync(PATH, "utf8"))
  : null;

describe("engine parity with Python", () => {
  it("has the Python traces (run npm run fixtures:engine)", () => {
    expect(data, `missing ${PATH}`).not.toBeNull();
    expect(data!.games.length).toBeGreaterThanOrEqual(50);
  });

  it("uses the same gravity table", () => {
    expect(gravity.bits).toEqual(data!.gravity_bits);
  });

  for (const [i, t] of (data?.games ?? []).entries()) {
    it(`game ${i}: ${t.kind}, seed ${t.seed}, level ${t.start_level}, ${t.ops.length} steps`, () => {
      const g = Game.create(t.seed, defaultConfig({ start_level: t.start_level }));
      expect(canon(g, g.popEvents())).toBe(t.initial);
      for (let k = 0; k < t.ops.length; k++) {
        const [op, arg] = t.ops[k];
        if (op === "m") g.apply(arg as Move);
        else if (op === "t") g.tick(fromBits(arg as string));
        else g.place(placementFromIndex(arg as number));
        const text = canon(g, g.popEvents());
        if (fnv1a(text) !== t.hashes[k]) {
          const next = Object.keys(t.full).map(Number).find((n) => n >= k);
          const hint = next === k ? t.full[String(k)] : `(next full state at step ${next})`;
          expect.fail(`step ${k} (${op} ${arg}) differs\n  ts: ${text}\n  py: ${hint}`);
        }
        if (t.full[String(k)] !== undefined) expect(text).toBe(t.full[String(k)]);
      }
    });
  }
});
