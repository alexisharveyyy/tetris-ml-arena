// Agent parity: every agent picks the same placement as its Python original on
// recorded board states (scripts/tools/web_fixtures.py agents).
//
// A different pick is allowed only at a near-tie: when Python's top two scores
// are within TIE_TOLERANCE of each other (relative to the top score's size, at
// least 1), and the TypeScript pick is one of Python's placements within that
// tolerance of the top. The number of near-ties is reported per agent.
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { enumerateAfterstates } from "../../src/agents/afterstates";
import { type AgentKey, createAgent, modelFiles, WEB_AGENT_KEYS } from "../../src/agents/agents";
import type { DTAgent } from "../../src/agents/dt";
import { Game, type GameDict, parseGameJson } from "../../src/engine/game";
import { placementFromIndex } from "../../src/engine/placement";

export const TIE_TOLERANCE = 1e-4;

const FIX = join(__dirname, "..", "fixtures", "agents");
const MODELS = join(__dirname, "..", "..", "public", "models");

type Top = [number, number][];
interface StateCase {
  state: GameDict;
  top: Top;
  played?: number;
}

function load<T>(name: string): T {
  return parseGameJson<T>(readFileSync(join(FIX, name), "utf8"));
}

function files(key: AgentKey): ArrayBuffer[] {
  return modelFiles(key).map((f) => {
    const b = readFileSync(join(MODELS, f));
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  });
}

const tol = (top: number) => TIE_TOLERANCE * Math.max(1, Math.abs(top));

interface Result {
  states: number;
  exact: number;
  nearTies: number;
  /** Near-ties where both picks give the same board (for example the O piece's rotations). */
  sameAfterstate: number;
  maxScoreDiff: number;
}

function sameAfterstate(g: Game, a: number, b: number): boolean {
  const afs = enumerateAfterstates(g, true, false);
  const x = afs.find((s) => s.index === a);
  const y = afs.find((s) => s.index === b);
  return !!x && !!y && x.placement.useHold === y.placement.useHold && x.board.join() === y.board.join();
}
const results: Record<string, Result> = {};

function check(key: string, g: Game, py: Top, ts: { index: number; score: number }[], r: Result, where: string): void {
  r.states += 1;
  expect(ts.length, `${where}: no placements`).toBeGreaterThan(0);
  const tsTop = ts[0];
  const pyScoreOfTsTop = ts.find((x) => x.index === py[0][0]);
  if (pyScoreOfTsTop) r.maxScoreDiff = Math.max(r.maxScoreDiff, Math.abs(pyScoreOfTsTop.score - py[0][1]));
  if (tsTop.index === py[0][0]) {
    r.exact += 1;
    return;
  }
  const t = tol(py[0][1]);
  const close = py.filter(([, s]) => Math.abs(py[0][1] - s) <= t).map(([i]) => i);
  const nearTie = py.length > 1 && Math.abs(py[0][1] - py[1][1]) <= t && close.includes(tsTop.index);
  expect(nearTie, `${key} ${where}: TS picks ${tsTop.index} (${tsTop.score}), Python ${py[0][0]} (${py[0][1]}); Python top 5 ${JSON.stringify(py)}`).toBe(true);
  r.nearTies += 1;
  if (sameAfterstate(g, tsTop.index, py[0][0])) r.sameAfterstate += 1;
}

describe("agent parity with Python", () => {
  const pool = load<{ states: GameDict[] }>("pool.json").states;

  for (const key of WEB_AGENT_KEYS) {
    it(`${key} picks the Python placement`, () => {
      const r: Result = { states: 0, exact: 0, nearTies: 0, sameAfterstate: 0, maxScoreDiff: 0 };
      results[key] = r;
      const agent = createAgent(key, files(key));
      if (key === "dt") {
        const data = load<{ trajectories: { steps: StateCase[] }[] }>("dt.json");
        for (const [ti, traj] of data.trajectories.entries()) {
          const dt = createAgent(key, files(key)) as DTAgent;
          traj.steps.forEach((s, si) => {
            const g = Game.fromDict(s.state);
            check(key, g, s.top, dt.rank(g), r, `trajectory ${ti} step ${si}`);
            dt.commit(g, placementFromIndex(s.played!));
          });
        }
      } else {
        const data = load<{ own: StateCase[]; pool_top: Top[] }>(`${key}.json`);
        data.own.forEach((s, i) => {
          const g = Game.fromDict(s.state);
          check(key, g, s.top, agent.rank(g), r, `own state ${i}`);
        });
        pool.forEach((s, i) => {
          const g = Game.fromDict(s);
          check(key, g, data.pool_top[i], agent.rank(g), r, `pool state ${i}`);
        });
      }
      expect(r.states).toBeGreaterThanOrEqual(60);
    });
  }

  afterAll(() => {
    const rows = Object.entries(results).map(([k, r]) => ({ agent: k, ...r }));
    console.table(rows);
    writeFileSync(
      join(__dirname, "..", "parity-report.json"),
      JSON.stringify({ tie_tolerance: TIE_TOLERANCE, rule: "relative to max(1, |top score|)", agents: results }, null, 1) + "\n",
    );
  });
});
