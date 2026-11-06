// Path check, as in the Python tests: on recorded states, the planner reaches
// every legal placement through real engine moves, and the result equals the
// instant placement.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { legalPlacements } from "../../src/agents/afterstates";
import { planPath } from "../../src/agents/planner";
import { Game, type GameDict, parseGameJson } from "../../src/engine/game";

const pool = parseGameJson<{ states: GameDict[] }>(readFileSync(join(__dirname, "..", "fixtures", "agents", "pool.json"), "utf8")).states;

describe("path planner", () => {
  it("reaches every legal placement on the recorded states", () => {
    let checked = 0;
    for (const s of pool) {
      const g = Game.fromDict(s);
      for (const [placement] of legalPlacements(g)) {
        const moves = planPath(g, placement);
        expect(moves, `no path for ${JSON.stringify(placement)}`).not.toBeNull();
        const live = g.copy();
        for (const m of moves!) live.apply(m);
        const instant = g.copy();
        instant.place(placement);
        expect(live.board.join(), JSON.stringify(placement)).toBe(instant.board.join());
        expect(live.piecesPlaced).toBe(instant.piecesPlaced);
        checked += 1;
      }
    }
    expect(checked).toBeGreaterThan(1000);
  });
});
