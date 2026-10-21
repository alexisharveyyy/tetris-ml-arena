import { describe, expect, it } from "vitest";
import type { PlayerSnap } from "../../src/api/types";
import { asciiInterval, boardLines } from "../../src/themes/terminal/board";
import { allThemes } from "../../src/themes/registry";

const player = (overrides: Partial<PlayerSnap> = {}): PlayerSnap =>
  ({
    board: ".".repeat(190) + "IIII.OOOOO",
    active: { p: "T", c: [[0, 3], [0, 4], [0, 5], [1, 4]] },
    ghost: [[17, 3], [17, 4], [17, 5], [18, 4]],
    cand: [{ rank: 1, hold: false, p: "I", c: [[19, 4], [18, 4], [17, 4], [16, 4]], score: 1 }],
    cand_current: true,
    status: { state: "playing", text: "Playing" },
    ...overrides,
  }) as PlayerSnap;

const text = (lines: [string, string][][]) => lines.map((runs) => runs.map(([t]) => t).join(""));

describe("terminal board", () => {
  it("draws walls, cells, floor and base as characters", () => {
    const rows = text(boardLines({ player: player(), agent: null, showThinking: false, showGhost: true, human: true }));
    expect(rows).toHaveLength(22);
    expect(rows.every((r) => r.length === 24)).toBe(true);
    expect(rows[0]).toBe("<! . . .[][][] . . . .!>");
    expect(rows[19]).toBe("<![][][][] .[][][][][]!>");
    expect(rows[17]).toContain("::::::");
    expect(rows[20]).toBe("<!====================!>");
    expect(rows[21]).toBe("  \\/\\/\\/\\/\\/\\/\\/\\/\\/\\/  ");
  });

  it("marks candidates with () and the rank in the first cell", () => {
    const rows = text(boardLines({ player: player(), agent: null, showThinking: true, showGhost: false, human: false }));
    expect(rows[16].slice(10, 12)).toBe("1)");
    expect(rows[17].slice(10, 12)).toBe("()");
  });

  it("draws the ASCII interval chart", () => {
    expect(asciiInterval(0.12, 0.18, 0.15, 0.5, 51)).toBe(`${".".repeat(12)}[==*==]${".".repeat(32)}`);
    const off = asciiInterval(0.99, 1, 1, 0.5, 20);
    expect(off.endsWith(">")).toBe(true);
    expect(off).not.toContain("]");
  });

  it("registers all four looks in switcher order", () => {
    expect(allThemes().map((t) => t.id)).toEqual(["tetromino", "terminal", "broadcast", "poster"]);
  });
});
