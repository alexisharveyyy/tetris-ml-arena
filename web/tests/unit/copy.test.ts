import { describe, expect, it } from "vitest";
import type { PlayerSnap, StandingRow } from "../../src/api/types";
import { agentStatusLine, endedText, resultsHeadline, resultsSummary, tickerLine } from "../../src/lib/copy";
import { clock, ordinal, ordinalWord } from "../../src/lib/format";
import { raceGoal } from "../../src/lib/race";

const row = (place: number, name: string, kind: "human" | "agent", score: number, lines: number, topped = false, finish: number | null = null): StandingRow => ({
  place, id: name, name, kind, agent_key: kind === "agent" ? name.toLowerCase() : null, preset: null,
  lines, score, pieces: 0, tetrises: 0, level: 1, pps: 1, topped_out: topped, finish_time: finish, survival_time: 0,
});

const rows = [
  row(1, "DQN", "agent", 98400, 127),
  row(2, "CEM", "agent", 24900, 61),
  row(3, "You", "human", 21460, 49, true, 167),
  row(4, "PPO", "agent", 13880, 39, true, 130),
];

describe("results copy", () => {
  it("states the place in words and in numerals", () => {
    expect(resultsHeadline(rows)).toBe("You finished third of four.");
    expect(resultsHeadline(rows, "numerals")).toBe("You finished 3rd of 4");
  });

  it("summarizes the gaps to the players around you", () => {
    expect(resultsSummary(rows)).toBe(
      "You topped out at 2:47 with 21,460 points. That beat PPO by 7,580. CEM was 3,440 ahead of you.",
    );
  });

  it("handles first and last place and race lines", () => {
    const first = [row(1, "You", "human", 500, 10), row(2, "GA", "agent", 300, 4)];
    expect(resultsSummary(first)).toBe("You finished with 500 points. That beat GA by 200.");
    const last = [row(1, "GA", "agent", 300, 12), row(2, "You", "human", 100, 5, true, 61)];
    expect(resultsSummary(last, "lines")).toBe("You topped out at 1:01 with 5 lines. GA was 7 ahead of you.");
  });

  it("says how each game ended", () => {
    expect(endedText(rows[0])).toBe("Still playing");
    expect(endedText(rows[3])).toBe("Topped out at 2:10");
  });
});

describe("ticker", () => {
  const p = (id: string, kind: "human" | "agent", name: string) => ({ id, kind, name }) as PlayerSnap;
  const players = new Map([
    ["human", p("human", "human", "You")],
    ["ai1", p("ai1", "agent", "DQN")],
    ["ai2", p("ai2", "agent", "CEM")],
  ]);
  it("writes plain lines", () => {
    expect(tickerLine({ type: "tetris", player: "ai1", time: 1, nth: 6 }, players, "human")).toBe("DQN clears its 6th Tetris of the match");
    expect(tickerLine({ type: "rank_change", player: "ai2", time: 1, from: 3, to: 2, gap_to_human: 1120 }, players, "human")).toBe(
      "CEM moves into 2nd, 1,120 ahead of you",
    );
    expect(tickerLine({ type: "fallback", player: "ai1", time: 1, reason: "timeout" }, players, "human")).toBe(
      "DQN model timed out, baseline taking over",
    );
    expect(tickerLine({ type: "hold", player: "human", time: 1, piece: "I" }, players, "human")).toBe("You held an I piece");
    expect(tickerLine({ type: "line_clear", player: "ai1", time: 1, count: 1 }, players, "human")).toBeNull();
  });
});

describe("status lines", () => {
  const base = { status: { state: "playing", text: "Playing" }, cand: [{ rank: 1, hold: false, p: "I", c: [], score: 1 }], cand_current: true } as unknown as PlayerSnap;
  it("explains the outlines only when they are shown", () => {
    expect(agentStatusLine(base, true)).toBe("Outlines show its top three landing spots for this I piece.");
    expect(agentStatusLine(base, false)).toBe("Playing.");
    expect(agentStatusLine({ ...base, cand_current: false }, true)).toContain("just placed");
  });
  it("puts a fallback in plain words", () => {
    const fb = { ...base, status: { state: "fallback", text: "The model missed its time limit." } } as PlayerSnap;
    expect(agentStatusLine(fb, true)).toBe("The model missed its time limit.");
  });
});

describe("formatting", () => {
  it("formats clocks and ordinals", () => {
    expect(clock(92.4)).toBe("1:32");
    expect(ordinal(1) + ordinal(2) + ordinal(3) + ordinal(11) + ordinal(22)).toBe("1st2nd3rd11th22nd");
    expect(ordinalWord(3)).toBe("third");
  });
  it("describes race rules", () => {
    expect(raceGoal({ condition: "lines_in_time", time_limit: 120, target_lines: 40 })).toBe("most lines in 2 minutes");
    expect(raceGoal({ condition: "first_to_lines", time_limit: 120, target_lines: 40 })).toBe("first to 40 lines");
  });
});
