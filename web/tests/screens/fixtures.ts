// Screenshot fixtures. The API responses (agents, presets, leaderboard...) were
// captured from a real server; the match and results states copy the sample
// match in the mockups (A3, A4) so screenshots can be compared side by side.
// These numbers exist only in tests.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));

const load = (name: string) => JSON.parse(readFileSync(join(HERE, "..", "fixtures", `${name}.json`), "utf8"));
const agents = load("agents");
const leaderboard = load("leaderboard");
const presets = load("presets");
const recordings = load("recordings");
const settings = load("settings");

export const MATCH_ID = "fixture0001";

export const api: Record<string, unknown> = {
  "/api/agents": agents,
  "/api/presets": presets,
  "/api/settings": settings,
  "/api/settings/defaults": settings,
  "/api/leaderboard": leaderboard,
  "/api/recordings": recordings,
  "/api/lobby/marathon": {
    mode: "marathon",
    slots: [{ agent: "cem", preset: "Medium", custom: null }, { agent: "dqn", preset: "Hard", custom: null }, null],
    race: { condition: "lines_in_time", time_limit: 120, target_lines: 40 },
  },
  "/api/lobby/race": {
    mode: "race",
    slots: [{ agent: "cem", preset: "Medium", custom: null }, { agent: "dqn", preset: "Hard", custom: null }, null],
    race: { condition: "lines_in_time", time_limit: 120, target_lines: 40 },
  },
};

// ------------------------------------------------------------------ boards

/** Build a 200-char board from bottom rows (strings of 10, top to bottom). */
function board(bottom: string[]): string {
  return ".".repeat(10 * (20 - bottom.length)) + bottom.join("");
}

const HUMAN = board([
  "I...LS....",
  "I.LLLSS...",
  "IJSSOOTZZ.",
  "IJJSSTTTZZ",
  "OOJJZZTLLL",
  "OOIIIIJLL.",
]);

const DQN = board([
  "......T...",
  ".....TTO..",
  "LL...ZJJJ.",
  "LTZZJJJOO.",
  "LTSSIIIIO.",
  "OOTTSZZLLO",
]);

const CEM = board([
  "LLTTZZSOOJ",
  "LTTTSSIOOJ",
  "LZZOSSJJJ.",
  "JJZZIIOOTT",
  "JOOTTSZZLL",
]);

const PPO = board([
  "......I...",
  "......I...",
  "......I.OO",
  "......I.OO",
  ".Z...L..T.",
  "ZZ..LLL.TTT",
].map((r) => r.slice(0, 10)).concat([
  "S.SS.SS..O",
  "SS.OO.ZZ.O",
  "J.JJT.TTOO",
  "JJ.TTTZ.SS",
  "OIIII.LLSS",
]));

const stats = (score: number, lines: number, level: number, pps: number, tetrises = 0, pieces = 100) => ({
  score,
  lines,
  level,
  pps,
  tetrises,
  pieces,
});

const cand = (rank: number, cells: [number, number][], p = "I") => ({ rank, hold: false, p, c: cells, score: 1 - rank * 0.1 });

export const snapshot = {
  t: "snap",
  f: 5520,
  mode: "marathon",
  seed: 48213,
  time: 92.4,
  countdown: 0,
  loading: false,
  paused: false,
  disconnected: false,
  finished: false,
  spectating: false,
  winner: null,
  standings: ["ai1", "ai2", "human", "ai3"],
  final: null,
  history_id: null,
  race: null,
  ack: null,
  ev: [],
  players: [
    {
      id: "human",
      kind: "human",
      name: "You",
      agent: null,
      preset: null,
      place: 3,
      board: HUMAN,
      active: { p: "T", c: [[1, 4], [1, 5], [1, 6], [2, 5]] },
      ghost: [[12, 4], [12, 5], [12, 6], [13, 5]],
      hold: "I",
      can_hold: true,
      next: ["S", "L", "O", "Z", "J"],
      stats: stats(12840, 31, 4, 1.21, 1, 112),
      status: { state: "playing", text: "Playing" },
      finish_time: null,
      cand: [],
      cand_current: false,
    },
    {
      id: "ai1",
      kind: "agent",
      name: "DQN",
      agent: "dqn",
      preset: "Hard",
      place: 1,
      board: DQN,
      active: { p: "I", c: [[-1, 3], [-1, 4], [-1, 5], [-1, 6]] },
      ghost: null,
      hold: null,
      can_hold: true,
      next: ["T", "O", "S", "Z", "L"],
      stats: stats(32200, 70, 8, 1.94, 6, 180),
      status: { state: "playing", text: "Playing" },
      finish_time: null,
      cand: [
        cand(1, [[15, 9], [16, 9], [17, 9], [18, 9]]),
        cand(2, [[12, 5], [13, 5], [14, 5], [15, 5]]),
        cand(3, [[14, 0], [14, 1], [14, 2], [14, 3]]),
      ],
      cand_current: true,
    },
    {
      id: "ai2",
      kind: "agent",
      name: "CEM",
      agent: "cem",
      preset: "Medium",
      place: 2,
      board: CEM,
      active: null,
      ghost: null,
      hold: null,
      can_hold: true,
      next: ["T", "O", "S", "Z", "L"],
      stats: stats(13960, 34, 4, 0.98, 2, 90),
      status: { state: "playing", text: "Playing" },
      finish_time: null,
      cand: [
        cand(1, [[15, 9], [16, 9], [17, 9], [18, 9]]),
        cand(2, [[14, 1], [14, 2], [14, 3], [14, 4]]),
        cand(3, [[12, 8], [13, 8], [14, 8], [15, 8]]),
      ],
      cand_current: true,
    },
    {
      id: "ai3",
      kind: "agent",
      name: "PPO",
      agent: "ppo",
      preset: "Medium",
      place: 4,
      board: PPO,
      active: null,
      ghost: null,
      hold: null,
      can_hold: true,
      next: ["T", "O", "S", "Z", "L"],
      stats: stats(9120, 27, 3, 0.97, 0, 88),
      status: {
        state: "fallback",
        text: "The model missed its time limit, so the baseline is playing for it until this match ends.",
        fallback: "timeout",
      },
      finish_time: null,
      cand: [],
      cand_current: false,
    },
  ],
};

export const tickerEvents = [
  { type: "tetris", player: "ai1", time: 80, nth: 6 },
  { type: "rank_change", player: "ai2", time: 85, from: 3, to: 2, passed: "human", gap_to_human: 1120 },
  { type: "fallback", player: "ai3", time: 88, reason: "timeout", text: "" },
  { type: "hold", player: "human", time: 90, piece: "I" },
];

const row = (place: number, id: string, name: string, kind: string, agent_key: string | null, preset: string | null,
  score: number, lines: number, tetrises: number, pps: number, level: number, topped: boolean, finish: number | null) => ({
  place, id, name, kind, agent_key, preset, score, lines, tetrises, pps, level, pieces: 0,
  attack_sent: 0, attack_received: 0, topped_out: topped, finish_time: finish, survival_time: finish ?? 167,
});

export const matchInfo = {
  match: MATCH_ID,
  mode: "marathon",
  seed: 48213,
  ws: `/ws?match=${MATCH_ID}`,
  finished: true,
  connected: false,
  recording: null,
  players: [],
  history_id: "h1",
  standings: [
    row(1, "ai1", "DQN", "agent", "dqn", "Hard", 98400, 127, 14, 1.94, 13, false, null),
    row(2, "ai2", "CEM", "agent", "cem", "Medium", 24900, 61, 2, 0.98, 7, false, null),
    row(3, "human", "You", "human", null, null, 21460, 49, 3, 1.21, 5, true, 167),
    row(4, "ai3", "PPO", "agent", "ppo", "Medium", 13880, 39, 0, 0.97, 4, true, 130),
  ],
};
