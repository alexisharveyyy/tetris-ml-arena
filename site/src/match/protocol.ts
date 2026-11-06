// Snapshot encoder (tetris/web/protocol.py): the same message the server sends
// over the WebSocket, built in the page.
import { boardToStrings, HIDDEN_ROWS, pieceCells } from "../engine/board";
import { letter } from "../engine/pieces";
import { alive, describeRace, FIRST_TO_LINES, type MatchController, type PlayerState } from "./controller";
import { FALLBACK_TEXT, type FeedEventOut } from "./feed";

type Cell = [number, number];

const visible = (cells: [number, number][]): Cell[] => cells.map(([r, c]) => [r - HIDDEN_ROWS, c]);

function clock(s: number): string {
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
}

const round = (x: number, d: number) => Math.round(x * 10 ** d) / 10 ** d;

function status(m: MatchController, p: PlayerState) {
  const g = p.game;
  const pid = p.spec.playerId;
  if (g.gameOver) return { state: "topped_out", text: `Topped out at ${clock(p.finishTime ?? m.time)}` };
  if (p.done) return { state: "done", text: "Finished" };
  if (p.driver !== null && m.runner !== null) {
    const reason = m.runner.fallbackReason(pid);
    if (reason !== null) return { state: "fallback", text: FALLBACK_TEXT[reason] ?? FALLBACK_TEXT.crash, fallback: reason };
    if (!m.ready) return { state: "loading", text: "Loading the model" };
    if (p.driver.thinking) return { state: "thinking", text: "Thinking" };
  }
  if (m.finished && !m.spectating) return { state: "playing", text: "Still playing" };
  return { state: "playing", text: "Playing" };
}

function encodePlayer(m: MatchController, p: PlayerState, place: number) {
  const g = p.game;
  const row = m.playerRow(p.spec.playerId);
  let active = null;
  let ghost = null;
  const a = g.active;
  if (a !== null && alive(p)) {
    active = { p: letter(a.piece), c: visible(pieceCells(a.piece, a.rot, a.row, a.col)) };
    const gr = g.ghostRow();
    if (gr !== null) ghost = visible(pieceCells(a.piece, a.rot, gr, a.col));
  }
  let cand: { rank: number; hold: boolean; p: string; c: Cell[]; score: number }[] = [];
  let current = false;
  if (p.driver !== null && alive(p)) {
    const [cells, cur] = p.driver.overlay(g);
    current = cur;
    cand = cells.map((c) => ({ rank: c.rank, hold: c.hold, p: c.piece, c: visible(c.cells), score: c.score }));
  }
  return {
    id: p.spec.playerId,
    kind: p.spec.kind,
    name: p.spec.name,
    agent: p.spec.agentKey,
    preset: p.spec.preset,
    place,
    board: boardToStrings(g.board).slice(HIDDEN_ROWS).join(""),
    active,
    ghost,
    hold: g.hold !== null ? letter(g.hold) : null,
    can_hold: g.canHold,
    next: g.nextQueue.map(letter),
    stats: {
      score: row.score,
      lines: row.lines,
      level: row.level,
      pps: round(row.pps, 3),
      tetrises: row.tetrises,
      pieces: row.pieces,
    },
    status: status(m, p),
    finish_time: p.finishTime,
    cand,
    cand_current: current,
    model: null,
  };
}

export function encodeSnapshot(
  m: MatchController,
  frame: number,
  opts: { ack: { seq: number; ts: number } | null; events: FeedEventOut[]; disconnected: boolean; historyId: string | null },
) {
  const order = m.liveStandings();
  const places = new Map(order.map((pid, i) => [pid, i + 1]));
  const rules = m.config.race;
  const race =
    m.config.mode === "race" && rules
      ? {
          ...rules,
          description: describeRace(rules),
          time_remaining: rules.condition !== FIRST_TO_LINES ? Math.max(0, rules.time_limit - m.time) : null,
        }
      : null;
  return {
    t: "snap" as const,
    f: frame,
    mode: m.config.mode,
    seed: m.config.seed,
    time: round(m.time, 3),
    countdown: round(m.countdown, 3),
    loading: !m.ready,
    paused: m.paused,
    disconnected: opts.disconnected,
    finished: m.finished,
    spectating: m.spectating,
    winner: m.winner,
    standings: order,
    final: m.finished ? m.standings : null,
    history_id: opts.historyId,
    race,
    ack: opts.ack,
    ev: opts.events,
    players: [...m.players.entries()].map(([pid, p]) => encodePlayer(m, p, places.get(pid)!)),
  };
}
