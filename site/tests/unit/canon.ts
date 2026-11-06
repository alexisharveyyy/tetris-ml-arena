// The canonical state text shared with scripts/tools/web_fixtures.py (canon()).
import { boardToStrings } from "../../src/engine/board";
import type { Game, GameEvent } from "../../src/engine/game";
import { letter } from "../../src/engine/pieces";
import { toBits } from "../../src/engine/scoring";

function eventText(ev: GameEvent): string {
  const d = ev.data as Record<string, unknown>;
  if (ev.type === "lock") {
    const cells = (d.cells as [number, number][]).map(([r, c]) => `${r},${c}`).join(".");
    return `lock:${d.piece},${d.lines},${d.combo},${d.b2b ? 1 : 0},${d.attack},${d.sent},${d.score},${cells}`;
  }
  if (ev.type === "spawn" || ev.type === "hold") return `${ev.type}:${d.piece}`;
  if (ev.type === "game_over") return `game_over:${d.reason}`;
  return ev.type;
}

export function canon(g: Game, events: GameEvent[]): string {
  const a = g.active;
  return [
    boardToStrings(g.board).join(""),
    g.bagRng.state.toString(),
    g.garbageRng.state.toString(),
    g.queue.map(letter).join(""),
    a ? `${letter(a.piece)},${a.rot},${a.row},${a.col}` : "-",
    g.hold !== null ? letter(g.hold) : "-",
    g.canHold ? "1" : "0",
    String(g.score),
    String(g.lines),
    String(g.level),
    String(g.combo),
    g.b2b ? "1" : "0",
    String(g.piecesPlaced),
    String(g.attackSent),
    g.clears.join(","),
    g.gameOver ? "1" : "0",
    toBits(g.time),
    toBits(g.gravityAcc),
    toBits(g.lockTimer),
    String(g.lockResets),
    String(g.lowestRow),
    events.map(eventText).join(";"),
  ].join("|");
}

export function fnv1a(text: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}
