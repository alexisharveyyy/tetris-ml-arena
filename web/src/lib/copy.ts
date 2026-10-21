// Plain-word sentences built from match data. Every number comes from the server.
import type { Agent, FeedEvent, PlayerSnap, StandingRow } from "../api/types";
import { clock, fmtInt, numberWord, ordinal, ordinalWord } from "./format";

export type Metric = "score" | "lines";

/** "You finished third of four." (words) or "You finished 3rd of 4" (numerals). */
export function resultsHeadline(rows: StandingRow[], style: "words" | "numerals" = "words"): string {
  const human = rows.find((r) => r.kind === "human");
  if (!human) return "The match is over.";
  if (style === "numerals") return `You finished ${ordinal(human.place)} of ${rows.length}`;
  return `You finished ${ordinalWord(human.place)} of ${numberWord(rows.length)}.`;
}

/**
 * One line on the gaps: who you beat by how much and who was ahead.
 * "You topped out at 2:47 with 21,460 points. That beat PPO by 7,580. CEM was 3,440 ahead of you."
 */
export function resultsSummary(rows: StandingRow[], metric: Metric = "score"): string {
  const human = rows.find((r) => r.kind === "human");
  if (!human) return "";
  const unit = metric === "score" ? "points" : "lines";
  const value = (r: StandingRow) => (metric === "score" ? r.score : r.lines);
  const how =
    human.topped_out && human.finish_time != null
      ? `You topped out at ${clock(human.finish_time)} with ${fmtInt(value(human))} ${unit}.`
      : `You finished with ${fmtInt(value(human))} ${unit}.`;
  const above = rows.find((r) => r.place === human.place - 1);
  const below = rows.find((r) => r.place === human.place + 1);
  const parts = [how];
  if (below) parts.push(`That beat ${below.name} by ${fmtInt(value(human) - value(below))}.`);
  if (above) parts.push(`${above.name} was ${fmtInt(value(above) - value(human))} ahead of you.`);
  if (!above && !below) parts.push("You were the only player.");
  return parts.join(" ");
}

/** How a game ended, for results tables. */
export function endedText(r: StandingRow): string {
  if (r.topped_out && r.finish_time != null) return `Topped out at ${clock(r.finish_time)}`;
  return "Still playing";
}

const PIECE_NAME: Record<string, string> = { I: "I", O: "O", T: "T", S: "S", Z: "Z", J: "J", L: "L" };

/** Ticker line for a feed event, or null for events not worth a line. */
export function tickerLine(ev: FeedEvent, players: Map<string, PlayerSnap>, humanId: string | null): string | null {
  const who = ev.player ? players.get(ev.player) : undefined;
  const name = who ? (who.kind === "human" ? "You" : who.name) : "";
  const you = who?.kind === "human";
  switch (ev.type) {
    case "tetris": {
      const nth = Number(ev.nth ?? 1);
      return nth === 1
        ? `${name} ${you ? "clear" : "clears"} a Tetris`
        : `${name} ${you ? "clear" : "clears"} ${you ? "your" : "its"} ${ordinal(nth)} Tetris of the match`;
    }
    case "b2b":
      return `${name} back-to-back Tetris`;
    case "line_clear":
      return Number(ev.count) === 3 ? `${name} ${you ? "clear" : "clears"} three lines` : null;
    case "rank_change": {
      const to = Number(ev.to);
      if (you) return `You move into ${ordinal(to)}`;
      const gap = ev.gap_to_human as number | null;
      if (gap == null || humanId == null) return `${name} moves into ${ordinal(to)}`;
      return gap >= 0
        ? `${name} moves into ${ordinal(to)}, ${fmtInt(gap)} ahead of you`
        : `${name} moves into ${ordinal(to)}, ${fmtInt(-gap)} behind you`;
    }
    case "lead_change":
      return you ? "You take the lead" : `${name} takes the lead`;
    case "top_out":
      return you ? "You topped out" : `${name} tops out`;
    case "fallback":
      return ev.reason === "timeout"
        ? `${name} model timed out, baseline taking over`
        : `${name} model stopped, baseline taking over`;
    case "hold":
      return you ? `You held ${article(String(ev.piece))} ${PIECE_NAME[String(ev.piece)] ?? ev.piece} piece` : null;
    case "match_end":
      return "Match over";
  }
  return null;
}

const article = (letter: string) => (["I", "O", "S", "L"].includes(letter) ? "an" : "a");

/** The plain status line under an agent's board. */
export function agentStatusLine(p: PlayerSnap, showThinking: boolean): string {
  const s = p.status;
  if (s.state === "fallback" || s.state === "topped_out" || s.state === "loading" || s.state === "done") return s.text;
  if (showThinking && p.cand.length) {
    const piece = p.cand[0].p;
    return p.cand_current
      ? `Outlines show its top three landing spots for this ${piece} piece.`
      : `Outlines show the top three spots it weighed for the ${piece} piece it just placed.`;
  }
  return s.state === "thinking" ? "Thinking." : "Playing.";
}

export function placeText(place: number, of: number): string {
  return `${ordinal(place)} of ${of}`;
}

export function aOrAn(word: string): string {
  return /^[aeiou]/i.test(word) ? "an" : "a";
}

/** Ticker lines, newest first: the new line goes in front and the oldest drops off. */
export function pushTicker(lines: string[], line: string, max: number): string[] {
  return [line, ...lines].slice(0, max);
}

/** The server is still running this agent's startup load check. */
export function isChecking(a: Agent | null | undefined): boolean {
  return a?.status?.state === "checking";
}

/** What an agent that cannot be picked shows instead of its controls. */
export function unavailableText(a: Agent | null | undefined): string {
  return isChecking(a) ? "Checking..." : "Not available yet";
}
