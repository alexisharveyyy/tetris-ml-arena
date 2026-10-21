import { cellBoard } from "../shared/cellBoard";
import type { Theme } from "../types";
import "./theme.css";
import { BroadcastHome } from "./layouts/Home";
import { BroadcastLeaderboard } from "./layouts/Leaderboard";
import { BroadcastMatch } from "./layouts/Match";
import { BroadcastOpponents } from "./layouts/Opponents";
import { BroadcastResults } from "./layouts/Results";

const pieceColors = {
  I: "#22D3E6",
  O: "#FFD23F",
  T: "#B07CFF",
  S: "#4BE08A",
  Z: "#FF5A5A",
  J: "#5B8CFF",
  L: "#FF9B3D",
  G: "#7D86C9",
};

export const YELLOW = "#FFD23F";

export const theme: Theme = {
  id: "broadcast",
  label: "Broadcast",
  order: 2,
  pieceColors,
  board: cellBoard({
    pieceColors,
    grid: "#0C1458",
    empty: "#16207A",
    gap: 1,
    cellBorder: { color: "rgba(12, 20, 88, 0.35)", width: 2 },
    ghost: { fill: "rgba(220, 225, 255, 0.10)", stroke: "rgba(220, 225, 255, 0.55)" },
    // Broadcast outlines every agent's candidates in the leader yellow.
    candidate: () => YELLOW,
    candidateFill: 0.15,
    badge: { fill: "#0C1458", text: "#FFFFFF", font: (px) => `700 ${px}px "Barlow", sans-serif` },
    toppedOut: "rgba(12, 20, 88, 0.6)",
  }),
  layouts: {
    Home: BroadcastHome,
    Opponents: BroadcastOpponents,
    Match: BroadcastMatch,
    Results: BroadcastResults,
    Leaderboard: BroadcastLeaderboard,
  },
};
