import type { Theme } from "../types";
import { terminalBoard } from "./board";
import "./theme.css";
import { TerminalHome } from "./layouts/Home";
import { TerminalLeaderboard } from "./layouts/Leaderboard";
import { TerminalMatch } from "./layouts/Match";
import { TerminalOpponents } from "./layouts/Opponents";
import { TerminalResults } from "./layouts/Results";

const G = "#39FF7A";

export const theme: Theme = {
  id: "terminal",
  label: "Terminal",
  order: 1,
  // Terminal draws pieces as text; these colors only reach non-board DOM previews.
  pieceColors: { I: G, O: G, T: G, S: G, Z: G, J: G, L: G, G },
  board: terminalBoard,
  layouts: {
    Home: TerminalHome,
    Opponents: TerminalOpponents,
    Match: TerminalMatch,
    Results: TerminalResults,
    Leaderboard: TerminalLeaderboard,
  },
};
