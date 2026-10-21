import { cellBoard } from "../shared/cellBoard";
import type { Theme } from "../types";
import "./theme.css";
import { TetrominoHome } from "./layouts/Home";
import { TetrominoLeaderboard } from "./layouts/Leaderboard";
import { TetrominoMatch } from "./layouts/Match";
import { TetrominoOpponents } from "./layouts/Opponents";
import { TetrominoResults } from "./layouts/Results";

const pieceColors = {
  I: "#1FB3C4",
  O: "#F2C318",
  T: "#8A5FDB",
  S: "#3BB065",
  Z: "#E04B39",
  J: "#3A6BDB",
  L: "#EE8A22",
  G: "#9AA0B4",
};

export const theme: Theme = {
  id: "tetromino",
  label: "Tetromino",
  order: 0,
  pieceColors,
  board: cellBoard({
    pieceColors,
    grid: "#D6DAE3",
    empty: "#FFFFFF",
    gap: 1,
    cellBorder: { color: "rgba(27, 31, 59, 0.2)", width: 2 },
    ghost: { fill: "#E6E8EE", stroke: "#9AA0B4" },
    candidate: (agent) => agent?.looks.tetromino.fill ?? "#1B1F3B",
    candidateFill: 0.15,
    badge: { fill: "#FFFFFF", text: "#1B1F3B", font: (px) => `700 ${px}px "Atkinson Hyperlegible", sans-serif` },
    toppedOut: "rgba(236, 238, 242, 0.55)",
  }),
  layouts: {
    Home: TetrominoHome,
    Opponents: TetrominoOpponents,
    Match: TetrominoMatch,
    Results: TetrominoResults,
    Leaderboard: TetrominoLeaderboard,
  },
};
