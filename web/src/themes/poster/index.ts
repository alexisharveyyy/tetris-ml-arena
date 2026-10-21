import type { Theme } from "../types";
import { posterBoard } from "./board";
import "./theme.css";
import { PosterHome } from "./layouts/Home";
import { PosterLeaderboard } from "./layouts/Leaderboard";
import { PosterMatch } from "./layouts/Match";
import { PosterOpponents } from "./layouts/Opponents";
import { PosterResults } from "./layouts/Results";

// Poster pieces take the color of the column they sit in; previews outside a
// column fall back to light text.
const T = "#F3F4F8";

export const theme: Theme = {
  id: "poster",
  label: "Poster",
  order: 3,
  pieceColors: { I: T, O: T, T, S: T, Z: T, J: T, L: T, G: T },
  board: posterBoard,
  layouts: {
    Home: PosterHome,
    Opponents: PosterOpponents,
    Match: PosterMatch,
    Results: PosterResults,
    Leaderboard: PosterLeaderboard,
  },
};
