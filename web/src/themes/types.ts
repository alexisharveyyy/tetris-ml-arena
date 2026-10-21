// What a look supplies. Screens own data and behavior; a look supplies tokens
// (its theme.css), a board renderer, and layout variants where the mockups
// differ in structure. Any layout a look leaves out falls back to Tetromino's.
import type { ComponentType } from "react";
import type { Agent, LookId, PlayerSnap } from "../api/types";
import type { HomeModel, LeaderboardModel, MatchModel, OpponentsModel, ResultsModel } from "../screens/models";

export interface BoardView {
  player: PlayerSnap;
  agent: Agent | null; // null for the human
  showThinking: boolean;
  showGhost: boolean;
  /** Board for the human (larger, look-specific accents). */
  human: boolean;
}

export interface BoardRenderer {
  /** CSS pixel size of the board drawing for a given cell size. */
  size(cell: number): { width: number; height: number };
  draw(ctx: CanvasRenderingContext2D, view: BoardView, cell: number): void;
}

export interface Layouts {
  Home: ComponentType<HomeModel>;
  Opponents: ComponentType<OpponentsModel>;
  Match: ComponentType<MatchModel>;
  Results: ComponentType<ResultsModel>;
  Leaderboard: ComponentType<LeaderboardModel>;
}

export interface Theme {
  id: LookId;
  label: string;
  /** Position in the look switcher. */
  order: number;
  /** Board piece colors by letter (I, O, T, S, Z, J, L, G). */
  pieceColors: Record<string, string>;
  board: BoardRenderer;
  layouts: Partial<Layouts>;
}
