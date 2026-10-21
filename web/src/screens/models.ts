// View models: what each screen hands to a look's layout. Layouts render these
// and call the callbacks; they never fetch data or talk to the server.
import type {
  Agent,
  LeaderboardResponse,
  LookId,
  PlayerSnap,
  Preset,
  PresetInfo,
  Snapshot,
  StandingRow,
} from "../api/types";
import type { SocketStatus } from "../api/socket";

export interface LookOption {
  id: LookId;
  label: string;
}

export interface ControlHint {
  keys: string[]; // key names (settings style)
  label: string; // "move", "rotate", ...
}

export interface HomeModel {
  lineup: Agent[];
  looks: LookOption[];
  look: LookId;
  setLook: (id: LookId) => void;
  controls: ControlHint[];
  /** Short lines from the survival leaderboard (no numbers when not loaded). */
  evalTicker: string[];
  raceMinutes: number;
  /** Start a Marathon (Enter on Home). */
  startMarathon: () => void;
  links: { marathon: string; race: string; watch: string; solo: string; record: string; leaderboard: string; settings: string };
}

export interface AgentRow {
  agent: Agent;
  inSlots: number;
  available: boolean;
  record: string; // "no games yet", "Medium 2W 1L, Hard 0W 1L"
}

export interface SlotView {
  agent: Agent;
  preset: Preset;
  presetText: string;
}

export interface OpponentsModel {
  mode: "marathon" | "race";
  setMode: (m: "marathon" | "race") => void;
  groups: { title: string | null; rows: AgentRow[] }[];
  slots: (SlotView | null)[];
  /** 1-based number of the slot the next Add fills, or null when all are full. */
  nextSlot: number | null;
  add: (key: string) => void;
  remove: (index: number) => void;
  setPreset: (index: number, preset: Preset) => void;
  presets: PresetInfo[];
  instruction: string;
  modeText: string;
  startLabel: string;
  canStart: boolean;
  start: () => void;
  starting: boolean;
  error: string | null;
  backHref: string;
}

export interface MatchModel {
  snap: Snapshot;
  status: SocketStatus;
  human: PlayerSnap | null;
  /** Agent players in seat order. */
  agents: PlayerSnap[];
  /** Every player by live place. */
  ranked: PlayerSnap[];
  agent: (key: string | null | undefined) => Agent | null;
  /** "1:32" (elapsed, or time left in a timed race). */
  timer: string;
  /** "Marathon, seed 48213" or "Race: most lines in 2 min, seed 48213". */
  modeLine: string;
  showThinking: boolean;
  toggleThinking: () => void;
  showGhost: boolean;
  togglePause: () => void;
  endMatch: () => void;
  /** Recent ticker lines, newest first. */
  ticker: string[];
  /** Demo mode only: end the match now and show the results (also F9). */
  demoSkip: (() => void) | null;
  /** Spectating after the match ended (Keep watching). */
  resultsHref: string;
}

export interface ResultsModel {
  rows: StandingRow[];
  mode: "marathon" | "race";
  seed: number;
  headline: string;
  headlineNumerals: string;
  summary: string;
  caption: string;
  agent: (key: string | null | undefined) => Agent | null;
  rematch: () => void;
  newSeed: () => void;
  keepWatching: () => void;
  opponentsHref: string;
  homeHref: string;
  busy: boolean;
}

export interface LeaderboardModel {
  tab: "survival" | "people";
  setTab: (t: "survival" | "people") => void;
  data: LeaderboardResponse | null;
  agent: (key: string | null | undefined) => Agent | null;
  /** One-line description of the survival evaluation, from the report's settings. */
  intro: string;
  preset: Preset;
  setPreset: (p: Preset) => void;
  presets: Preset[];
  homeHref: string;
}
