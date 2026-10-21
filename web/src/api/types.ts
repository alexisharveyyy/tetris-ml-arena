// Types for the REST API and WebSocket protocol v1 (tetris/web/protocol.py, tetris/web/api.py).

export type LookId = "tetromino" | "terminal" | "broadcast" | "poster";
export type Preset = "Easy" | "Medium" | "Hard" | "Unlimited";
export const PRESETS: Preset[] = ["Easy", "Medium", "Hard", "Unlimited"];
export type Action = "left" | "right" | "soft" | "hard" | "cw" | "ccw" | "hold";
export type Cell = [number, number];

export interface Record3 {
  wins: number;
  losses: number;
  ties: number;
  text: string;
}

export interface AgentLooks {
  tetromino: { fill: string; text: string };
  terminal: { letter: string };
  broadcast: { fill: string };
  poster: { field: string; text: string };
}

export interface Agent {
  key: string;
  name: string;
  family: string;
  blurb: string;
  piece: string | null;
  letter: string;
  group: "main" | "variant" | "leaderboard" | "human";
  planned: boolean;
  cells: Cell[];
  looks: AgentLooks;
  registered?: boolean;
  available?: boolean;
  status?: { state: string; detail: string };
  playable?: boolean;
  on_leaderboard?: boolean;
  description?: string;
  record?: Record<Preset, Record3>;
}

export interface AgentsResponse {
  agents: Agent[];
  human: Agent;
  groups: string[];
  looks: LookId[];
}

export interface PresetInfo {
  name: Preset;
  description: string;
  params: Record<string, number | null>;
}

export interface Settings {
  bindings: Record<string, string[]>;
  das_ms: number;
  arr_ms: number;
  soft_drop_rate: number;
  show_ghost: boolean;
  fps: number;
  look: LookId;
  show_thinking: boolean;
  last_match: Record<string, unknown>;
}

export interface Slot {
  agent: string;
  preset: Preset;
}

export interface LobbyResponse {
  mode: "marathon" | "race";
  slots: (Slot | null)[];
  race: { condition: string; time_limit: number; target_lines: number };
}

export interface SurvivalRow {
  key: string;
  evaluated: boolean;
  status: "valid" | "pending_rerun" | "not_evaluated";
  reason?: string;
  source?: string;
  games?: number;
  top_out_rate?: number;
  lo?: number;
  hi?: number;
  off_scale?: boolean;
  pieces_survived?: number;
  score_per_piece?: number;
}

export interface MarathonPreset {
  games: number;
  score: Record<string, number | null>;
  lines: Record<string, number | null>;
  alive: Record<string, number | null>;
  time_to_top_out: number | null;
  time_is_lower_bound: boolean;
  cap_hit_rate: number;
}

export interface MarathonRow {
  key: string;
  evaluated: boolean;
  status: "valid" | "pending_rerun" | "not_evaluated";
  reason?: string;
  presets: Record<Preset, MarathonPreset | null>;
}

export interface LeaderboardResponse {
  survival: {
    scale_max: number;
    setting: { garbage_rows?: number; garbage_every?: number; max_pieces?: number };
    method: string;
    significance: string;
    rows: SurvivalRow[];
  };
  marathon: { minutes: number[]; presets: Preset[]; rows: MarathonRow[] };
  records: Record<string, Record<Preset, Record3>>;
}

export interface PlayerInfo {
  id: string;
  kind: "human" | "agent";
  name: string;
  agent: string | null;
  preset: Preset | null;
}

export interface MatchInfo {
  match: string;
  mode: "marathon" | "race" | "watch" | "solo" | "record";
  seed: number;
  ws: string;
  finished: boolean;
  connected: boolean;
  players: PlayerInfo[];
  recording?: { setting: string; on_space: number; off_space: number; saved: string | null } | null;
  standings?: StandingRow[] | null;
  history_id?: string | null;
  totals?: RecordingTotals;
}

export interface RecordingTotals {
  games: number;
  on_space: number;
  off_space: number;
}

export interface StandingRow {
  place: number;
  id: string;
  name: string;
  kind: "human" | "agent";
  agent_key: string | null;
  preset: Preset | null;
  lines: number;
  score: number;
  pieces: number;
  tetrises: number;
  level: number;
  pps: number;
  topped_out: boolean;
  finish_time: number | null;
  survival_time: number;
}

// ---------------------------------------------------------------- snapshot

export interface PieceCells {
  p: string;
  c: Cell[];
}

export interface Candidate {
  rank: number;
  hold: boolean;
  p: string;
  c: Cell[];
  score: number;
}

export interface PlayerSnap {
  id: string;
  kind: "human" | "agent";
  name: string;
  agent: string | null;
  preset: Preset | null;
  place: number;
  board: string;
  active: PieceCells | null;
  ghost: Cell[] | null;
  hold: string | null;
  can_hold: boolean;
  next: string[];
  stats: { score: number; lines: number; level: number; pps: number; tetrises: number; pieces: number };
  status: { state: "playing" | "thinking" | "topped_out" | "done" | "fallback" | "loading"; text: string; fallback?: string | null };
  finish_time: number | null;
  cand: Candidate[];
  cand_current: boolean;
}

export interface FeedEvent {
  type: "line_clear" | "tetris" | "b2b" | "rank_change" | "lead_change" | "top_out" | "fallback" | "hold" | "match_end";
  player: string | null;
  time: number;
  [key: string]: unknown;
}

export interface Snapshot {
  t: "snap";
  f: number;
  mode: string;
  seed: number;
  time: number;
  countdown: number;
  loading: boolean;
  paused: boolean;
  disconnected: boolean;
  finished: boolean;
  spectating: boolean;
  winner: string | null;
  standings: string[];
  final: StandingRow[] | null;
  history_id: string | null;
  race: { condition: string; time_limit: number; target_lines: number; description: string; time_remaining: number | null } | null;
  ack: { seq: number; ts: number } | null;
  ev: FeedEvent[];
  players: PlayerSnap[];
}

/** GET /api/demo: demo mode settings (configs/demo.yaml), when --demo is on. */
export interface DemoInfo {
  enabled: boolean;
  look?: LookId;
  seed?: number;
  opponents?: { agent: string; preset: Preset }[];
}
