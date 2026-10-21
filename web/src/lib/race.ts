// Race rules in words (mirrors tetris/match/rules.py RaceRules.describe).
export interface RaceRules {
  condition: string;
  time_limit: number;
  target_lines: number;
}

export function raceGoal(r: RaceRules): string {
  if (r.condition === "first_to_lines") return `first to ${r.target_lines} lines`;
  const minutes = r.time_limit / 60;
  const what = r.condition === "score_in_time" ? "highest score" : "most lines";
  return `${what} in ${minutes % 1 === 0 ? minutes : minutes.toFixed(1)} minute${minutes === 1 ? "" : "s"}`;
}

export const raceMetric = (r: RaceRules | null | undefined): "score" | "lines" =>
  r && r.condition === "score_in_time" ? "score" : r ? "lines" : "score";
