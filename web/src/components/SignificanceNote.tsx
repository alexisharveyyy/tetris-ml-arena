import type { LeaderboardResponse } from "../api/types";

/** Method, the grouped bootstrap note and a link to the full comparison table,
 * one paragraph each. Looks pass their own paragraph class. */
export function SignificanceNote({ survival, className }: { survival: LeaderboardResponse["survival"]; className?: string }) {
  const n = survival.comparisons.length;
  return (
    <>
      <p className={className}>{survival.method}</p>
      {survival.significance.map((line) => (
        <p key={line.label} className={className}>
          <strong>{line.label}:</strong> {line.text}
        </p>
      ))}
      {n ? (
        <p className={className}>
          <a className="link" href="#/leaderboard/comparisons">
            Full comparison table ({n} pairs)
          </a>
        </p>
      ) : null}
    </>
  );
}
