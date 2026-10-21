import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Comparison, Interval, LeaderboardResponse } from "../api/types";
import { useApp } from "../state/app";
import { PageHead } from "./PageHead";

const signed = (v: number, digits: number) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(digits)}`;

export function points(d: Interval | null): string {
  if (!d) return "-";
  const p = (v: number) => signed(v * 100, 1);
  return `${p(d.mean)} pts [${p(d.lo)}, ${p(d.hi)}]`;
}

export function fitness(d: Interval | null): string {
  if (!d) return "-";
  return `${signed(d.mean, 3)} [${signed(d.lo, 3)}, ${signed(d.hi, 3)}]`;
}

/** Every pair of evaluated agents from the paired bootstrap, shared by all looks. */
export function ComparisonsScreen() {
  const app = useApp();
  const [data, setData] = useState<LeaderboardResponse | null>(null);
  useEffect(() => {
    api.leaderboard().then(setData).catch(() => setData(null));
  }, []);
  const name = (k: string) => app.agent(k)?.name ?? k;
  const rows: Comparison[] = data?.survival.comparisons ?? [];
  return (
    <main className="screen comparisons">
      <PageHead title="All comparisons" back="#/leaderboard" backLabel="Leaderboard" />
      {!data ? (
        <p>Loading the comparisons...</p>
      ) : !rows.length ? (
        <p>No paired comparisons in the current reports.</p>
      ) : (
        <>
          <p className="comparisons__intro">
            Each row is the first agent minus the second, on the same seeds under garbage: the mean difference and its 95% interval
            from a paired bootstrap. A lower top-out rate is better, and a higher fitness is better. A difference is significant when
            its interval excludes 0.
          </p>
          <div className="table-scroll">
            <table className="keys-table comparisons-table">
              <thead>
                <tr>
                  <th scope="col">Comparison</th>
                  <th scope="col" className="num">
                    Top-out rate difference
                  </th>
                  <th scope="col">Top-out gap</th>
                  <th scope="col" className="num">
                    Fitness difference
                  </th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={`${r.a}-${r.b}`}>
                    <th scope="row">
                      {name(r.a)} minus {name(r.b)}
                    </th>
                    <td className="num">{points(r.top_out_rate)}</td>
                    <td>{r.top_out_rate ? (r.top_out_rate.significant ? "Significant" : "Tied") : "-"}</td>
                    <td className="num">{fitness(r.fitness)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </main>
  );
}
