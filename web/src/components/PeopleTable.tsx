// "Against people" tab: what a human has to beat at each point of a Marathon,
// per agent at one difficulty, plus the human's record. Shared by every look.
import type { Agent, LeaderboardResponse, Preset } from "../api/types";
import { clock, fmtInt } from "../lib/format";
import { PieceIcon } from "./PieceIcon";
import { Segmented } from "./Segmented";

interface Props {
  data: LeaderboardResponse;
  agent: (key: string) => Agent | null;
  preset: Preset;
  setPreset: (p: Preset) => void;
  presets: Preset[];
  /** Terminal draws letters instead of piece icons. */
  icons?: boolean;
}

export function peopleIntro(data: LeaderboardResponse, preset: Preset): string {
  const games = data.marathon.rows.map((r) => r.presets[preset]?.games).find((g) => g);
  const per = games ? `, over ${fmtInt(games)} games per agent` : "";
  return (
    `Each agent's mean score and lines after 1, 3, 5 and 10 minutes of a Marathon at ${preset}${per}. ` +
    "A Marathon ends when you top out, so these are the numbers to beat at that moment. " +
    'Time to top-out is the mean time until the agent tops out; "at least" means some games reached the piece cap, so the real mean is longer.'
  );
}

export function PeopleTable({ data, agent, preset, setPreset, presets, icons = true }: Props) {
  const minutes = data.marathon.minutes;
  return (
    <div className="people">
      <Segmented
        label="Difficulty"
        className="seg--difficulty"
        value={preset}
        onChange={setPreset}
        options={presets.map((p) => ({ value: p, label: p }))}
      />
      <p className="people__intro">{peopleIntro(data, preset)}</p>
      <div className="table-scroll">
        <table className="people-table">
          <thead>
            <tr>
              <th scope="col" rowSpan={2}>
                Agent
              </th>
              <th scope="colgroup" colSpan={minutes.length} className="num">
                Score after
              </th>
              <th scope="colgroup" colSpan={minutes.length} className="num">
                Lines after
              </th>
              <th scope="col" rowSpan={2} className="num">
                Time to top-out
              </th>
              <th scope="col" rowSpan={2}>
                Your record
              </th>
            </tr>
            <tr>
              {minutes.map((m) => (
                <th key={`s${m}`} scope="col" className="num">
                  {m} min
                </th>
              ))}
              {minutes.map((m) => (
                <th key={`l${m}`} scope="col" className="num">
                  {m} min
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.marathon.rows.map((row) => {
              const a = agent(row.key);
              const cell = row.presets[preset];
              const record = data.records[row.key]?.[preset]?.text ?? "no games yet";
              const name = (
                <th scope="row" className="people-table__agent">
                  {icons && a ? <PieceIcon agent={a} cell={12} /> : null}
                  {!icons && a ? <span className="piece-letter">{a.looks.terminal.letter}</span> : null}
                  <span>{a?.name ?? row.key}</span>
                </th>
              );
              if (row.status !== "valid" || !cell) {
                const text =
                  row.status === "pending_rerun"
                    ? "Pending rerun"
                    : row.status === "valid"
                      ? `Not evaluated at ${preset} yet`
                      : "Not evaluated yet";
                return (
                  <tr key={row.key} className="is-pending">
                    {name}
                    <td colSpan={minutes.length * 2 + 1} className="people-table__pending">
                      {text}
                    </td>
                    <td>{record}</td>
                  </tr>
                );
              }
              return (
                <tr key={row.key}>
                  {name}
                  {minutes.map((m) => (
                    <td key={`s${m}`} className="num">
                      {cell.score[String(m)] != null ? fmtInt(cell.score[String(m)]!) : "-"}
                    </td>
                  ))}
                  {minutes.map((m) => (
                    <td key={`l${m}`} className="num">
                      {cell.lines[String(m)] != null ? fmtInt(cell.lines[String(m)]!) : "-"}
                    </td>
                  ))}
                  <td className="num">
                    {cell.time_to_top_out != null ? `${cell.time_is_lower_bound ? "at least " : ""}${clock(cell.time_to_top_out)}` : "-"}
                  </td>
                  <td>{record}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** Footnote lines explaining each pending result. */
export function pendingNotes(data: LeaderboardResponse, agent: (key: string) => Agent | null): string[] {
  return data.survival.rows
    .filter((r) => r.status === "pending_rerun")
    .map((r) => `${agent(r.key)?.name ?? r.key}: pending a rerun because ${r.reason}.`);
}
