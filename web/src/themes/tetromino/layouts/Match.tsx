import { Button } from "../../../components/Button";
import { ConnectionBanner, MatchState } from "../../../components/Overlays";
import { AgentCard, HumanPanel } from "../../../components/Panels";
import { PieceIcon } from "../../../components/PieceIcon";
import { fmtInt, ordinal } from "../../../lib/format";
import type { MatchModel } from "../../../screens/models";

export function SpectatingBar({ m }: { m: MatchModel }) {
  if (!m.snap.spectating) return null;
  return (
    <div className="spectating" role="status">
      <span>The final standings are frozen. The agents keep playing.</span>
      <Button variant="secondary" href={m.resultsHref}>
        Back to results
      </Button>
    </div>
  );
}

export function TetrominoMatch(m: MatchModel) {
  const n = m.snap.players.length;
  return (
    <div className="screen match">
      <ConnectionBanner status={m.status} />
      <header className="match__head">
        <div className="match__clock">
          <span className="match__timer" role="timer" aria-label={`Time ${m.timer}`}>
            {m.timer}
          </span>
          <span className="match__mode">{m.modeLine}</span>
        </div>
        <ol className="standings" aria-label="Live standings">
          {m.ranked.map((p) => (
            <li key={p.id} className={`standing${p.kind === "human" ? " standing--you" : ""}`}>
              <span className="standing__place">{p.place}</span>
              {p.kind === "agent" ? <PieceIcon agent={m.agent(p.agent)} cell={7} /> : <span className="standing__you-mark" aria-hidden="true" />}
              <span className="standing__name">{p.kind === "human" ? "You" : p.name}</span>
              <span className="standing__score">{fmtInt(p.stats.score)}</span>
            </li>
          ))}
        </ol>
      </header>
      <div className="match__controls">
        <Button onClick={m.toggleThinking} pressed={m.showThinking}>
          {m.showThinking ? "Hide model thinking" : "Show model thinking"}
        </Button>
        <Button variant="secondary" onClick={m.togglePause}>
          Pause (Esc)
        </Button>
      </div>
      <SpectatingBar m={m} />
      <div className="match__boards">
        {m.human ? <HumanPanel player={m.human} cell={31} place={`${ordinal(m.human.place)} of ${n}`} /> : null}
        <div className="match__agents">
          {m.agents.map((p) => (
            <AgentCard key={p.id} player={p} agent={m.agent(p.agent)} cell={16} showThinking={m.showThinking} of={n} />
          ))}
        </div>
      </div>
      <MatchState snap={m.snap} onResume={m.togglePause} onEnd={m.endMatch} />
    </div>
  );
}
