// Solo practice and Record your games: one human board, no agents.
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { MatchInfo, RecordingTotals } from "../api/types";
import { Button } from "../components/Button";
import { ConnectionBanner, MatchState } from "../components/Overlays";
import { HumanPanel } from "../components/Panels";
import { Segmented } from "../components/Segmented";
import { clock, fmtInt } from "../lib/format";
import { go } from "../router";
import { useApp } from "../state/app";
import { useGameKeys, useMatchSession, useShortcuts } from "../state/match";
import { PageHead } from "./PageHead";

function useSoloSession(matchId: string) {
  const app = useApp();
  const session = useMatchSession(matchId);
  const { socket, snap } = session;
  const togglePause = () => {
    const s = socket?.latest;
    if (s && !s.finished && !s.disconnected) socket!.pause(!s.paused);
  };
  const live = !!snap && !snap.paused && !snap.finished && snap.countdown <= 0;
  useGameKeys(socket, app.settings?.bindings, { enabled: live, onPause: togglePause });
  return { ...session, togglePause };
}

export function SoloScreen({ matchId }: { matchId: string | null }) {
  useEffect(() => {
    if (matchId) return;
    api.startMatch({ mode: "solo" }).then((info) => go(`/solo/${info.match}`)).catch(() => undefined);
  }, [matchId]);
  if (!matchId) return <main className="screen"><p>Starting...</p></main>;
  return <SoloLive matchId={matchId} />;
}

function SoloLive({ matchId }: { matchId: string }) {
  const { snap, status, togglePause } = useSoloSession(matchId);
  const again = () => void api.rematch(matchId, true);
  useShortcuts({ r: again }, !!snap?.finished);
  useEffect(() => () => void api.closeMatch(matchId).catch(() => undefined), [matchId]);
  const p = snap?.players[0];
  return (
    <main className="screen solo">
      <ConnectionBanner status={status} />
      <PageHead title="Solo practice">
        <span className="match__timer match__timer--small">{snap ? clock(snap.time) : "0:00"}</span>
        <Button variant="secondary" onClick={togglePause}>
          Pause (Esc)
        </Button>
      </PageHead>
      {p ? (
        <div className="solo__body">
          <HumanPanel player={p} cell={31} />
          {snap!.finished ? (
            <section className="game-over" role="status">
              <h2>Game over</h2>
              <p>
                {fmtInt(p.stats.score)} points, {p.stats.lines} lines, level {p.stats.level}, {p.stats.pps.toFixed(2)} pieces per second.
              </p>
              <Button onClick={again} shortcut="r">
                Play again
              </Button>
              <Button variant="link" href="#/">
                Home
              </Button>
            </section>
          ) : null}
        </div>
      ) : null}
      {snap ? <MatchState snap={snap} onResume={togglePause} /> : null}
    </main>
  );
}

const SETTING_TEXT: Record<string, string> = {
  plain: "Plain: a normal game with level gravity and no garbage.",
  garbage: "Garbage: a garbage row arrives every three pieces, as in the agents' training.",
};

export function RecordScreen({ matchId }: { matchId: string | null }) {
  return matchId ? <RecordLive matchId={matchId} /> : <RecordSetup />;
}

function Totals({ totals }: { totals: RecordingTotals | null }) {
  if (!totals) return null;
  return (
    <p className="record__totals">
      Saved so far: {totals.games} game{totals.games === 1 ? "" : "s"}, {fmtInt(totals.on_space)} usable placements
      {totals.off_space ? `, ${fmtInt(totals.off_space)} tucks or spins the cloning agent skips` : ""}.
    </p>
  );
}

function RecordSetup() {
  const [setting, setSetting] = useState<"plain" | "garbage">("plain");
  const [totals, setTotals] = useState<RecordingTotals | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.recordings().then((r) => setTotals(r.totals)).catch(() => undefined);
  }, []);
  const start = () =>
    api
      .startMatch({ mode: "record", record_setting: setting })
      .then((info) => go(`/record/${info.match}`))
      .catch((e: Error) => setError(e.message));
  return (
    <main className="screen record">
      <PageHead title="Record your games" />
      <p className="lead">
        Play a normal game. Every piece you place is saved with the board it went on, as training data for the behavior cloning
        agent fine-tuned on your play, BC (you). Nothing is saved until you choose to keep the game.
      </p>
      <div className="form-row">
        <span className="form-row__label">Setting</span>
        <Segmented
          label="Recording setting"
          value={setting}
          onChange={setSetting}
          options={[
            { value: "plain", label: "Plain" },
            { value: "garbage", label: "Garbage" },
          ]}
        />
      </div>
      <p className="hint">{SETTING_TEXT[setting]}</p>
      <Totals totals={totals} />
      {error ? (
        <p className="form-error" role="alert">
          {error}
        </p>
      ) : null}
      <Button onClick={start} shortcut="return">
        Start recording
      </Button>
    </main>
  );
}

function RecordLive({ matchId }: { matchId: string }) {
  const { snap, status, togglePause } = useSoloSession(matchId);
  const [info, setInfo] = useState<MatchInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const finished = !!snap?.finished;
  useEffect(() => {
    if (finished) api.match(matchId).then(setInfo).catch(() => undefined);
  }, [finished, matchId]);
  useEffect(() => () => void api.closeMatch(matchId).catch(() => undefined), [matchId]);
  const save = () => {
    setBusy(true);
    api
      .saveRecording(matchId)
      .then(setInfo)
      .finally(() => setBusy(false));
  };
  const again = () => {
    setInfo(null);
    void api.rematch(matchId, true);
  };
  const saved = info?.recording?.saved;
  useShortcuts({ s: () => !saved && save(), d: again, return: again }, finished && !busy);
  const p = snap?.players[0];
  return (
    <main className="screen record">
      <ConnectionBanner status={status} />
      <PageHead title="Recording" back="#/record" backLabel="Stop">
        <span className="match__timer match__timer--small">{snap ? clock(snap.time) : "0:00"}</span>
        <Button variant="secondary" onClick={togglePause}>
          Pause (Esc)
        </Button>
      </PageHead>
      {p ? (
        <div className="solo__body">
          <HumanPanel player={p} cell={31} title="Recording" />
          {finished ? (
            <section className="game-over" role="status">
              <h2>Keep this game?</h2>
              <p>
                {fmtInt(p.stats.score)} points, {p.stats.lines} lines.{" "}
                {info?.recording
                  ? `${info.recording.on_space} placements the cloning agent can learn from${
                      info.recording.off_space ? `, ${info.recording.off_space} tucks or spins it skips` : ""
                    }.`
                  : null}
              </p>
              {saved ? (
                <>
                  <p>Saved to {saved}.</p>
                  <Totals totals={info?.totals ?? null} />
                  <Button onClick={again} shortcut="return">
                    Record another
                  </Button>
                </>
              ) : (
                <>
                  <Button onClick={save} shortcut="s" disabled={busy}>
                    Save it
                  </Button>
                  <Button variant="secondary" onClick={again} shortcut="d">
                    Discard and play again
                  </Button>
                </>
              )}
            </section>
          ) : null}
        </div>
      ) : null}
      {snap ? <MatchState snap={snap} onResume={togglePause} /> : null}
    </main>
  );
}
