import { useEffect, useRef } from "react";
import type { SocketStatus } from "../api/socket";
import type { Snapshot } from "../api/types";
import { Button } from "./Button";

/** Plain banner while the connection is down (the server has paused the match). */
export function ConnectionBanner({ status }: { status: SocketStatus }) {
  if (status === "lost" || status === "connecting") {
    return (
      <div className="banner" role="status">
        {status === "lost" ? "Connection lost. Reconnecting..." : "Connecting..."}
      </div>
    );
  }
  if (status === "gone") {
    return (
      <div className="banner" role="alert">
        This match is no longer on the server. <a href="#/">Go home</a>
      </div>
    );
  }
  return null;
}

interface StateProps {
  snap: Snapshot;
  onResume: () => void;
  onEnd?: () => void;
}

/** Loading, countdown and pause states over the match. */
export function MatchState({ snap, onResume, onEnd }: StateProps) {
  // Pausing moves focus into the pause menu (Tab reaches End the match next);
  // resuming puts it back where it was.
  const box = useRef<HTMLDivElement>(null);
  const paused = snap.paused && !snap.loading && !snap.disconnected;
  useEffect(() => {
    if (!paused) return;
    const before = document.activeElement as HTMLElement | null;
    box.current?.querySelector<HTMLElement>("button, a")?.focus();
    return () => {
      if (before?.isConnected) before.focus();
    };
  }, [paused]);
  let body = null;
  if (snap.loading) body = <p className="overlay__title">Loading the models</p>;
  else if (snap.disconnected) body = null;
  else if (snap.paused)
    body = (
      <>
        <p className="overlay__title">Paused</p>
        <div className="overlay__actions">
          <Button onClick={onResume} shortcut="escape">
            Resume
          </Button>
          {onEnd ? (
            <Button variant="secondary" onClick={onEnd}>
              End the match
            </Button>
          ) : null}
          <Button variant="link" href="#/">
            Home
          </Button>
        </div>
      </>
    );
  else if (snap.countdown > 0)
    body = (
      <p className="overlay__title" aria-live="assertive">
        {Math.ceil(snap.countdown)}
      </p>
    );
  if (!body) return null;
  return (
    <div className="overlay" role="dialog" aria-modal="false" aria-label="Match state">
      <div className="overlay__box" ref={box}>
        {body}
      </div>
    </div>
  );
}
