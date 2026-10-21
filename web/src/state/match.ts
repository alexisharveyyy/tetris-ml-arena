// Live match hooks: the socket, the latest snapshot (at most one render per
// animation frame) and keyboard input mapped through the user's bindings.
import { useEffect, useRef, useState } from "react";
import { MatchSocket, type SocketStatus } from "../api/socket";
import type { Action, FeedEvent, Snapshot } from "../api/types";
import { fromFormField, GAME_ACTIONS, keyName, keymap } from "../lib/keys";

export interface MatchSession {
  socket: MatchSocket | null;
  snap: Snapshot | null;
  status: SocketStatus;
}

export function useMatchSession(matchId: string | null, onEvent?: (e: FeedEvent) => void): MatchSession {
  const [socket, setSocket] = useState<MatchSocket | null>(null);
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [status, setStatus] = useState<SocketStatus>("connecting");
  const eventRef = useRef(onEvent);
  eventRef.current = onEvent;

  useEffect(() => {
    if (!matchId) return;
    const s = new MatchSocket(matchId);
    setSocket(s);
    let frame = 0;
    const offSnap = s.onSnapshot(() => {
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        setSnap(s.latest);
      });
    });
    const offEvent = s.onEvent((e) => eventRef.current?.(e));
    const offStatus = s.onStatus(setStatus);
    return () => {
      offSnap();
      offEvent();
      offStatus();
      if (frame) cancelAnimationFrame(frame);
      s.close();
      setSocket(null);
      setSnap(null);
    };
  }, [matchId]);

  return { socket, snap, status };
}

interface KeyOptions {
  enabled: boolean;
  onPause?: () => void;
}

/** Sends game keys to the server. Browser key repeat is ignored: DAS and ARR run server side. */
export function useGameKeys(socket: MatchSocket | null, bindings: Record<string, string[]> | undefined, opts: KeyOptions): void {
  const optsRef = useRef(opts);
  optsRef.current = opts;
  useEffect(() => {
    if (!socket || !bindings) return;
    const map = keymap(bindings);
    const held = new Set<Action>();
    const down = (e: KeyboardEvent) => {
      if (fromFormField(e)) return;
      const action = map.get(keyName(e));
      if (!action) return;
      if (action === "pause") {
        if (!e.repeat) optsRef.current.onPause?.();
        e.preventDefault();
        return;
      }
      if (!optsRef.current.enabled || !GAME_ACTIONS.includes(action as Action)) return;
      e.preventDefault();
      if (e.repeat || held.has(action as Action)) return;
      held.add(action as Action);
      socket.input(action as Action, true);
    };
    const up = (e: KeyboardEvent) => {
      const action = map.get(keyName(e)) as Action | undefined;
      if (!action || !held.has(action)) return;
      held.delete(action);
      socket.input(action, false);
    };
    const blur = () => {
      for (const a of held) socket.input(a, false);
      held.clear();
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
      blur();
    };
  }, [socket, bindings]);
}

/** Single-key shortcuts (R, N, W ...) outside of play. */
export function useShortcuts(map: Record<string, () => void>, enabled = true): void {
  const ref = useRef(map);
  ref.current = map;
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (fromFormField(e) || e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
      const name = keyName(e);
      const tag = (e.target as HTMLElement | null)?.tagName;
      // Enter and Space on a focused control belong to that control.
      if ((name === "return" || name === "space") && (tag === "A" || tag === "BUTTON")) return;
      const fn = ref.current[name];
      if (fn) {
        e.preventDefault();
        fn();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [enabled]);
}
