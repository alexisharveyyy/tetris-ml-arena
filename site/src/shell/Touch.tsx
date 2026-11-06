// On-screen controls for touch screens, shown on play screens with a human
// board. Holding a button repeats like a held key (DAS and ARR apply).
import { useEffect, useState, type PointerEvent } from "react";
import type { Action } from "../../../web/src/api/types";
import { activeSocket } from "../local/socket";

const BUTTONS: { action: Action | "pause"; label: string; aria: string }[] = [
  { action: "hold", label: "Hold", aria: "Hold" },
  { action: "ccw", label: "Rotate L", aria: "Rotate counterclockwise" },
  { action: "cw", label: "Rotate R", aria: "Rotate clockwise" },
  { action: "pause", label: "Pause", aria: "Pause" },
  { action: "left", label: "Left", aria: "Move left" },
  { action: "soft", label: "Down", aria: "Soft drop" },
  { action: "right", label: "Right", aria: "Move right" },
  { action: "hard", label: "Drop", aria: "Hard drop" },
];

export function useCoarsePointer(): boolean {
  const query = "(pointer: coarse)";
  const touch = () => (typeof matchMedia !== "undefined" && matchMedia(query).matches) || navigator.maxTouchPoints > 0;
  const [coarse, setCoarse] = useState(touch);
  useEffect(() => {
    const m = matchMedia(query);
    const on = () => setCoarse(touch());
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);
  return coarse;
}

function press(action: Action | "pause", down: boolean): void {
  if (action === "pause") {
    // The play screen owns pausing; send it the pause key.
    if (down) window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", code: "Escape" }));
    return;
  }
  activeSocket()?.input(action, down);
}

export function TouchControls() {
  const down = (a: Action | "pause") => (e: PointerEvent<HTMLButtonElement>) => {
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    press(a, true);
  };
  const up = (a: Action | "pause") => (e: PointerEvent<HTMLButtonElement>) => {
    e.preventDefault();
    press(a, false);
  };
  return (
    <div className="touch" role="group" aria-label="Game controls">
      {BUTTONS.map((b) => (
        <button
          key={b.action}
          type="button"
          className={`touch__btn touch__btn--${b.action}`}
          aria-label={b.aria}
          onPointerDown={down(b.action)}
          onPointerUp={up(b.action)}
          onPointerCancel={up(b.action)}
          onContextMenu={(e) => e.preventDefault()}
        >
          {b.label}
        </button>
      ))}
    </div>
  );
}
