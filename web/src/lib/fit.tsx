// Play screens (Match, Solo, Watch) must fit the window: you cannot scroll
// while playing. FitToViewport finds the largest factor (0.4 to 1) for board
// and piece-preview cell sizes at which the page fits, by binary search with
// one render per step. Text keeps its size, so the 18px floor holds. The
// search runs again from full size when the window is resized or the web fonts
// finish loading (fallback fonts have other widths), and from the current size
// when the page outgrows the window during play (a longer status line, say).
import { createContext, useContext, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";

const MIN_SCALE = 0.4;
const STEPS = 7;
const FitCtx = createContext(1);

/** Factor to apply to board and preview cell sizes on play screens. */
export function useFitScale(): number {
  return useContext(FitCtx);
}

/** Cell size after the play-screen fit, at least ``min`` px. */
export function useFitCell(cell: number, min = 4): number {
  return Math.max(min, Math.round(cell * useFitScale()));
}

function fits(): boolean {
  const el = document.documentElement;
  return el.scrollHeight <= window.innerHeight + 1 && el.scrollWidth <= window.innerWidth + 1;
}

interface Search {
  lo: number; // largest scale known to fit
  hi: number; // smallest scale known not to fit
  step: number;
}

export function FitToViewport({ children }: { children: ReactNode }) {
  const [scale, setScale] = useState(1);
  const search = useRef<Search | null>(null);
  const scaleRef = useRef(scale);
  scaleRef.current = scale;

  const check = useRef<() => void>(() => undefined);
  check.current = () => {
    const s = search.current;
    const ok = fits();
    if (!s) {
      // Not searching: start one if the page has outgrown the window.
      if (ok || scaleRef.current <= MIN_SCALE) return;
      search.current = { lo: MIN_SCALE, hi: scaleRef.current, step: 0 };
    } else {
      if (ok) s.lo = scaleRef.current;
      else s.hi = scaleRef.current;
      s.step += 1;
    }
    const cur = search.current!;
    const mid = Math.floor(((cur.lo + cur.hi) / 2) * 100) / 100;
    // Done when the steps run out or the midpoint cannot differ from a bound
    // (setting the same scale again would not render, and the search would stall).
    if (cur.step >= STEPS || mid <= cur.lo || mid >= cur.hi) {
      search.current = null;
      if (scaleRef.current !== cur.lo) setScale(cur.lo);
      return;
    }
    setScale(mid);
  };

  // After each render of this component (each search step) and whenever the
  // page changes size (new snapshots re-render the screen below, not this).
  useLayoutEffect(() => {
    check.current();
    // For tests: the chosen scale, and whether a search is still running.
    document.documentElement.dataset.fitScale = String(scale);
    document.documentElement.dataset.fit = search.current ? "searching" : "done";
  }, [scale]);
  useEffect(() => {
    const idle = () => {
      if (!search.current) check.current();
    };
    const ro = new ResizeObserver(idle);
    // The app root grows with its content; <body> may not.
    ro.observe(document.getElementById("root") ?? document.body);
    const timer = setInterval(idle, 1000);
    const reset = () => {
      search.current = null;
      if (scaleRef.current === 1) check.current();
      else setScale(1);
    };
    window.addEventListener("resize", reset);
    // fonts.ready can resolve before the web fonts have started loading, so
    // listen for each batch of loaded fonts as well.
    let live = true;
    document.fonts?.ready.then(() => live && reset());
    document.fonts?.addEventListener("loadingdone", reset);
    return () => {
      live = false;
      document.fonts?.removeEventListener("loadingdone", reset);
      ro.disconnect();
      clearInterval(timer);
      window.removeEventListener("resize", reset);
    };
  }, []);

  return <FitCtx.Provider value={scale}>{children}</FitCtx.Provider>;
}
