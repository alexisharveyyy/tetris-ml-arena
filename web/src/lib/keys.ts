// Key names follow the settings file (Pygame names), so bindings are shared with the desktop app.
import type { Action } from "../api/types";

const CODE_NAMES: Record<string, string> = {
  ArrowLeft: "left",
  ArrowRight: "right",
  ArrowUp: "up",
  ArrowDown: "down",
  Space: "space",
  ShiftLeft: "left shift",
  ShiftRight: "right shift",
  ControlLeft: "left ctrl",
  ControlRight: "right ctrl",
  AltLeft: "left alt",
  AltRight: "right alt",
  MetaLeft: "left meta",
  MetaRight: "right meta",
  Escape: "escape",
  Enter: "return",
  NumpadEnter: "enter",
  Tab: "tab",
  Backspace: "backspace",
  Comma: ",",
  Period: ".",
  Slash: "/",
  Semicolon: ";",
  Quote: "'",
  BracketLeft: "[",
  BracketRight: "]",
  Minus: "-",
  Equal: "=",
};

/** Settings key name for a keyboard event. */
export function keyName(e: Pick<KeyboardEvent, "code" | "key">): string {
  if (CODE_NAMES[e.code]) return CODE_NAMES[e.code];
  if (e.code.startsWith("Key")) return e.code.slice(3).toLowerCase();
  if (e.code.startsWith("Digit")) return e.code.slice(5);
  if (e.code.startsWith("Numpad")) return `[${e.code.slice(6).toLowerCase()}]`;
  if (/^F\d+$/.test(e.code)) return e.code.toLowerCase();
  return e.key.toLowerCase();
}

const LABELS: Record<string, string> = {
  left: "←",
  right: "→",
  up: "↑",
  down: "↓",
  space: "Space",
  escape: "Esc",
  return: "Enter",
  "left shift": "Shift",
  "right shift": "Right Shift",
};

/** How a key is shown on a key cap. */
export function keyLabel(name: string): string {
  return LABELS[name] ?? (name.length === 1 ? name.toUpperCase() : name.replace(/\b\w/g, (c) => c.toUpperCase()));
}

export const GAME_ACTIONS: Action[] = ["left", "right", "soft", "hard", "cw", "ccw", "hold"];

export function keymap(bindings: Record<string, string[]>): Map<string, string> {
  const map = new Map<string, string>();
  for (const [action, keys] of Object.entries(bindings)) for (const k of keys) map.set(k, action);
  return map;
}

/** Ignore key events typed into form fields. */
export function fromFormField(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  return !!t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA" || t.isContentEditable);
}
