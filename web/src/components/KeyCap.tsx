import { keyLabel } from "../lib/keys";

/** A key drawn as a key cap. ``name`` is a settings key name or a literal label. */
export function KeyCap({ name }: { name: string }) {
  return <kbd className="keycap">{keyLabel(name)}</kbd>;
}
