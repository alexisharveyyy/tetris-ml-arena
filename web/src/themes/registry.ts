// Every folder under themes/ with an index.ts exporting `theme` is a look.
import type { LookId } from "../api/types";
import type { Layouts, Theme } from "./types";

const modules = import.meta.glob<{ theme: Theme }>("./*/index.ts", { eager: true });

export const DEFAULT_LOOK: LookId = "tetromino";

// Built on first use: theme modules import shared code that imports this
// registry, so reading them while modules are still loading would miss some.
let cache: { list: Theme[]; byId: Map<LookId, Theme> } | null = null;

function registry() {
  if (!cache) {
    const list = Object.values(modules)
      .map((m) => m.theme)
      .sort((a, b) => a.order - b.order);
    cache = { list, byId: new Map(list.map((t) => [t.id, t])) };
  }
  return cache;
}

/** Every look, in switcher order. */
export function allThemes(): Theme[] {
  return registry().list;
}

export function getTheme(id: string | null | undefined): Theme {
  const { byId } = registry();
  return byId.get(id as LookId) ?? byId.get(DEFAULT_LOOK)!;
}

export function isLook(id: unknown): id is LookId {
  return typeof id === "string" && registry().byId.has(id as LookId);
}

export function layout<K extends keyof Layouts>(theme: Theme, name: K): Layouts[K] {
  return (theme.layouts[name] ?? getTheme(DEFAULT_LOOK).layouts[name])!;
}
