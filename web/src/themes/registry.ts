// Every folder under themes/ with an index.ts exporting `theme` is a look.
import type { LookId } from "../api/types";
import type { Layouts, Theme } from "./types";

const modules = import.meta.glob<{ theme: Theme }>("./*/index.ts", { eager: true });

export const THEMES: Theme[] = Object.values(modules)
  .map((m) => m.theme)
  .sort((a, b) => a.order - b.order);

const BY_ID = new Map(THEMES.map((t) => [t.id, t]));
export const DEFAULT_LOOK: LookId = "tetromino";

export function getTheme(id: string | null | undefined): Theme {
  return BY_ID.get(id as LookId) ?? BY_ID.get(DEFAULT_LOOK)!;
}

export function isLook(id: unknown): id is LookId {
  return typeof id === "string" && BY_ID.has(id as LookId);
}

export function layout<K extends keyof Layouts>(theme: Theme, name: K): Layouts[K] {
  return (theme.layouts[name] ?? getTheme(DEFAULT_LOOK).layouts[name])!;
}
