// Playwright global setup: fail the run if the real settings, Marathon history
// or recordings change while it runs. Live servers started by
// playwright.config.ts keep their state in a temporary directory, so they
// never should; a server started by hand (LIVE_URL) is checked the same way.
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
/** Same files as tetris/state.py (REAL_SETTINGS, REAL_HISTORY, REAL_RECORDINGS). */
export const GUARDED = [
  join(ROOT, "configs", "user_settings.yaml"),
  join(ROOT, "configs", "user_history.json"),
  join(ROOT, "data", "recordings"),
];

function files(path: string): string[] {
  if (!existsSync(path)) return [path];
  if (!statSync(path).isDirectory()) return [path];
  return readdirSync(path, { recursive: true, withFileTypes: true })
    .filter((d) => d.isFile())
    .map((d) => join(d.parentPath, d.name))
    .sort();
}

export function fingerprint(paths: string[] = GUARDED): Map<string, string> {
  const out = new Map<string, string>();
  for (const f of paths.flatMap(files)) {
    out.set(f, existsSync(f) ? createHash("sha256").update(readFileSync(f)).digest("hex") : "missing");
  }
  return out;
}

export function changed(before: Map<string, string>, after: Map<string, string>): string[] {
  return [...new Set([...before.keys(), ...after.keys()])].filter((k) => before.get(k) !== after.get(k)).sort();
}

export default function setup() {
  const before = fingerprint();
  return () => {
    const touched = changed(before, fingerprint());
    if (touched.length) throw new Error(`Tests changed the real user state files: ${touched.join(", ")}`);
  };
}
