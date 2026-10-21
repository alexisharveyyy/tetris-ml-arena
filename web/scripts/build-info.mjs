// Fingerprint of everything the frontend build is made from. The build writes
// it to build-info.json next to index.html; tetris/web/build.py computes the
// same value to tell whether the committed build matches web/src.
//
// hash = sha256 over sorted relative paths, one line per file:
//   "<path>\0<sha256 of the file>\n"
import { createHash } from "node:crypto";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";

export const INPUTS = ["index.html", "package.json", "package-lock.json", "tsconfig.json", "vite.config.ts", "scripts/build-info.mjs", "src"];
const SKIP = new Set([".DS_Store"]);

function walk(root, path, out) {
  const full = join(root, path);
  if (statSync(full).isDirectory()) {
    for (const name of readdirSync(full)) if (!SKIP.has(name)) walk(root, join(path, name), out);
  } else {
    out.push(relative(root, full).split(sep).join("/"));
  }
}

export function sourceHash(root) {
  const files = [];
  for (const input of INPUTS) walk(root, input, files);
  files.sort();
  const h = createHash("sha256");
  for (const f of files) {
    const digest = createHash("sha256").update(readFileSync(join(root, f))).digest("hex");
    h.update(`${f}\0${digest}\n`);
  }
  return { hash: h.digest("hex"), files: files.length };
}
