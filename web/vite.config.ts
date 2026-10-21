/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { copyFileSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import { sourceHash } from "./scripts/build-info.mjs";

const ROOT = dirname(fileURLToPath(import.meta.url));
const OUT = join(ROOT, "..", "tetris", "web", "static");

// Writes build-info.json (the source fingerprint the server checks at launch)
// and ships each font's Open Font License next to the built fonts.
function buildInfo(): Plugin {
  return {
    name: "arena-build-info",
    apply: "build",
    closeBundle() {
      const fonts = join(ROOT, "src", "fonts");
      for (const family of readdirSync(fonts, { withFileTypes: true })) {
        if (!family.isDirectory()) continue;
        mkdirSync(join(OUT, "fonts", family.name), { recursive: true });
        copyFileSync(join(fonts, family.name, "OFL.txt"), join(OUT, "fonts", family.name, "OFL.txt"));
      }
      const info = { ...sourceHash(ROOT), note: "Written by npm run build. tetris/web/build.py compares hash with web/src." };
      writeFileSync(join(OUT, "build-info.json"), JSON.stringify(info, null, 2) + "\n");
    },
  };
}

// The production build goes where the FastAPI server serves it from.
export default defineConfig({
  plugins: [react(), buildInfo()],
  build: { outDir: OUT, emptyOutDir: true, assetsInlineLimit: 0 },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8765",
      "/ws": { target: "ws://127.0.0.1:8765", ws: true },
    },
  },
  test: { environment: "jsdom", include: ["tests/unit/**/*.test.{ts,tsx}"] },
});
