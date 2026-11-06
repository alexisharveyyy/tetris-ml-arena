/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

const ROOT = dirname(fileURLToPath(import.meta.url));
const WEB_SRC = resolve(ROOT, "..", "web", "src");

// The web client talks to the Python server through two modules. The static
// build answers both in the page instead, so web/src stays unchanged.
const SWAPS: Record<string, string> = {
  [join(WEB_SRC, "api", "client.ts")]: join(ROOT, "src", "local", "client.ts"),
  [join(WEB_SRC, "api", "socket.ts")]: join(ROOT, "src", "local", "socket.ts"),
  // Same fit-to-window logic, with a lower floor so four boards fit a phone.
  [join(WEB_SRC, "lib", "fit.tsx")]: join(ROOT, "src", "shell", "fit.tsx"),
};

function localServer(): Plugin {
  return {
    name: "arena-local-server",
    enforce: "pre",
    async resolveId(source, importer, options) {
      if (!importer || !source.startsWith(".")) return null;
      const resolved = await this.resolve(source, importer, { ...options, skipSelf: true });
      if (resolved && SWAPS[resolved.id]) return SWAPS[resolved.id];
      return null;
    },
  };
}

export default defineConfig({
  // Served from https://alexisalineharvey.com/tetris-ml-arena/
  base: process.env.SITE_BASE ?? "/tetris-ml-arena/",
  plugins: [localServer(), react()],
  resolve: {
    // web/src imports React from its own folder; use this package's copy.
    alias: [
      { find: /^react$/, replacement: join(ROOT, "node_modules", "react") },
      { find: /^react\/(.*)$/, replacement: join(ROOT, "node_modules", "react", "$1") },
      { find: /^react-dom$/, replacement: join(ROOT, "node_modules", "react-dom") },
      { find: /^react-dom\/(.*)$/, replacement: join(ROOT, "node_modules", "react-dom", "$1") },
    ],
  },
  server: { fs: { allow: [resolve(ROOT, "..")] } },
  worker: { format: "es" },
  build: { outDir: "dist", emptyOutDir: true, assetsInlineLimit: 0, target: "es2022" },
});
