/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The production build goes where the FastAPI server serves it from.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "../tetris/web/static", emptyOutDir: true, assetsInlineLimit: 0 },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8765",
      "/ws": { target: "ws://127.0.0.1:8765", ws: true },
    },
  },
  test: { environment: "jsdom", include: ["tests/unit/**/*.test.{ts,tsx}"] },
});
