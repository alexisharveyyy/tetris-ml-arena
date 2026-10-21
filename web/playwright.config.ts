import { defineConfig } from "@playwright/test";

// Screenshot runs against the production build (npm run build), with every API
// call and the WebSocket mocked from tests/screens/fixtures.ts. No game server
// or agent process is started.
export default defineConfig({
  testDir: "tests/screens",
  workers: 1,
  timeout: 30_000,
  reporter: [["list"]],
  use: { browserName: "chromium", viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 },
  webServer: {
    command: "npx vite preview --host 127.0.0.1 --port 4173 --strictPort",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: true,
  },
});
