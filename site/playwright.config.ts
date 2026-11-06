import { defineConfig, devices } from "@playwright/test";

// Browser checks of the built site (tests/e2e). SITE_URL selects the target;
// without it, a local preview of the production build is started.
const local = !process.env.SITE_URL;

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 120_000,
  workers: 1,
  reporter: [["list"]],
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } },
    { name: "phone", use: { ...devices["iPhone 13"], browserName: "chromium" } },
  ],
  webServer: local
    ? { command: "npx vite preview --port 4180 --strictPort", url: "http://localhost:4180/tetris-ml-arena/", reuseExistingServer: true }
    : undefined,
});
