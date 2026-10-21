// Live check against a running server (no mocks): Solo practice end to end.
// Runs only with LIVE_URL set, for example
//   LIVE_URL=http://127.0.0.1:8799 npx playwright test live
import { expect, test } from "@playwright/test";

const LIVE = process.env.LIVE_URL;

test.skip(!LIVE, "set LIVE_URL to run against a real server");

test("solo practice plays over the real WebSocket", async ({ page }) => {
  await page.goto(`${LIVE}/#/solo`);
  const board = page.locator("canvas.board").first();
  await expect(board).toBeVisible();
  const pieces = async () => {
    const label = (await board.getAttribute("aria-label")) ?? "";
    return label;
  };
  const before = await pieces();
  for (let i = 0; i < 4; i++) {
    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("Space");
  }
  await expect.poll(pieces, { timeout: 5000 }).not.toBe(before);
  await page.keyboard.press("Escape");
  await expect(page.getByText("Paused")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByText("Paused")).toBeHidden();
});
