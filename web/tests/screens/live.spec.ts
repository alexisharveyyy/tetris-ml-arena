// Live check against a running server (no mocks): Solo practice end to end.
// Runs only with LIVE=1 or LIVE_URL set (see playwright.config.ts), for example
//   LIVE=1 TETRIS_CHECKPOINT_DIR=~/tetris-ml-arena/checkpoints npx playwright test live
import { expect, test } from "@playwright/test";

const LIVE = process.env.LIVE_URL;

test.skip(!LIVE, "set LIVE=1 to start a server with temporary state, or LIVE_URL");

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
