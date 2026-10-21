// Screenshots of every screen in every look, saved to reports/ui/.
//   npx playwright test                     all looks and screens
//   LOOKS=tetromino SCREENS=home npx playwright test
import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { BASE, mock, SHOTS } from "./mock";

const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "reports", "ui");
const LETTER: Record<string, string> = { tetromino: "A", terminal: "B", broadcast: "C", poster: "D" };
const WIDTH = Number(process.env.WIDTH ?? 1440);
const HEIGHT = Number(process.env.HEIGHT ?? 900);
const LOOKS = (process.env.LOOKS ?? "tetromino,terminal,broadcast,poster").split(",");
const SCREENS = (process.env.SCREENS ?? "home,opponents,match,results,leaderboard,people,comparisons,watch,solo,record,settings").split(",");

mkdirSync(OUT, { recursive: true });

for (const look of LOOKS) {
  for (const shot of SHOTS.filter((s) => SCREENS.includes(s.name))) {
    test(`${look} ${shot.name}`, async ({ page }) => {
      await page.setViewportSize({ width: WIDTH, height: HEIGHT });
      await mock(page, look, shot.name);
      await page.goto(BASE + shot.hash);
      await expect(page.locator(shot.ready).first()).toBeVisible();
      await page.evaluate(() => document.fonts.ready);
      await page.waitForTimeout(250);
      // Play screens size their boards to the window (src/lib/fit.tsx).
      await page.waitForFunction(() => document.documentElement.dataset.fit !== "searching");
      const suffix = WIDTH === 1440 ? "" : `-${WIDTH}x${HEIGHT}`;
      await page.screenshot({ path: join(OUT, `${LETTER[look]}${shot.file}${suffix}.png`), fullPage: true });
    });
  }
}
