// The LLM agent's board shows how many of its moves fell back to the GA, in every look.
//   npx playwright test model
import { expect, test } from "@playwright/test";
import { snapshot } from "./fixtures";
import { BASE, mock, SHOTS } from "./mock";

const llm = {
  ...snapshot.players[3],
  name: "LLM (novelty)",
  agent: "llm",
  status: { state: "playing", text: "Playing" },
  model: { moves: 41, fallbacks: 3, by_cause: { timeout: 2, late: 1 } },
};
snapshot.players[3] = llm as (typeof snapshot.players)[number];

const match = SHOTS.find((s) => s.name === "match")!;

for (const look of ["tetromino", "terminal", "broadcast", "poster"]) {
  test(`${look} shows the LLM fallback count`, async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await mock(page, look, "match");
    await page.goto(BASE + match.hash);
    const line = page.getByTestId("model-fallbacks");
    await expect(line).toHaveCount(1);
    await expect(line).toContainText("GA fallbacks: 3 of 41 moves");
    await expect(line).toHaveAttribute("title", "2 timed out, 1 locked while thinking");
  });
}
