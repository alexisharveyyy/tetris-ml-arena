// Keyboard-only run through Home, Choose opponents, Match and Results, against
// a running server (no mocks), in every look. Every focused control must show
// a visible focus ring.
//   LIVE=1 npx playwright test keyboard
import { expect, test, type Page } from "@playwright/test";
import { LOOK_IDS } from "./mock";

const LIVE = process.env.LIVE_URL;
const LOOKS = (process.env.LOOKS ?? LOOK_IDS.join(",")).split(",");

test.skip(!LIVE, "set LIVE=1 to start a server with temporary state, or LIVE_URL");
test.setTimeout(90_000);

/** Visible focus ring: an outline or box-shadow of at least 2px. */
async function focusRing(page: Page): Promise<string | null> {
  return page.evaluate(() => {
    const el = document.activeElement as HTMLElement | null;
    if (!el || el === document.body) return "nothing focused";
    const cs = getComputedStyle(el);
    const outline = cs.outlineStyle !== "none" && parseFloat(cs.outlineWidth) >= 2;
    const shadow = cs.boxShadow !== "none";
    return outline || shadow ? null : `${el.tagName.toLowerCase()} "${el.textContent?.trim().slice(0, 30)}" has no focus ring`;
  });
}

/** Press Tab until the focused element's accessible text matches, checking each ring. */
async function tabTo(page: Page, name: RegExp, problems: string[], max = 60): Promise<void> {
  for (let i = 0; i < max; i++) {
    await page.keyboard.press("Tab");
    const ring = await focusRing(page);
    if (ring) problems.push(`${ring} (Tab ${i + 1} toward ${name}, at ${new URL(page.url()).hash})`);
    const text = await page.evaluate(() => {
      const el = document.activeElement as HTMLElement | null;
      return el ? `${el.getAttribute("aria-label") ?? ""} ${el.textContent ?? ""}`.trim() : "";
    });
    if (name.test(text)) return;
  }
  throw new Error(`Tab never reached ${name}`);
}

for (const look of LOOKS) {
  test(`keyboard only: home, opponents, match, results (${look})`, async ({ page }) => {
    const problems: string[] = [];
    await page.addInitScript((l) => {
      localStorage.setItem("arena.look", l);
      sessionStorage.setItem("arena.demo.started", "1");
    }, look);
    await page.goto(`${LIVE}/#/`);
    await expect(page.locator("h1").first()).toBeVisible();

    await tabTo(page, /marathon/i, problems);
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/#\/play\/marathon/);
    await expect(page.locator("section[aria-label=Agents]")).toBeVisible();
    // The demo opponents fill the slots once /api/demo answers; until then
    // Start is disabled and Tab skips it.
    await expect(page.getByRole("button", { name: /^start marathon/i })).toBeEnabled();

    await tabTo(page, /^start marathon/i, problems);
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/#\/match\//, { timeout: 30_000 });
    const board = page.locator("canvas.board").first();
    await expect(board).toBeVisible();

    // Wait out the countdown and agent loading, then play a few pieces.
    const label = async () => (await board.getAttribute("aria-label")) ?? "";
    await page.waitForTimeout(4500);
    const before = await label();
    for (let i = 0; i < 3; i++) {
      await page.keyboard.press("ArrowLeft");
      await page.keyboard.press("ArrowUp");
      await page.keyboard.press("Space");
    }
    await expect.poll(label, { timeout: 10_000 }).not.toBe(before);

    // Pause with Esc, then end the match from the pause menu.
    await page.keyboard.press("Escape");
    await expect(page.getByText("Paused")).toBeVisible();
    // Pausing moves focus to Resume; End the match is the next stop.
    await expect(page.getByRole("button", { name: /resume/i })).toBeFocused();
    await tabTo(page, /end the match/i, problems, 3);
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/#\/results\//, { timeout: 10_000 });
    // Results shows its heading before the match record loads; tab once the
    // actions are there, or a press can land on the browser bar.
    await expect(page.getByRole("button", { name: /rematch/i })).toBeVisible();
    await tabTo(page, /rematch/i, problems, 30);

    expect(problems, problems.join("; ")).toEqual([]);
  });
}

// Demo keys, against a server started with --demo: the demo opponents are
// preselected, F10 switches looks mid-match without breaking play, and F9
// skips to the results.
test("demo mode: preselected opponents, F10 mid-match, F9 to results", async ({ page, request }) => {
  const demo = await (await request.get(`${LIVE}/api/demo`)).json();
  test.skip(!demo.enabled, "server not started with --demo");
  await page.goto(`${LIVE}/#/play/marathon`);
  const slots = page.locator("section[aria-label=Agents]");
  await expect(slots).toBeVisible();
  const startLabel = new RegExp(`^start marathon.*${demo.opponents.length}`, "i");
  await expect(page.getByRole("button", { name: startLabel })).toBeVisible();
  await page.getByRole("button", { name: startLabel }).press("Enter");
  await expect(page).toHaveURL(/#\/match\//, { timeout: 30_000 });
  const board = page.locator("canvas.board").first();
  await expect(board).toBeVisible();
  await page.waitForTimeout(4500);

  const looks: string[] = [];
  for (let i = 0; i < 4; i++) {
    looks.push((await page.evaluate(() => document.documentElement.dataset.look)) ?? "");
    const before = (await board.getAttribute("aria-label")) ?? "";
    await page.keyboard.press("ArrowLeft");
    await page.keyboard.press("Space");
    await expect.poll(async () => (await page.locator("canvas.board").first().getAttribute("aria-label")) ?? "", { timeout: 5000 }).not.toBe(before);
    await page.keyboard.press("F10");
  }
  expect(new Set(looks).size).toBe(4);
  await expect(page.getByText(`seed ${demo.seed}`, { exact: false }).first()).toBeVisible();

  await page.keyboard.press("F9");
  await expect(page).toHaveURL(/#\/results\//, { timeout: 10_000 });
});
