// Accessibility audit of every screen in every look, at both projector sizes:
//   - axe-core WCAG 2 A/AA rules, color contrast included;
//   - rendered text size: anything a viewer reads is at least 18px; axis tick
//     labels and key caps at least 16px;
//   - the project's contrast rule, stricter than WCAG's large-text rule: 4.5:1
//     below 24px and 3:1 from 24px (axe allows 3:1 for bold text from 18.7px);
//   - every board canvas has role="img" and a label;
//   - play screens (match, watch, solo) fit the window: no scrolling, and
//     every board fully visible and not covered by anything (such as a ticker).
// Writes reports/ui/audit.json; each test fails on its own screen's problems.
//   npx playwright test audit            (LOOKS=, SCREENS=, SIZES=1280x720 to narrow)
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { BASE, LOOK_IDS, mock, SHOTS } from "./mock";

const OUT = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "reports", "ui");
const REPORT = join(OUT, "audit.json");
const LOOKS = (process.env.LOOKS ?? LOOK_IDS.join(",")).split(",");
const SCREENS = (process.env.SCREENS ?? SHOTS.map((s) => s.name).join(",")).split(",");
const SIZES = (process.env.SIZES ?? "1280x720,1920x1080").split(",").map((s) => s.split("x").map(Number));
/** Elements allowed down to 16px. */
const SMALL_OK = ".keycap, .axis-tick, .bc-tick, .po-tick, [data-small-ok]";

mkdirSync(OUT, { recursive: true });
const results: Record<string, unknown> = {};
try {
  Object.assign(results, JSON.parse(readFileSync(REPORT, "utf8")));
} catch {
  /* first run */
}

for (const [w, h] of SIZES) {
  for (const look of LOOKS) {
    for (const shot of SHOTS.filter((s) => SCREENS.includes(s.name))) {
      test(`audit ${look} ${shot.name} ${w}x${h}`, async ({ page }) => {
        await page.setViewportSize({ width: w, height: h });
        await mock(page, look, shot.name);
        await page.goto(BASE + shot.hash);
        await expect(page.locator(shot.ready).first()).toBeVisible();
        await page.evaluate(() => document.fonts.ready);
        await page.waitForTimeout(250);
        // Play screens size their boards to the window (src/lib/fit.tsx).
        await page.waitForFunction(() => document.documentElement.dataset.fit !== "searching");

        const axe = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
        const violations = axe.violations.map((v) => ({
          id: v.id,
          impact: v.impact,
          help: v.help,
          nodes: v.nodes.slice(0, 5).map((n) => ({ target: n.target.join(" "), summary: n.failureSummary })),
          count: v.nodes.length,
        }));

        const small = await page.evaluate((smallOk) => {
          const out: { text: string; px: number; min: number; where: string }[] = [];
          const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
          const seen = new Set<Element>();
          for (let n = walker.nextNode(); n; n = walker.nextNode()) {
            const el = n.parentElement;
            if (!el || seen.has(el) || !n.textContent?.trim()) continue;
            seen.add(el);
            if (el.closest("[aria-hidden=true], .sr-only, .visually-hidden, script, style")) continue;
            const cs = getComputedStyle(el);
            const r = el.getBoundingClientRect();
            if (cs.visibility !== "visible" || Number(cs.opacity) === 0 || r.width === 0 || r.height === 0) continue;
            const px = parseFloat(cs.fontSize);
            const min = el.closest(smallOk) ? 16 : 18;
            if (px + 0.01 < min) {
              const cls = typeof el.className === "string" && el.className ? "." + el.className.split(" ").join(".") : "";
              out.push({ text: n.textContent.trim().slice(0, 40), px, min, where: el.tagName.toLowerCase() + cls });
            }
          }
          return out;
        }, SMALL_OK);

        const lowContrast = await page.evaluate(() => {
          type RGBA = [number, number, number, number];
          const parse = (c: string): RGBA => {
            const m = c.match(/rgba?\(([^)]+)\)/);
            if (!m) return [0, 0, 0, 0];
            const p = m[1].split(/[ ,/]+/).filter(Boolean).map(Number);
            return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1];
          };
          const over = (top: RGBA, under: RGBA): RGBA => {
            const a = top[3] + under[3] * (1 - top[3]);
            if (a === 0) return [0, 0, 0, 0];
            const ch = (i: number) => (top[i] * top[3] + under[i] * under[3] * (1 - top[3])) / a;
            return [ch(0), ch(1), ch(2), a];
          };
          const lum = (c: RGBA) => {
            const f = (v: number) => {
              const s = v / 255;
              return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
            };
            return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2]);
          };
          // Background behind el: blend ancestor background colors until opaque.
          // Returns null when an image, gradient or canvas is involved.
          const background = (el: Element): RGBA | null => {
            const layers: RGBA[] = [];
            for (let e: Element | null = el; e; e = e.parentElement) {
              const cs = getComputedStyle(e);
              if (cs.backgroundImage !== "none") return null;
              const bg = parse(cs.backgroundColor);
              if (bg[3] > 0) layers.push(bg);
              if (bg[3] >= 1) break;
            }
            let out: RGBA = [255, 255, 255, 1];
            for (const l of layers.reverse()) out = over(l, out);
            return out;
          };
          const out: { text: string; px: number; ratio: number; need: number; fg: string; bg: string; where: string }[] = [];
          const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
          const seen = new Set<Element>();
          for (let n = walker.nextNode(); n; n = walker.nextNode()) {
            const el = n.parentElement;
            if (!el || seen.has(el) || !n.textContent?.trim()) continue;
            seen.add(el);
            if (el.closest("[aria-hidden=true], [disabled], script, style")) continue;
            const cs = getComputedStyle(el);
            const r = el.getBoundingClientRect();
            if (cs.visibility !== "visible" || r.width === 0 || r.height === 0) continue;
            const bg = background(el);
            if (!bg) continue;
            let opacity = 1;
            for (let e: Element | null = el; e; e = e.parentElement) opacity *= Number(getComputedStyle(e).opacity);
            const fg0 = parse(cs.color);
            const fg = over([fg0[0], fg0[1], fg0[2], fg0[3] * opacity], bg);
            const [hi, lo] = [lum(fg), lum(bg)].sort((a, b) => b - a);
            const ratio = (hi + 0.05) / (lo + 0.05);
            const px = parseFloat(cs.fontSize);
            const need = px >= 24 ? 3 : 4.5;
            if (ratio + 0.005 < need) {
              const cls = typeof el.className === "string" && el.className ? "." + el.className.split(" ").join(".") : "";
              const hex = (c: RGBA) => "#" + c.slice(0, 3).map((v) => Math.round(v).toString(16).padStart(2, "0")).join("");
              out.push({ text: n.textContent.trim().slice(0, 40), px, ratio: Math.round(ratio * 100) / 100, need, fg: hex(fg), bg: hex(bg), where: el.tagName.toLowerCase() + cls });
            }
          }
          return out;
        });

        const boards = await page.evaluate(() =>
          [...document.querySelectorAll("canvas")].map((c) => ({
            role: c.getAttribute("role"),
            label: c.getAttribute("aria-label") ?? "",
          })),
        );
        const badBoards = boards.filter((b) => b.role !== "img" || !b.label.trim());

        const play = ["match", "watch", "solo"].includes(shot.name);
        const fit = play
          ? await page.evaluate(() => {
              const out: string[] = [];
              const el = document.documentElement;
              if (el.scrollHeight > innerHeight + 1) out.push(`page is ${el.scrollHeight}px tall`);
              if (el.scrollWidth > innerWidth + 1) out.push(`page is ${el.scrollWidth}px wide`);
              for (const c of document.querySelectorAll("canvas")) {
                const r = c.getBoundingClientRect();
                // 10% in from each corner: a plate over the frame is fine, a ticker over
                // the stack is not.
                const dx = r.width * 0.1;
                const dy = r.height * 0.1;
                const pts = [[r.left + dx, r.top + dy], [r.right - dx, r.top + dy], [r.left + dx, r.bottom - dy], [r.right - dx, r.bottom - dy]];
                for (const [x, y] of pts) {
                  if (x < 0 || y < 0 || x > innerWidth || y > innerHeight) {
                    out.push(`${c.getAttribute("aria-label")?.split(":")[0]} is outside the window`);
                    break;
                  }
                  const top = document.elementFromPoint(x, y);
                  if (top !== c) {
                    out.push(`${c.getAttribute("aria-label")?.split(":")[0]} is covered by ${top?.tagName.toLowerCase()}.${top?.className}`);
                    break;
                  }
                }
              }
              return out;
            })
          : [];
        results[`${look} ${shot.name} ${w}x${h}`] = { violations, small, lowContrast, boards: boards.length, badBoards, fit };
        writeFileSync(REPORT, JSON.stringify(results, null, 1) + "\n");

        expect.soft(violations, "axe violations").toEqual([]);
        expect.soft(small, "text below the size floor").toEqual([]);
        expect.soft(lowContrast, "text below 4.5:1 (3:1 from 24px)").toEqual([]);
        expect.soft(fit, "play screen does not fit the window").toEqual([]);
        expect.soft(badBoards, "boards without role=img and a label").toEqual([]);
      });
    }
  }
}
