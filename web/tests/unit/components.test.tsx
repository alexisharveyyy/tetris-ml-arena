import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Segmented } from "../../src/components/Segmented";
import { Tabs } from "../../src/components/Tabs";
import { keyLabel, keyName } from "../../src/lib/keys";
import { pos } from "../../src/themes/tetromino/layouts/Leaderboard";
import { getTheme, THEMES } from "../../src/themes/registry";

describe("toggles and tabs", () => {
  it("marks the chosen option with aria-pressed", () => {
    const onChange = vi.fn();
    render(<Segmented label="Mode" value="a" onChange={onChange} options={[{ value: "a", label: "A" }, { value: "b", label: "B" }]} />);
    expect(screen.getByRole("button", { name: "A" }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByRole("button", { name: "B" }).getAttribute("aria-pressed")).toBe("false");
    fireEvent.click(screen.getByRole("button", { name: "B" }));
    expect(onChange).toHaveBeenCalledWith("b");
  });

  it("uses tabs with aria-selected and arrow keys", () => {
    const onChange = vi.fn();
    render(<Tabs label="Board" idPrefix="t" value="x" onChange={onChange} tabs={[{ id: "x", label: "X" }, { id: "y", label: "Y" }]} />);
    const x = screen.getByRole("tab", { name: "X" });
    expect(x.getAttribute("aria-selected")).toBe("true");
    act(() => {
      fireEvent.keyDown(x, { key: "ArrowRight" });
    });
    expect(onChange).toHaveBeenCalledWith("y");
  });
});

describe("keys", () => {
  it("maps browser codes to settings key names", () => {
    expect(keyName({ code: "ArrowLeft", key: "ArrowLeft" })).toBe("left");
    expect(keyName({ code: "Space", key: " " })).toBe("space");
    expect(keyName({ code: "KeyX", key: "x" })).toBe("x");
    expect(keyName({ code: "ShiftLeft", key: "Shift" })).toBe("left shift");
    expect(keyName({ code: "Escape", key: "Escape" })).toBe("escape");
    expect(keyLabel("escape")).toBe("Esc");
    expect(keyLabel("z")).toBe("Z");
  });
});

describe("themes", () => {
  it("registers every theme folder and falls back to Tetromino", () => {
    expect(THEMES.map((t) => t.id)).toContain("tetromino");
    expect(getTheme("nope").id).toBe("tetromino");
    for (const t of THEMES) {
      expect(t.board.size(10).width).toBeGreaterThan(0);
      for (const letter of "IOTSZJLG") expect(t.pieceColors[letter]).toBeTruthy();
    }
  });

  it("clips the interval chart to its scale", () => {
    expect(pos(0.15, 0.5)).toBeCloseTo(30);
    expect(pos(1, 0.5)).toBe(100);
    expect(pos(-0.1, 0.5)).toBe(0);
  });
});
