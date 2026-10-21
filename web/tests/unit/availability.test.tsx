// The lineup must update by itself when the server's startup checks finish.
import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../../src/api/client";
import type { AgentsResponse } from "../../src/api/types";
import { AppProvider, CHECK_POLL_MS, stillChecking, useApp } from "../../src/state/app";

function response(state: "checking" | "ok"): AgentsResponse {
  const agent = { key: "bc", name: "BC", available: state === "ok", playable: true, group: "main", status: { state, detail: "" } };
  return { agents: [agent], human: null, groups: [], looks: [] } as unknown as AgentsResponse;
}

function Lineup() {
  const app = useApp();
  return <p>{app.agents.map((a) => `${a.name}: ${a.available ? "available" : "Not available yet"}`).join(", ")}</p>;
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("agent availability", () => {
  it("knows when checks are still running", () => {
    expect(stillChecking(response("checking"))).toBe(true);
    expect(stillChecking(response("ok"))).toBe(false);
    expect(stillChecking(null)).toBe(false);
  });

  it("updates the lineup when the startup checks finish, without a reload", async () => {
    vi.useFakeTimers();
    const agents = vi
      .spyOn(api, "agents")
      .mockResolvedValueOnce(response("checking"))
      .mockResolvedValueOnce(response("checking"))
      .mockResolvedValue(response("ok"));
    vi.spyOn(api, "presets").mockResolvedValue([]);
    vi.spyOn(api, "settings").mockResolvedValue({ look: "tetromino" } as never);
    vi.spyOn(api, "saveSettings").mockResolvedValue({} as never);

    render(
      <AppProvider>
        <Lineup />
      </AppProvider>,
    );
    await act(async () => undefined);
    expect(screen.getByText("BC: Not available yet")).toBeTruthy();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(CHECK_POLL_MS[0]);
    });
    expect(screen.getByText("BC: Not available yet")).toBeTruthy();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(CHECK_POLL_MS[1]);
    });
    expect(screen.getByText("BC: available")).toBeTruthy();
    expect(agents).toHaveBeenCalledTimes(3);

    // Once nothing is checking, polling stops.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10_000);
    });
    expect(agents).toHaveBeenCalledTimes(3);
  });
});
