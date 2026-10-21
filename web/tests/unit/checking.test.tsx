// While the server's startup check for an agent is running, every look says
// "Checking..." for it; once a check has failed, "Not available yet".
import { act, cleanup, render, screen, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../../src/api/client";
import type { Agent, AgentsResponse } from "../../src/api/types";
import { unavailableText } from "../../src/lib/copy";
import { AppProvider } from "../../src/state/app";
import type { AgentRow, HomeModel, OpponentsModel } from "../../src/screens/models";
import { allThemes, layout } from "../../src/themes/registry";
import fixture from "../fixtures/agents.json";

const all = (fixture as unknown as AgentsResponse).agents;
const lineup: Agent[] = all
  .filter((a) => a.group === "main")
  .map((a, i) =>
    i === 0
      ? { ...a, available: false, status: { state: "checking", detail: "" } }
      : i === 1
        ? { ...a, available: false, status: { state: "failed", detail: "could not load" } }
        : a,
  );
const [checking, failed] = lineup;

const noop = () => undefined;

function homeModel(look: string): HomeModel {
  return {
    lineup,
    looks: [],
    look: look as HomeModel["look"],
    setLook: noop,
    controls: [],
    evalTicker: [],
    raceMinutes: 2,
    startMarathon: noop,
    links: { marathon: "#", race: "#", watch: "#", solo: "#", record: "#", leaderboard: "#", settings: "#" },
  };
}

function opponentsModel(): OpponentsModel {
  const row = (agent: Agent): AgentRow => ({ agent, inSlots: 0, available: !!agent.available, record: "no games yet" });
  return {
    mode: "marathon",
    setMode: noop,
    groups: [{ title: null, rows: lineup.map(row) }],
    slots: [null, null, null],
    nextSlot: 1,
    add: noop,
    remove: noop,
    setPreset: noop,
    presets: [],
    instruction: "",
    modeText: "",
    startLabel: "Start",
    canStart: false,
    start: noop,
    starting: false,
    error: null,
    backHref: "#/",
  };
}

beforeEach(() => {
  vi.spyOn(api, "agents").mockResolvedValue({ ...(fixture as unknown as AgentsResponse), agents: lineup });
  vi.spyOn(api, "presets").mockResolvedValue([]);
  vi.spyOn(api, "settings").mockResolvedValue({ look: "tetromino" } as never);
  vi.spyOn(api, "saveSettings").mockResolvedValue({} as never);
  vi.spyOn(api, "demo").mockResolvedValue({ enabled: false } as never);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

async function show(node: ReactNode) {
  const out = render(<AppProvider>{node}</AppProvider>);
  await act(async () => undefined);
  return out;
}

describe("agents whose startup check is still running", () => {
  it("say Checking... rather than Not available yet", () => {
    expect(unavailableText(checking)).toBe("Checking...");
    expect(unavailableText(failed)).toBe("Not available yet");
  });

  for (const theme of allThemes()) {
    it(`show Checking... on the ${theme.id} opponent screen`, async () => {
      const Opponents = layout(theme, "Opponents");
      await show(<Opponents {...opponentsModel()} />);
      const rowOf = (a: Agent) => screen.getAllByText(a.name)[0].closest("li")!;
      expect(within(rowOf(checking)).getByText("Checking...")).toBeTruthy();
      expect(within(rowOf(failed)).getByText("Not available yet")).toBeTruthy();
      expect(within(rowOf(checking)).queryByRole("button", { name: new RegExp(`Add ${checking.name}\\b`) })).toBeNull();
    });

    it(`show Checking... in the ${theme.id} home lineup`, async () => {
      const Home = layout(theme, "Home");
      const { container } = await show(<Home {...homeModel(theme.id)} />);
      if (theme.id === "terminal") {
        expect(container.querySelector('abbr[title="Checking..."]')).toBeTruthy();
        expect(container.querySelector('abbr[title="Not available yet"]')).toBeTruthy();
      } else {
        expect(screen.getAllByText("Checking...").length).toBeGreaterThan(0);
        expect(screen.getAllByText("Not available yet").length).toBeGreaterThan(0);
      }
    });
  }
});
