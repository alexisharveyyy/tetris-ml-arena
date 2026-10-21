import { ComparisonsScreen } from "./screens/Comparisons";
import { HomeScreen } from "./screens/Home";
import { LeaderboardScreen } from "./screens/Leaderboard";
import { MatchScreen } from "./screens/Match";
import { OpponentsScreen } from "./screens/Opponents";
import { ResultsScreen } from "./screens/Results";
import { SettingsScreen } from "./screens/Settings";
import { RecordScreen, SoloScreen } from "./screens/Solo";
import { WatchScreen } from "./screens/Watch";
import { FitToViewport } from "./lib/fit";
import { useRoute } from "./router";
import { useApp } from "./state/app";
import { useShortcuts } from "./state/match";

export function App() {
  const route = useRoute();
  const app = useApp();
  // Demo mode: F10 cycles the four looks on any screen.
  useShortcuts({ f10: app.cycleLook }, !!app.demo);
  if (app.error) {
    return (
      <main className="screen screen--message">
        <h1 className="page-title">The server is not answering</h1>
        <p>{app.error}</p>
        <p>Start it with python -m tetris.web, then reload this page.</p>
      </main>
    );
  }
  if (!app.settings || !app.agents.length) {
    return (
      <main className="screen screen--message" aria-busy="true">
        <p>Loading...</p>
      </main>
    );
  }
  const [a] = route.params;
  switch (route.name) {
    case "play":
      return <OpponentsScreen mode={a === "race" ? "race" : "marathon"} />;
    case "match":
      return (
        <FitToViewport key={a}>
          <MatchScreen matchId={a} />
        </FitToViewport>
      );
    case "results":
      return <ResultsScreen key={a} matchId={a} />;
    case "leaderboard":
      if (a === "comparisons") return <ComparisonsScreen />;
      return <LeaderboardScreen tab={a === "people" ? "people" : "survival"} />;
    case "watch":
      return a ? (
        <FitToViewport key={a}>
          <WatchScreen matchId={a} />
        </FitToViewport>
      ) : (
        <WatchScreen key="setup" matchId={null} />
      );
    case "solo":
      return (
        <FitToViewport key={a ?? "new"}>
          <SoloScreen matchId={a ?? null} />
        </FitToViewport>
      );
    case "record":
      return a ? (
        <FitToViewport key={a}>
          <RecordScreen matchId={a} />
        </FitToViewport>
      ) : (
        <RecordScreen key="setup" matchId={null} />
      );
    case "settings":
      return <SettingsScreen />;
    default:
      return <HomeScreen />;
  }
}
