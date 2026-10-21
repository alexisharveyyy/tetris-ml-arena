import { HomeScreen } from "./screens/Home";
import { LeaderboardScreen } from "./screens/Leaderboard";
import { MatchScreen } from "./screens/Match";
import { OpponentsScreen } from "./screens/Opponents";
import { ResultsScreen } from "./screens/Results";
import { SettingsScreen } from "./screens/Settings";
import { RecordScreen, SoloScreen } from "./screens/Solo";
import { WatchScreen } from "./screens/Watch";
import { useRoute } from "./router";
import { useApp } from "./state/app";

export function App() {
  const route = useRoute();
  const app = useApp();
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
      return <MatchScreen key={a} matchId={a} />;
    case "results":
      return <ResultsScreen key={a} matchId={a} />;
    case "leaderboard":
      return <LeaderboardScreen tab={a === "people" ? "people" : "survival"} />;
    case "watch":
      return <WatchScreen key={a ?? "setup"} matchId={a ?? null} />;
    case "solo":
      return <SoloScreen key={a ?? "new"} matchId={a ?? null} />;
    case "record":
      return <RecordScreen key={a ?? "setup"} matchId={a ?? null} />;
    case "settings":
      return <SettingsScreen />;
    default:
      return <HomeScreen />;
  }
}
