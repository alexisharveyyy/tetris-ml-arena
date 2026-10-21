import { createRoot } from "react-dom/client";
import { App } from "./App";
import { PerfHud } from "./components/PerfHud";
import "./fonts/fonts.css";
import "./styles/global.css";
import { AppProvider } from "./state/app";

// No StrictMode: its double-run effects would start and close matches twice.
createRoot(document.getElementById("root")!).render(
  <AppProvider>
    <App />
    <PerfHud />
  </AppProvider>,
);
