// Entry point of the static site: the web client's app, unchanged, with its
// API and socket answered in the page (see vite.config.ts), plus an About
// page, touch controls and a short note on the one screen that needs the
// local app (Record your games).
import { createRoot } from "react-dom/client";
import { useEffect, useState } from "react";
import { App } from "../../../web/src/App";
import { PerfHud } from "../../../web/src/components/PerfHud";
import "../../../web/src/fonts/fonts.css";
import "../../../web/src/styles/global.css";
import { AppProvider } from "../../../web/src/state/app";
import { About } from "./About";
import { REPO_URL } from "./links";
import { TouchControls, useCoarsePointer } from "./Touch";
import "./shell.css";
import "./phone.css";

function useHash(): string[] {
  const parse = () => location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const [parts, setParts] = useState(parse);
  useEffect(() => {
    const on = () => setParts(parse());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return parts;
}

function RecordNote() {
  return (
    <main className="about">
      <a className="about__back" href="#/">
        Back to the game
      </a>
      <h1>Record your games</h1>
      <p>
        Recording games for behavior cloning saves them to your computer, so it is part of the local app only. To record, run the project
        locally (see the README), then train BC (you) on your games.
      </p>
      <p>
        <a href={REPO_URL}>Get the code on GitHub</a>
      </p>
    </main>
  );
}

function Shell() {
  const [name, id] = useHash();
  const coarse = useCoarsePointer();
  // Route names of play screens with a live board.
  const playing = (name === "match" || name === "solo" || name === "watch") && (name === "solo" || !!id);
  const humanBoard = playing && name !== "watch";
  if (name === "about") return <About />;
  if (name === "record") return <RecordNote />;
  return (
    <>
      <App />
      {humanBoard && coarse && <TouchControls />}
      {!playing && (
        <footer className="site-footer">
          <a href="#/about">About this project</a>
          <a href={REPO_URL}>Source on GitHub</a>
        </footer>
      )}
    </>
  );
}

createRoot(document.getElementById("root")!).render(
  <AppProvider>
    <Shell />
    <PerfHud />
  </AppProvider>,
);
