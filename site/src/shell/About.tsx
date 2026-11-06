// About page. Every number comes from reports/FINAL_REPORT.md (section 3.1).
import { REPO_URL, REPORT_URL } from "./links";

const ROWS: [string, string, string, string][] = [
  ["1", "DQN (afterstates)", "1.345", "0.13"],
  ["2", "PPO", "1.339", "0.12"],
  ["3", "Cross-entropy method", "1.318", "0.15"],
  ["4", "Decision Transformer (v1)", "1.309", "0.16"],
  ["5", "Imitation (pure BC)", "1.296", "0.20"],
  ["6", "GA heuristic", "1.289", "0.18"],
  ["7", "NEAT", "1.280", "0.19"],
  ["8", "Imitation (BC + DAgger)", "1.280", "0.21"],
  ["9", "Dellacherie (baseline)", "1.196", "0.31"],
  ["10", "BC (you)", "0.147", "1.00"],
  ["11", "PPO (raw actions)", "0.103", "1.00"],
];

const FAMILIES: [string, string][] = [
  ["Hand-tuned baseline", "Dellacherie: fixed linear weights over board features such as holes and height."],
  ["Evolved linear weights", "GA heuristic and cross-entropy method (CEM): the same features, with the weights optimized."],
  ["Neuroevolution", "NEAT: a small network whose shape and weights evolved together."],
  ["Value-based RL", "DQN: a CNN that values the board after each possible placement."],
  ["Policy-gradient RL", "PPO, which scores each placement's resulting board, and PPO (raw actions), which picks among all 80 moves from the board alone."],
  ["Imitation learning", "Behavior cloning of the GA's games, with and without DAgger, and BC (you), fine-tuned on recorded human games."],
  ["Offline RL", "Decision Transformer: a return-conditioned transformer trained on the other agents' games."],
];

export function About() {
  return (
    <main className="about">
      <a className="about__back" href="#/">
        Back to the game
      </a>
      <h1>About Tetris ML Arena</h1>
      <p>
        Tetris ML Arena puts agents from very different machine learning methods on the same deterministic Tetris engine, compares them on 500
        held-out games under one protocol, and lets you play against them in real time. This page is the browser version: the engine and every
        agent run on your device, with nothing sent to a server.
      </p>

      <h2>The agents</h2>
      <p>
        Every agent chooses one hard-drop placement per piece. Most decide by scoring the board that each possible placement would leave.
      </p>
      <dl className="about__families">
        {FAMILIES.map(([name, text]) => (
          <div key={name}>
            <dt>{name}</dt>
            <dd>{text}</dd>
          </div>
        ))}
      </dl>
      <p>The project also has an LLM agent as a novelty opponent. It is not part of the browser version.</p>

      <h2>The main finding</h2>
      <p>
        The top eight agents form a chain of statistical ties: each is tied with its neighbors in the ranking, though the ends of the chain
        differ. All eight beat the hand-tuned Dellacherie baseline. Built by six different methods, they land within 0.065 fitness of each other.
        Their shared action space (one hard-drop placement per piece, chosen by scoring the board after the placement) appears to set the
        ceiling, not the learning method. The only agent that learns raw actions is far below
        them.
      </p>
      <div className="about__table" role="region" aria-label="Leaderboard" tabIndex={0}>
        <table>
          <caption>Garbage leaderboard, 500 held-out games per agent</caption>
          <thead>
            <tr>
              <th scope="col">Rank</th>
              <th scope="col">Agent</th>
              <th scope="col">Fitness</th>
              <th scope="col">Top-out rate</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map(([rank, name, fit, top]) => (
              <tr key={rank} className={Number(rank) <= 8 ? "about__tied" : undefined}>
                <td>{rank}</td>
                <td>{name}</td>
                <td>{fit}</td>
                <td>{top}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>How the browser version works</h2>
      <p>
        The engine was ported to TypeScript and replays the Python engine's games move for move. The neural agents were exported to ONNX, and
        their weights run in a small TypeScript inference module inside Web Workers, so the game keeps its frame rate while they think. A test
        checks that every agent picks the same placement as its Python original on a fixed set of recorded board states. Model files download
        only when you choose that agent.
      </p>

      <h2>Links</h2>
      <ul>
        <li>
          <a href={REPO_URL}>Source code on GitHub</a>
        </li>
        <li>
          <a href={REPORT_URL}>Final report</a>, with the full method, every result and how to reproduce it
        </li>
      </ul>
    </main>
  );
}
