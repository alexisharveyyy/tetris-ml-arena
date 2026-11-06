"""Parity fixtures for the browser version (site/).

``engine``: seeded games played through the Python engine, move for move.
Each game is a list of operations (a player move, a gravity tick of an exact
``dt``, or an instant placement) and, after every operation, a hash of the
full game state plus the events it emitted. The TypeScript engine replays the
same operations and must produce the same hash at every step. Needs NumPy only,
so CI regenerates this file from the Python engine on every run.

``agents``: board states with each agent's Python ranking (top five
placements and their scores) and, for the Decision Transformer, whole
trajectories (it keeps a history). Needs the trained checkpoints, so it runs
locally and the output is committed. ``--only`` limits it to some agents, and
``--check`` compares with the committed files instead of writing them (CI does
this for the linear and NEAT agents, whose weights are committed): placements
must match exactly and scores to 1e-9.

Usage:
  python scripts/tools/web_fixtures.py engine [--out PATH]
  python scripts/tools/web_fixtures.py agents [--only ga,cem] [--check]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tetris.engine import scoring  # noqa: E402
from tetris.engine.board import board_to_strings  # noqa: E402
from tetris.engine.game import Game, GameConfig, Move  # noqa: E402
from tetris.engine.rng import Rng, derive_seed  # noqa: E402

FIXTURES = ROOT / "site" / "tests" / "fixtures"
ENGINE_OUT = FIXTURES / "engine_traces.json"
AGENT_DIR = FIXTURES / "agents"

MOVES = ("left", "right", "cw", "ccw", "soft", "hard", "hold")
MOVE_WEIGHTS = (3, 3, 2, 2, 2, 1, 1)
DTS = (1 / 60, 1 / 60, 1 / 60, 1 / 60, 1 / 30, 1 / 120, 0.005, 0.25)


def bits(x: float) -> str:
    return struct.pack(">d", float(x)).hex()


def fnv1a(text: str) -> str:
    h = 0x811C9DC5
    for b in text.encode("ascii"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def _event(ev) -> str:
    d = ev.data
    if ev.type == "lock":
        cells = ".".join(f"{r},{c}" for r, c in d["cells"])
        return (f"lock:{d['piece']},{d['lines']},{d['combo']},{int(d['b2b'])},{d['attack']},"
                f"{d['sent']},{d['score']},{cells}")
    if ev.type in ("spawn", "hold"):
        return f"{ev.type}:{d['piece']}"
    if ev.type == "game_over":
        return f"game_over:{d['reason']}"
    return ev.type


def canon(g: Game, events) -> str:
    """Every field of the game state, in a fixed text form shared with TypeScript."""
    a = g.active
    return "|".join(
        [
            "".join(board_to_strings(g.board)),
            str(g.bag_rng.state),
            str(g.garbage_rng.state),
            "".join(p.letter for p in g.queue),
            f"{a.piece.letter},{a.rot},{a.row},{a.col}" if a else "-",
            g.hold.letter if g.hold else "-",
            str(int(g.can_hold)),
            str(g.score),
            str(g.lines),
            str(g.level),
            str(g.combo),
            str(int(g.b2b)),
            str(g.pieces_placed),
            str(g.attack_sent),
            ",".join(map(str, g.clears)),
            str(int(g.game_over)),
            bits(g.time),
            bits(g._gravity_acc),
            bits(g._lock_timer),
            str(g._lock_resets),
            str(g._lowest_row),
            ";".join(_event(e) for e in events),
        ]
    )


class Trace:
    def __init__(self, kind: str, seed: int, start_level: int) -> None:
        self.game = Game(seed, GameConfig(start_level=start_level))
        self.ops: list[list] = []
        self.hashes: list[str] = []
        self.full: dict[int, str] = {}
        first = canon(self.game, self.game.pop_events())
        self.meta = {"kind": kind, "seed": seed, "start_level": start_level,
                     "initial": first, "initial_hash": fnv1a(first)}

    def record(self, op: list) -> None:
        g = self.game
        text = canon(g, g.pop_events())
        self.ops.append(op)
        self.hashes.append(fnv1a(text))
        if "lock:" in text or g.game_over:
            self.full[len(self.ops) - 1] = text

    def move(self, move: str) -> None:
        self.game.apply(move)
        self.record(["m", move])

    def tick(self, dt: float) -> None:
        self.game.tick(dt)
        self.record(["t", bits(dt)])

    def to_dict(self) -> dict:
        return {**self.meta, "ops": self.ops, "hashes": self.hashes,
                "full": {str(k): v for k, v in self.full.items()}}


def _pick(rng: Rng, weights) -> int:
    total = sum(weights)
    x = rng.randbelow(total)
    for i, w in enumerate(weights):
        if x < w:
            return i
        x -= w
    return len(weights) - 1


def random_game(seed: int) -> dict:
    rng = Rng(derive_seed(seed, 77))
    t = Trace("random", seed, 1 + rng.randbelow(20))
    for _ in range(2500):
        if t.game.game_over:
            break
        if rng.random() < 0.6:
            t.move(MOVES[_pick(rng, MOVE_WEIGHTS)])
        else:
            t.tick(DTS[rng.randbelow(len(DTS))])
    return t.to_dict()


def agent_game(seed: int, max_pieces: int = 200) -> dict:
    """Real-time play: the GA agent's placements, executed by the path planner."""
    from tetris.agents.planner import PlanExecutor
    from tetris.agents.registry import create_agent

    agent = create_agent("ga")
    rng = Rng(derive_seed(seed, 78))
    t = Trace("agent", seed, 1 + rng.randbelow(12))
    g = t.game
    while not g.game_over and g.pieces_placed < max_pieces:
        ranked = agent.rank_placements(g)
        if not ranked:
            t.move("hard")
            continue
        i = 0 if rng.random() < 0.8 else min(len(ranked) - 1, 1 + rng.randbelow(3))
        ex = PlanExecutor(g, ranked[i].placement)
        count = g.pieces_placed
        while not ex.done and not g.game_over and g.pieces_placed == count:
            for _ in range(rng.randbelow(4)):
                t.tick(DTS[rng.randbelow(len(DTS))])
                if g.pieces_placed != count or g.game_over:
                    break
            if g.pieces_placed != count or g.game_over:
                break
            move = ex.next_move(g)
            if move is None:
                if ex.done:
                    break
                g.apply(Move.HARD)
                t.record(["m", "hard"])
                break
            t.record(["m", str(move)])
    return t.to_dict()


def placement_game(seed: int, max_pieces: int = 300) -> dict:
    """Instant placements through Game.place, hold placements included."""
    from tetris.agents.registry import create_agent

    agent = create_agent("cem")
    rng = Rng(derive_seed(seed, 79))
    t = Trace("place", seed, 1 + rng.randbelow(10))
    g = t.game
    while not g.game_over and g.pieces_placed < max_pieces:
        ranked = agent.rank_placements(g)
        if not ranked:
            break
        i = 0 if rng.random() < 0.7 else min(len(ranked) - 1, 1 + rng.randbelow(4))
        p = ranked[i].placement
        g.place(p)
        t.record(["p", p.index])
    return t.to_dict()


def engine(out: Path) -> None:
    games = []
    for i in range(25):
        games.append(random_game(1000 + i))
    for i in range(20):
        games.append(agent_game(2000 + i))
    for i in range(5):
        games.append(placement_game(3000 + i))
    data = {
        "note": "Generated by scripts/tools/web_fixtures.py engine. Not committed.",
        "gravity_bits": [bits(scoring.gravity_rows_per_second(lv)) for lv in range(1, 21)],
        "games": games,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    steps = sum(len(g["ops"]) for g in games)
    print(f"{len(games)} games, {steps} steps -> {out}")


# ----------------------------------------------------------------- agents

#: Agents in the browser version (the LLM agent is not).
AGENT_KEYS = ("dellacherie", "ga", "cem", "neat", "dqn", "ppo", "ppo_raw", "bc", "bc_dagger",
              "bc_you", "dt")
TOP = 5
OWN_GAMES = 2
OWN_STATES = 60  # per game
POOL_STATES = 60


def _ranked_top(ranked) -> list[list]:
    return [[r.placement.index, float(r.score)] for r in ranked[:TOP]]


def _agent(key: str, noise: float, epsilon: float, seed: int):
    from tetris.agents.difficulty import DifficultyParams
    from tetris.agents.registry import create_agent

    return create_agent(key, difficulty=DifficultyParams(noise=noise, epsilon=epsilon), seed=seed)


def shared_pool() -> list[dict]:
    """Messy boards: Dellacherie with frequent random placements."""
    agent = _agent("dellacherie", 0.3, 0.35, 11)
    states = []
    seed = 500
    g = Game(seed)
    step = 0
    while len(states) < POOL_STATES:
        if g.game_over:
            seed += 1
            g = Game(seed)
        ranked = agent.rank_placements(g)
        if not ranked:
            seed += 1
            g = Game(seed)
            continue
        if step % 4 == 3:
            states.append(g.to_dict())
        g.place(agent._choose(g, ranked))
        step += 1
    return states


def own_states(key: str) -> list[dict]:
    """States from the agent's own games (Medium-like noise, so boards vary)."""
    out = []
    for k in range(OWN_GAMES):
        agent = _agent(key, 0.15, 0.02, 100 + k)
        g = Game(derive_seed(4000 + k, AGENT_KEYS.index(key)))
        step = 0
        taken = 0
        while taken < OWN_STATES and not g.game_over:
            ranked = agent.rank_placements(g)
            if not ranked:
                break
            if step % 3 == 0:
                out.append({"state": g.to_dict(), "top": _ranked_top(ranked)})
                taken += 1
            g.place(agent._choose(g, ranked))
            step += 1
    return out


def dt_trajectories() -> list[dict]:
    """Whole games of the Decision Transformer, with what it actually played."""
    trajs = []
    for k in range(2):
        agent = _agent("dt", 0.15, 0.03, 200 + k)
        g = Game(derive_seed(5000 + k, 1))
        steps = []
        while len(steps) < 150 and not g.game_over:
            state = g.to_dict()
            ranked = agent.rank_placements(g)
            if not ranked:
                break
            placement = agent._choose(g, ranked)
            steps.append({"state": state, "top": _ranked_top(ranked), "played": placement.index})
            g.place(placement)
        trajs.append({"seed": k, "steps": steps})
    return trajs


def _first_diff(a, b, path: str = "", rel: float = 1e-9) -> str | None:
    """Where ``a`` and ``b`` first differ: equal structure and integers, floats
    within ``rel`` (BLAS summation order can change a score's last bits)."""
    if isinstance(a, float) or isinstance(b, float):
        ok = isinstance(a, (int, float)) and isinstance(b, (int, float))
        ok = ok and abs(a - b) <= rel * max(1.0, abs(a), abs(b))
        return None if ok else f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{path}: keys differ"
        for k in a:
            d = _first_diff(a[k], b[k], f"{path}.{k}", rel)
            if d:
                return d
        return None
    if _is_ranking(a) and _is_ranking(b):
        return _ranking_diff(a, b, path, rel)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_diff(x, y, f"{path}[{i}]", rel)
            if d:
                return d
        return None
    return None if a == b else f"{path}: {str(a)[:200]} != {str(b)[:200]}"


def _is_ranking(x) -> bool:
    """A list of [action index, score] pairs, best first."""
    return isinstance(x, list) and len(x) > 0 and all(
        isinstance(e, list) and len(e) == 2 and isinstance(e[0], int) and isinstance(e[1], (int, float))
        for e in x
    )


def _ranking_diff(a, b, path: str, rel: float) -> str | None:
    """Scores must match in order. Placements must match, except that within a
    group of equal scores (a tie) their order may differ."""
    if len(a) != len(b):
        return f"{path}: length {len(a)} != {len(b)}"
    close = lambda x, y: abs(x - y) <= rel * max(1.0, abs(x), abs(y))  # noqa: E731
    for i, ((_, sa), (_, sb)) in enumerate(zip(a, b)):
        if not close(sa, sb):
            return f"{path}[{i}]: score {sa!r} != {sb!r}"
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and close(a[j][1], a[i][1]):
            j += 1
        # A tie group that runs to the end of the list may continue past it, so
        # only its size is comparable there.
        if j < len(a) and sorted(e[0] for e in a[i:j]) != sorted(e[0] for e in b[i:j]):
            return f"{path}[{i}:{j}]: placements {a[i:j]} != {b[i:j]}"
        i = j
    return None


def _same(a, b) -> bool:
    return _first_diff(a, b) is None


def agents(only: list[str] | None, check: bool) -> int:
    keys = [k for k in AGENT_KEYS if not only or k in only]
    pool_path = AGENT_DIR / "pool.json"
    pool = shared_pool()
    outputs: dict[Path, str] = {pool_path: json.dumps({"states": pool}, separators=(",", ":"))}
    for key in keys:
        if key == "dt":
            data = {"agent": key, "trajectories": dt_trajectories()}
        else:
            agent = _agent(key, 0.0, 0.0, 0)
            pool_top = [_ranked_top(agent.rank_placements(Game.from_dict(s))) for s in pool]
            data = {"agent": key, "own": own_states(key), "pool_top": pool_top}
        outputs[AGENT_DIR / f"{key}.json"] = json.dumps(data, separators=(",", ":"))
    bad = 0
    for path, text in outputs.items():
        if check:
            diff = "missing" if not path.exists() else _first_diff(json.loads(path.read_text()), json.loads(text))
            if diff:
                print(f"differs from the Python agents: {path.relative_to(ROOT)}: {diff}")
                bad += 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"wrote {path.relative_to(ROOT)} ({len(text) / 1e3:.0f} kB)")
    if check and not bad:
        print(f"fixtures match for {', '.join(keys)}")
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("engine")
    e.add_argument("--out", type=Path, default=ENGINE_OUT)
    a = sub.add_parser("agents")
    a.add_argument("--only", default=None, help="comma-separated agent keys")
    a.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.cmd == "engine":
        engine(args.out)
    else:
        sys.exit(agents(args.only.split(",") if args.only else None, args.check))


if __name__ == "__main__":
    main()
