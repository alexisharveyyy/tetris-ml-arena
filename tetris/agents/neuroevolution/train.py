"""NEAT over the 13 afterstate features, with the GA/CEM fitness and seed protocol.

Usage:
    python -m tetris.agents.neuroevolution.train [--config PATH] [--budget-minutes M]
        [--seed S] [--workers N] [--run-name NAME] [--no-promote] [--resume]

Each generation every genome plays the same ``games_per_eval`` new training
seeds (common random numbers) in the garbage setting, and its fitness is the
leaderboard fitness. Every ``select_every`` generations that generation's best
genome is evaluated on the 30 selection seeds; at the end the genome with the
highest selection fitness becomes ``checkpoints/neat/best.json``, with its
topology drawn to ``reports/neat_topology.png``. Held-out results come from
``scripts/evaluate.py``. CPU only.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

import neat

from tetris.agents.evaluation import (
    EvalSetting,
    GameStats,
    default_workers,
    fitness,
    play_game,
    summarize,
)
from tetris.agents.linear_training import (
    generation_seeds,
    load_config,
    selection_seeds,
    setting_from_config,
)
from tetris.agents.neuroevolution.agent import best_placement_index, save_genome_checkpoint
from tetris.agents.neuroevolution.network import GenomeNet
from tetris.agents.registry import CHECKPOINT_DIR, ROOT
from tetris.rl.training import Budget, CurveLogger, EvalRecord, linear_references

AGENT_KEY = "neat"
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def neat_config(path: str | Path) -> neat.Config:
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(ROOT / path),
    )


def _play(args: tuple[dict[str, Any], int, dict[str, Any]]) -> dict[str, Any]:
    net_d, seed, setting_d = args
    net = GenomeNet.from_dict(net_d)
    stats = play_game(
        lambda g: best_placement_index(g, net), seed, EvalSetting.from_dict(setting_d)
    )
    return stats.to_dict()


def evaluate_nets(
    pool, nets: list[GenomeNet], seeds: list[int], setting: EvalSetting
) -> list[list[GameStats]]:
    """Every network on every seed (one task per game, for load balance)."""
    tasks = [(n.to_dict(), s, setting.to_dict()) for n in nets for s in seeds]
    rows = pool.map(_play, tasks, chunksize=1)
    k = len(seeds)
    return [[GameStats(**r) for r in rows[i * k : (i + 1) * k]] for i in range(len(nets))]


def train(
    cfg: dict[str, Any], resume: bool = False, run_name: str | None = None, promote: bool = True
) -> dict[str, Any]:
    seed = int(cfg["seed"])
    run_name = run_name or f"seed{seed}"
    run_dir = CHECKPOINT_DIR / AGENT_KEY / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = neat_config(cfg["neat_config"])
    if cfg.get("pop_size"):
        config.pop_size = int(cfg["pop_size"])  # smoke tests
    setting = setting_from_config(cfg)
    sel_seeds = selection_seeds(int(cfg["selection_games"]))
    state_path = run_dir / "state.pkl"
    if resume and state_path.exists():
        with state_path.open("rb") as f:
            state = pickle.load(f)
        population, elapsed0, records = state["population"], state["elapsed"], state["records"]
        print(f"resumed {run_name} at generation {population.generation}", flush=True)
    else:
        population = neat.Population(config, seed=seed)
        elapsed0, records = 0.0, []
    budget = Budget(float(cfg["budget_minutes"]) * 60, elapsed0)
    logger = CurveLogger(AGENT_KEY, run_name)
    workers = cfg.get("workers") or default_workers()
    games = int(cfg["games_per_eval"])
    print(
        f"neat {run_name}: population {config.pop_size}, {games} games per genome, "
        f"{workers} workers, budget {cfg['budget_minutes']} min",
        flush=True,
    )

    with mp.get_context("spawn").Pool(workers) as pool:
        while not budget.done:
            gen = population.generation
            t0 = time.perf_counter()
            seeds = generation_seeds(seed, gen, games)
            nets: dict[int, GenomeNet] = {}
            fits: dict[int, float] = {}

            def evaluate(genomes, config, seeds=seeds, nets=nets, fits=fits):
                for gid, genome in genomes:
                    nets[gid] = GenomeNet.from_genome(genome, config)
                results = evaluate_nets(pool, [nets[g] for g, _ in genomes], seeds, setting)
                for (gid, genome), stats in zip(genomes, results, strict=True):
                    genome.fitness = fits[gid] = fitness(stats)

            # One generation: evaluate, then reproduce (population.generation advances).
            population.run(evaluate, 1)
            best_gid = max(fits, key=fits.get)
            best_net, best_fit = nets[best_gid], fits[best_gid]
            minutes = budget.elapsed / 60
            logger.scalar("train/best_fitness", best_fit, gen)
            logger.scalar("train/hidden_nodes", len(best_net.hidden), gen)
            logger.scalar("train/connections", best_net.n_connections, gen)
            logger.scalar("train/species", len(population.species.species), gen)
            line = (
                f"gen {gen:4d}  {minutes:6.1f} min  best {best_fit:.3f}  "
                f"hidden {len(best_net.hidden)}  conns {best_net.n_connections}  "
                f"species {len(population.species.species)}  "
                f"{time.perf_counter() - t0:.0f}s/gen"
            )
            if gen % int(cfg["select_every"]) == 0 or budget.done:
                path = run_dir / f"gen_{gen:04d}.json"
                save_genome_checkpoint(path, best_net, generation=gen, train_fitness=best_fit)
                stats = evaluate_nets(pool, [best_net], sel_seeds, setting)[0]
                rec = EvalRecord(str(path), gen, budget.elapsed, fitness(stats), summarize(stats))
                records.append(rec)
                logger.eval_record(rec)
                line += f"  selection {rec.fitness:.3f}"
            print(line, flush=True)
            with state_path.open("wb") as f:
                pickle.dump(
                    {"population": population, "elapsed": budget.elapsed, "records": records}, f
                )

    best = max(records, key=lambda r: r.fitness)
    summary = {
        "agent": AGENT_KEY,
        "selected_checkpoint": best.checkpoint,
        "selected_generation": best.step,
        "selected_minutes": best.elapsed / 60,
        "selection_fitness": best.fitness,
        "selection_summary": best.summary,
        "generations": population.generation,
        "evaluations": [
            {"checkpoint": r.checkpoint, "generation": r.step, "fitness": r.fitness}
            for r in records
        ],
    }
    (run_dir / "selection.json").write_text(json.dumps(summary, indent=2))
    from tetris.agents.neuroevolution.plot import plot_topology

    topo = plot_topology(best.checkpoint, run_dir / "topology.png")
    if promote:
        shutil.copy(best.checkpoint, CHECKPOINT_DIR / AGENT_KEY / "best.json")
        (CHECKPOINT_DIR / AGENT_KEY / "selection.json").write_text(json.dumps(summary, indent=2))
        shutil.copy(topo, ROOT / "reports" / "neat_topology.png")
    logger.plot(references=linear_references())
    logger.close()
    print(
        f"selected generation {best.step} (selection fitness {best.fitness:.3f}); topology {topo}",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--budget-minutes", dest="budget_minutes", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--pop-size", dest="pop_size", type=int)
    parser.add_argument("--games", dest="games_per_eval", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    overrides = {
        k: getattr(args, k)
        for k in ("budget_minutes", "seed", "workers", "pop_size", "games_per_eval")
    }
    train(load_config(args.config, overrides), args.resume, args.run_name, not args.no_promote)


if __name__ == "__main__":
    main()
