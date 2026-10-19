"""Genetic algorithm over linear feature weights.

Usage:
    python -m tetris.agents.heuristic_ga.train [--config PATH] [--resume]
        [--generations N] [--population N] [--games N] [--workers N] [--seed N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from tetris.agents.evaluation import LinearEvaluator, fitness
from tetris.agents.linear import save_linear_checkpoint
from tetris.agents.linear_training import (
    HallOfFame,
    TrainingLogger,
    finalize,
    generation_seeds,
    load_config,
    normalize,
    setting_from_config,
    weights_table,
)
from tetris.agents.registry import CHECKPOINT_DIR, ROOT
from tetris.env.features import NUM_FEATURES

AGENT_KEY = "ga"
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def _tournament(rng: np.random.Generator, fit: np.ndarray, k: int) -> int:
    idx = rng.choice(len(fit), size=k, replace=False)
    return int(idx[np.argmax(fit[idx])])


def next_population(
    rng: np.random.Generator, pop: np.ndarray, fit: np.ndarray, cfg: dict[str, Any], sigma: float
) -> np.ndarray:
    """Elitism + tournament selection + fitness-weighted blend crossover + Gaussian mutation."""
    n = len(pop)
    order = np.argsort(-fit)
    children = [pop[i].copy() for i in order[: cfg["elite"]]]
    min_fit = fit.min()
    while len(children) < n:
        a = _tournament(rng, fit, cfg["tournament"])
        b = _tournament(rng, fit, cfg["tournament"])
        if rng.random() < cfg["crossover_rate"]:
            fa, fb = fit[a] - min_fit + 1e-6, fit[b] - min_fit + 1e-6
            child = (fa * pop[a] + fb * pop[b]) / (fa + fb)
        else:
            child = pop[a].copy()
        mask = rng.random(NUM_FEATURES) < cfg["mutation_rate"]
        child = child + mask * rng.normal(0.0, sigma, NUM_FEATURES)
        children.append(normalize(child))
    return np.array(children)


def _save_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def train(cfg: dict[str, Any], resume: bool = False) -> dict[str, Any]:
    out_dir = CHECKPOINT_DIR / AGENT_KEY
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    setting = setting_from_config(cfg)

    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng"]
        pop = np.array(state["population"])
        start_gen = state["generation"] + 1
        sigma = state["sigma"]
        hof = HallOfFame(state["hall_of_fame"])
        run_name = state["run_name"]
        print(f"resuming from generation {start_gen}", flush=True)
    else:
        rng = np.random.default_rng(cfg["seed"])
        pop = np.array([normalize(rng.normal(size=NUM_FEATURES)) for _ in range(cfg["population"])])
        start_gen = 0
        sigma = cfg["mutation_sigma"]
        hof = HallOfFame()
        run_name = time.strftime("%Y%m%d_%H%M%S")

    logger = TrainingLogger(ROOT / "runs" / AGENT_KEY / run_name)
    with LinearEvaluator(cfg.get("workers")) as evaluator:
        for gen in range(start_gen, cfg["generations"]):
            t0 = time.time()
            seeds = generation_seeds(cfg["seed"], gen, cfg["games_per_eval"])
            results = evaluator.evaluate(pop, seeds, setting)
            fit = np.array([fitness(r) for r in results])
            best = int(np.argmax(fit))
            hof.add(gen, pop[best], float(fit[best]))
            logger.generation(gen, fit, results[best], sigma=sigma, seconds=time.time() - t0)
            save_linear_checkpoint(
                out_dir / "best.json", pop[best], agent=AGENT_KEY, generation=gen, interim=True
            )
            if gen + 1 < cfg["generations"]:
                pop = next_population(rng, pop, fit, cfg, sigma)
                sigma = max(cfg["mutation_sigma_min"], sigma * cfg["mutation_sigma_decay"])
            if (gen + 1) % cfg.get("checkpoint_every", 1) == 0 or gen + 1 == cfg["generations"]:
                _save_state(
                    state_path,
                    {
                        "generation": gen,
                        "population": pop.tolist(),
                        "rng": rng.bit_generator.state,
                        "sigma": sigma,
                        "hall_of_fame": hof.entries,
                        "run_name": run_name,
                        "config": cfg,
                    },
                )
        report = finalize(AGENT_KEY, out_dir, evaluator, hof, setting, cfg)
    logger.close()
    print(
        "final weights:\n"
        + weights_table(np.array(json.loads((out_dir / "best.json").read_text())["weights"]))
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generations", type=int)
    parser.add_argument("--population", type=int)
    parser.add_argument("--games", dest="games_per_eval", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k not in ("config", "resume")}
    train(load_config(args.config, overrides), resume=args.resume)


if __name__ == "__main__":
    main()
