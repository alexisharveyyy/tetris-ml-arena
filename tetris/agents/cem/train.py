"""Noisy cross-entropy method over linear feature weights.

Usage:
    python -m tetris.agents.cem.train [--config PATH] [--resume]
        [--generations N] [--samples N] [--games N] [--workers N] [--seed N]
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
    setting_from_config,
    weights_table,
)
from tetris.agents.registry import CHECKPOINT_DIR, ROOT
from tetris.env.features import NUM_FEATURES

AGENT_KEY = "cem"
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def extra_variance(cfg: dict[str, Any], it: int) -> float:
    frac = max(0.0, 1.0 - it / max(1, cfg["extra_noise_iters"]))
    return cfg["extra_noise"] * frac


def train(cfg: dict[str, Any], resume: bool = False) -> dict[str, Any]:
    out_dir = CHECKPOINT_DIR / AGENT_KEY
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "state.json"
    setting = setting_from_config(cfg)
    n_elite = max(2, int(round(cfg["samples"] * cfg["elite_frac"])))

    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng"]
        mean, std = np.array(state["mean"]), np.array(state["std"])
        start = state["generation"] + 1
        hof = HallOfFame(state["hall_of_fame"])
        run_name = state["run_name"]
        print(f"resuming from iteration {start}", flush=True)
    else:
        rng = np.random.default_rng(cfg["seed"])
        mean = np.zeros(NUM_FEATURES)
        std = np.full(NUM_FEATURES, cfg["init_std"])
        start = 0
        hof = HallOfFame()
        run_name = time.strftime("%Y%m%d_%H%M%S")

    logger = TrainingLogger(ROOT / "runs" / AGENT_KEY / run_name)
    with LinearEvaluator(cfg.get("workers")) as evaluator:
        for it in range(start, cfg["generations"]):
            t0 = time.time()
            samples = mean + std * rng.normal(size=(cfg["samples"], NUM_FEATURES))
            # The current mean is evaluated alongside the samples (it is the CEM estimate).
            candidates = np.vstack([samples, mean[None]])
            seeds = generation_seeds(cfg["seed"], it, cfg["games_per_eval"])
            results = evaluator.evaluate(candidates, seeds, setting)
            fit = np.array([fitness(r) for r in results])
            sample_fit = fit[:-1]
            best = int(np.argmax(fit))
            hof.add(it, candidates[best], float(fit[best]))
            logger.generation(
                it,
                sample_fit,
                results[best],
                mean_fitness=float(fit[-1]),
                std=float(std.mean()),
                seconds=time.time() - t0,
            )
            save_linear_checkpoint(
                out_dir / "best.json",
                candidates[best],
                agent=AGENT_KEY,
                generation=it,
                interim=True,
            )
            elite = samples[np.argsort(-sample_fit)[:n_elite]]
            mean = elite.mean(axis=0)
            std = np.sqrt(elite.var(axis=0) + extra_variance(cfg, it))
            if (it + 1) % cfg.get("checkpoint_every", 1) == 0 or it + 1 == cfg["generations"]:
                tmp = state_path.with_suffix(".tmp")
                tmp.write_text(
                    json.dumps(
                        {
                            "generation": it,
                            "mean": mean.tolist(),
                            "std": std.tolist(),
                            "rng": rng.bit_generator.state,
                            "hall_of_fame": hof.entries,
                            "run_name": run_name,
                            "config": cfg,
                        }
                    )
                )
                tmp.replace(state_path)
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
    parser.add_argument("--samples", type=int)
    parser.add_argument("--games", dest="games_per_eval", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k not in ("config", "resume")}
    train(load_config(args.config, overrides), resume=args.resume)


if __name__ == "__main__":
    main()
