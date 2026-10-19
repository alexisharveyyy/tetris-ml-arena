"""Training scaffolding shared by the GA and CEM trainers.

Seed protocol:

* Each generation draws a fresh set of training seeds, and every individual in
  that generation plays the same seeds (common random numbers).
* After training, a hall of fame (the best candidate of each generation) is
  re-evaluated on a fixed *selection* seed set to pick the final weights.
* The chosen weights are then reported on the fixed *held-out* seed set,
  which is never used for any decision.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tetris.agents.evaluation import (
    HELD_OUT_SEEDS,
    PLAIN_SETTING,
    SETTINGS,
    EvalSetting,
    GameStats,
    LinearEvaluator,
    fitness,
    summarize,
)
from tetris.agents.linear import save_linear_checkpoint
from tetris.env.features import FEATURE_NAMES

TRAIN_SEED_RANGE = 500_000
SELECTION_SEED_BASE = 800_000


def generation_seeds(run_seed: int, generation: int, n: int) -> list[int]:
    """Training seeds for one generation (shared by all individuals in it)."""
    rng = np.random.default_rng([run_seed, generation])
    return [int(s) for s in rng.choice(TRAIN_SEED_RANGE, size=n, replace=False)]


def selection_seeds(n: int) -> list[int]:
    return list(range(SELECTION_SEED_BASE, SELECTION_SEED_BASE + n))


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    return cfg


def setting_from_config(cfg: dict[str, Any]) -> EvalSetting:
    base = SETTINGS[cfg.get("setting", "garbage")]
    overrides = {k: cfg[k] for k in ("max_pieces", "garbage_every", "garbage_rows") if k in cfg}
    return EvalSetting(**{**base.to_dict(), **overrides})


def normalize(w: np.ndarray) -> np.ndarray:
    """Unit-length weights (a linear evaluation is invariant to positive scale)."""
    n = np.linalg.norm(w)
    return w / n if n > 0 else w


@dataclass
class HallOfFame:
    """Best candidate of each generation."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def add(self, generation: int, weights: np.ndarray, train_fitness: float) -> None:
        self.entries.append(
            {
                "generation": generation,
                "weights": [float(x) for x in weights],
                "fitness": train_fitness,
            }
        )

    def last(self, n: int) -> list[dict[str, Any]]:
        return self.entries[-n:]


class TrainingLogger:
    """TensorBoard plus a plain-text progress line per generation."""

    def __init__(self, log_dir: Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(str(log_dir))

    def generation(
        self, gen: int, fitnesses: np.ndarray, best_stats: list[GameStats], **extra: float
    ) -> None:
        m = summarize(best_stats)
        self.writer.add_scalar("fitness/max", float(fitnesses.max()), gen)
        self.writer.add_scalar("fitness/mean", float(fitnesses.mean()), gen)
        for key in (
            "mean_pieces",
            "mean_lines",
            "score_per_piece",
            "lines_per_piece",
            "top_out_rate",
        ):
            self.writer.add_scalar(f"best/{key}", m[key], gen)
        for key, value in extra.items():
            self.writer.add_scalar(f"search/{key}", value, gen)
        self.writer.flush()
        print(
            f"gen {gen:3d}  fit max {fitnesses.max():.3f} mean {fitnesses.mean():.3f}  "
            f"best: pieces {m['mean_pieces']:.0f} spp {m['score_per_piece']:.0f} "
            f"top-out {m['top_out_rate']:.2f}",
            flush=True,
        )

    def close(self) -> None:
        self.writer.close()


def finalize(
    agent_key: str,
    out_dir: Path,
    evaluator: LinearEvaluator,
    hof: HallOfFame,
    setting: EvalSetting,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Pick the final weights on selection seeds and report on held-out seeds."""
    candidates = hof.last(int(cfg.get("final_candidates", 10)))
    sel_seeds = selection_seeds(int(cfg.get("selection_games", 30)))
    t0 = time.time()
    results = evaluator.evaluate([np.array(c["weights"]) for c in candidates], sel_seeds, setting)
    sel_fit = [fitness(r) for r in results]
    best_i = int(np.argmax(sel_fit))
    best = candidates[best_i]
    weights = np.array(best["weights"])
    print(
        f"selection: picked generation {best['generation']} "
        f"(selection fitness {sel_fit[best_i]:.3f}, {time.time() - t0:.0f}s)",
        flush=True,
    )

    held = list(HELD_OUT_SEEDS[: int(cfg.get("final_eval_games", 100))])
    report = {
        "agent": agent_key,
        "selected_generation": best["generation"],
        "selection_fitness": sel_fit[best_i],
        "held_out": {},
    }
    for s in (setting, PLAIN_SETTING):
        stats = evaluator.evaluate([weights], held, s)[0]
        report["held_out"][s.name] = {"setting": s.to_dict(), **summarize(stats)}
    save_linear_checkpoint(
        out_dir / "best.json",
        weights,
        agent=agent_key,
        generation=best["generation"],
        config=cfg,
        report=report,
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["held_out"], indent=2), flush=True)
    return report


def weights_table(weights: np.ndarray) -> str:
    return "\n".join(f"  {n:18s} {w:+.4f}" for n, w in zip(FEATURE_NAMES, weights, strict=True))
