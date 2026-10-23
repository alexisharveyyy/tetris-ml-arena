"""Shared scaffolding for the DQN and PPO trainers.

* Wall-clock budget and run directories.
* Periodic evaluation of saved checkpoints on the 30 *selection* seeds of the
  garbage setting, in a separate process pool so training does not pause.
* Learning curves to TensorBoard, a CSV and a PNG under ``reports/``.
* Final checkpoint choice by selection fitness only; held-out seeds are used
  later by ``scripts/evaluate.py``.
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tetris.agents.evaluation import GARBAGE_SETTING, EvalSetting, GameStats, fitness, summarize
from tetris.agents.linear_training import selection_seeds
from tetris.agents.registry import CHECKPOINT_DIR, ROOT


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    return cfg


def setting_for(cfg: dict[str, Any]) -> EvalSetting:
    base = GARBAGE_SETTING.to_dict()
    for k in ("max_pieces", "garbage_every", "garbage_rows", "include_hold"):
        if k in cfg:
            base[k] = cfg[k]
    return EvalSetting(**base)


class Budget:
    """Wall-clock budget that survives resume (elapsed time is carried over)."""

    def __init__(self, seconds: float, already_elapsed: float = 0.0) -> None:
        self.seconds = seconds
        self._start = time.perf_counter() - already_elapsed

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    @property
    def fraction(self) -> float:
        return min(1.0, self.elapsed / self.seconds)

    @property
    def done(self) -> bool:
        return self.elapsed >= self.seconds


def _eval_task(args: tuple[str, str, int, dict[str, Any]]) -> dict[str, Any]:
    from tetris.agents.evaluation import _agent_task

    return _agent_task(args)


@dataclass
class EvalRecord:
    checkpoint: str
    step: int
    elapsed: float
    fitness: float
    summary: dict[str, float]


@dataclass
class SelectionEvaluator:
    """Evaluates checkpoints on the selection seeds in background processes."""

    agent_key: str
    workers: int = 3
    games: int = 30
    setting: EvalSetting = GARBAGE_SETTING
    records: list[EvalRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._pool = mp.get_context("spawn").Pool(self.workers)
        self._pending: list[tuple[str, int, float, Any]] = []
        self.seeds = selection_seeds(self.games)

    def submit(self, checkpoint: Path, step: int, elapsed: float) -> None:
        tasks = [(self.agent_key, str(checkpoint), s, self.setting.to_dict()) for s in self.seeds]
        self._pending.append(
            (str(checkpoint), step, elapsed, self._pool.map_async(_eval_task, tasks))
        )

    def poll(self, block: bool = False) -> list[EvalRecord]:
        done = []
        still = []
        for ckpt, step, elapsed, res in self._pending:
            if block:
                res.wait()
            if res.ready():
                stats = [GameStats(**r) for r in res.get()]
                rec = EvalRecord(ckpt, step, elapsed, fitness(stats), summarize(stats))
                self.records.append(rec)
                done.append(rec)
            else:
                still.append((ckpt, step, elapsed, res))
        self._pending = still
        return done

    @property
    def pending(self) -> int:
        return len(self._pending)

    def best(self) -> EvalRecord | None:
        return max(self.records, key=lambda r: r.fitness) if self.records else None

    def close(self) -> None:
        self._pool.close()
        self._pool.join()


class CurveLogger:
    """TensorBoard scalars plus a CSV and PNG of the selection-seed learning curve."""

    def __init__(self, agent_key: str, run_name: str) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.agent_key = agent_key
        self.run_name = run_name
        self.writer = SummaryWriter(str(ROOT / "runs" / agent_key / run_name))
        self.reports = ROOT / "reports" / "training" / "curves"
        self.reports.mkdir(parents=True, exist_ok=True)
        self.csv_path = CHECKPOINT_DIR / agent_key / run_name / "selection_curve.csv"

    def scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def eval_record(self, rec: EvalRecord) -> None:
        m = rec.summary
        for key in ("fitness", "mean_pieces", "mean_lines", "score_per_piece", "top_out_rate"):
            self.writer.add_scalar(f"selection/{key}", m[key], rec.step)
        self.writer.add_scalar("selection/fitness_vs_minutes", m["fitness"], int(rec.elapsed / 60))
        self.writer.flush()
        new = not self.csv_path.exists()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(
                    [
                        "step",
                        "minutes",
                        "fitness",
                        "mean_pieces",
                        "top_out_rate",
                        "score_per_piece",
                        "checkpoint",
                    ]
                )
            w.writerow(
                [
                    rec.step,
                    f"{rec.elapsed / 60:.1f}",
                    f"{m['fitness']:.4f}",
                    f"{m['mean_pieces']:.1f}",
                    f"{m['top_out_rate']:.3f}",
                    f"{m['score_per_piece']:.0f}",
                    rec.checkpoint,
                ]
            )
        self.plot()

    def plot(self, out: Path | None = None, references: dict[str, float] | None = None) -> Path:
        """PNG of selection fitness and top-out rate against training minutes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = list(csv.DictReader(self.csv_path.open())) if self.csv_path.exists() else []
        minutes = [float(r["minutes"]) for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(
            minutes, [float(r["fitness"]) for r in rows], marker="o", label=self.agent_key.upper()
        )
        for name, value in (references or {}).items():
            ax1.axhline(value, linestyle="--", linewidth=1, label=name)
        ax1.set_xlabel("training minutes")
        ax1.set_ylabel("fitness (30 selection seeds)")
        ax1.legend(loc="lower right")
        ax1.grid(alpha=0.3)
        ax2.plot(minutes, [float(r["top_out_rate"]) for r in rows], marker="o", color="tab:red")
        ax2.set_xlabel("training minutes")
        ax2.set_ylabel("top-out rate")
        ax2.set_ylim(-0.02, 1.02)
        ax2.grid(alpha=0.3)
        fig.suptitle(f"{self.agent_key.upper()} learning curve ({self.run_name}), garbage setting")
        fig.tight_layout()
        out = out or self.reports / f"curve_{self.agent_key}_{self.run_name}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        return out

    def close(self) -> None:
        self.writer.close()


def finalize_selection(
    agent_key: str, evaluator: SelectionEvaluator, run_dir: Path, promote: bool
) -> dict[str, Any]:
    """Pick the best evaluated checkpoint by selection fitness; optionally make it best.pt."""
    evaluator.poll(block=True)
    best = evaluator.best()
    if best is None:
        raise RuntimeError("no checkpoint was evaluated")
    summary = {
        "agent": agent_key,
        "selected_checkpoint": best.checkpoint,
        "selected_step": best.step,
        "selected_minutes": best.elapsed / 60,
        "selection_fitness": best.fitness,
        "selection_summary": best.summary,
        "evaluations": [
            {
                "checkpoint": r.checkpoint,
                "step": r.step,
                "minutes": r.elapsed / 60,
                "fitness": r.fitness,
            }
            for r in sorted(evaluator.records, key=lambda r: r.step)
        ],
    }
    (run_dir / "selection.json").write_text(json.dumps(summary, indent=2))
    if promote:
        shutil.copy(best.checkpoint, CHECKPOINT_DIR / agent_key / "best.pt")
        (CHECKPOINT_DIR / agent_key / "selection.json").write_text(json.dumps(summary, indent=2))
    return summary


def linear_references() -> dict[str, float]:
    """Selection-seed fitness of the linear agents, for the curve plots (cached)."""
    cache = CHECKPOINT_DIR / "selection_references.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from tetris.agents.evaluation import evaluate_agent

    seeds = selection_seeds(30)
    refs = {}
    for key, label in (("cem", "CEM"), ("ga", "GA"), ("dellacherie", "Dellacherie")):
        refs[label] = fitness(evaluate_agent(key, seeds, GARBAGE_SETTING))
    cache.write_text(json.dumps(refs, indent=2))
    return refs


def mean_episode_stats(episodes: list[dict[str, Any]]) -> dict[str, float]:
    if not episodes:
        return {}
    return {
        "pieces": float(np.mean([e["pieces"] for e in episodes])),
        "lines": float(np.mean([e["lines"] for e in episodes])),
        "return": float(np.mean([e["return"] for e in episodes])),
        "top_out_rate": float(np.mean([e["topped_out"] for e in episodes])),
    }
