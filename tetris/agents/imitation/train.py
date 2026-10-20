"""Behavior cloning of the GA heuristic (pure and with DAgger) and of your recorded games.

Usage:
    python -m tetris.agents.imitation.train pure    [--budget-minutes M] [--config PATH]
    python -m tetris.agents.imitation.train dagger  [--rounds N] [--round-minutes M]
    python -m tetris.agents.imitation.train choose  # bc = better of bc_pure, bc_dagger
    python -m tetris.agents.imitation.train you     # needs about 2,000 recorded placements

The policy is the PPO afterstate policy (each legal action's logit scores its
afterstate). The loss is the negative log of the total probability of the
actions whose afterstate equals the teacher's (see dataset.py).

* ``pure``: trains on the logged GA games only. Checkpoints are evaluated on
  the 30 selection seeds during training and the best becomes
  ``checkpoints/bc_pure/best.pt``.
* ``dagger``: starts from the pure BC model. Each round the current policy
  plays new games, the GA labels every state the policy visits, and training
  continues on the GA games plus all DAgger data. The best checkpoint of the
  rounds (selection seeds) becomes ``checkpoints/bc_dagger/best.pt``.
* ``choose``: the playable ``bc`` agent is whichever of the two has the higher
  selection fitness; recorded in ``checkpoints/bc/version_choice.json``.
* ``you``: fine-tunes the chosen model on your recordings (on-space steps
  only). Games are split into train, validation (picks the checkpoint) and
  held-out test games, and top-1 and top-3 agreement on the test games are
  reported in ``reports/bc_you.json``.

Every run on MPS starts with the gradient pre-flight check and runs under the
parameter-change watchdog (tetris/rl/safeguards.py).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tetris.agents.evaluation import default_workers
from tetris.agents.imitation.dataset import (
    ImitationData,
    encode_files,
    log_files,
    split_episodes,
)
from tetris.agents.registry import CHECKPOINT_DIR, ROOT
from tetris.engine.placement import NUM_ACTIONS
from tetris.rl.nets import (
    MASKED_LOGIT,
    build,
    load_checkpoint,
    save_checkpoint,
    score_afterstate_rows,
    select_device,
)
from tetris.rl.safeguards import ParameterChange, ProgressWatchdog, preflight
from tetris.rl.training import (
    Budget,
    CurveLogger,
    SelectionEvaluator,
    finalize_selection,
    linear_references,
    load_config,
)

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
DAGGER_SEED_BASE = 650_000
DAGGER_DIR = ROOT / "data" / "logs" / "dagger"


# ----------------------------------------------------------------- batches


def batch_tensors(
    data: ImitationData, states: np.ndarray, device: torch.device
) -> list[torch.Tensor]:
    """Rows of ``states`` as tensors: boards, extras, owning state, action, target."""
    rows, owner = data.rows_of(states)
    return [
        torch.from_numpy(data.boards[rows]).to(device),
        torch.from_numpy(data.extras[rows].astype(np.float32)).to(device),
        torch.from_numpy(owner.astype(np.int64)).to(device),
        torch.from_numpy(data.actions[rows].astype(np.int64)).to(device),
        torch.from_numpy(data.target[rows]).to(device),
    ]


def batch_logits(net: torch.nn.Module, batch: list[torch.Tensor]) -> torch.Tensor:
    """Masked logits ``(B, 80)`` for a batch from :func:`batch_tensors`."""
    boards, extras, owner, actions, _ = batch
    n_states = int(owner[-1]) + 1  # owners are sorted
    scores = score_afterstate_rows(net.scorer, boards, extras)
    logits = torch.full((n_states, NUM_ACTIONS), MASKED_LOGIT, device=scores.device)
    return logits.index_put((owner, actions), scores)


def bc_loss(net: torch.nn.Module, batch: list[torch.Tensor]) -> torch.Tensor:
    """-log P(any action equivalent to the teacher's), averaged over states."""
    logits = batch_logits(net, batch)
    _, _, owner, actions, target = batch
    target_mask = torch.zeros_like(logits, dtype=torch.bool).index_put(
        (owner[target], actions[target]), torch.ones_like(owner[target], dtype=torch.bool)
    )
    target_logits = logits.masked_fill(~target_mask, MASKED_LOGIT)
    return (torch.logsumexp(logits, 1) - torch.logsumexp(target_logits, 1)).mean()


@torch.no_grad()
def agreement(
    net: torch.nn.Module,
    data: ImitationData,
    states: np.ndarray,
    device: torch.device,
    batch_states: int = 512,
) -> dict[str, float]:
    """Top-1 and top-3 agreement with the teacher, and the loss, on ``states``.

    Ranks are over distinct afterstates: equivalent actions have identical
    inputs and therefore the same score, so they count once. A state is a
    top-k hit when fewer than k distinct non-teacher afterstates score higher
    than the teacher's afterstate.
    """
    net.eval()
    top1 = top3 = 0
    loss_sum = 0.0
    for i in range(0, len(states), batch_states):
        chunk = states[i : i + batch_states]
        batch = batch_tensors(data, chunk, device)
        loss_sum += float(bc_loss(net, batch)) * len(chunk)
        scores = score_afterstate_rows(net.scorer, batch[0], batch[1]).cpu().numpy()
        owner = batch[2].cpu().numpy()
        target = batch[4].cpu().numpy()
        bounds = np.searchsorted(owner, np.arange(len(chunk) + 1))
        for j in range(len(chunk)):
            s, t = scores[bounds[j] : bounds[j + 1]], target[bounds[j] : bounds[j + 1]]
            best_target = s[t].max()
            others = np.unique(np.round(s[~t], 6))
            better = int((others > best_target + 1e-6).sum())
            top1 += better == 0
            top3 += better < 3
    net.train()
    n = max(len(states), 1)
    return {"top1": float(top1 / n), "top3": float(top3 / n), "loss": loss_sum / n}


# ------------------------------------------------------------------ data


def cached_dataset(files: list[Path], cache_dir: Path, name: str) -> ImitationData:
    """Encode ``files`` once; the cache is keyed by the file list and sizes."""
    key = hashlib.sha1("\n".join(f"{p}:{p.stat().st_size}" for p in files).encode()).hexdigest()[
        :12
    ]
    path = ROOT / cache_dir / f"{name}_{key}.npz"
    if path.exists():
        return ImitationData.load(path)
    t0 = time.time()
    data = encode_files(files, workers=default_workers())
    data.save(path)
    print(
        f"encoded {len(files)} episodes: {len(data)} states, {data.n_rows} afterstate rows "
        f"in {time.time() - t0:.0f}s -> {path}",
        flush=True,
    )
    return data


def teacher_data(cfg: dict[str, Any]) -> ImitationData:
    files = log_files([ROOT / d for d in cfg["log_dirs"]])
    if not files:
        raise FileNotFoundError(f"no logs under {cfg['log_dirs']}; run scripts/log_games.py")
    return cached_dataset(files, Path(cfg["cache_dir"]), cfg["teacher"])


# -------------------------------------------------------------- training


def cosine(lr: float, lr_final: float, fraction: float) -> float:
    f = min(1.0, max(0.0, fraction))
    return lr_final + 0.5 * (lr - lr_final) * (1.0 + math.cos(math.pi * f))


class Trainer:
    """Minibatch BC on a fixed dataset for a wall-clock budget."""

    def __init__(
        self, cfg: dict[str, Any], agent_key: str, run_name: str, init: str | None = None
    ) -> None:
        self.cfg = cfg
        self.agent_key = agent_key
        self.run_name = run_name
        self.run_dir = CHECKPOINT_DIR / agent_key / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(int(cfg["seed"]))
        torch.set_num_threads(int(cfg.get("torch_threads", 2)))
        self.rng = np.random.default_rng(int(cfg["seed"]))
        self.net = build("ppo", cfg.get("model", {}))
        if init is not None:
            src, _ = load_checkpoint(init, "ppo", "cpu")
            self.net.load_state_dict(src.state_dict())
        self.device = select_device(cfg.get("device", "auto"))
        self.steps = 0
        self.elapsed = 0.0
        self.logger = CurveLogger(agent_key, run_name)
        self.watchdog = ProgressWatchdog(
            f"{agent_key} {run_name}", "relative parameter change", 1e-6, 10
        )

    def check_device(self, data: ImitationData) -> None:
        """MPS pre-flight on one real batch; switches to the CPU if it fails."""
        states = np.arange(min(len(data), int(self.cfg["batch_states"])))
        batch = batch_tensors(data, states, torch.device("cpu"))
        check = preflight(
            f"{self.agent_key} {self.run_name}",
            self.net,
            batch,
            bc_loss,
            self.device,
            f"batch {len(states)} states ({len(batch[0]):,} afterstates)",
            self.run_dir,
        )
        self.device = torch.device(check.final_device)
        self.net.to(self.device)

    def fit(
        self,
        data: ImitationData,
        train_states: np.ndarray,
        minutes: float,
        lr: float,
        lr_final: float,
        on_tick=None,
        tick_seconds: float = 180.0,
        first_tick: float | None = None,
    ) -> None:
        """Train for ``minutes``; ``on_tick(trainer)`` runs every ``tick_seconds``."""
        cfg = self.cfg
        net = self.net
        net.train()
        opt = torch.optim.AdamW(
            net.scorer.parameters(), lr=lr, weight_decay=float(cfg["weight_decay"])
        )
        budget = Budget(minutes * 60)
        change = ParameterChange(net.scorer)
        bsz = int(cfg["batch_states"])
        perm, pos = self.rng.permutation(train_states), 0
        next_log = 30.0
        next_tick = first_tick if first_tick is not None else tick_seconds
        losses: list[float] = []
        while not budget.done:
            if pos + bsz > len(perm):
                perm, pos = self.rng.permutation(train_states), 0
            states = np.sort(perm[pos : pos + bsz])
            pos += bsz
            for g in opt.param_groups:
                g["lr"] = cosine(lr, lr_final, budget.fraction)
            loss = bc_loss(net, batch_tensors(data, states, self.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.scorer.parameters(), float(cfg["grad_clip"]))
            opt.step()
            self.steps += 1
            losses.append(float(loss.detach()))
            if budget.elapsed >= next_log:
                delta = change()
                self.watchdog.update(delta)
                mean_loss = float(np.mean(losses))
                losses.clear()
                total = self.elapsed + budget.elapsed
                self.logger.scalar("train/loss", mean_loss, self.steps)
                self.logger.scalar("train/param_change", delta, self.steps)
                self.logger.scalar("train/lr", opt.param_groups[0]["lr"], self.steps)
                print(
                    f"{total / 60:6.1f} min  step {self.steps:7d}  loss {mean_loss:.4f}"
                    f"  lr {opt.param_groups[0]['lr']:.1e}  param change {delta:.1e}",
                    flush=True,
                )
                next_log = budget.elapsed + 30.0
            if on_tick is not None and budget.elapsed >= next_tick:
                on_tick(self, self.elapsed + budget.elapsed)
                next_tick = budget.elapsed + tick_seconds
        self.elapsed += budget.elapsed

    def save(self, name: str, **meta: Any) -> Path:
        path = self.run_dir / name
        save_checkpoint(path, "ppo", self.net, steps=self.steps, run=self.run_name, **meta)
        return path


def selection_tick(evaluator: SelectionEvaluator, val: tuple[ImitationData, np.ndarray] | None):
    """Every tick: save a checkpoint, queue it for the selection seeds, log agreement."""

    def tick(trainer: Trainer, elapsed: float) -> None:
        path = trainer.save(f"eval_{int(elapsed / 60):04d}min_{trainer.steps}.pt")
        evaluator.submit(path, trainer.steps, elapsed)
        if val is not None:
            m = agreement(trainer.net, val[0], val[1], trainer.device)
            for k, v in m.items():
                trainer.logger.scalar(f"val/{k}", v, trainer.steps)
            print(
                f"   val top-1 {m['top1']:.3f}  top-3 {m['top3']:.3f}  loss {m['loss']:.3f}",
                flush=True,
            )
        for rec in evaluator.poll():
            trainer.logger.eval_record(rec)
            print(f"   eval {Path(rec.checkpoint).name}: fitness {rec.fitness:.3f}", flush=True)

    return tick


def finish(trainer: Trainer, evaluator: SelectionEvaluator, promote: bool = True) -> dict:
    path = trainer.save(f"eval_{int(trainer.elapsed / 60):04d}min_{trainer.steps}.pt")
    evaluator.submit(path, trainer.steps, trainer.elapsed)
    for rec in evaluator.poll(block=True):
        trainer.logger.eval_record(rec)
    summary = finalize_selection(trainer.agent_key, evaluator, trainer.run_dir, promote)
    trainer.logger.plot(references=linear_references())
    trainer.logger.close()
    evaluator.close()
    print(
        f"selected {Path(summary['selected_checkpoint']).name} "
        f"(selection fitness {summary['selection_fitness']:.3f})",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------- variants


def train_pure(cfg: dict[str, Any], run_name: str = "seed0") -> dict:
    data = teacher_data(cfg)
    train_idx, val_idx = split_episodes(data, float(cfg["val_fraction"]), int(cfg["seed"]))
    print(
        f"pure BC: {len(train_idx)} train / {len(val_idx)} validation states "
        f"({data.n_rows} afterstate rows)",
        flush=True,
    )
    trainer = Trainer(cfg, "bc_pure", run_name)
    trainer.check_device(data)
    evaluator = SelectionEvaluator("bc_pure", workers=int(cfg["eval_workers"]))
    trainer.fit(
        data,
        train_idx,
        float(cfg["budget_minutes"]),
        float(cfg["lr"]),
        float(cfg["lr_final"]),
        selection_tick(evaluator, (data, val_idx)),
        float(cfg["eval_every_minutes"]) * 60,
        float(cfg["first_eval_minutes"]) * 60,
    )
    summary = finish(trainer, evaluator)
    write_agreement_report(cfg, "bc_pure", summary, data, val_idx)
    return summary


def train_dagger(cfg: dict[str, Any], run_name: str = "seed0") -> dict:
    from tetris.agents.imitation.dagger import collect_round

    init = CHECKPOINT_DIR / "bc_pure" / "best.pt"
    if not init.exists():
        raise FileNotFoundError("DAgger starts from the pure BC model; run `pure` first")
    base = teacher_data(cfg)
    train_idx, val_idx = split_episodes(base, float(cfg["val_fraction"]), int(cfg["seed"]))
    trainer = Trainer(cfg, "bc_dagger", run_name, init=str(init))
    trainer.check_device(base)
    evaluator = SelectionEvaluator("bc_dagger", workers=int(cfg["eval_workers"]))
    parts = [base.subset(train_idx)]
    current = str(init)
    for r in range(1, int(cfg["dagger_rounds"]) + 1):
        t0 = time.time()
        files = collect_round(
            current,
            r,
            int(cfg["dagger_games_per_setting"]),
            DAGGER_DIR / run_name,
            DAGGER_SEED_BASE + 1000 * r,
            default_workers(),
        )
        parts.append(cached_dataset(files, Path(cfg["cache_dir"]), f"dagger_{run_name}_r{r}"))
        data = ImitationData.concat(parts)
        print(
            f"DAgger round {r}: {len(files)} games labeled by the GA in {time.time() - t0:.0f}s; "
            f"training on {len(data)} states",
            flush=True,
        )
        trainer.fit(
            data,
            np.arange(len(data)),
            float(cfg["dagger_round_minutes"]),
            float(cfg["lr"]) * 0.5,
            float(cfg["lr_final"]),
            selection_tick(evaluator, (base, val_idx)),
            float(cfg["eval_every_minutes"]) * 60,
        )
        current = str(trainer.save(f"round{r}.pt", dagger_round=r))
    summary = finish(trainer, evaluator)
    write_agreement_report(cfg, "bc_dagger", summary, base, val_idx)
    return summary


def write_agreement_report(
    cfg: dict[str, Any], key: str, summary: dict, data: ImitationData, states: np.ndarray
) -> None:
    net, _ = load_checkpoint(summary["selected_checkpoint"], "ppo", "cpu")
    m = agreement(net, data, states, torch.device("cpu"))
    out = {**summary, "validation_agreement_with_ga": m}
    (CHECKPOINT_DIR / key / "selection.json").write_text(json.dumps(out, indent=2))
    print(f"{key}: GA agreement on validation games: {m}", flush=True)


def choose_version() -> dict:
    """Playable ``bc`` = higher selection fitness of bc_pure and bc_dagger (ties: pure)."""
    fits = {}
    for key in ("bc_pure", "bc_dagger"):
        sel = CHECKPOINT_DIR / key / "selection.json"
        if sel.exists():
            fits[key] = json.loads(sel.read_text())["selection_fitness"]
    if not fits:
        raise FileNotFoundError("no BC variant has been trained")
    chosen = max(fits, key=lambda k: (fits[k], k == "bc_pure"))
    out = CHECKPOINT_DIR / "bc"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(CHECKPOINT_DIR / chosen / "best.pt", out / "best.pt")
    choice = {"selection_fitness": fits, "chosen": chosen}
    (out / "version_choice.json").write_text(json.dumps(choice, indent=2))
    print(f"Imitation (BC) = {chosen}: {fits}", flush=True)
    return choice


def train_you(cfg: dict[str, Any], run_name: str = "seed0") -> dict:
    """Fine-tune on your recordings; report agreement on held-out recorded games."""
    from tetris.match.recording import recording_files, recording_totals

    report_path = ROOT / "reports" / "bc_you.json"
    totals = recording_totals()
    need = int(cfg["you_min_placements"])
    if totals["on_space"] < need:
        msg = f"{totals['on_space']} on-space placements recorded; BC (you) needs about {need}"
        print(f"skipping BC (you): {msg}", flush=True)
        report_path.write_text(json.dumps({"skipped": msg, "recordings": totals}, indent=2))
        return {"skipped": msg}
    init = CHECKPOINT_DIR / "bc" / "best.pt"
    files = recording_files()
    data = cached_dataset(files, Path(cfg["cache_dir"]), "human")
    train_idx, val_idx, test_idx = split_recordings(
        data, float(cfg["you_val_fraction"]), float(cfg["you_test_fraction"])
    )
    trainer = Trainer(cfg, "bc_you", run_name, init=str(init))
    trainer.check_device(data)
    before = agreement(trainer.net, data, test_idx, trainer.device)
    best = {"top1": -1.0}
    best_state: dict[str, torch.Tensor] = {}

    def tick(tr: Trainer, elapsed: float) -> None:
        nonlocal best, best_state
        m = agreement(tr.net, data, val_idx, tr.device)
        print(f"   validation top-1 {m['top1']:.3f}  top-3 {m['top3']:.3f}", flush=True)
        if m["top1"] > best["top1"]:
            best = {**m, "minutes": elapsed / 60}
            best_state = copy.deepcopy({k: v.cpu() for k, v in tr.net.state_dict().items()})

    trainer.fit(
        data,
        train_idx,
        float(cfg["you_minutes"]),
        float(cfg["you_lr"]),
        float(cfg["you_lr"]) * 0.1,
        tick,
        tick_seconds=30.0,
        first_tick=0.0,
    )
    tick(trainer, trainer.elapsed)
    trainer.net.load_state_dict(best_state)
    after = agreement(trainer.net, data, test_idx, trainer.device)
    trainer.net.cpu()
    path = trainer.save("best.pt", validation=best)
    shutil.copy(path, CHECKPOINT_DIR / "bc_you" / "best.pt")
    report = {
        "recordings": totals,
        "states": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "test_agreement_before_fine_tuning": before,
        "test_agreement": after,
        "validation_best": best,
        "initialized_from": str(init),
    }
    report_path.write_text(json.dumps(report, indent=2))
    trainer.logger.close()
    print(json.dumps(report, indent=2), flush=True)
    return report


def split_recordings(
    data: ImitationData, val_fraction: float, test_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train / validation / test state indices by recorded game, in recording order.

    The most recent games are the test set, the ones before them validation.
    With fewer than three games, each split is a contiguous block of states.
    """
    episodes = np.unique(data.episode)
    if len(episodes) >= 3:
        n_test = max(1, round(test_fraction * len(episodes)))
        n_val = max(1, round(val_fraction * len(episodes)))
        test_eps = episodes[-n_test:]
        val_eps = episodes[-n_test - n_val : -n_test]
        test = np.flatnonzero(np.isin(data.episode, test_eps))
        val = np.flatnonzero(np.isin(data.episode, val_eps))
        train = np.flatnonzero(~np.isin(data.episode, np.concatenate([test_eps, val_eps])))
        return train, val, test
    n = len(data)
    n_test, n_val = max(1, int(test_fraction * n)), max(1, int(val_fraction * n))
    idx = np.arange(n)
    return idx[: n - n_test - n_val], idx[n - n_test - n_val : n - n_test], idx[n - n_test :]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("variant", choices=["pure", "dagger", "choose", "you"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--budget-minutes", dest="budget_minutes", type=float)
    parser.add_argument("--rounds", dest="dagger_rounds", type=int)
    parser.add_argument("--round-minutes", dest="dagger_round_minutes", type=float)
    parser.add_argument("--games-per-setting", dest="dagger_games_per_setting", type=int)
    parser.add_argument("--you-minutes", dest="you_minutes", type=float)
    parser.add_argument("--device")
    parser.add_argument("--run-name", default="seed0")
    args = parser.parse_args(argv)
    keys = (
        "budget_minutes",
        "dagger_rounds",
        "dagger_round_minutes",
        "dagger_games_per_setting",
        "you_minutes",
        "device",
    )
    cfg = load_config(args.config, {k: getattr(args, k) for k in keys})
    if args.variant == "pure":
        train_pure(cfg, args.run_name)
    elif args.variant == "dagger":
        train_dagger(cfg, args.run_name)
    elif args.variant == "choose":
        choose_version()
    else:
        train_you(cfg, args.run_name)


if __name__ == "__main__":
    main()
