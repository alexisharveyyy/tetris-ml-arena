"""Decision Transformer training, checkpoint selection and target-return calibration.

Usage:
    python -m tetris.agents.decision_transformer.train train [--budget-minutes M] [--run-name N]
        [--resume]
    python -m tetris.agents.decision_transformer.train calibrate [--run-name N --report R]
    python -m tetris.agents.decision_transformer.train choose [--run-name seed0_v2]

``train`` fits the model on the collected trajectories (``collect.py``) for a
wall-clock budget on MPS, with the gradient pre-flight check and the
parameter-change watchdog. Every ``eval_every_minutes`` a checkpoint plays the
30 selection seeds at the selection target; the best becomes
``checkpoints/dt/best.pt``.

``calibrate`` plays the selected model at every target in
``calibration_targets`` on the selection seeds and writes the achieved return
against the target (``checkpoints/dt/calibration.json``,
``reports/analysis/dt_calibration.md`` and ``.png``). The target with the highest
selection fitness becomes the model's default target, used on the leaderboard.
With ``--run-name`` it calibrates the model selected in that run instead
(``calibration.json`` next to it, ``reports/<report>.md``).

``choose`` keeps the promoted model as ``versions/v1``, stores the model
selected in ``--run-name`` as ``versions/v2``, and promotes v2 only if its
selection fitness is strictly higher (``checkpoints/dt/version_choice.json``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tetris.agents.decision_transformer.dataset import DTData, encode_dt_files
from tetris.agents.decision_transformer.model import DecisionTransformer
from tetris.agents.difficulty import DifficultyParams
from tetris.agents.evaluation import default_workers, evaluate_agent, fitness, summarize
from tetris.agents.imitation.dataset import log_files, split_episodes
from tetris.agents.linear_training import selection_seeds
from tetris.agents.registry import CHECKPOINT_DIR, ROOT
from tetris.engine.placement import NUM_ACTIONS
from tetris.rl.nets import MASKED_LOGIT, load_checkpoint, save_checkpoint, select_device, to_cpu
from tetris.rl.safeguards import ParameterChange, ProgressWatchdog, preflight
from tetris.rl.training import (
    Budget,
    CurveLogger,
    EvalRecord,
    SelectionEvaluator,
    finalize_selection,
    linear_references,
    load_config,
    setting_for,
)

AGENT_KEY = "dt"
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


# ------------------------------------------------------------------ data


def cached_dt_dataset(cfg: dict[str, Any], files: list[Path] | None = None) -> DTData:
    """Encode the collected trajectories once; the cache is keyed by the file list."""
    if files is None:
        files = log_files([ROOT / cfg["collect"]["out_dir"]])
    if not files:
        raise FileNotFoundError("no trajectories; run tetris.agents.decision_transformer.collect")
    key = hashlib.sha1("\n".join(f"{p}:{p.stat().st_size}" for p in files).encode()).hexdigest()[
        :12
    ]
    path = ROOT / cfg["cache_dir"] / f"dt_{key}.npz"
    if path.exists():
        return DTData.load(path)
    t0 = time.time()
    data = encode_dt_files(files, int(cfg["return_cap"]), workers=default_workers())
    data.save(path)
    print(
        f"encoded {len(files)} episodes: {len(data)} states, {data.steps.n_rows} afterstate rows "
        f"in {time.time() - t0:.0f}s -> {path}",
        flush=True,
    )
    return data


def selection_target(cfg: dict[str, Any], data: DTData) -> float:
    """``selection_target`` from the config, or the 90th percentile of episode returns."""
    value = cfg.get("selection_target", "auto")
    if value == "auto":
        return float(np.percentile(data.episode_returns(), 90))
    return float(value)


# --------------------------------------------------------------- batches


def window_states(data: DTData, ends: np.ndarray, context: int) -> list[np.ndarray]:
    """For each end state, the states of its window (at most ``context``, same episode)."""
    starts = np.maximum(data.episode_starts()[ends], ends - context + 1)
    return [np.arange(s, e + 1) for s, e in zip(starts, ends, strict=True)]


def dt_batch(
    data: DTData, ends: np.ndarray, context: int, device: torch.device
) -> list[torch.Tensor]:
    """Tensors for windows ending at ``ends`` (left-padded to ``context``).

    Order: rtg, obs_boards, obs_extras, timesteps, valid, slot, row boards,
    row extras, row owner, row action, row target, chosen row position. ``slot``
    maps each valid state to its flat ``B * K`` position; ``owner`` maps each
    afterstate row to its valid state.
    """
    windows = window_states(data, ends, context)
    b = len(windows)
    rtg = np.zeros((b, context), np.float32)
    obs_b = np.zeros((b, context, data.obs_boards.shape[1]), np.uint8)
    obs_e = np.zeros((b, context, data.obs_extras.shape[1]), np.float32)
    ts = np.zeros((b, context), np.int64)
    valid = np.zeros((b, context), bool)
    slots = []
    for i, w in enumerate(windows):
        j = context - len(w)
        rtg[i, j:], obs_b[i, j:], obs_e[i, j:] = data.rtg[w], data.obs_boards[w], data.obs_extras[w]
        ts[i, j:] = data.timestep[w]
        valid[i, j:] = True
        slots.append(i * context + np.arange(j, context))
    states = np.concatenate(windows)
    rows, owner = data.steps.rows_of(states)
    counts = data.steps.offsets[states + 1] - data.steps.offsets[states]
    first = np.concatenate([[0], np.cumsum(counts)[:-1]])
    chosen_pos = first + (data.chosen[states] - data.steps.offsets[states])
    arrays = [
        rtg,
        obs_b,
        obs_e,
        ts,
        valid,
        np.concatenate(slots).astype(np.int64),
        data.steps.boards[rows],
        data.steps.extras[rows].astype(np.float32),
        owner.astype(np.int64),
        data.steps.actions[rows].astype(np.int64),
        data.steps.target[rows],
        chosen_pos.astype(np.int64),
    ]
    return [torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in arrays]


def dt_logits(net: DecisionTransformer, batch: list[torch.Tensor]) -> torch.Tensor:
    """Masked logits ``(M, 80)`` for every valid state of a :func:`dt_batch`."""
    rtg, obs_b, obs_e, ts, valid, slot, boards, extras, owner, actions, _, chosen = batch
    b, k = rtg.shape
    emb = net.embed_afterstates(boards, extras)  # (N, d)
    act = torch.zeros(b * k, emb.shape[1], device=emb.device, dtype=emb.dtype)
    act = act.index_put((slot,), emb[chosen])
    queries = net(rtg, obs_b, obs_e, act.reshape(b, k, -1), ts, valid).reshape(b * k, -1)[slot]
    scores = net.action_logits(queries[owner], emb)
    logits = torch.full((len(slot), NUM_ACTIONS), MASKED_LOGIT, device=emb.device)
    return logits.index_put((owner, actions), scores)


def dt_loss(net: DecisionTransformer, batch: list[torch.Tensor]) -> torch.Tensor:
    """-log P(any action equivalent to the logged one), averaged over valid states."""
    logits = dt_logits(net, batch)
    owner, actions, target = batch[8], batch[9], batch[10]
    target_mask = torch.zeros_like(logits, dtype=torch.bool).index_put(
        (owner[target], actions[target]), torch.ones_like(owner[target], dtype=torch.bool)
    )
    target_logits = logits.masked_fill(~target_mask, MASKED_LOGIT)
    return (torch.logsumexp(logits, 1) - torch.logsumexp(target_logits, 1)).mean()


def preflight_loss(net: DecisionTransformer, batch: list[torch.Tensor]) -> torch.Tensor:
    """The training loss with dropout off, so the device and CPU runs are comparable."""
    net.eval()
    return dt_loss(net, batch)


@torch.no_grad()
def top1_agreement(
    net: DecisionTransformer, data: DTData, ends: np.ndarray, device: torch.device
) -> float:
    """Share of window-final states where the top action is equivalent to the logged one."""
    net.eval()
    hits = 0
    for i in range(0, len(ends), 64):
        batch = dt_batch(data, ends[i : i + 64], net.context, device)
        logits = dt_logits(net, batch)
        owner, actions, target = batch[8], batch[9], batch[10]
        last = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
        slot = batch[5]
        last[(slot % net.context) == net.context - 1] = True  # window-final states
        best = logits.argmax(1)
        is_target = torch.zeros_like(logits, dtype=torch.bool).index_put(
            (owner[target], actions[target]), torch.ones_like(owner[target], dtype=torch.bool)
        )
        ok = is_target.gather(1, best.unsqueeze(1)).squeeze(1)
        hits += int(ok[last].sum())
    net.train()
    return hits / max(len(ends), 1)


def lr_at(step: int, fraction: float, cfg: dict[str, Any]) -> float:
    lr, lr_final, warm = float(cfg["lr"]), float(cfg["lr_final"]), int(cfg["warmup_steps"])
    if step < warm:
        return lr * (step + 1) / warm
    return lr_final + 0.5 * (lr - lr_final) * (1 + math.cos(math.pi * min(1.0, fraction)))


# -------------------------------------------------------------- training


def train(
    cfg: dict[str, Any], run_name: str = "seed0", promote: bool = True, resume: bool = False
) -> dict[str, Any]:
    """Train for the budget; with ``resume``, continue from the run's ``state.pt``.

    The state (weights, optimizer, step count, elapsed time, RNG states and
    the selection evaluations so far) is saved at every evaluation.
    """
    torch.manual_seed(int(cfg["seed"]))
    torch.set_num_threads(int(cfg.get("torch_threads", 2)))
    rng = np.random.default_rng(int(cfg["seed"]))
    data = cached_dt_dataset(cfg)
    train_idx, val_idx = split_episodes(data.steps, float(cfg["val_fraction"]), int(cfg["seed"]))
    target = selection_target(cfg, data)
    cap = int(cfg["return_cap"])
    returns = data.episode_returns()
    print(
        f"DT: {len(train_idx)} train / {len(val_idx)} validation states, "
        f"{len(returns)} episodes, returns {returns.min():.3f} to {returns.max():.3f} "
        f"(median {np.median(returns):.3f}); selection target {target:.3f}",
        flush=True,
    )
    run_dir = CHECKPOINT_DIR / AGENT_KEY / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    net = DecisionTransformer(**cfg["model"])
    device = select_device(cfg.get("device", "auto"))
    bsz, context = int(cfg["batch_windows"]), int(cfg["model"]["context"])
    sample = dt_batch(data, rng.choice(train_idx, bsz), context, torch.device("cpu"))
    check = preflight(
        f"{AGENT_KEY} {run_name}",
        net,
        sample,
        preflight_loss,
        device,
        f"batch {bsz} windows x {context} pieces ({len(sample[6]):,} afterstates)",
        run_dir,
    )
    device = torch.device(check.final_device)
    net.to(device).train()
    meta = {
        "default_target": target,
        "selection_target": target,
        "return_cap": cap,
        "run": run_name,
    }
    opt = torch.optim.AdamW(
        net.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])
    )
    logger = CurveLogger(AGENT_KEY, run_name)
    evaluator = SelectionEvaluator(
        AGENT_KEY, workers=int(cfg["eval_workers"]), setting=setting_for(cfg)
    )
    watchdog = ProgressWatchdog(f"{AGENT_KEY} {run_name}", "relative parameter change", 1e-6, 10)
    change = ParameterChange(net)
    val_ends = rng.choice(val_idx, min(2000, len(val_idx)), replace=False)
    steps, losses, elapsed0 = 0, [], 0.0
    next_eval = float(cfg["first_eval_minutes"]) * 60
    state_path = run_dir / "state.pt"
    if resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        steps, elapsed0, next_eval = state["steps"], state["elapsed"], state["next_eval"]
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"])
        evaluator.records = [EvalRecord(**r) for r in state["records"]]
        for ckpt, step, at in state["pending"]:
            evaluator.submit(Path(ckpt), step, at)
        change = ParameterChange(net)
        print(f"resumed {run_name} at step {steps}, {elapsed0 / 60:.1f} min", flush=True)
    budget = Budget(float(cfg["budget_minutes"]) * 60, elapsed0)
    next_log = budget.elapsed + 30.0

    def save_and_submit() -> None:
        path = run_dir / f"eval_{int(budget.elapsed / 60):04d}min_{steps}.pt"
        save_checkpoint(path, AGENT_KEY, net, steps=steps, **meta)
        evaluator.submit(path, steps, budget.elapsed)
        torch.save(
            {
                "model": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "optimizer": to_cpu(opt.state_dict()),
                "steps": steps,
                "elapsed": budget.elapsed,
                "next_eval": budget.elapsed + float(cfg["eval_every_minutes"]) * 60,
                "rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "records": [asdict(r) for r in evaluator.records],
                "pending": [(c, s, e) for c, s, e, _ in evaluator._pending],
            },
            state_path,
        )

    while not budget.done:
        for g in opt.param_groups:
            g["lr"] = lr_at(steps, budget.fraction, cfg)
        loss = dt_loss(net, dt_batch(data, rng.choice(train_idx, bsz), context, device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), float(cfg["grad_clip"]))
        opt.step()
        steps += 1
        losses.append(float(loss.detach()))
        if budget.elapsed >= next_log:
            delta = change()
            watchdog.update(delta)
            mean_loss = float(np.mean(losses))
            losses.clear()
            logger.scalar("train/loss", mean_loss, steps)
            logger.scalar("train/param_change", delta, steps)
            logger.scalar("train/lr", opt.param_groups[0]["lr"], steps)
            print(
                f"{budget.elapsed / 60:6.1f} min  step {steps:7d}  loss {mean_loss:.4f}"
                f"  lr {opt.param_groups[0]['lr']:.1e}  param change {delta:.1e}",
                flush=True,
            )
            next_log = budget.elapsed + 30.0
        if budget.elapsed >= next_eval:
            save_and_submit()
            top1 = top1_agreement(net, data, val_ends, device)
            logger.scalar("val/top1", top1, steps)
            print(f"   validation top-1 agreement with the logged action {top1:.3f}", flush=True)
            for rec in evaluator.poll():
                logger.eval_record(rec)
                print(f"   eval {Path(rec.checkpoint).name}: fitness {rec.fitness:.3f}", flush=True)
            next_eval = budget.elapsed + float(cfg["eval_every_minutes"]) * 60
    save_and_submit()
    for rec in evaluator.poll(block=True):
        logger.eval_record(rec)
    summary = finalize_selection(AGENT_KEY, evaluator, run_dir, promote)
    summary["selection_target"] = target
    (run_dir / "selection.json").write_text(json.dumps(summary, indent=2))
    if promote:
        (CHECKPOINT_DIR / AGENT_KEY / "selection.json").write_text(json.dumps(summary, indent=2))
    logger.plot(references=linear_references())
    logger.close()
    evaluator.close()
    print(
        f"selected {Path(summary['selected_checkpoint']).name} "
        f"(selection fitness {summary['selection_fitness']:.3f} at target {target:.3f})",
        flush=True,
    )
    return summary


# ----------------------------------------------------------- calibration


def spearman(x: list[float], y: list[float]) -> float:
    """Rank correlation (no ties expected among targets)."""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def calibrate(
    cfg: dict[str, Any],
    checkpoint: Path | None = None,
    workers: int | None = None,
    run_name: str | None = None,
    report: str = "dt_calibration",
    title: str = "Decision Transformer",
) -> dict[str, Any]:
    """Achieved return at each target on the selection seeds; sets the default target.

    The checkpoint is ``checkpoint``, else the one selected in ``run_name``,
    else the promoted ``best.pt``. ``calibration.json`` goes next to it and
    the report to ``reports/<report>.md``.
    """
    if checkpoint is None and run_name is not None:
        selection = json.loads(
            (CHECKPOINT_DIR / AGENT_KEY / run_name / "selection.json").read_text()
        )
        checkpoint = Path(selection["selected_checkpoint"])
    checkpoint = checkpoint or CHECKPOINT_DIR / AGENT_KEY / "best.pt"
    seeds = selection_seeds(int(cfg["calibration_games"]))
    setting = setting_for(cfg)
    rows = []
    for target in [float(t) for t in cfg["calibration_targets"]]:
        stats = evaluate_agent(
            AGENT_KEY,
            seeds,
            setting,
            workers,
            str(checkpoint),
            DifficultyParams(target_return=target),
        )
        s = summarize(stats)
        rows.append(
            {
                "target": target,
                "achieved_fitness": fitness(stats),
                "top_out_rate": s["top_out_rate"],
                "mean_pieces": s["mean_pieces"],
                "score_per_piece": s["score_per_piece"],
            }
        )
        print(f"target {target:.2f}: achieved {rows[-1]['achieved_fitness']:.3f}", flush=True)
    best = max(rows, key=lambda r: r["achieved_fitness"])
    rho = spearman([r["target"] for r in rows], [r["achieved_fitness"] for r in rows])
    result = {
        "checkpoint": str(checkpoint),
        "games_per_target": len(seeds),
        "rows": rows,
        "spearman": rho,
        "leaderboard_target": best["target"],
    }
    net, meta = load_checkpoint(checkpoint, AGENT_KEY, "cpu")
    extra = {k: v for k, v in meta.items() if k not in ("kind", "config", "state_dict")}
    extra["default_target"] = best["target"]
    save_checkpoint(checkpoint, AGENT_KEY, net, **extra)
    out_dir = checkpoint.parent
    (out_dir / "calibration.json").write_text(json.dumps(result, indent=2))
    write_calibration_report(result, report, title)
    return result


def write_calibration_report(
    result: dict[str, Any], report: str = "dt_calibration", title: str = "Decision Transformer"
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["rows"]
    reports = ROOT / "reports" / "analysis"
    reports.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    t = [r["target"] for r in rows]
    ax.plot(t, [r["achieved_fitness"] for r in rows], marker="o", label="achieved")
    ax.plot(t, t, linestyle="--", color="gray", label="achieved = target")
    ax.set_xlabel("target return (fitness units)")
    ax.set_ylabel(f"achieved fitness ({result['games_per_target']} selection seeds)")
    ax.set_title(f"{title}: target return vs achieved")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports / f"{report}.png", dpi=110)
    plt.close(fig)
    lines = [
        f"# {title}: target-return calibration",
        "",
        "The model chooses actions by afterstate matching (each placement's logit is the "
        "match between the transformer's output and an embedding of that placement's "
        "afterstate) rather than with an 80-way action classifier. Raw-action PPO already "
        "shows the raw action space failing here: held-out fitness 0.103, against 1.280 to "
        "1.345 for the seven top agents, all of which score afterstates.",
        "",
        f"The selected model played {result['games_per_target']} selection seeds (garbage "
        "setting) at each target return. Returns are in fitness units: a game's return "
        "equals its fitness when it reaches the 500-piece cap.",
        "",
        "| Target | Achieved fitness | Top-out rate | Mean pieces | Score/piece |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['target']:.2f} | {r['achieved_fitness']:.3f} | {r['top_out_rate']:.2f} "
            f"| {r['mean_pieces']:.0f} | {r['score_per_piece']:.0f} |"
        )
    lines += [
        "",
        f"Rank correlation between target and achieved fitness: {result['spearman']:+.2f} "
        "(+1 means a higher target always played better).",
        "",
        f"Default target (highest achieved fitness, used on the leaderboard): "
        f"{result['leaderboard_target']:.2f}.",
        "",
        f"![calibration]({report}.png)",
        "",
    ]
    (reports / f"{report}.md").write_text("\n".join(lines))


# ------------------------------------------------------- version choice


def version_dir(version: str) -> Path:
    """``checkpoints/dt/versions/<version>/dt/``; its parent is usable as a
    TETRIS_CHECKPOINT_DIR (``<dir>/dt/best.pt``)."""
    return CHECKPOINT_DIR / AGENT_KEY / "versions" / version / AGENT_KEY


def keep_version(version: str, checkpoint: Path, calibration: Path, selection: Path) -> Path:
    """Copy a calibrated model with its calibration and selection records."""
    out = version_dir(version)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(checkpoint, out / "best.pt")
    shutil.copy(calibration, out / "calibration.json")
    shutil.copy(selection, out / "selection.json")
    return out


def choose_version(new_run: str = "seed0_v2") -> dict[str, Any]:
    """Keep the promoted model as v1, store ``new_run`` as v2, and promote v2 only if
    its selection fitness is strictly higher (both chosen on the selection seeds)."""
    base = CHECKPOINT_DIR / AGENT_KEY
    if not (version_dir("v1") / "best.pt").exists():
        keep_version("v1", base / "best.pt", base / "calibration.json", base / "selection.json")
    run_dir = base / new_run
    selection = json.loads((run_dir / "selection.json").read_text())
    keep_version(
        "v2",
        Path(selection["selected_checkpoint"]),
        Path(selection["selected_checkpoint"]).parent / "calibration.json",
        run_dir / "selection.json",
    )
    fits = {
        v: json.loads((version_dir(v) / "selection.json").read_text())["selection_fitness"]
        for v in ("v1", "v2")
    }
    chosen = "v2" if fits["v2"] > fits["v1"] else "v1"
    src = version_dir(chosen)
    for name in ("best.pt", "calibration.json", "selection.json"):
        shutil.copy(src / name, base / name)
    result = {"selection_fitness": fits, "chosen": chosen, "rule": "v2 if strictly higher"}
    (base / "version_choice.json").write_text(json.dumps(result, indent=2))
    print(f"DT versions: {fits} -> {chosen} is playable", flush=True)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("command", choices=["train", "calibrate", "choose"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--budget-minutes", dest="budget_minutes", type=float)
    parser.add_argument("--run-name", default="seed0")
    parser.add_argument("--no-promote", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", default="dt_calibration", help="calibrate: report name")
    parser.add_argument("--title", default="Decision Transformer", help="calibrate: report title")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, {"budget_minutes": args.budget_minutes})
    if args.command == "train":
        train(cfg, args.run_name, not args.no_promote, args.resume)
    elif args.command == "calibrate":
        # The promoted model, or with --run-name the model selected in that run.
        run_name = args.run_name if args.run_name != "seed0" else None
        calibrate(cfg, run_name=run_name, report=args.report, title=args.title)
    else:
        choose_version(args.run_name)


if __name__ == "__main__":
    main()
