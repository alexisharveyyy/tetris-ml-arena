"""PPO over the masked 80-action placement space with vectorized environments.

Usage:
    python -m tetris.agents.ppo.train [--config PATH] [--budget-minutes M] [--seed S]
        [--run-name NAME] [--no-promote] [--resume]

Illegal actions get a large negative logit (-1e9, not -inf) and the entropy is
computed over legal actions only, so the masked distribution never produces NaN.

Termination vs truncation: a top-out ends the return (no bootstrap). Hitting
the piece cap is truncation: the step's reward is augmented with
``gamma * V(final observation)`` and the advantage chain is cut there, so the
value of the unfinished game is bootstrapped instead of treated as zero.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tetris.agents.registry import CHECKPOINT_DIR
from tetris.engine.placement import NUM_ACTIONS
from tetris.rl.encoding import OBS_EXTRA_DIM
from tetris.rl.nets import ActorCritic, masked_log_probs, save_checkpoint, select_device, to_cpu
from tetris.rl.training import (
    Budget,
    CurveLogger,
    SelectionEvaluator,
    finalize_selection,
    linear_references,
    load_config,
    mean_episode_stats,
    setting_for,
)
from tetris.rl.vec_env import VecEnv

AGENT_KEY = "ppo"
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    final_values: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and returns for ``(T, N)`` rollouts.

    ``final_values[t, n]`` is V(final observation) where step ``t`` was truncated.
    A terminated step bootstraps nothing; a truncated step bootstraps from its
    final observation; both cut the chain to the next (new) episode.
    """
    steps = rewards.shape[0]
    adv = np.zeros_like(rewards)
    last = np.zeros_like(last_values)
    for t in reversed(range(steps)):
        next_values = last_values if t == steps - 1 else values[t + 1]
        ended = terminated[t] | truncated[t]
        r = rewards[t] + gamma * final_values[t] * truncated[t]
        delta = r + gamma * next_values * (~ended) - values[t]
        last = delta + gamma * lam * (~ended) * last
        adv[t] = last
    return adv, adv + values


def _stack(packages: list[dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([p["board"] for p in packages]),
        np.stack([p["extras"] for p in packages]),
        np.stack([p["mask"] for p in packages]),
    )


def train(
    cfg: dict[str, Any], resume: bool = False, run_name: str | None = None, promote: bool = True
) -> dict:
    seed = int(cfg["seed"])
    run_name = run_name or f"seed{seed}"
    run_dir = CHECKPOINT_DIR / AGENT_KEY / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.set_num_threads(int(cfg.get("torch_threads", 2)))
    device = select_device(cfg.get("device", "auto"))
    setting = setting_for(cfg)
    net = ActorCritic(**cfg.get("model", {})).to(device)
    fused = device.type in ("cuda", "mps")
    opt = torch.optim.Adam(net.parameters(), lr=float(cfg["lr"]), eps=1e-5, fused=fused)
    gamma, lam = float(cfg["gamma"]), float(cfg["gae_lambda"])
    scale = float(cfg["reward_scale"])

    elapsed0, steps, n_evals, updates = 0.0, 0, 0, 0
    latest = run_dir / "latest.pt"
    evaluator = SelectionEvaluator(AGENT_KEY, workers=cfg["eval_workers"])
    if resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        net.load_state_dict(state["net"])
        opt.load_state_dict(state["opt"])
        elapsed0, steps, n_evals, updates = (
            state["elapsed"],
            state["steps"],
            state["n_evals"],
            state["updates"],
        )
        evaluator.records = state["eval_records"]
        print(f"resumed {run_name} at {elapsed0 / 60:.1f} min", flush=True)

    logger = CurveLogger(AGENT_KEY, run_name)
    budget = Budget(cfg["budget_minutes"] * 60, elapsed0)
    vec = VecEnv("obs", cfg["n_workers"], cfg["envs_per_worker"], setting, seed + 1000 * n_evals)
    n_envs, horizon = vec.n_envs, int(cfg["rollout_steps"])
    packages = vec.reset()
    episodes: list[dict[str, Any]] = []
    next_eval = (
        budget.elapsed + (cfg["eval_every_minutes"] if n_evals else cfg["first_eval_minutes"]) * 60
    )
    print(
        f"ppo {run_name}: device {device}, {n_envs} envs x {horizon} steps,"
        f" budget {cfg['budget_minutes']} min",
        flush=True,
    )

    def tensors(b: np.ndarray, e: np.ndarray, m: np.ndarray):
        return (
            torch.from_numpy(b).to(device).float(),
            torch.from_numpy(e).to(device),
            torch.from_numpy(m).to(device),
        )

    def save_eval_checkpoint() -> Path:
        path = run_dir / f"eval_{int(budget.elapsed / 60):04d}min_{steps}.pt"
        save_checkpoint(
            path, AGENT_KEY, net, steps=steps, minutes=budget.elapsed / 60, run=run_name
        )
        return path

    def save_latest() -> None:
        torch.save(
            {
                "net": to_cpu(net.state_dict()),
                "opt": to_cpu(opt.state_dict()),
                "elapsed": budget.elapsed,
                "steps": steps,
                "n_evals": n_evals,
                "updates": updates,
                "eval_records": evaluator.records,
                "config": cfg,
            },
            latest,
        )

    buf_b = np.zeros((horizon, n_envs, 24, 10), np.uint8)
    buf_e = np.zeros((horizon, n_envs, OBS_EXTRA_DIM), np.float32)
    buf_m = np.zeros((horizon, n_envs, NUM_ACTIONS), bool)
    buf_a = np.zeros((horizon, n_envs), np.int64)
    buf_logp = np.zeros((horizon, n_envs), np.float32)
    buf_v = np.zeros((horizon, n_envs), np.float32)
    buf_r = np.zeros((horizon, n_envs), np.float32)
    buf_term = np.zeros((horizon, n_envs), bool)
    buf_trunc = np.zeros((horizon, n_envs), bool)
    buf_final_v = np.zeros((horizon, n_envs), np.float32)

    try:
        while not budget.done:
            t0 = time.perf_counter()
            frac = budget.fraction
            for g in opt.param_groups:
                g["lr"] = float(cfg["lr"]) * max(0.05, 1.0 - frac)
            net.eval()
            for t in range(horizon):
                b, e, m = _stack(packages)
                with torch.no_grad():
                    logits, v = net(*tensors(b, e, m))
                    logp_all = torch.log_softmax(logits, -1)
                    a = torch.multinomial(logp_all.exp(), 1).squeeze(1)
                    logp = logp_all.gather(1, a[:, None]).squeeze(1)
                actions = a.cpu().numpy()
                buf_b[t], buf_e[t], buf_m[t], buf_a[t] = b, e, m, actions
                buf_logp[t], buf_v[t] = logp.cpu().numpy(), v.cpu().numpy()
                results = vec.step(actions)
                finals = []
                for i, res in enumerate(results):
                    buf_r[t, i] = res.reward * scale
                    buf_term[t, i] = res.terminated
                    buf_trunc[t, i] = res.truncated
                    buf_final_v[t, i] = 0.0
                    if res.truncated:
                        finals.append((i, res.final))
                    if res.episode is not None:
                        episodes.append(res.episode)
                if finals:
                    fb, fe, fm = _stack([f for _, f in finals])
                    with torch.no_grad():
                        _, fv = net(*tensors(fb, fe, fm))
                    for (i, _), value in zip(finals, fv.cpu().numpy(), strict=True):
                        buf_final_v[t, i] = value
                packages = [r.package for r in results]
            steps += horizon * n_envs
            with torch.no_grad():
                _, last_v = net(*tensors(*_stack(packages)))
            adv, ret = compute_gae(
                buf_r, buf_v, buf_term, buf_trunc, buf_final_v, last_v.cpu().numpy(), gamma, lam
            )
            t_roll = time.perf_counter() - t0

            # ---- PPO update
            net.train()
            flat = horizon * n_envs
            fb = torch.from_numpy(buf_b.reshape(flat, 24, 10)).to(device).float()
            fe = torch.from_numpy(buf_e.reshape(flat, -1)).to(device)
            fm = torch.from_numpy(buf_m.reshape(flat, -1)).to(device)
            fa = torch.from_numpy(buf_a.reshape(flat)).to(device)
            flogp = torch.from_numpy(buf_logp.reshape(flat)).to(device)
            fadv = torch.from_numpy(adv.reshape(flat)).to(device)
            fret = torch.from_numpy(ret.reshape(flat)).to(device)
            fval = torch.from_numpy(buf_v.reshape(flat)).to(device)
            mb = int(cfg["minibatch_size"])
            log = {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_frac": 0.0,
            }
            n_mb = 0
            for _ in range(int(cfg["epochs"])):
                perm = torch.randperm(flat, device=device)
                for start in range(0, flat, mb):
                    j = perm[start : start + mb]
                    logits, v = net(fb[j], fe[j], fm[j])
                    logp_all, entropy = masked_log_probs(logits, fm[j])
                    logp = logp_all.gather(1, fa[j][:, None]).squeeze(1)
                    ratio = (logp - flogp[j]).exp()
                    a_n = fadv[j]
                    a_n = (a_n - a_n.mean()) / (a_n.std() + 1e-8)
                    pg = torch.max(
                        -a_n * ratio, -a_n * ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"])
                    ).mean()
                    v_clip = fval[j] + (v - fval[j]).clamp(-cfg["value_clip"], cfg["value_clip"])
                    vl = 0.5 * torch.max((v - fret[j]) ** 2, (v_clip - fret[j]) ** 2).mean()
                    ent = entropy.mean()
                    loss = pg + cfg["vf_coef"] * vl - cfg["ent_coef"] * ent
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
                    opt.step()
                    with torch.no_grad():
                        log["policy_loss"] += pg
                        log["value_loss"] += vl
                        log["entropy"] += ent
                        log["approx_kl"] += (flogp[j] - logp).mean()
                        log["clip_frac"] += ((ratio - 1).abs() > cfg["clip"]).float().mean()
                    n_mb += 1
            updates += 1
            vals = {k: float(v) / n_mb for k, v in log.items()}
            if any(np.isnan(v) for v in vals.values()):
                raise FloatingPointError(f"NaN in PPO update: {vals}")
            ep = mean_episode_stats(episodes[-200:])
            for k, v in {**vals, **{f"train_{k}": v for k, v in ep.items()}}.items():
                logger.scalar(f"train/{k}", v, steps)
            sps = horizon * n_envs / (time.perf_counter() - t0)
            logger.scalar("train/steps_per_second", sps, steps)
            if updates % 5 == 0:
                print(
                    f"{budget.elapsed / 60:6.1f} min  steps {steps:9d}  {sps:6.0f}/s"
                    f" (rollout {t_roll:.1f}s)  pieces {ep.get('pieces', 0):6.1f}"
                    f"  top-out {ep.get('top_out_rate', 0):.2f}  ent {vals['entropy']:.2f}"
                    f"  kl {vals['approx_kl']:.4f}  vloss {vals['value_loss']:.3f}",
                    flush=True,
                )
            if budget.elapsed >= next_eval:
                evaluator.submit(save_eval_checkpoint(), steps, budget.elapsed)
                n_evals += 1
                save_latest()
                next_eval = budget.elapsed + cfg["eval_every_minutes"] * 60
            for rec in evaluator.poll():
                logger.eval_record(rec)
                name, top = Path(rec.checkpoint).name, rec.summary["top_out_rate"]
                print(f"   eval {name}: fitness {rec.fitness:.3f}  top-out {top:.2f}", flush=True)
    finally:
        vec.close()
    evaluator.submit(save_eval_checkpoint(), steps, budget.elapsed)
    save_latest()
    for rec in evaluator.poll(block=True):
        logger.eval_record(rec)
    summary = finalize_selection(AGENT_KEY, evaluator, run_dir, promote)
    logger.plot(references=linear_references())
    logger.close()
    evaluator.close()
    name, fit = Path(summary["selected_checkpoint"]).name, summary["selection_fitness"]
    print(f"selected {name} (selection fitness {fit:.3f})", flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--budget-minutes", dest="budget_minutes", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", dest="n_workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--run-name")
    parser.add_argument(
        "--no-promote", action="store_true", help="do not copy the result to best.pt"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    overrides = {k: getattr(args, k) for k in ("budget_minutes", "seed", "n_workers", "device")}
    train(load_config(args.config, overrides), args.resume, args.run_name, not args.no_promote)


if __name__ == "__main__":
    main()
