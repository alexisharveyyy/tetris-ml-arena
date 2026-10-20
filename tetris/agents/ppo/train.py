"""PPO over the masked 80-action placement space with vectorized environments.

Two policies share this trainer (``policy`` in the config):

* ``afterstate`` (agent ``ppo``): each action's logit is a shared network's
  score of that action's afterstate. Still a distribution over the 80 actions.
* ``raw`` (agent ``ppo_raw``): logits for the 80 actions from the board and
  piece observation alone (the originally specified design).

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
from tetris.rl.encoding import AFTERSTATE_EXTRA_DIM, OBS_EXTRA_DIM, PACKED_BOARD_BYTES
from tetris.rl.nets import build, masked_log_probs, save_checkpoint, select_device, to_cpu
from tetris.rl.safeguards import ProgressWatchdog, preflight
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


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    """1 - Var(returns - values) / Var(returns); 1 is perfect, 0 is no better than a constant."""
    var = float(np.var(returns))
    return float("nan") if var == 0 else 1.0 - float(np.var(returns - values)) / var


def policy_parameters(net: torch.nn.Module):
    """Parameters that produce the logits: the afterstate scorer, or the raw policy's own."""
    if hasattr(net, "scorer"):
        return net.scorer.parameters()
    return (p for n, p in net.named_parameters() if not n.startswith(("value", "critic")))


#: Package fields fed to each policy type, in network-argument order (mask is third).
POLICY_INPUTS = {
    "raw": ("board", "extras", "mask"),
    "afterstate": ("board", "extras", "mask", "cand_boards", "cand_extras"),
}
VEC_MODE = {"raw": "obs", "afterstate": "action_afterstate"}
NET_KIND = {"raw": "ppo_raw", "afterstate": "ppo"}


def preflight_batch(policy: str, minibatch: int, seed: int = 0) -> list[torch.Tensor]:
    """Random policy inputs for one real minibatch, plus actions, advantages and returns.

    About 70% of the 80 actions are legal, close to real boards (about 57).
    """
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(minibatch, NUM_ACTIONS, generator=g) < 0.7
    mask[:, 0] = True
    batch = [
        torch.randint(0, 2, (minibatch, 24, 10), generator=g).float(),
        torch.rand(minibatch, OBS_EXTRA_DIM, generator=g),
        mask,
    ]
    if policy == "afterstate":
        batch += [
            torch.randint(
                0, 256, (minibatch, NUM_ACTIONS, PACKED_BOARD_BYTES), dtype=torch.uint8, generator=g
            ),
            torch.rand(minibatch, NUM_ACTIONS, AFTERSTATE_EXTRA_DIM, generator=g),
        ]
    actions = torch.multinomial(mask.float(), 1, generator=g).squeeze(1)
    return [
        *batch,
        actions,
        torch.randn(minibatch, generator=g),
        torch.rand(minibatch, generator=g),
    ]


def preflight_loss(net: torch.nn.Module, batch: list[torch.Tensor]) -> torch.Tensor:
    """Policy-gradient, value and entropy terms of the PPO loss on one minibatch."""
    *inputs, actions, adv, ret = batch
    logits, v = net(*inputs)
    logp_all, entropy = masked_log_probs(logits, inputs[2])
    logp = logp_all.gather(1, actions[:, None]).squeeze(1)
    return -(adv * logp).mean() + 0.5 * ((v - ret) ** 2).mean() - 0.01 * entropy.mean()


def _stack(packages: list[dict[str, np.ndarray]], fields: tuple[str, ...]) -> list[np.ndarray]:
    return [np.stack([p[f] for p in packages]) for f in fields]


def train(
    cfg: dict[str, Any], resume: bool = False, run_name: str | None = None, promote: bool = True
) -> dict:
    seed = int(cfg["seed"])
    policy = cfg.get("policy", "afterstate")
    fields = POLICY_INPUTS[policy]
    AGENT_KEY = cfg.get("agent_key", NET_KIND[policy])  # noqa: N806 - matches the DQN trainer
    run_name = run_name or f"seed{seed}"
    run_dir = CHECKPOINT_DIR / AGENT_KEY / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    torch.set_num_threads(int(cfg.get("torch_threads", 2)))
    device = select_device(cfg.get("device", "auto"))
    setting = setting_for(cfg)
    net = build(NET_KIND[policy], cfg.get("model", {}))
    if cfg.get("preflight", True):
        mb = int(cfg["minibatch_size"])
        check = preflight(
            f"{AGENT_KEY} {run_name}",
            net,
            preflight_batch(policy, mb),
            preflight_loss,
            device,
            f"minibatch {mb} states"
            + (
                f" (about {int(mb * 0.7 * NUM_ACTIONS):,} afterstates)"
                if policy == "afterstate"
                else ""
            ),
            run_dir,
        )
        device = torch.device(check.final_device)
    net = net.to(device)
    # Abort if the policy stops changing: per-update KL ((r - 1) - log r estimator, >= 0).
    # Measured on MPS: the broken afterstate run settled at about 1.2e-7, a healthy one
    # started at 3e-6 and rose to 2e-4. The threshold scales with (lr / lr0)^2.
    watchdog = ProgressWatchdog(
        f"{AGENT_KEY} {run_name}",
        "policy KL per update",
        float(cfg.get("watchdog_kl", 1e-6)),
        int(cfg.get("watchdog_patience", 30)),
    )
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
    vec = VecEnv(
        VEC_MODE[policy], cfg["n_workers"], cfg["envs_per_worker"], setting, seed + 1000 * n_evals
    )
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

    def tensors(arrays: list[np.ndarray]) -> list[torch.Tensor]:
        out = [torch.from_numpy(a).to(device) for a in arrays]
        out[0] = out[0].float()  # board occupancy
        return out

    def save_eval_checkpoint() -> Path:
        path = run_dir / f"eval_{int(budget.elapsed / 60):04d}min_{steps}.pt"
        save_checkpoint(
            path, NET_KIND[policy], net, steps=steps, minutes=budget.elapsed / 60, run=run_name
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

    shapes = {
        "board": ((24, 10), np.uint8),
        "extras": ((OBS_EXTRA_DIM,), np.float32),
        "mask": ((NUM_ACTIONS,), bool),
        "cand_boards": ((NUM_ACTIONS, PACKED_BOARD_BYTES), np.uint8),
        "cand_extras": ((NUM_ACTIONS, AFTERSTATE_EXTRA_DIM), np.float32),
    }
    bufs = [np.zeros((horizon, n_envs, *shapes[f][0]), shapes[f][1]) for f in fields]
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
                inputs = _stack(packages, fields)
                with torch.no_grad():
                    logits, v = net(*tensors(inputs))
                    logp_all = torch.log_softmax(logits, -1)
                    a = torch.multinomial(logp_all.exp(), 1).squeeze(1)
                    logp = logp_all.gather(1, a[:, None]).squeeze(1)
                actions = a.cpu().numpy()
                for buf, arr in zip(bufs, inputs, strict=True):
                    buf[t] = arr
                buf_a[t] = actions
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
                    with torch.no_grad():
                        _, fv = net(*tensors(_stack([f for _, f in finals], fields)))
                    for (i, _), value in zip(finals, fv.cpu().numpy(), strict=True):
                        buf_final_v[t, i] = value
                packages = [r.package for r in results]
            steps += horizon * n_envs
            with torch.no_grad():
                _, last_v = net(*tensors(_stack(packages, fields)))
            adv, ret = compute_gae(
                buf_r, buf_v, buf_term, buf_trunc, buf_final_v, last_v.cpu().numpy(), gamma, lam
            )
            t_roll = time.perf_counter() - t0

            # ---- PPO update
            net.train()
            flat = horizon * n_envs
            flat_in = tensors([b.reshape(flat, *b.shape[2:]) for b in bufs])
            fm = flat_in[2]
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
                "kl_k3": 0.0,
                "clip_frac": 0.0,
                "policy_grad_norm": 0.0,
            }
            n_mb = 0
            # Ratio check: at the first minibatch of the first epoch the policy has
            # not changed yet, so recomputed log-probs must equal the rollout ones.
            first_ratio_dev = None
            policy_params = list(policy_parameters(net))
            for _ in range(int(cfg["epochs"])):
                perm = torch.randperm(flat, device=device)
                for start in range(0, flat, mb):
                    j = perm[start : start + mb]
                    logits, v = net(*[x[j] for x in flat_in])
                    logp_all, entropy = masked_log_probs(logits, fm[j])
                    logp = logp_all.gather(1, fa[j][:, None]).squeeze(1)
                    ratio = (logp - flogp[j]).exp()
                    if first_ratio_dev is None:
                        first_ratio_dev = (ratio.detach() - 1).abs().max()
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
                    pgn = torch.linalg.vector_norm(
                        torch.stack(
                            [
                                torch.linalg.vector_norm(q.grad)
                                for q in policy_params
                                if q.grad is not None
                            ]
                        )
                    )
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["max_grad_norm"])
                    opt.step()
                    with torch.no_grad():
                        log["policy_loss"] += pg
                        log["value_loss"] += vl
                        log["entropy"] += ent
                        log["approx_kl"] += (flogp[j] - logp).mean()
                        log_ratio = logp - flogp[j]
                        log["kl_k3"] += (log_ratio.exp() - 1 - log_ratio).mean()
                        log["clip_frac"] += ((ratio - 1).abs() > cfg["clip"]).float().mean()
                        log["policy_grad_norm"] += pgn
                    n_mb += 1
            updates += 1
            vals = {k: float(v) / n_mb for k, v in log.items()}
            vals["first_ratio_dev"] = float(first_ratio_dev)
            watchdog.update(vals["kl_k3"], scale=max(0.05, 1.0 - frac) ** 2)
            vals["explained_variance"] = explained_variance(buf_v, ret)
            vals["adv_mean"], vals["adv_std"] = float(adv.mean()), float(adv.std())
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
                    f"  kl {vals['approx_kl']:.4f}  clip {vals['clip_frac']:.3f}"
                    f"  ratio0 {vals['first_ratio_dev']:.1e}  pgrad {vals['policy_grad_norm']:.2e}"
                    f"  ev {vals['explained_variance']:.2f}  vloss {vals['value_loss']:.3f}",
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
    parser.add_argument("--lr", type=float)
    parser.add_argument("--ent-coef", dest="ent_coef", type=float)
    args = parser.parse_args(argv)
    keys = ("budget_minutes", "seed", "n_workers", "device", "lr", "ent_coef")
    overrides = {k: getattr(args, k) for k in keys}
    train(load_config(args.config, overrides), args.resume, args.run_name, not args.no_promote)


if __name__ == "__main__":
    main()
