"""Double DQN over afterstates with prioritized replay.

Usage:
    python -m tetris.agents.dqn.train [--config PATH] [--budget-minutes M] [--seed S]
        [--run-name NAME] [--no-promote] [--resume]

The network values afterstates V(after). Acting picks
``argmax_a r(s, a) + gamma * V(after(s, a))`` (V = 0 for a top-out afterstate).
Each stored transition keeps the chosen afterstate and the top ``K`` candidate
afterstates of the next state (ranked by the online network when collected).
The Double DQN target selects among those candidates with the online network
and evaluates the choice with the target network:

    y = 0                                         if the next state is terminal
    y = r(s', a*) + gamma * V_target(after(s', a*)) otherwise, where
    a* = argmax_c r(s', c) + gamma * V_online(after(s', c))

Hitting the piece cap is truncation, not termination: the next state is the
final state of the episode and the target bootstraps from it.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tetris.agents.registry import CHECKPOINT_DIR
from tetris.rl.nets import (
    AfterstateValueNet,
    save_checkpoint,
    select_device,
    to_cpu,
    unpack_boards_torch,
)
from tetris.rl.replay import PrioritizedReplay
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

AGENT_KEY = "dqn"
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def double_dqn_targets(
    v_online_next: torch.Tensor,
    v_target_next: torch.Tensor,
    next_reward: torch.Tensor,
    next_top: torch.Tensor,
    next_valid: torch.Tensor,
    partial_return: torch.Tensor,
    discount: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """n-step Double DQN targets for V(after_t).

    Candidate tensors are ``(B, K)``; ``partial_return`` and ``discount`` are ``(B,)``.
    ``y = partial_return + discount * Q_target(c*)`` with ``c*`` chosen by the
    online network. Terminated transitions have ``discount == 0`` (no bootstrap);
    truncated ones keep their discount and bootstrap from the final state.
    """
    alive = (~next_top).float()
    q_online = next_reward + gamma * v_online_next * alive
    q_online = q_online.masked_fill(~next_valid, float("-inf"))
    best = q_online.argmax(dim=1, keepdim=True)
    q_eval = next_reward.gather(1, best) + gamma * v_target_next.gather(1, best) * alive.gather(
        1, best
    )
    q_eval = q_eval.squeeze(1)
    bootstrap = torch.where(discount > 0, discount * q_eval, torch.zeros_like(q_eval))
    return partial_return + bootstrap


class Learner:
    def __init__(self, cfg: dict[str, Any], device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.gamma = float(cfg["gamma"])
        self.online = AfterstateValueNet(**cfg.get("model", {})).to(device)
        self.target = copy.deepcopy(self.online)
        # Fused Adam avoids per-parameter host syncs (about 2x faster per update on MPS).
        fused = device.type in ("cuda", "mps")
        self.opt = torch.optim.Adam(self.online.parameters(), lr=float(cfg["lr"]), fused=fused)
        self.updates = 0

    @torch.no_grad()
    def values(self, boards_packed: np.ndarray, extras: np.ndarray) -> np.ndarray:
        """V(after) for packed boards on the training device (for acting)."""
        b = unpack_boards_torch(torch.from_numpy(boards_packed).to(self.device))
        e = torch.from_numpy(extras.astype(np.float32)).to(self.device)
        return self.online(b, e).float().cpu().numpy()

    def update(self, replay: PrioritizedReplay, beta: float) -> dict[str, float] | None:
        """One gradient step. Returns stats from the *previous* step, if any.

        All inputs cross to the device as two buffers (uint8 and float32), and
        priorities are written back one step late so the CPU never waits for
        the GPU right after queuing work.
        """
        cfg = self.cfg
        bsz, k = cfg["batch_size"], replay.k
        idx, weights = replay.sample(bsz, beta)
        nbytes = replay.after_board.shape[1]
        ne_dim = replay.next_extra.shape[2]

        u8 = np.concatenate(
            [
                replay.after_board[idx].reshape(-1),
                replay.next_board[idx].reshape(-1),
                replay.next_top[idx].reshape(-1).view(np.uint8),
                replay.next_valid[idx].reshape(-1).view(np.uint8),
            ]
        )
        f32 = np.concatenate(
            [
                replay.after_extra[idx].astype(np.float32).reshape(-1),
                replay.next_extra[idx].astype(np.float32).reshape(-1),
                replay.next_reward[idx].reshape(-1),
                weights,
                replay.partial_return[idx],
                replay.discount[idx],
            ]
        )
        u8_t = torch.from_numpy(u8).to(self.device, non_blocking=True)
        f32_t = torch.from_numpy(f32).to(self.device, non_blocking=True)

        o = 0
        ab = u8_t[o : o + bsz * nbytes].view(bsz, nbytes)
        o += bsz * nbytes
        nb = u8_t[o : o + bsz * k * nbytes].view(bsz * k, nbytes)
        o += bsz * k * nbytes
        n_top = u8_t[o : o + bsz * k].view(bsz, k).bool()
        o += bsz * k
        n_valid = u8_t[o : o + bsz * k].view(bsz, k).bool()
        e_dim = replay.after_extra.shape[1]
        o = 0
        ae = f32_t[o : o + bsz * e_dim].view(bsz, e_dim)
        o += bsz * e_dim
        ne = f32_t[o : o + bsz * k * ne_dim].view(bsz * k, ne_dim)
        o += bsz * k * ne_dim
        n_rew = f32_t[o : o + bsz * k].view(bsz, k)
        o += bsz * k
        w = f32_t[o : o + bsz]
        o += bsz
        part = f32_t[o : o + bsz]
        o += bsz
        disc = f32_t[o : o + bsz]

        with torch.no_grad():
            nb_f = unpack_boards_torch(nb)
            v_on = self.online(nb_f, ne).view(bsz, k)
            v_tg = self.target(nb_f, ne).view(bsz, k)
            y = double_dqn_targets(v_on, v_tg, n_rew, n_top, n_valid, part, disc, self.gamma)
        pred = self.online(unpack_boards_torch(ab), ae)
        td = (pred - y).detach()
        loss = (w * torch.nn.functional.smooth_l1_loss(pred, y, reduction="none")).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), cfg["grad_clip"])
        self.opt.step()
        self.updates += 1
        if self.updates % cfg["target_update"] == 0:
            self.target.load_state_dict(self.online.state_dict())

        out = self.flush(replay)
        self._prev = (idx, td, loss.detach(), pred.detach().mean())
        return out

    def flush(self, replay: PrioritizedReplay) -> dict[str, float] | None:
        """Apply the pending priority update and return the previous step's stats."""
        prev = getattr(self, "_prev", None)
        if prev is None:
            return None
        idx, td, loss, v_mean = prev
        self._prev = None
        stats = torch.stack([loss, v_mean]).cpu().numpy()
        replay.update_priorities(idx, td.cpu().numpy())
        return {"loss": float(stats[0]), "v_mean": float(stats[1])}

    def state(self) -> dict[str, Any]:
        return to_cpu(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "opt": self.opt.state_dict(),
                "updates": self.updates,
            }
        )

    def load_state(self, s: dict[str, Any]) -> None:
        self.online.load_state_dict(s["online"])
        self.target.load_state_dict(s["target"])
        self.opt.load_state_dict(s["opt"])
        self.updates = s["updates"]


def _top_k(q: np.ndarray, k: int) -> np.ndarray:
    if len(q) <= k:
        return np.argsort(-q)
    part = np.argpartition(-q, k)[:k]
    return part[np.argsort(-q[part])]


class NStepTracker:
    """Builds n-step transitions per environment.

    For the afterstate chosen at state ``s_t`` the target bootstraps from
    ``s_{t+n}``: ``R = r_{t+1} + ... + gamma^(n-2) r_{t+n-1}`` (rewards of the
    actions taken in between) plus ``gamma^(n-1) * max_c Q(s_{t+n}, c)``. If the
    episode is truncated earlier it bootstraps from the final state with the
    discount reached so far; if it terminates, only ``R`` remains.
    """

    def __init__(self, n_envs: int, n: int, gamma: float, replay: PrioritizedReplay) -> None:
        self.n = n
        self.gamma = gamma
        self.replay = replay
        # Per env: list of [after_board, after_extra, partial_return, rewards_collected].
        self.pending: list[list[list[Any]]] = [[] for _ in range(n_envs)]
        self.last_action = [0] * n_envs

    def _complete(
        self, entry: list[Any], pkg: dict[str, np.ndarray], q: np.ndarray, truncated: bool
    ) -> None:
        top = _top_k(q, self.replay.k)
        self.replay.add(
            entry[0],
            entry[1],
            False,
            truncated,
            pkg["boards"][top],
            pkg["extras"][top],
            pkg["rewards"][top],
            pkg["tops"][top],
            partial_return=entry[2],
            discount=self.gamma ** entry[3],
        )

    def observe(
        self, i: int, pkg: dict[str, np.ndarray], q: np.ndarray, explore: bool, rng
    ) -> None:
        """Called at every non-final state: completes due transitions and picks an action."""
        keep = []
        for entry in self.pending[i]:
            if entry[3] == self.n - 1:
                self._complete(entry, pkg, q, truncated=False)
            else:
                keep.append(entry)
        a = int(rng.integers(len(q))) if explore else int(np.argmax(q))
        r = float(pkg["rewards"][a])
        for entry in keep:
            entry[2] += (self.gamma ** entry[3]) * r
            entry[3] += 1
        keep.append([pkg["boards"][a], pkg["extras"][a], 0.0, 0])
        self.pending[i] = keep
        self.last_action[i] = int(pkg["actions"][a])

    def end(
        self, i: int, final_pkg: dict[str, np.ndarray] | None, final_q: np.ndarray | None
    ) -> None:
        """Episode ended: terminate (``final_pkg is None``) or bootstrap from the final state."""
        for entry in self.pending[i]:
            if final_pkg is None:
                self.replay.add(entry[0], entry[1], True, False, partial_return=entry[2])
            else:
                self._complete(entry, final_pkg, final_q, truncated=True)
        self.pending[i] = []


def epsilon(cfg: dict[str, Any], fraction: float) -> float:
    frac = min(1.0, fraction / cfg["eps_decay_frac"])
    return cfg["eps_start"] + (cfg["eps_end"] - cfg["eps_start"]) * frac


def train(
    cfg: dict[str, Any], resume: bool = False, run_name: str | None = None, promote: bool = True
) -> dict:
    seed = int(cfg["seed"])
    run_name = run_name or f"seed{seed}"
    run_dir = CHECKPOINT_DIR / AGENT_KEY / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(int(cfg.get("torch_threads", 2)))
    device = select_device(cfg.get("device", "auto"))
    setting = setting_for(cfg)
    learner = Learner(cfg, device)
    replay = PrioritizedReplay(
        cfg["replay_capacity"], cfg["next_candidates"], cfg["per_alpha"], seed=seed
    )
    gamma = learner.gamma

    elapsed0, steps, n_evals = 0.0, 0, 0
    latest = run_dir / "latest.pt"
    evaluator = SelectionEvaluator(AGENT_KEY, workers=cfg["eval_workers"])
    if resume and latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        learner.load_state(state["learner"])
        elapsed0, steps, n_evals = state["elapsed"], state["steps"], state["n_evals"]
        rng.bit_generator.state = state["rng"]
        evaluator.records = state["eval_records"]
        print(
            f"resumed {run_name} at {elapsed0 / 60:.1f} min; replay refills from scratch",
            flush=True,
        )

    logger = CurveLogger(AGENT_KEY, run_name)
    budget = Budget(cfg["budget_minutes"] * 60, elapsed0)
    vec = VecEnv(
        "afterstate", cfg["n_workers"], cfg["envs_per_worker"], setting, seed + 1000 * n_evals
    )
    packages = vec.reset()
    n_envs = vec.n_envs
    tracker = NStepTracker(n_envs, int(cfg["n_step"]), gamma, replay)
    episodes: list[dict[str, Any]] = []
    stats: list[dict[str, float]] = []
    next_log = budget.elapsed + 30
    next_eval = (
        budget.elapsed + cfg["eval_every_minutes"] * 60
        if n_evals
        else budget.elapsed + cfg["first_eval_minutes"] * 60
    )
    t_rate, steps_rate = time.perf_counter(), steps
    print(
        f"dqn {run_name}: device {device}, {n_envs} envs, budget {cfg['budget_minutes']} min",
        flush=True,
    )

    def save_eval_checkpoint() -> Path:
        path = run_dir / f"eval_{int(budget.elapsed / 60):04d}min_{steps}.pt"
        save_checkpoint(
            path,
            AGENT_KEY,
            learner.online,
            gamma=gamma,
            steps=steps,
            minutes=budget.elapsed / 60,
            run=run_name,
        )
        return path

    def save_latest() -> None:
        torch.save(
            {
                "learner": learner.state(),
                "elapsed": budget.elapsed,
                "steps": steps,
                "n_evals": n_evals,
                "rng": rng.bit_generator.state,
                "eval_records": evaluator.records,
                "config": cfg,
            },
            latest,
        )

    try:
        while not budget.done:
            eps = epsilon(cfg, budget.fraction)
            counts = [len(p["rewards"]) for p in packages]
            all_boards = np.concatenate([p["boards"] for p in packages])
            all_extras = np.concatenate([p["extras"] for p in packages])
            v_all = learner.values(all_boards, all_extras)
            actions = []
            start = 0
            for i, p in enumerate(packages):
                v = v_all[start : start + counts[i]]
                start += counts[i]
                q = p["rewards"] + gamma * v * (~p["tops"])
                tracker.observe(i, p, q, rng.random() < eps, rng)
                actions.append(tracker.last_action[i])
            vec.step_async(actions)
            if len(replay) >= cfg["warmup"]:
                beta = cfg["per_beta_start"] + (1.0 - cfg["per_beta_start"]) * budget.fraction
                for _ in range(cfg["updates_per_step"]):
                    out = learner.update(replay, beta)
                    if out is not None:
                        stats.append(out)
            results = vec.step_wait()
            steps += n_envs
            for i, res in enumerate(results):
                if res.terminated:
                    tracker.end(i, None, None)
                elif res.truncated:
                    f = res.final
                    q = f["rewards"] + gamma * learner.values(f["boards"], f["extras"]) * (
                        ~f["tops"]
                    )
                    tracker.end(i, f, q)
                if res.episode is not None:
                    episodes.append(res.episode)
            packages = [r.package for r in results]

            if budget.elapsed >= next_log:
                now = time.perf_counter()
                rate = (steps - steps_rate) / (now - t_rate)
                t_rate, steps_rate = now, steps
                ep = mean_episode_stats(episodes[-200:])
                for key, value in ep.items():
                    logger.scalar(f"train/{key}", value, steps)
                if stats:
                    logger.scalar("train/loss", float(np.mean([s["loss"] for s in stats])), steps)
                    logger.scalar(
                        "train/v_mean", float(np.mean([s["v_mean"] for s in stats])), steps
                    )
                logger.scalar("train/epsilon", eps, steps)
                logger.scalar("train/steps_per_second", rate, steps)
                loss = np.mean([s["loss"] for s in stats]) if stats else 0.0
                print(
                    f"{budget.elapsed / 60:6.1f} min  steps {steps:9d}  {rate:6.0f}/s"
                    f"  updates {learner.updates:7d}  eps {eps:.3f}"
                    f"  pieces {ep.get('pieces', 0):6.1f}  top-out {ep.get('top_out_rate', 0):.2f}"
                    f"  loss {loss:.3f}",
                    flush=True,
                )
                stats.clear()
                next_log = budget.elapsed + 30
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
