"""Benchmark MPS vs CPU for the DQN and PPO workloads.

Usage: python scripts/bench_device.py [--repeats 30]
"""

from __future__ import annotations

import argparse
import time

import torch

from tetris.engine.placement import NUM_ACTIONS
from tetris.rl.encoding import AFTERSTATE_EXTRA_DIM, OBS_EXTRA_DIM
from tetris.rl.nets import ActorCritic, AfterstateValueNet, masked_log_probs


def timeit(fn, device: torch.device, repeats: int) -> float:
    for _ in range(3):
        fn()
    if device.type == "mps":
        torch.mps.synchronize()
    t = time.perf_counter()
    for _ in range(repeats):
        fn()
    if device.type == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t) / repeats * 1000


def bench(
    device: torch.device, repeats: int, batch: int, k: int, envs: int, cands: int
) -> dict[str, float]:
    torch.manual_seed(0)
    dqn = AfterstateValueNet().to(device)
    tgt = AfterstateValueNet().to(device)
    opt = torch.optim.Adam(dqn.parameters(), lr=1e-4)
    n_inf = envs * cands
    b_inf = torch.rand(n_inf, 24, 10, device=device).round()
    e_inf = torch.rand(n_inf, AFTERSTATE_EXTRA_DIM, device=device)
    b = torch.rand(batch, 24, 10, device=device).round()
    e = torch.rand(batch, AFTERSTATE_EXTRA_DIM, device=device)
    nb = torch.rand(batch * k, 24, 10, device=device).round()
    ne = torch.rand(batch * k, AFTERSTATE_EXTRA_DIM, device=device)

    def dqn_infer():
        with torch.no_grad():
            dqn(b_inf, e_inf).cpu()

    def dqn_update():
        with torch.no_grad():
            a = dqn(nb, ne).view(batch, k)
            v = tgt(nb, ne).view(batch, k)
            y = v.gather(1, a.argmax(1, keepdim=True)).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(dqn(b, e), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss.item()

    ppo = ActorCritic().to(device)
    popt = torch.optim.Adam(ppo.parameters(), lr=1e-4)
    ob = torch.rand(envs, 24, 10, device=device).round()
    oe = torch.rand(envs, OBS_EXTRA_DIM, device=device)
    om = torch.rand(envs, NUM_ACTIONS, device=device) > 0.4
    mb = 1024
    mbb = torch.rand(mb, 24, 10, device=device).round()
    mbe = torch.rand(mb, OBS_EXTRA_DIM, device=device)
    mbm = torch.rand(mb, NUM_ACTIONS, device=device) > 0.4
    acts = torch.multinomial(mbm.float(), 1).squeeze(1)

    def ppo_infer():
        with torch.no_grad():
            logits, v = ppo(ob, oe, om)
            torch.distributions.Categorical(logits=logits).sample().cpu()

    def ppo_update():
        logits, v = ppo(mbb, mbe, mbm)
        logp, ent = masked_log_probs(logits, mbm)
        loss = -logp.gather(1, acts[:, None]).mean() + v.pow(2).mean() - 0.01 * ent.mean()
        popt.zero_grad()
        loss.backward()
        popt.step()
        loss.item()

    return {
        f"dqn_inference ({n_inf} afterstates)": timeit(dqn_infer, device, repeats),
        f"dqn_update (batch {batch}, {k} next candidates)": timeit(dqn_update, device, repeats),
        f"ppo_inference ({envs} obs)": timeit(ppo_infer, device, repeats),
        f"ppo_update (minibatch {mb})": timeit(ppo_update, device, repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--cands", type=int, default=50)
    args = parser.parse_args()
    rows = {}
    for name, dev, threads in (("cpu x4", "cpu", 4), ("cpu x8", "cpu", 8), ("mps", "mps", 4)):
        torch.set_num_threads(threads)
        if dev == "mps" and not torch.backends.mps.is_available():
            continue
        rows[name] = bench(
            torch.device(dev), args.repeats, args.batch, args.k, args.envs, args.cands
        )
    keys = list(next(iter(rows.values())))
    print(f"{'workload':48s}" + "".join(f"{n:>12s}" for n in rows))
    for key in keys:
        print(f"{key:48s}" + "".join(f"{rows[n][key]:10.2f}ms" for n in rows))


if __name__ == "__main__":
    main()
