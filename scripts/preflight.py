"""Run the MPS-versus-CPU gradient pre-flight for the learned agents at their real batch sizes.

The same check runs automatically at the start of every training run (see
tetris/rl/safeguards.py). This script reports it for all three trainers, and
with ``--show-bug`` also for afterstate PPO with chunking disabled (the
configuration of the broken run), which must fail.

Usage: python scripts/preflight.py [--show-bug]
"""

from __future__ import annotations

import argparse

import torch

from tetris.agents.dqn import train as dqn_train
from tetris.agents.ppo import train as ppo_train
from tetris.rl import nets
from tetris.rl.nets import AfterstateValueNet, build
from tetris.rl.safeguards import gradient_difference, preflight
from tetris.rl.training import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--show-bug", action="store_true")
    args = parser.parse_args()
    device = nets.select_device("mps")
    torch.manual_seed(0)
    cfg = load_config(str(dqn_train.DEFAULT_CONFIG), {})
    preflight(
        "dqn",
        AfterstateValueNet(**cfg.get("model", {})),
        dqn_train.preflight_batch(cfg),
        dqn_train.preflight_loss,
        device,
        f"batch {cfg['batch_size']} x {cfg['next_candidates']} candidates",
    )
    for name, path in (
        ("ppo_raw", "tetris/agents/ppo/config_raw.yaml"),
        ("ppo", str(ppo_train.DEFAULT_CONFIG)),
    ):
        cfg = load_config(path, {})
        policy, mb = cfg["policy"], int(cfg["minibatch_size"])
        preflight(
            name,
            build(ppo_train.NET_KIND[policy], cfg.get("model", {})),
            ppo_train.preflight_batch(policy, mb),
            ppo_train.preflight_loss,
            device,
            f"minibatch {mb} states",
        )
    if args.show_bug:
        cfg = load_config(str(ppo_train.DEFAULT_CONFIG), {})
        saved, nets.SCORER_CHUNK = nets.SCORER_CHUNK, 10**9
        rel, _ = gradient_difference(
            build("ppo", cfg.get("model", {})),
            ppo_train.preflight_batch("afterstate", int(cfg["minibatch_size"])),
            ppo_train.preflight_loss,
            device,
        )
        nets.SCORER_CHUNK = saved
        print(f"ppo with chunking disabled (the broken run): relative difference {rel:.2e}")


if __name__ == "__main__":
    main()
