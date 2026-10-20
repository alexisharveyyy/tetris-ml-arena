"""Subprocess vector environment for the learned agents.

Each worker process runs several games and returns, per environment, either
the candidate afterstates (``mode="afterstate"``, DQN), an observation and
legal-action mask (``mode="obs"``, raw PPO), or both an observation and the
afterstate of each of the 80 actions (``mode="action_afterstate"``, PPO). Episodes end with:

* ``terminated``: the game topped out (or has no legal placement). Terminal.
* ``truncated``: the piece cap was reached. Not terminal: the package of the
  final state is returned as ``final`` so the learner can bootstrap from it,
  and the environment then resets.

Workers never import torch. Stepping is split into :meth:`VecEnv.step_async` and
:meth:`VecEnv.step_wait` so the learner can train while workers simulate.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tetris.agents.evaluation import EvalSetting
from tetris.engine.game import Game
from tetris.engine.placement import Placement
from tetris.engine.rng import Rng, derive_seed
from tetris.env.afterstates import action_mask, enumerate_afterstates
from tetris.env.tetris_env import inject_scheduled_garbage
from tetris.rl.encoding import (
    action_afterstate_arrays,
    afterstate_arrays,
    observation_arrays,
    reward_for_score,
)

#: Training seeds come from [0, TRAIN_SEED_RANGE); selection and held-out seeds are above it.
TRAIN_SEED_RANGE = 500_000


@dataclass
class StepResult:
    """Per-environment result of one step."""

    reward: float
    terminated: bool
    truncated: bool
    package: dict[str, np.ndarray]  # state to act on next (after any reset)
    final: dict[str, np.ndarray] | None = None  # final state when truncated
    episode: dict[str, Any] | None = None  # stats when the episode ended
    info: dict[str, Any] = field(default_factory=dict)


def _package(game: Game, mode: str, include_hold: bool) -> dict[str, np.ndarray]:
    if mode == "afterstate":
        afs = enumerate_afterstates(game, include_hold=include_hold, dedupe=True)
        boards, extras, rewards, tops, actions = afterstate_arrays(game, afs)
        return {
            "boards": boards,
            "extras": extras,
            "rewards": rewards,
            "tops": tops,
            "actions": actions,
        }
    board, extras = observation_arrays(game)
    if mode == "action_afterstate":
        cand_boards, cand_extras, mask = action_afterstate_arrays(game)
        if not include_hold:
            mask[40:] = False
        return {
            "board": board,
            "extras": extras,
            "mask": mask,
            "cand_boards": cand_boards,
            "cand_extras": cand_extras,
        }
    return {"board": board, "extras": extras, "mask": action_mask(game, include_hold)}


class IllegalActionError(RuntimeError):
    """A policy sent an action the environment's own legality check rejects."""


class _EnvSlot:
    def __init__(self, rng: Rng, setting: EvalSetting) -> None:
        self.rng = rng
        self.setting = setting
        self.game: Game | None = None
        self.episode_reward = 0.0
        self.seed: int | None = None
        #: Steps since this env's last reset, and how its previous episode ended.
        self.steps_since_reset = 0
        self.last_end = "start"

    def reset(self, reason: str = "start") -> Game:
        self.seed = self.rng.randbelow(TRAIN_SEED_RANGE)
        self.game = Game(self.seed)
        self.episode_reward = 0.0
        self.steps_since_reset = 0
        self.last_end = reason
        return self.game

    def illegal_report(self, index: int, action: int) -> dict[str, Any]:
        g = self.game
        return {
            "env": index,
            "action": int(action),
            "placement": repr(Placement.from_index(int(action))),
            "piece": g.active.piece.letter if g.active else None,
            "hold": g.hold.letter if g.hold else None,
            "can_hold": g.can_hold,
            "legal_actions": int(action_mask(g, self.setting.include_hold).sum()),
            "steps_since_reset": self.steps_since_reset,
            "previous_episode_end": self.last_end,
            "game_seed": self.seed,
            "pieces_placed": g.pieces_placed,
        }

    def has_moves(self) -> bool:
        g = self.game
        return (
            g is not None
            and not g.game_over
            and bool(action_mask(g, self.setting.include_hold).any())
        )


def _worker(conn, mode: str, n_envs: int, seed: int, setting_d: dict[str, Any]) -> None:
    setting = EvalSetting.from_dict(setting_d)
    slots = [_EnvSlot(Rng(derive_seed(seed, i)), setting) for i in range(n_envs)]
    while True:
        cmd, payload = conn.recv()
        if cmd == "reset":
            packages = []
            for slot in slots:
                slot.reset()
                packages.append(_package(slot.game, mode, setting.include_hold))
            conn.send(packages)
        elif cmd == "step":
            # Check every action against the env's own legality first, independent of
            # the mask the policy used; report instead of crashing the worker.
            illegal = [
                slot.illegal_report(i, a)
                for i, (slot, a) in enumerate(zip(slots, payload, strict=True))
                if not action_mask(slot.game, slot.setting.include_hold)[int(a)]
            ]
            if illegal:
                conn.send({"illegal": illegal})
                continue
            results = []
            for slot, action in zip(slots, payload, strict=True):
                g = slot.game
                before = g.score
                g.place(Placement.from_index(int(action)))
                inject_scheduled_garbage(g, setting.garbage_every, setting.garbage_rows)
                reward = float(reward_for_score(g.score - before))
                slot.episode_reward += reward
                terminated = not slot.has_moves()
                truncated = not terminated and g.pieces_placed >= setting.max_pieces
                final = episode = None
                if terminated or truncated:
                    if truncated:
                        final = _package(g, mode, setting.include_hold)
                    episode = {
                        "pieces": g.pieces_placed,
                        "lines": g.lines,
                        "score": g.score,
                        "topped_out": terminated,
                        "return": slot.episode_reward,
                    }
                    slot.reset("terminated" if terminated else "truncated")
                else:
                    slot.steps_since_reset += 1
                results.append(
                    StepResult(
                        reward,
                        terminated,
                        truncated,
                        _package(slot.game, mode, setting.include_hold),
                        final,
                        episode,
                    )
                )
            conn.send(results)
        elif cmd == "setting":
            setting = EvalSetting.from_dict(payload)
            for slot in slots:
                slot.setting = setting
            conn.send(True)
        elif cmd == "close":
            conn.close()
            return


class VecEnv:
    """``n_workers`` processes with ``envs_per_worker`` games each."""

    def __init__(
        self,
        mode: str,
        n_workers: int,
        envs_per_worker: int,
        setting: EvalSetting,
        seed: int,
    ) -> None:
        if mode not in ("afterstate", "obs", "action_afterstate"):
            raise ValueError(mode)
        ctx = mp.get_context("spawn")
        self.mode = mode
        self.n_envs = n_workers * envs_per_worker
        self.envs_per_worker = envs_per_worker
        self._conns = []
        self._procs = []
        for w in range(n_workers):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(
                    child,
                    mode,
                    envs_per_worker,
                    derive_seed(seed, 10_000 + w),
                    setting.to_dict(),
                ),
                daemon=True,
            )
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)
        self._waiting = False

    def reset(self) -> list[dict[str, np.ndarray]]:
        for c in self._conns:
            c.send(("reset", None))
        return [p for c in self._conns for p in c.recv()]

    def step_async(self, actions: list[int] | np.ndarray) -> None:
        actions = list(actions)
        k = self.envs_per_worker
        for i, c in enumerate(self._conns):
            c.send(("step", actions[i * k : (i + 1) * k]))
        self._waiting = True

    def step_wait(self) -> list[StepResult]:
        self._waiting = False
        replies = [c.recv() for c in self._conns]
        illegal = []
        for w, reply in enumerate(replies):
            if isinstance(reply, dict) and "illegal" in reply:
                for r in reply["illegal"]:
                    illegal.append({**r, "env": w * self.envs_per_worker + r["env"]})
        if illegal:
            details = "\n".join(str(r) for r in illegal)
            raise IllegalActionError(
                f"{len(illegal)} env(s) rejected the policy's action (env state unchanged):\n"
                f"{details}"
            )
        return [r for reply in replies for r in reply]

    def step(self, actions: list[int] | np.ndarray) -> list[StepResult]:
        self.step_async(actions)
        return self.step_wait()

    def set_setting(self, setting: EvalSetting) -> None:
        """Change the environment setting (curricula). Takes effect for new placements."""
        for c in self._conns:
            c.send(("setting", setting.to_dict()))
        for c in self._conns:
            c.recv()

    def close(self) -> None:
        if self._waiting:
            self.step_wait()
        for c in self._conns:
            try:
                c.send(("close", None))
            except (OSError, BrokenPipeError):
                pass
        for p in self._procs:
            p.join(timeout=2)
            if p.is_alive():
                p.kill()
