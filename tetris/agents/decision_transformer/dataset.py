"""Decision Transformer data: trajectories as states, afterstates and returns-to-go.

Each logged step becomes:

* the afterstate rows of its legal actions, with the logged action's
  equivalence set as targets (as in :mod:`tetris.agents.imitation.dataset`);
* the state observation the transformer reads (packed board and the PPO
  observation extras);
* the reward of the move, its return-to-go and its step index in the game.

Reward per piece is ``(1 + 0.25 * points / 1000) / cap`` with ``cap`` the
evaluation cap (500), so a game's return equals its leaderboard fitness
whenever it reaches the cap (and is a close lower bound otherwise).
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tetris.agents.game_log import read_episode
from tetris.agents.imitation.dataset import ImitationData, encode_step
from tetris.engine.game import Game
from tetris.rl.encoding import (
    OBS_EXTRA_DIM,
    PACKED_BOARD_BYTES,
    observation_arrays,
    pack_boards,
    reward_for_score,
)


@dataclass
class DTData:
    steps: ImitationData  # afterstate rows, targets and episode ids of every state
    obs_boards: np.ndarray  # (S, 30) uint8, packed board before the move
    obs_extras: np.ndarray  # (S, 62) float16
    rewards: np.ndarray  # (S,) float32, in return units
    rtg: np.ndarray  # (S,) float32, return-to-go including this step's reward
    timestep: np.ndarray  # (S,) int32, piece index in the game
    chosen: np.ndarray  # (S,) int64, global row of the logged action

    def __len__(self) -> int:
        return len(self.rewards)

    @property
    def episode(self) -> np.ndarray:
        return self.steps.episode

    @classmethod
    def empty(cls) -> DTData:
        return cls(
            ImitationData.empty(),
            np.zeros((0, PACKED_BOARD_BYTES), np.uint8),
            np.zeros((0, OBS_EXTRA_DIM), np.float16),
            np.zeros(0, np.float32),
            np.zeros(0, np.float32),
            np.zeros(0, np.int32),
            np.zeros(0, np.int64),
        )

    def episode_starts(self) -> np.ndarray:
        """Index of the first state of each state's episode, ``(S,)``."""
        ep = self.episode
        first = np.ones(len(ep), bool)
        first[1:] = ep[1:] != ep[:-1]
        idx = np.where(first, np.arange(len(ep)), 0)
        return np.maximum.accumulate(idx)

    def episode_returns(self) -> np.ndarray:
        """Return of every episode (the return-to-go of its first state)."""
        ep = self.episode
        first = np.ones(len(ep), bool)
        first[1:] = ep[1:] != ep[:-1]
        return self.rtg[first]

    @classmethod
    def concat(cls, parts: Sequence[DTData]) -> DTData:
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        steps = ImitationData.concat([p.steps for p in parts])
        row_base = np.cumsum([0] + [p.steps.n_rows for p in parts[:-1]])
        return cls(
            steps,
            np.concatenate([p.obs_boards for p in parts]),
            np.concatenate([p.obs_extras for p in parts]),
            np.concatenate([p.rewards for p in parts]),
            np.concatenate([p.rtg for p in parts]),
            np.concatenate([p.timestep for p in parts]),
            np.concatenate([p.chosen + b for p, b in zip(parts, row_base, strict=True)]),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"steps_{k}": getattr(self.steps, k) for k in ImitationData.__dataclass_fields__}
        for k in self.__dataclass_fields__:
            if k != "steps":
                arrays[k] = getattr(self, k)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> DTData:
        with np.load(path) as d:
            steps = ImitationData(
                **{k: d[f"steps_{k}"] for k in ImitationData.__dataclass_fields__}
            )
            rest = {k: d[k] for k in cls.__dataclass_fields__ if k != "steps"}
        return cls(steps, **rest)


def step_rewards(scores: Sequence[int], final_score: int, cap: int) -> np.ndarray:
    """Per-move rewards from the score before each move and the final score."""
    after = np.append(np.asarray(scores[1:], np.float64), final_score)
    delta = after - np.asarray(scores, np.float64)
    return (reward_for_score(delta) / cap).astype(np.float32)


def returns_to_go(rewards: np.ndarray) -> np.ndarray:
    return np.cumsum(rewards[::-1])[::-1].astype(np.float32)


def encode_dt_episode(args: tuple[str, int]) -> DTData:
    """Encode one game log. Every logged action must be legal (agent logs)."""
    path, cap = args
    parts, obs_b, obs_e, scores, final = [], [], [], [], None
    for rec in read_episode(path):
        if rec["type"] == "result":
            final = int(rec["score"])
        if rec["type"] != "step":
            continue
        rows = encode_step(rec["state"], rec["action"])
        if rows is None:
            raise ValueError(f"{path}: step {rec['step']} logs an illegal action")
        parts.append((rows, rec["action"]))
        board, extras = observation_arrays(Game.from_dict(rec["state"]))
        obs_b.append(pack_boards(board.astype(bool)))
        obs_e.append(extras.astype(np.float16))
        scores.append(int(rec["state"]["score"]))
    if not parts or final is None:
        return DTData.empty()
    boards = np.concatenate([r[0] for r, _ in parts])
    extras = np.concatenate([r[1] for r, _ in parts])
    actions = np.concatenate([r[2] for r, _ in parts])
    target = np.concatenate([r[3] for r, _ in parts])
    counts = np.array([len(r[2]) for r, _ in parts])
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    teacher = np.array([a for _, a in parts], np.int16)
    chosen = np.array(
        [offsets[i] + int(np.flatnonzero(r[2] == a)[0]) for i, (r, a) in enumerate(parts)],
        np.int64,
    )
    steps = ImitationData(
        boards, extras, actions, target, offsets, np.zeros(len(parts), np.int32), teacher
    )
    rewards = step_rewards(scores, final, cap)
    return DTData(
        steps,
        np.stack(obs_b),
        np.stack(obs_e),
        rewards,
        returns_to_go(rewards),
        np.arange(len(parts), dtype=np.int32),
        chosen,
    )


def encode_dt_files(paths: Iterable[str | Path], cap: int, workers: int = 0) -> DTData:
    """Encode game logs (one episode each), in parallel when ``workers > 0``."""
    tasks = [(str(p), cap) for p in paths]
    if workers > 0 and len(tasks) > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(encode_dt_episode, tasks, chunksize=1)
    else:
        parts = [encode_dt_episode(t) for t in tasks]
    return DTData.concat(parts)
