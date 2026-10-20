"""Imitation datasets: logged states encoded as their legal afterstates.

Each logged step (state before the move, teacher's 80-space action) becomes
the rows of its legal afterstates (packed board and the 29 afterstate extras,
as the PPO policy sees them) plus a target flag per row.

Several actions can produce the same afterstate: the O piece's rotations, a
horizontal I in two rotations, or holding when the held piece equals the
current one. All rows identical to the teacher's afterstate are targets, and
the loss maximizes their total probability, so the network is not penalized
for choosing an equivalent action.

Storage is compressed-sparse: rows of all states are concatenated, and
``offsets[s]:offsets[s + 1]`` are the rows of state ``s``.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tetris.agents.game_log import read_episode
from tetris.engine.game import Game
from tetris.rl.encoding import AFTERSTATE_EXTRA_DIM, PACKED_BOARD_BYTES, action_afterstate_arrays


@dataclass
class ImitationData:
    boards: np.ndarray  # (R, 30) uint8, packed afterstate boards of legal actions
    extras: np.ndarray  # (R, 29) float16
    actions: np.ndarray  # (R,) uint8, 80-space action of each row
    target: np.ndarray  # (R,) bool, row's afterstate equals the teacher's
    offsets: np.ndarray  # (S + 1,) int64
    episode: np.ndarray  # (S,) int32, episode id of each state
    teacher: np.ndarray  # (S,) int16, the teacher's action

    def __len__(self) -> int:
        return len(self.teacher)

    @property
    def n_rows(self) -> int:
        return len(self.actions)

    @classmethod
    def empty(cls) -> ImitationData:
        return cls(
            np.zeros((0, PACKED_BOARD_BYTES), np.uint8),
            np.zeros((0, AFTERSTATE_EXTRA_DIM), np.float16),
            np.zeros(0, np.uint8),
            np.zeros(0, bool),
            np.zeros(1, np.int64),
            np.zeros(0, np.int32),
            np.zeros(0, np.int16),
        )

    def rows_of(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row indices of ``states`` and, per row, its position in ``states``."""
        starts, ends = self.offsets[states], self.offsets[states + 1]
        counts = ends - starts
        owner = np.repeat(np.arange(len(states)), counts)
        first = np.repeat(starts - np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
        return first + np.arange(counts.sum()), owner

    def subset(self, states: np.ndarray) -> ImitationData:
        rows, _ = self.rows_of(states)
        counts = self.offsets[states + 1] - self.offsets[states]
        return ImitationData(
            self.boards[rows],
            self.extras[rows],
            self.actions[rows],
            self.target[rows],
            np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
            self.episode[states],
            self.teacher[states],
        )

    @classmethod
    def concat(cls, parts: Sequence[ImitationData]) -> ImitationData:
        """Concatenate; episode ids are renumbered so they stay distinct."""
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        offsets, episodes, base_row, base_ep = [np.zeros(1, np.int64)], [], 0, 0
        for p in parts:
            offsets.append(p.offsets[1:] + base_row)
            episodes.append(p.episode + base_ep)
            base_row += p.n_rows
            base_ep += int(p.episode.max()) + 1
        return cls(
            np.concatenate([p.boards for p in parts]),
            np.concatenate([p.extras for p in parts]),
            np.concatenate([p.actions for p in parts]),
            np.concatenate([p.target for p in parts]),
            np.concatenate(offsets),
            np.concatenate(episodes).astype(np.int32),
            np.concatenate([p.teacher for p in parts]),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, **{k: getattr(self, k) for k in self.__dataclass_fields__})
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> ImitationData:
        with np.load(path) as d:
            return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def encode_step(state: dict, action: int):
    """Rows for one logged step, or None if the teacher's action is not legal there."""
    game = Game.from_dict(state)
    cand_boards, cand_extras, mask = action_afterstate_arrays(game)
    if not 0 <= action < len(mask) or not mask[action]:
        return None
    legal = np.flatnonzero(mask)
    boards, extras = cand_boards[legal], cand_extras[legal]
    target = (boards == cand_boards[action]).all(1) & (extras == cand_extras[action]).all(1)
    return boards, extras.astype(np.float16), legal.astype(np.uint8), target


def encode_episode(path: str | Path) -> ImitationData:
    """Encode the on-space steps of one log file (off-space human steps are skipped)."""
    boards, extras, actions, target, counts, teacher = [], [], [], [], [], []
    for rec in read_episode(path):
        if rec["type"] != "step" or rec["action"] < 0:
            continue
        rows = encode_step(rec["state"], rec["action"])
        if rows is None:
            continue
        b, e, a, t = rows
        boards.append(b)
        extras.append(e)
        actions.append(a)
        target.append(t)
        counts.append(len(a))
        teacher.append(rec["action"])
    if not counts:
        return ImitationData.empty()
    return ImitationData(
        np.concatenate(boards),
        np.concatenate(extras),
        np.concatenate(actions),
        np.concatenate(target),
        np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        np.zeros(len(counts), np.int32),
        np.array(teacher, np.int16),
    )


def encode_files(paths: Iterable[str | Path], workers: int = 0) -> ImitationData:
    """Encode log files (one episode each), in parallel when ``workers > 0``."""
    paths = [str(p) for p in paths]
    if workers > 0 and len(paths) > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(encode_episode, paths, chunksize=1)
    else:
        parts = [encode_episode(p) for p in paths]
    return ImitationData.concat(parts)  # one episode id per non-empty file


def log_files(dirs: Iterable[str | Path]) -> list[Path]:
    """Every episode log under the given directories, sorted."""
    files: list[Path] = []
    for d in dirs:
        files += sorted(Path(d).glob("**/*.jsonl.gz"))
    return files


def split_episodes(
    data: ImitationData, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """State indices of (train, held-out) episodes; at least one episode is held out."""
    episodes = np.unique(data.episode)
    rng = np.random.default_rng(seed)
    n_out = max(1, round(fraction * len(episodes))) if len(episodes) > 1 else 0
    out = set(rng.choice(episodes, size=n_out, replace=False).tolist())
    held = np.isin(data.episode, list(out))
    return np.flatnonzero(~held), np.flatnonzero(held)
