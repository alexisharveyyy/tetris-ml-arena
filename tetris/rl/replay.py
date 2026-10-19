"""Prioritized replay (proportional, sum tree) for afterstate DQN.

A transition stores the chosen afterstate, an n-step partial return, and up to
``K`` candidate afterstates of the bootstrap state (with their rewards and
top-out flags), so the Double DQN target
``R + gamma^k max_a' [r(s', a') + gamma V(after(s', a'))]`` can be computed
without re-simulating the game. Boards are bit-packed (30 bytes each).
"""

from __future__ import annotations

import numpy as np

from tetris.rl.encoding import AFTERSTATE_EXTRA_DIM, PACKED_BOARD_BYTES


class SumTree:
    """Array-backed binary sum tree with vectorized update and sampling."""

    def __init__(self, capacity: int) -> None:
        size = 1
        while size < capacity:
            size *= 2
        self.size = size
        self.tree = np.zeros(2 * size, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.tree[1])

    def update(self, idx: np.ndarray, values: np.ndarray) -> None:
        pos = np.asarray(idx) + self.size
        self.tree[pos] = values
        pos = np.unique(pos // 2)
        while pos[0] >= 1:
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]
            if pos[0] == 1:
                break
            pos = np.unique(pos // 2)

    def update_one(self, idx: int, value: float) -> None:
        """Fast path for a single leaf (no array allocation)."""
        pos = idx + self.size
        tree = self.tree
        tree[pos] = value
        pos //= 2
        while pos >= 1:
            tree[pos] = tree[2 * pos] + tree[2 * pos + 1]
            pos //= 2

    def find(self, mass: np.ndarray) -> np.ndarray:
        """Leaf index for each cumulative mass value."""
        pos = np.ones(len(mass), dtype=np.int64)
        mass = mass.astype(np.float64).copy()
        while pos[0] < self.size:
            left = 2 * pos
            go_right = mass > self.tree[left]
            mass = np.where(go_right, mass - self.tree[left], mass)
            pos = np.where(go_right, left + 1, left)
        return pos - self.size

    def get(self, idx: np.ndarray) -> np.ndarray:
        return self.tree[np.asarray(idx) + self.size]


class PrioritizedReplay:
    def __init__(
        self, capacity: int, k: int, alpha: float = 0.6, eps: float = 1e-3, seed: int = 0
    ) -> None:
        self.capacity = capacity
        self.k = k
        self.alpha = alpha
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.pos = 0
        self.full = False
        b, e = PACKED_BOARD_BYTES, AFTERSTATE_EXTRA_DIM
        self.after_board = np.zeros((capacity, b), np.uint8)
        self.after_extra = np.zeros((capacity, e), np.float16)
        self.next_board = np.zeros((capacity, k, b), np.uint8)
        self.next_extra = np.zeros((capacity, k, e), np.float16)
        self.next_reward = np.zeros((capacity, k), np.float32)
        self.next_top = np.zeros((capacity, k), bool)
        self.next_valid = np.zeros((capacity, k), bool)
        self.terminated = np.zeros(capacity, bool)
        self.truncated = np.zeros(capacity, bool)
        #: n-step: discounted rewards collected before the bootstrap state, and
        #: the discount applied to the bootstrap (0 when terminated).
        self.partial_return = np.zeros(capacity, np.float32)
        self.discount = np.ones(capacity, np.float32)

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def add(
        self,
        after_board: np.ndarray,
        after_extra: np.ndarray,
        terminated: bool,
        truncated: bool,
        next_board: np.ndarray | None = None,
        next_extra: np.ndarray | None = None,
        next_reward: np.ndarray | None = None,
        next_top: np.ndarray | None = None,
        partial_return: float = 0.0,
        discount: float = 1.0,
    ) -> None:
        """Store one transition. Next-state candidates are required unless terminated.

        The target is ``partial_return + discount * max_c Q(c)`` over the stored
        candidates, or just ``partial_return`` when terminated.
        """
        i = self.pos
        self.partial_return[i] = partial_return
        self.discount[i] = 0.0 if terminated else discount
        self.after_board[i] = after_board
        self.after_extra[i] = after_extra
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        self.next_valid[i] = False
        if not terminated:
            if next_board is None or len(next_board) == 0:
                raise ValueError("non-terminal transitions need next-state candidates")
            n = min(self.k, len(next_board))
            self.next_board[i, :n] = next_board[:n]
            self.next_extra[i, :n] = next_extra[:n]
            self.next_reward[i, :n] = next_reward[:n]
            self.next_top[i, :n] = next_top[:n]
            self.next_valid[i, :n] = True
        self.tree.update_one(i, self.max_priority**self.alpha)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch: int, beta: float) -> tuple[np.ndarray, np.ndarray]:
        """Indices and normalized importance weights (stratified sampling)."""
        total = self.tree.total
        bounds = np.linspace(0.0, total, batch + 1)
        mass = self.rng.uniform(bounds[:-1], bounds[1:])
        idx = np.minimum(self.tree.find(mass), len(self) - 1)
        probs = self.tree.get(idx) / total
        weights = (len(self) * np.maximum(probs, 1e-12)) ** (-beta)
        return idx, (weights / weights.max()).astype(np.float32)

    def update_priorities(self, idx: np.ndarray, td_error: np.ndarray) -> None:
        p = np.abs(td_error) + self.eps
        self.max_priority = max(self.max_priority, float(p.max()))
        self.tree.update(idx, p**self.alpha)
