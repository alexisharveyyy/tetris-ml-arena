"""Hard cap on API spending, shared by every process through a ledger file.

Before each API call the caller reserves the call's worst-case cost (the
prompt's tokens estimated from its length, plus the maximum output tokens).
If the reservation would take the total past the cap, the call is refused
with :class:`BudgetExhausted` and nothing is sent. After the call the
reservation is replaced by the actual cost from the reported token usage.

The ledger (``data/llm/spend.json``) is updated under an exclusive file lock,
so parallel processes cannot overspend together.
"""

from __future__ import annotations

import fcntl
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Reserved prompt tokens are an upper bound: a token never covers less than
#: one byte, and the prompts are ASCII, so there are at most as many tokens as
#: characters, plus the chat template's per-message overhead. (Real prompts use
#: about a third of that.) The cap therefore holds even when a reply reports
#: more tokens than expected.
MESSAGE_OVERHEAD_TOKENS = 64


class BudgetExhausted(RuntimeError):
    """An API call would exceed the spending cap; it was not sent."""


@dataclass(frozen=True)
class Price:
    input: float  # USD per million tokens
    output: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input + output_tokens * self.output) / 1e6


def estimate_tokens(text: str) -> int:
    """Upper bound on the prompt's tokens (see :data:`MESSAGE_OVERHEAD_TOKENS`)."""
    return len(text.encode()) + MESSAGE_OVERHEAD_TOKENS


class SpendingLedger:
    def __init__(self, path: Path, cap_usd: float) -> None:
        self.path = Path(path)
        self.cap_usd = float(cap_usd)

    @contextmanager
    def _locked(self) -> Iterator[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(".lock")
        with lock.open("a") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                state = (
                    json.loads(self.path.read_text())
                    if self.path.exists()
                    else {"spent_usd": 0.0, "reserved_usd": 0.0, "calls": 0, "by_model": {}}
                )
                yield state
                state["cap_usd"] = self.cap_usd
                state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(state, indent=2))
                tmp.replace(self.path)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def snapshot(self) -> dict[str, Any]:
        with self._locked() as state:
            return dict(state)

    @property
    def spent(self) -> float:
        return float(self.snapshot()["spent_usd"])

    def reserve(self, amount: float) -> None:
        """Reserve ``amount`` USD, or raise :class:`BudgetExhausted`."""
        with self._locked() as state:
            committed = state["spent_usd"] + state["reserved_usd"]
            if committed + amount > self.cap_usd:
                raise BudgetExhausted(
                    f"API spending cap ${self.cap_usd:.2f} reached: ${state['spent_usd']:.4f} "
                    f"spent, ${state['reserved_usd']:.4f} reserved, next call up to ${amount:.4f}"
                )
            state["reserved_usd"] += amount

    def settle(self, reserved: float, actual: float, model: str, tokens: tuple[int, int]) -> None:
        """Replace a reservation with the actual cost of the call."""
        if actual > reserved:
            print(
                f"warning: {model} call cost ${actual:.6f}, more than the ${reserved:.6f} "
                "reserved; the token estimate is too low",
                flush=True,
            )
        with self._locked() as state:
            state["reserved_usd"] = max(0.0, state["reserved_usd"] - reserved)
            state["spent_usd"] += actual
            state["calls"] += 1
            m = state["by_model"].setdefault(
                model, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}
            )
            m["calls"] += 1
            m["input_tokens"] += tokens[0]
            m["output_tokens"] += tokens[1]
            m["usd"] += actual
