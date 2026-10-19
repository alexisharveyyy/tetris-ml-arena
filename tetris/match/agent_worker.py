"""Entry point of an agent worker process.

One worker serves one agent slot for a whole match: it builds the agent and
loads its checkpoint once, reports ``ready``, then answers decision requests
until told to close. Inference runs on the CPU.
"""

from __future__ import annotations

import time
import traceback
from multiprocessing.connection import Connection
from typing import Any


def worker_main(
    conn: Connection,
    factory: str,
    checkpoint: str | None,
    difficulty: dict[str, Any],
    seed: int | None,
) -> None:
    t0 = time.perf_counter()
    try:
        from tetris.agents.difficulty import DifficultyParams
        from tetris.agents.registry import create_agent_from_factory
        from tetris.match.agent_runtime import decide

        agent = create_agent_from_factory(
            factory, checkpoint, DifficultyParams.from_dict(difficulty), seed
        )
        try:
            import sys

            if "torch" in sys.modules:
                sys.modules["torch"].set_num_threads(1)
        except Exception:  # noqa: BLE001 - thread tuning is best effort
            pass
        conn.send(("ready", time.perf_counter() - t0))
    except Exception:  # noqa: BLE001 - reported to the parent, which falls back
        conn.send(("error", traceback.format_exc()))
        return

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        kind = msg[0]
        if kind == "decide":
            _, request_id, state, top_k = msg
            conn.send(("decision", decide(agent, request_id, state, top_k)))
        elif kind == "difficulty":
            agent.set_difficulty(DifficultyParams.from_dict(msg[1]))
        elif kind == "close":
            return
