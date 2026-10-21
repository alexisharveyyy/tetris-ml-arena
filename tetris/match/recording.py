"""Recording mode: turn a human game into an imitation-learning log.

Every locked piece becomes one step in the same format as the agents' game
logs (:mod:`tetris.agents.game_log`): the full game state when the piece
spawned (before any hold), the 80-action legal mask, and the 80-space action
that produces the same locked cells.

A human can also place pieces that no hard-drop action can reach: tucks under
an overhang, spins, or slides after a soft drop. Those steps are saved with
``action = -1``, ``off_space = true`` and the locked cells, and the imitation
trainer skips them.

Files: ``data/recordings/human/<setting>/<timestamp>_seed_<seed>.jsonl.gz``.
"""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tetris import state
from tetris.agents.evaluation import GARBAGE_SETTING, PLAIN_SETTING, EvalSetting
from tetris.agents.game_log import mask_to_str
from tetris.engine.board import piece_cells
from tetris.engine.game import Game
from tetris.env.afterstates import action_mask, enumerate_afterstates

#: data/recordings, or under TETRIS_STATE_DIR (:mod:`tetris.state`).
RECORDINGS_DIR = state.recordings_dir()
#: Recording settings: plain (level gravity, no garbage) or the garbage training setting.
RECORD_SETTINGS: dict[str, EvalSetting] = {"plain": PLAIN_SETTING, "garbage": GARBAGE_SETTING}


def match_action(state: Game, piece_letter: str, cells: list[tuple[int, int]]) -> int:
    """The 80-space action whose hard-drop placement locks exactly ``cells``, or -1.

    ``state`` is the game when the piece spawned (before any hold). When
    several actions give the same cells (for example the O piece's rotations),
    the lowest index is returned.
    """
    target = sorted((int(r), int(c)) for r, c in cells)
    for af in sorted(enumerate_afterstates(state), key=lambda a: a.placement.index):
        t = af.target
        if t.piece.letter != piece_letter:
            continue
        if sorted(piece_cells(t.piece, t.rotation, t.row, t.box_col)) == target:
            return af.placement.index
    return -1


@dataclass
class RecordedStep:
    state: dict[str, Any]
    mask: str
    action: int
    off_space: bool
    cells: list[list[int]]

    def to_record(self, step: int) -> dict[str, Any]:
        record = {
            "type": "step",
            "step": step,
            "action": self.action,
            "mask": self.mask,
            "state": self.state,
            "off_space": self.off_space,
        }
        if self.off_space:
            record["cells"] = self.cells
        return record


@dataclass
class GameRecorder:
    """Records one human game from a :class:`~tetris.match.MatchController`.

    Call :meth:`start` with the game after its first piece spawns, and register
    :meth:`on_lock` as a controller lock listener.
    """

    player_id: str
    seed: int
    setting: str = "plain"
    steps: list[RecordedStep] = field(default_factory=list)
    _pending: Game | None = None

    def start(self, game: Game) -> None:
        self.steps.clear()
        self._pending = game.copy()

    def on_lock(self, player_id: str, game: Game, data: dict[str, Any]) -> None:
        """Lock listener: finish the step for the piece that just locked."""
        if player_id != self.player_id or self._pending is None:
            return
        before = self._pending
        cells = [tuple(c) for c in data["cells"]]
        action = match_action(before, data["piece"], cells)
        self.steps.append(
            RecordedStep(
                before.to_dict(),
                mask_to_str(action_mask(before)),
                action,
                action < 0,
                [list(c) for c in cells],
            )
        )
        # The next piece has already spawned (and any garbage was applied).
        self._pending = None if game.game_over else game.copy()

    @property
    def on_space(self) -> int:
        return sum(1 for s in self.steps if not s.off_space)

    @property
    def off_space(self) -> int:
        return sum(1 for s in self.steps if s.off_space)

    def save(self, game: Game, root: Path | None = None) -> Path:
        """Write the log under ``root`` (default :data:`RECORDINGS_DIR`); returns its path."""
        root = root if root is not None else RECORDINGS_DIR
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = root / "human" / self.setting / f"{stamp}_seed_{self.seed}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            header = {
                "type": "episode",
                "agent": "human",
                "seed": self.seed,
                "setting": RECORD_SETTINGS[self.setting].to_dict(),
                "checkpoint": None,
                "recorded_at": stamp,
            }
            f.write(json.dumps(header) + "\n")
            for i, step in enumerate(self.steps):
                f.write(json.dumps(step.to_record(i), separators=(",", ":")) + "\n")
            result = {
                "type": "result",
                "seed": self.seed,
                "pieces": game.pieces_placed,
                "lines": game.lines,
                "score": game.score,
                "topped_out": game.game_over,
                "on_space": self.on_space,
                "off_space": self.off_space,
            }
            f.write(json.dumps(result) + "\n")
        tmp.replace(path)
        return path


def recording_files(root: Path | None = None) -> list[Path]:
    """Saved human recordings, oldest first (file names start with the timestamp)."""
    root = root if root is not None else RECORDINGS_DIR
    return sorted((root / "human").glob("*/*.jsonl.gz"), key=lambda p: p.name)


def recording_totals(root: Path | None = None) -> dict[str, int]:
    """Saved human games and their on-space and off-space placements."""
    totals = {"games": 0, "on_space": 0, "off_space": 0}
    for path in recording_files(root):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec["type"] == "result":
                    totals["games"] += 1
                    totals["on_space"] += rec["on_space"]
                    totals["off_space"] += rec["off_space"]
    return totals
