"""Local record of the human's Marathon results against each agent and preset.

A win means the human's score beat that agent's score in a finished match;
equal scores are recorded as ties. Stored as JSON next to the user settings.

File format (version 2)::

    {"version": 2,
     "marathon": {"<agent>|<preset>": {"wins": n, "losses": n, "ties": n}},
     "matches": [{"id", "finished_at", "mode", "seed",
                  "opponents": [{"agent", "preset"}], "players": [standings rows]}]}

``marathon`` holds the running totals (version 1 files had only this) and
``matches`` one entry per finished match, every player's final row included.
Writes go to a temporary file first and replace the old one, so a crash never
leaves a half-written history.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

DEFAULT_HISTORY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "user_history.json"
)
VERSION = 2

#: Standings fields kept for each player in a match record.
PLAYER_FIELDS = (
    "place",
    "id",
    "name",
    "kind",
    "agent_key",
    "preset",
    "score",
    "lines",
    "level",
    "pieces",
    "tetrises",
    "pps",
    "topped_out",
    "finish_time",
    "survival_time",
)


def record_key(agent_key: str, preset: str | None) -> str:
    return f"{agent_key}|{preset or 'Custom'}"


class MarathonHistory:
    def __init__(self, path: Path = DEFAULT_HISTORY_PATH) -> None:
        self.path = path
        self.records: dict[str, dict[str, int]] = {}
        self.matches: list[dict[str, Any]] = []
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.records = data.get("marathon", {})
                self.matches = data.get("matches", [])
            except (json.JSONDecodeError, OSError, AttributeError):
                self.records, self.matches = {}, []

    def record_match(
        self, standings: list[dict[str, Any]], seed: int | None = None, mode: str = "marathon"
    ) -> str | None:
        """Add one finished Marathon match (standings from the controller).

        Returns the new match id, or None when there was no human player.
        """
        human = next((r for r in standings if r["kind"] == "human"), None)
        if human is None:
            return None
        for row in standings:
            if row["kind"] != "agent" or row.get("agent_key") is None:
                continue
            rec = self.records.setdefault(
                record_key(row["agent_key"], row.get("preset")), {"wins": 0, "losses": 0, "ties": 0}
            )
            if human["score"] > row["score"]:
                rec["wins"] += 1
            elif human["score"] < row["score"]:
                rec["losses"] += 1
            else:
                rec["ties"] += 1
        match_id = uuid.uuid4().hex[:12]
        self.matches.append(
            {
                "id": match_id,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "mode": mode,
                "seed": seed,
                "opponents": [
                    {"agent": r["agent_key"], "preset": r.get("preset")}
                    for r in standings
                    if r["kind"] == "agent"
                ],
                "players": [{k: r.get(k) for k in PLAYER_FIELDS} for r in standings],
            }
        )
        self.save()
        return match_id

    def get(self, agent_key: str, preset: str | None) -> dict[str, int]:
        return self.records.get(record_key(agent_key, preset), {"wins": 0, "losses": 0, "ties": 0})

    def describe(self, agent_key: str, preset: str | None) -> str:
        r = self.get(agent_key, preset)
        if not any(r.values()):
            return "no games yet"
        text = f"{r['wins']}W {r['losses']}L"
        return text + (f" {r['ties']}T" if r["ties"] else "")

    def agent_records(self, agent_key: str) -> dict[str, dict[str, int]]:
        """Records against one agent, keyed by preset."""
        prefix = f"{agent_key}|"
        return {k[len(prefix) :]: v for k, v in self.records.items() if k.startswith(prefix)}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A temporary name per writer, so two processes sharing configs/ (two
        # servers, or the server and the Pygame app) never rename each other's file.
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(
            json.dumps(
                {"version": VERSION, "marathon": self.records, "matches": self.matches}, indent=2
            )
        )
        os.replace(tmp, self.path)
