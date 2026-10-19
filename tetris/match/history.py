"""Local record of the human's Marathon results against each agent and preset.

A win means the human's score beat that agent's score in a finished match;
equal scores are recorded as ties. Stored as JSON next to the user settings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_HISTORY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "user_history.json"
)


def record_key(agent_key: str, preset: str | None) -> str:
    return f"{agent_key}|{preset or 'Custom'}"


class MarathonHistory:
    def __init__(self, path: Path = DEFAULT_HISTORY_PATH) -> None:
        self.path = path
        self.records: dict[str, dict[str, int]] = {}
        if path.exists():
            try:
                self.records = json.loads(path.read_text()).get("marathon", {})
            except (json.JSONDecodeError, OSError):
                self.records = {}

    def record_match(self, standings: list[dict[str, Any]]) -> None:
        """Add one finished Marathon match (standings from the controller)."""
        human = next((r for r in standings if r["kind"] == "human"), None)
        if human is None:
            return
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
        self.save()

    def get(self, agent_key: str, preset: str | None) -> dict[str, int]:
        return self.records.get(record_key(agent_key, preset), {"wins": 0, "losses": 0, "ties": 0})

    def describe(self, agent_key: str, preset: str | None) -> str:
        r = self.get(agent_key, preset)
        if not any(r.values()):
            return "no games yet"
        text = f"{r['wins']}W {r['losses']}L"
        return text + (f" {r['ties']}T" if r["ties"] else "")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"marathon": self.records}, indent=2))
