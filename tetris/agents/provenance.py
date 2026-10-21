"""Which checkpoint produced each leaderboard result, and is that result still current?

``scripts/evaluate.py`` writes ``reports/leaderboard_sources.json`` next to the
CSVs: for each report section (``garbage``, ``plain``, ``marathon``) and agent,
the checkpoint it evaluated (path relative to its checkpoint directory, sha256
and size). Readers call :func:`result_status` before showing a number:

* ``valid``: the agent needs no checkpoint, or the recorded sha256 equals the
  agent's current promoted checkpoint.
* ``pending_rerun``: the source is marked invalid (for example a file named
  ``*mpsbug*``), the promoted checkpoint has changed or is missing, or the
  report has no source recorded for that agent. The number must not be shown.

The file only ever reads checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from tetris.agents.registry import AGENTS, read_checkpoint_dirs

SOURCES_FILE = "leaderboard_sources.json"
#: A checkpoint whose file name contains one of these was produced by a known bug.
INVALID_MARKERS: tuple[str, ...] = ("mpsbug",)

VALID = "valid"
PENDING = "pending_rerun"

_hash_cache: dict[tuple, str] = {}


def file_sha256(path: Path) -> str:
    """sha256 of a file, cached per (path, size, modification time)."""
    st = path.stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _hash_cache:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _hash_cache[key] = h.hexdigest()
    return _hash_cache[key]


def _relative(path: Path) -> str:
    for d in read_checkpoint_dirs():
        try:
            return str(path.relative_to(d))
        except ValueError:
            continue
    return str(path)


def checkpoint_source(key: str) -> dict[str, Any]:
    """The source record for an agent's current promoted checkpoint."""
    path = AGENTS[key].checkpoint_path
    if path is None:
        return {"checkpoint": None}
    return {"checkpoint": _relative(path), "sha256": file_sha256(path), "size": path.stat().st_size}


def load_sources(reports: Path) -> dict[str, Any]:
    path = reports / SOURCES_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def write_sources(reports: Path, section: str, sources: dict[str, dict[str, Any]]) -> None:
    """Replace one section of the sources file, keeping the others."""
    data = load_sources(reports)
    data.setdefault("sections", {})[section] = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "basis": "recorded by scripts/evaluate.py",
        "agents": sources,
    }
    (reports / SOURCES_FILE).write_text(json.dumps(data, indent=2) + "\n")


#: Plain sentences for each pending reason code (UI copy).
REASON_TEXT: dict[str, str] = {
    "unrecorded": "the report does not say which checkpoint it evaluated",
    "invalid": "the evaluated checkpoint was produced by a known bug",
    "buggy": "the evaluated checkpoint was produced by a known bug",
    "unregistered": "the agent is not registered",
    "missing": "the evaluated checkpoint is not on this computer",
    "changed": "the agent's checkpoint changed after it was evaluated",
}


def result_status(sources: dict[str, Any], section: str, key: str) -> tuple[str, str, str]:
    """``(VALID or PENDING, reason code, detail)`` for one agent's result in one section.

    Codes: ``no_checkpoint`` and ``evaluated`` (valid); ``unrecorded``,
    ``invalid``, ``buggy``, ``unregistered``, ``missing`` and ``changed``
    (pending; see :data:`REASON_TEXT`).
    """
    entry = sources.get("sections", {}).get(section, {}).get("agents", {}).get(key)
    if entry is None:
        return PENDING, "unrecorded", "no source recorded for this agent"
    name = entry.get("checkpoint")
    if entry.get("invalid"):
        return PENDING, "invalid", f"{name}: {entry['invalid']}"
    if name is not None and any(m in Path(name).name for m in INVALID_MARKERS):
        return PENDING, "buggy", f"{name} is marked as produced by a bug"
    spec = AGENTS.get(key)
    if spec is None:
        return PENDING, "unregistered", f"{key} is not in the registry"
    current = spec.checkpoint_path
    if name is None and current is None:
        return VALID, "no_checkpoint", "no checkpoint needed"
    if current is None or not current.exists():
        return PENDING, "missing", f"no promoted checkpoint found for {key}"
    if file_sha256(current) != entry.get("sha256"):
        return PENDING, "changed", f"{current} differs from the evaluated {name}"
    return VALID, "evaluated", f"evaluated {name}"
