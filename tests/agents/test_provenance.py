from __future__ import annotations

import os
import shutil
import time

from tetris.agents import provenance
from tetris.agents.provenance import (
    PENDING,
    VALID,
    checkpoint_source,
    load_sources,
    result_status,
    write_sources,
)
from tetris.agents.registry import AGENTS, CHECKPOINT_ENV


def test_sources_round_trip_and_staleness(tmp_path, monkeypatch) -> None:
    ckpt = tmp_path / "ckpt"
    (ckpt / "ga").mkdir(parents=True)
    shutil.copy(AGENTS["ga"].checkpoint_path, ckpt / "ga" / "best.json")
    monkeypatch.setenv(CHECKPOINT_ENV, str(ckpt))
    reports = tmp_path / "reports"
    reports.mkdir()
    write_sources(reports, "garbage", {k: checkpoint_source(k) for k in ("ga", "dellacherie")})
    write_sources(reports, "marathon", {"ga": checkpoint_source("ga")})
    sources = load_sources(reports)
    assert set(sources["sections"]) == {"garbage", "marathon"}
    assert sources["sections"]["garbage"]["agents"]["ga"]["checkpoint"] == "ga/best.json"
    assert result_status(sources, "garbage", "ga") == (VALID, "evaluated", "evaluated ga/best.json")
    assert result_status(sources, "garbage", "dellacherie")[0] == VALID
    # No source recorded for this agent or section.
    assert result_status(sources, "garbage", "cem")[0] == PENDING
    assert result_status(sources, "plain", "ga")[0] == PENDING

    # The promoted file changes after the evaluation: the result is stale.
    path = ckpt / "ga" / "best.json"
    path.write_text(path.read_text() + "\n")
    t = time.time() + 5
    os.utime(path, (t, t))
    status, code, _ = result_status(sources, "garbage", "ga")
    assert status == PENDING and code == "changed"

    # The promoted file is gone.
    path.unlink()
    monkeypatch.setattr("tetris.agents.registry.CHECKPOINT_DIR", tmp_path / "none")
    assert result_status(sources, "garbage", "ga")[0] == PENDING


def test_buggy_checkpoints_are_never_valid(tmp_path) -> None:
    src = checkpoint_source("ga")
    sources = {
        "sections": {
            "garbage": {
                "agents": {
                    "ga": {**src, "checkpoint": "ga/best_mpsbug.json"},
                    "cem": {**checkpoint_source("cem"), "invalid": "changed during evaluation"},
                }
            }
        }
    }
    status, code, _ = result_status(sources, "garbage", "ga")
    assert status == PENDING and code == "buggy"
    status, code, detail = result_status(sources, "garbage", "cem")
    assert status == PENDING and code == "invalid" and "changed during evaluation" in detail
    assert "mpsbug" in provenance.INVALID_MARKERS


def test_repo_reports_record_their_sources() -> None:
    sources = load_sources(provenance.Path(__file__).resolve().parents[2] / "reports")
    for section in ("garbage", "plain", "marathon"):
        agents = sources["sections"][section]["agents"]
        assert agents["ppo"]["invalid"] and agents["ppo"]["checkpoint"] == "ppo/best_mpsbug.pt"
        assert agents["dellacherie"] == {"checkpoint": None}
