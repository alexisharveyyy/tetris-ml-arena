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
    repo = provenance.Path(__file__).resolve().parents[2]
    sources = load_sources(repo / "reports" / "leaderboard")
    for section in ("garbage", "plain", "marathon"):
        agents = sources["sections"][section]["agents"]
        # The Phase 5 reports evaluated the fixed PPO, not the MPS-bug checkpoint.
        assert agents["ppo"]["checkpoint"] == "ppo/best.pt" and "invalid" not in agents["ppo"]
        assert agents["dellacherie"] == {"checkpoint": None}
        # Explicit list: other tests register fake agents in AGENTS.
        evaluated = {"dellacherie", "ga", "cem", "dqn", "ppo", "ppo_raw"}
        evaluated |= {"bc_pure", "bc_dagger", "bc_you", "neat"}
        assert evaluated <= set(agents), evaluated - set(agents)


def test_backfill_records_only_checkpoints_older_than_the_report(tmp_path, monkeypatch) -> None:
    import importlib.util
    from datetime import datetime, timedelta

    spec = importlib.util.spec_from_file_location(
        "backfill_sources",
        provenance.Path(__file__).parents[2] / "scripts" / "analysis" / "backfill_sources.py",
    )
    backfill = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backfill)

    ckpt = tmp_path / "ckpt"
    for key in ("ga", "cem"):
        (ckpt / key).mkdir(parents=True)
        shutil.copy(AGENTS[key].checkpoint_path, ckpt / key / "best.json")
    monkeypatch.setenv(CHECKPOINT_ENV, str(ckpt))
    report = tmp_path / "leaderboard_garbage.csv"
    report.write_text("agent,key\nGA,ga\nCEM,cem\nDellacherie,dellacherie\n")
    written = datetime.now() + timedelta(seconds=5)
    # CEM's checkpoint is newer than the report, so it cannot be its source.
    later = (written + timedelta(minutes=1)).timestamp()
    os.utime(ckpt / "cem" / "best.json", (later, later))
    agents, problems = backfill.infer_section(report, written)
    assert set(agents) == {"ga", "dellacherie"}
    assert agents["ga"]["checkpoint"] == "ga/best.json"
    assert agents["dellacherie"] == {"checkpoint": None}
    assert len(problems) == 1 and problems[0].startswith("cem:")
