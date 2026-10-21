from __future__ import annotations

import json
import os
import time
import zipfile

import pytest

from tetris.agents import checkpoint_check as cc
from tetris.agents.checkpoint_check import CheckpointVerifier, quick_check
from tetris.agents.registry import AGENTS, CHECKPOINT_ENV, AgentSpec


def _age(path, seconds: float = 60.0) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t))


def spec(key: str, name: str, factory: str = "tetris.agents.cem.agent:CEMAgent") -> AgentSpec:
    return AgentSpec(key, key, "test", factory, name)


def test_override_dir_searched_first_and_falls_back(tmp_path, monkeypatch) -> None:
    (tmp_path / "cem").mkdir()
    (tmp_path / "cem" / "best.json").write_text(AGENTS["cem"].checkpoint_path.read_text())
    monkeypatch.setenv(CHECKPOINT_ENV, str(tmp_path))
    assert AGENTS["cem"].checkpoint_path == tmp_path / "cem" / "best.json"
    # Not in the override: the repo's own checkpoint is used.
    assert AGENTS["ga"].checkpoint_path.parent.parent.name == "checkpoints"
    assert AGENTS["ga"].checkpoint_path.exists()


def test_quick_check_states(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(CHECKPOINT_ENV, str(tmp_path))
    monkeypatch.setenv(cc.SETTLE_ENV, "10")
    d = tmp_path / "x"
    d.mkdir()
    assert quick_check(spec("x", "best.pt")).state == cc.MISSING
    pt = d / "best.pt"
    pt.write_bytes(b"PK\x03\x04 truncated")
    assert quick_check(spec("x", "best.pt")).state == cc.WRITING  # just modified
    _age(pt)
    assert quick_check(spec("x", "best.pt")).state == cc.CORRUPT
    with zipfile.ZipFile(pt, "w") as z:
        z.writestr("data.pkl", b"x")
    _age(pt)
    assert quick_check(spec("x", "best.pt")) is None  # complete: needs a load test
    js = d / "best.json"
    js.write_text("{not json")
    _age(js)
    assert quick_check(spec("x", "best.json")).state == cc.CORRUPT
    assert quick_check(AGENTS["dellacherie"]).ok


def test_load_failure_means_not_available(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(CHECKPOINT_ENV, str(tmp_path))
    (tmp_path / "bad").mkdir()
    path = tmp_path / "bad" / "best.json"
    path.write_text(json.dumps({"weights": "nope"}))
    _age(path)
    good = tmp_path / "cem"
    good.mkdir()
    (good / "best.json").write_text(
        AGENTS["cem"].checkpoint_path.parent.joinpath("best.json").read_text()
    )
    _age(good / "best.json")
    verifier = CheckpointVerifier(inline=True)
    assert verifier.status(spec("bad", "best.json")).state == cc.LOAD_FAILED
    assert verifier.status(spec("cem", "best.json")).ok
    # A changed file is checked again.
    path.write_text((good / "best.json").read_text())
    _age(path, 30.0)
    assert verifier.status(spec("bad", "best.json")).ok


def test_background_load_check_in_worker_process() -> None:
    verifier = CheckpointVerifier(load_timeout=60.0)
    try:
        results = verifier.check_all([AGENTS["ga"], AGENTS["dellacherie"]])
        assert results["ga"].ok and results["dellacherie"].ok
    finally:
        verifier.close()


def test_settle_window_is_configurable(monkeypatch) -> None:
    monkeypatch.delenv(cc.SETTLE_ENV, raising=False)
    assert cc.settle_seconds() == cc.SETTLE_SECONDS == 10.0
    monkeypatch.setenv(cc.SETTLE_ENV, "0")
    assert cc.settle_seconds() == 0.0
    monkeypatch.setenv(cc.SETTLE_ENV, "2.5")
    assert cc.settle_seconds() == 2.5
    for bad in ("soon", "-1"):
        monkeypatch.setenv(cc.SETTLE_ENV, bad)
        with pytest.raises(ValueError, match=cc.SETTLE_ENV):
            cc.settle_seconds()
