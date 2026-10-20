from __future__ import annotations

import pytest
import torch

from tetris.agents.dqn import train as dqn_train
from tetris.agents.ppo import train as ppo_train
from tetris.rl import nets, safeguards
from tetris.rl.nets import AfterstateValueNet, build
from tetris.rl.safeguards import (
    ParameterChange,
    ProgressWatchdog,
    StalledTrainingError,
    gradient_difference,
    preflight,
)


def test_watchdog_aborts_after_patience_and_resets_on_progress() -> None:
    dog = ProgressWatchdog("x", "kl", 1e-8, patience=3)
    dog.update(0.0)
    dog.update(1e-12)
    dog.update(1e-3)  # progress resets the count
    dog.update(0.0)
    dog.update(0.0)
    with pytest.raises(StalledTrainingError):
        dog.update(float("nan"))


def test_parameter_change_measures_relative_update() -> None:
    net = torch.nn.Linear(4, 1)
    change = ParameterChange(net)
    assert change() == 0.0
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(1.1)
    assert change() == pytest.approx(0.1, rel=1e-4)


def test_preflight_on_cpu_passes_without_checking(tmp_path) -> None:
    result = preflight(
        "t",
        AfterstateValueNet(),
        [],
        lambda n, b: torch.zeros(()),
        torch.device("cpu"),
        "none",
        tmp_path,
    )
    assert result.passed and result.final_device == "cpu"
    assert (tmp_path / "preflight.json").exists()


def test_preflight_tries_smaller_chunks_then_falls_back_to_cpu(monkeypatch, capsys) -> None:
    calls = []

    def fake(net, batch, loss_fn, device):
        calls.append(nets.SCORER_CHUNK)
        return 5.0, 0.0  # always wrong

    monkeypatch.setattr(safeguards, "gradient_difference", fake)
    before = nets.SCORER_CHUNK
    result = preflight("t", build("ppo", {}), [], None, torch.device("mps"), "fake")
    assert calls == [before, *[c for c in safeguards.FALLBACK_CHUNKS if c < before]]
    assert not result.passed and result.final_device == "cpu"
    assert nets.SCORER_CHUNK == before
    assert "FALLS BACK TO THE CPU" in capsys.readouterr().out


def test_preflight_keeps_the_first_chunk_that_passes(monkeypatch) -> None:
    def fake(net, batch, loss_fn, device):
        return (5.0 if nets.SCORER_CHUNK > 2048 else 1e-6), 0.0

    monkeypatch.setattr(safeguards, "gradient_difference", fake)
    before = nets.SCORER_CHUNK
    try:
        result = preflight("t", build("ppo", {}), [], None, torch.device("mps"), "fake")
        assert result.passed and result.final_device == "mps" and result.scorer_chunk == 2048
    finally:
        nets.SCORER_CHUNK = before


def test_preflight_losses_run_on_cpu_at_small_size() -> None:
    cfg = {"batch_size": 4, "next_candidates": 3}
    rel, _ = gradient_difference(
        AfterstateValueNet(), dqn_train.preflight_batch(cfg), dqn_train.preflight_loss, "cpu"
    )
    assert rel == 0.0
    for policy in ("raw", "afterstate"):
        net = build(ppo_train.NET_KIND[policy], {})
        rel, _ = gradient_difference(
            net, ppo_train.preflight_batch(policy, 6), ppo_train.preflight_loss, "cpu"
        )
        assert rel == 0.0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_preflight_catches_the_unchunked_mps_bug(monkeypatch) -> None:
    monkeypatch.setattr(nets, "SCORER_CHUNK", 10**9)
    rel, _ = gradient_difference(
        build("ppo", {}),
        ppo_train.preflight_batch("afterstate", 2048),
        ppo_train.preflight_loss,
        torch.device("mps"),
    )
    assert rel > safeguards.PREFLIGHT_TOLERANCE


def test_watchdog_threshold_scales() -> None:
    dog = ProgressWatchdog("x", "kl", 1e-6, patience=1)
    dog.update(1e-7, scale=0.05**2)  # 1e-7 is above the scaled threshold 2.5e-9
    with pytest.raises(StalledTrainingError):
        dog.update(1e-7)
