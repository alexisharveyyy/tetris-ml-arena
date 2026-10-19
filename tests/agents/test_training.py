from __future__ import annotations

import json

import numpy as np
import pytest

from tetris.agents import registry
from tetris.agents.cem import train as cem_train
from tetris.agents.heuristic_ga import train as ga_train
from tetris.agents.linear_training import generation_seeds, load_config


@pytest.fixture
def tmp_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(ga_train, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(cem_train, "CHECKPOINT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(ga_train, "ROOT", tmp_path)
    monkeypatch.setattr(cem_train, "ROOT", tmp_path)
    return tmp_path / "checkpoints"


TINY = {
    "workers": 0,
    "games_per_eval": 2,
    "max_pieces": 20,
    "final_candidates": 2,
    "selection_games": 2,
    "final_eval_games": 2,
}


def test_generation_seeds_are_shared_and_fresh() -> None:
    assert generation_seeds(0, 3, 5) == generation_seeds(0, 3, 5)
    assert generation_seeds(0, 3, 5) != generation_seeds(0, 4, 5)


def test_ga_trains_resumes_and_is_deterministic(tmp_checkpoints) -> None:
    cfg = load_config(ga_train.DEFAULT_CONFIG, {**TINY, "population": 6, "generations": 2})
    ga_train.train(cfg)
    first = json.loads((tmp_checkpoints / "ga" / "best.json").read_text())["weights"]

    # Train 1 generation, then resume to 2: must match the uninterrupted run.
    (tmp_checkpoints / "ga" / "state.json").unlink()
    ga_train.train({**cfg, "generations": 1})
    ga_train.train(cfg, resume=True)
    resumed = json.loads((tmp_checkpoints / "ga" / "best.json").read_text())["weights"]
    np.testing.assert_allclose(first, resumed)
    report = json.loads((tmp_checkpoints / "ga" / "report.json").read_text())
    assert set(report["held_out"]) == {"garbage", "plain"}


def test_cem_trains(tmp_checkpoints) -> None:
    cfg = load_config(cem_train.DEFAULT_CONFIG, {**TINY, "samples": 6, "generations": 2})
    cem_train.train(cfg)
    data = json.loads((tmp_checkpoints / "cem" / "best.json").read_text())
    assert len(data["weights"]) == 13
    registry_agent = registry.create_agent("cem", tmp_checkpoints / "cem" / "best.json")
    assert registry_agent.weights.shape == (13,)
