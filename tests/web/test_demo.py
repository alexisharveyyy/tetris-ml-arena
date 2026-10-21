"""Demo mode: configs/demo.yaml, preselected opponents, fixed seed."""

from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from tetris.web.app import create_app
from tetris.web.demo import DEFAULT_DEMO_PATH, DemoConfig, DemoConfigError

from .conftest import start


def test_shipped_demo_config() -> None:
    demo = DemoConfig.load(DEFAULT_DEMO_PATH)
    assert demo.look == "tetromino"
    assert demo.opponents == (("bc_you", "Medium"), ("cem", "Easy"), ("ppo", "Medium"))
    assert isinstance(demo.seed, int)


@pytest.mark.parametrize(
    "data",
    [
        {"look": "neon", "opponents": [{"agent": "cem"}]},
        {"opponents": [{"agent": "nobody"}]},
        {"opponents": [{"agent": "bc_pure"}]},  # leaderboard only
        {"opponents": [{"agent": "cem", "preset": "Brutal"}]},
        {"opponents": []},
        {"opponents": [{"agent": "cem"}] * 4},
        {"opponents": [{"agent": "cem"}], "seed": -1},
    ],
)
def test_bad_demo_configs_are_refused(data) -> None:
    with pytest.raises(DemoConfigError):
        DemoConfig.from_dict(data)


@pytest.fixture
def demo_client(web_config):
    demo = DemoConfig.from_dict(
        {
            "look": "poster",
            "seed": 777,
            "opponents": [
                {"agent": "dellacherie", "preset": "Hard"},
                {"agent": "ga", "preset": "Easy"},
            ],
        }
    )
    with TestClient(create_app(replace(web_config, demo=demo))) as c:
        yield c


def test_demo_endpoint(client, demo_client) -> None:
    assert client.get("/api/demo").json() == {"enabled": False}
    info = demo_client.get("/api/demo").json()
    assert info["enabled"] is True
    assert info["look"] == "poster"
    assert info["seed"] == 777


def test_demo_preselects_opponents_and_seed(demo_client, web_config) -> None:
    lobby = demo_client.get("/api/lobby/marathon").json()
    assert [s and (s["agent"], s["preset"]) for s in lobby["slots"]] == [
        ("dellacherie", "Hard"),
        ("ga", "Easy"),
        None,
    ]
    info = start(demo_client, opponents=(("dellacherie", "Easy"),))
    assert info["seed"] == 777
    # A change during the demo is remembered for this run only.
    again = demo_client.get("/api/lobby/marathon").json()
    assert again["slots"][0]["preset"] == "Easy"
    assert (
        not web_config.settings_path.exists()
        or "dellacherie" not in web_config.settings_path.read_text()
    )
    # An explicit seed still wins.
    assert start(demo_client, opponents=(("dellacherie", "Easy"),), seed=5)["seed"] == 5
