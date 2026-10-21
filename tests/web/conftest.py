from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tetris.agents.registry import CHECKPOINT_ENV
from tetris.match.agent_runtime import InlineRunner
from tetris.web.app import WebConfig, create_app


@pytest.fixture
def web_config(tmp_path, monkeypatch) -> WebConfig:
    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)
    return WebConfig(
        settings_path=tmp_path / "settings.yaml",
        history_path=tmp_path / "history.json",
        runner_factory=InlineRunner,
        start_countdown=0.0,
        inline_checks=True,
    )


@pytest.fixture
def client(web_config):
    with TestClient(create_app(web_config)) as c:
        yield c


def start(client, mode="marathon", opponents=(("dellacherie", "Easy"),), **extra) -> dict:
    body = {"mode": mode, "opponents": [{"agent": a, "preset": p} for a, p in opponents], **extra}
    res = client.post("/api/matches", json=body)
    assert res.status_code == 201, res.text
    return res.json()
