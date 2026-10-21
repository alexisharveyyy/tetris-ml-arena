"""WebSocket protocol: connect, play, receive snapshots and events, reconnect."""

from __future__ import annotations

import time

import pytest
from starlette.websockets import WebSocketDisconnect

from tetris.web.protocol import server_message
from tetris.web.session import RESUME_COUNTDOWN
from tetris.web.ws import CLOSE_NO_MATCH

from .conftest import start


def receive_snap(ws, until=lambda s: True, limit: int = 600) -> dict:
    for _ in range(limit):
        msg = ws.receive_json()
        server_message.validate_python(msg)
        if msg["t"] == "snap" and until(msg):
            return msg
    raise AssertionError("expected snapshot never arrived")


def human(snap: dict) -> dict:
    return next(p for p in snap["players"] if p["kind"] == "human")


def test_play_over_websocket(client) -> None:
    info = start(client, opponents=(("dellacherie", "Unlimited"),))
    with client.websocket_connect(info["ws"]) as ws:
        welcome = ws.receive_json()
        server_message.validate_python(welcome)
        assert welcome["t"] == "welcome" and welcome["protocol"] == 1
        assert [p["id"] for p in welcome["players"]] == ["human", "ai1"]
        ws.send_json({"t": "hello", "last_frame": None})
        first = receive_snap(ws)
        assert len(human(first)["board"]) == 200 and len(human(first)["next"]) == 5

        ws.send_json({"t": "input", "seq": 1, "action": "hold", "down": True, "ts": 10.0})
        ws.send_json({"t": "input", "seq": 2, "action": "hold", "down": False, "ts": 11.0})
        events = []
        snap = receive_snap(
            ws, lambda s: events.extend(s["ev"]) or (s["ack"] or {}).get("seq") == 2
        )
        assert snap["ack"] == {"seq": 2, "ts": 11.0}
        snap = receive_snap(ws, lambda s: events.extend(s["ev"]) or human(s)["hold"] is not None)
        assert any(e["type"] == "hold" and e["player"] == "human" for e in events)

        before = human(snap)["stats"]["pieces"]
        for i in range(3):
            ws.send_json({"t": "input", "seq": 10 + 2 * i, "action": "hard", "down": True, "ts": 0})
            ws.send_json(
                {"t": "input", "seq": 11 + 2 * i, "action": "hard", "down": False, "ts": 0}
            )
        snap = receive_snap(ws, lambda s: human(s)["stats"]["pieces"] >= before + 3)

        agent = receive_snap(ws, lambda s: s["players"][1]["stats"]["pieces"] > 3)["players"][1]
        assert agent["kind"] == "agent" and agent["status"]["state"] in ("playing", "thinking")

        ws.send_json({"t": "ping", "ts": 5.0})
        for _ in range(120):
            msg = ws.receive_json()
            if msg["t"] == "pong":
                assert msg["ts"] == 5.0
                break
        else:
            raise AssertionError("no pong")

        ws.send_json({"t": "pause", "paused": True})
        paused = receive_snap(ws, lambda s: s["paused"])
        t = paused["time"]
        assert receive_snap(ws, lambda s: True)["time"] == t
        ws.send_json({"t": "pause", "paused": False})
        receive_snap(ws, lambda s: not s["paused"] and s["time"] > t)


def test_das_runs_on_the_server(client) -> None:
    info = start(client, mode="solo", opponents=())
    with client.websocket_connect(info["ws"]) as ws:
        ws.receive_json()
        start_col = min(
            c for _, c in human(receive_snap(ws, lambda s: human(s)["active"]))["active"]["c"]
        )
        ws.send_json({"t": "input", "seq": 1, "action": "left", "down": True, "ts": 0})
        # One press and no browser auto-repeat: holding still slides to the wall (DAS + ARR).
        snap = receive_snap(ws, lambda s: min(c for _, c in human(s)["active"]["c"]) == 0)
        assert start_col > 0 and snap["players"][0]["active"]
        ws.send_json({"t": "input", "seq": 2, "action": "left", "down": False, "ts": 0})


def test_disconnect_pauses_and_reconnect_resumes(client) -> None:
    info = start(client, opponents=(("dellacherie", "Hard"),))
    session = client.app.state.arena.sessions[info["match"]]
    with client.websocket_connect(info["ws"]) as ws:
        ws.receive_json()
        receive_snap(ws, lambda s: s["time"] > 0.2)
    deadline = time.time() + 5
    while not session.disconnected and time.time() < deadline:
        time.sleep(0.01)
    assert session.disconnected and session.match.paused
    frozen = session.match.time
    time.sleep(0.2)
    assert session.match.time == frozen

    with client.websocket_connect(info["ws"]) as ws:
        ws.receive_json()
        snap = receive_snap(ws)
        assert not snap["disconnected"] and not snap["paused"]
        assert 0 < snap["countdown"] <= RESUME_COUNTDOWN
        assert snap["time"] == pytest.approx(frozen, abs=1e-3)
        session.match.countdown = 0.0  # skip the wait
        receive_snap(ws, lambda s: s["time"] > frozen)


def test_user_pause_survives_reconnect(client) -> None:
    info = start(client, opponents=(("dellacherie", "Hard"),))
    with client.websocket_connect(info["ws"]) as ws:
        ws.receive_json()
        ws.send_json({"t": "pause", "paused": True})
        receive_snap(ws, lambda s: s["paused"])
    with client.websocket_connect(info["ws"]) as ws:
        ws.receive_json()
        snap = receive_snap(ws, lambda s: not s["disconnected"])
        assert snap["paused"]


def test_bad_message_and_unknown_match(client) -> None:
    info = start(client)
    with client.websocket_connect(info["ws"]) as ws:
        ws.receive_json()
        ws.send_json({"t": "input", "seq": 1, "action": "teleport", "down": True, "ts": 0})
        for _ in range(120):
            msg = ws.receive_json()
            if msg["t"] == "error":
                assert "bad message" in msg["message"]
                break
        else:
            raise AssertionError("no error reply")
    with client.websocket_connect("/ws?match=nope") as ws:
        assert ws.receive_json()["t"] == "error"
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_json()
        assert exc.value.code == CLOSE_NO_MATCH
