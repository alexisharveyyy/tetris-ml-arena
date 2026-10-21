from __future__ import annotations

import json

from tetris.match.history import MarathonHistory


def rows(human: int, agent: int) -> list[dict]:
    return [
        {
            "place": 1,
            "id": "human",
            "name": "You",
            "kind": "human",
            "agent_key": None,
            "preset": None,
            "score": human,
            "lines": 3,
            "level": 1,
            "pieces": 20,
            "tetrises": 0,
            "pps": 1.0,
            "topped_out": True,
            "finish_time": 20.0,
            "survival_time": 20.0,
        },
        {
            "place": 2,
            "id": "ai1",
            "name": "CEM",
            "kind": "agent",
            "agent_key": "cem",
            "preset": "Hard",
            "score": agent,
            "lines": 2,
            "level": 1,
            "pieces": 30,
            "tetrises": 0,
            "pps": 1.5,
            "topped_out": False,
            "finish_time": None,
            "survival_time": 20.0,
        },
    ]


def test_match_records_with_seed_opponents_and_stats(tmp_path) -> None:
    path = tmp_path / "h.json"
    h = MarathonHistory(path)
    match_id = h.record_match(rows(500, 300), seed=48213)
    data = json.loads(path.read_text())
    assert data["version"] == 2
    m = data["matches"][0]
    assert m["id"] == match_id and m["seed"] == 48213 and m["mode"] == "marathon"
    assert m["opponents"] == [{"agent": "cem", "preset": "Hard"}]
    assert [p["score"] for p in m["players"]] == [500, 300]
    assert data["marathon"]["cem|Hard"] == {"wins": 1, "losses": 0, "ties": 0}
    assert MarathonHistory(path).agent_records("cem") == {
        "Hard": {"wins": 1, "losses": 0, "ties": 0}
    }
    assert not path.with_name("h.json.tmp").exists()


def test_version_1_file_keeps_its_totals(tmp_path) -> None:
    path = tmp_path / "h.json"
    path.write_text(json.dumps({"marathon": {"ga|Easy": {"wins": 2, "losses": 1, "ties": 0}}}))
    h = MarathonHistory(path)
    assert h.describe("ga", "Easy") == "2W 1L" and h.matches == []
    h.record_match([{**rows(1, 2)[0]}, {**rows(1, 2)[1], "agent_key": "ga", "preset": "Easy"}])
    again = MarathonHistory(path)
    assert again.get("ga", "Easy") == {"wins": 2, "losses": 2, "ties": 0}
    assert len(again.matches) == 1


def test_concurrent_saves_do_not_collide(tmp_path) -> None:
    """Two writers (two servers sharing configs/) each use their own temp file."""
    import threading

    path = tmp_path / "history.json"
    errors: list[BaseException] = []

    def writer() -> None:
        h = MarathonHistory(path)
        try:
            for _ in range(50):
                h.save()
        except BaseException as exc:  # noqa: BLE001 - collected for the assert
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert list(tmp_path.glob("*.tmp")) == []
