from __future__ import annotations

import json
import shutil

from tetris.agents import registry
from tetris.agents.registry import AGENTS, ROOT, AgentSpec, register_agent
from tetris.web.leaderboard import bootstrap_table, comparisons, significance_note, tie_groups

from .conftest import start

#: Promoted checkpoints committed to git (the rest are local training outputs).
COMMITTED_CHECKPOINTS = ("cem/best.json", "ga/best.json", "neat/best.json")


def use_checkpoint_dir(monkeypatch, path, committed=()) -> None:
    """Point the registry at ``path`` instead of the checkout's checkpoints, so a
    test sees the same agents in every checkout. ``committed`` copies those
    git-tracked files into it."""
    path.mkdir()
    for name in committed:
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "checkpoints" / name, path / name)
    monkeypatch.setattr(registry, "CHECKPOINT_DIR", path)


def agents_by_key(client) -> dict:
    return {a["key"]: a for a in client.get("/api/agents").json()["agents"]}


def test_agents_lineup_groups_and_availability(client) -> None:
    data = client.get("/api/agents").json()
    agents = {a["key"]: a for a in data["agents"]}
    main = [a["key"] for a in data["agents"] if a["group"] == "main"]
    assert main[:10] == [
        "dqn",
        "cem",
        "ga",
        "ppo",
        "bc",
        "bc_you",
        "neat",
        "dt",
        "dellacherie",
        "llm",
    ]
    assert agents["ppo_raw"]["group"] == "variant"
    assert agents["bc_pure"]["group"] == "leaderboard" and not agents["bc_pure"]["playable"]
    for key in ("dt", "llm"):
        # "planned" before Phase 6 registers them; "missing" or "disabled" after
        # (no checkpoint in this checkout, LLM off by default).
        assert agents[key]["status"]["state"] in ("planned", "missing", "disabled")
        assert not agents[key]["available"]
    for key in ("dellacherie", "ga", "cem"):
        assert agents[key]["available"], agents[key]["status"]
    assert agents["bc_you"]["looks"]["tetromino"]["fill"] == "#729410"
    assert agents["bc_you"]["looks"]["terminal"]["letter"] == "j"
    assert agents["dqn"]["record"]["Hard"]["text"] == "no games yet"
    assert data["human"]["looks"]["terminal"]["letter"] == "@"


def test_new_registry_agent_appears_automatically(client) -> None:
    register_agent(
        AgentSpec(
            "web_test_agent", "Web test agent", "t", "tetris.agents.linear:DellacherieAgent", None
        )
    )
    try:
        entry = agents_by_key(client)["web_test_agent"]
        assert entry["name"] == "Web test agent" and entry["available"] and entry["group"] == "main"
    finally:
        AGENTS.pop("web_test_agent")


def test_missing_checkpoint_shows_not_available(client, monkeypatch, tmp_path) -> None:
    from tetris.agents import registry

    monkeypatch.setattr(registry, "CHECKPOINT_DIR", tmp_path)
    agents = agents_by_key(client)
    assert agents["ga"]["status"]["state"] == "missing" and not agents["ga"]["available"]


def test_presets(client) -> None:
    presets = client.get("/api/presets").json()
    assert [p["name"] for p in presets] == ["Easy", "Medium", "Hard", "Unlimited"]
    assert presets[1]["description"] == "About one piece per second."


def test_settings_round_trip_and_validation(client, web_config) -> None:
    assert client.get("/api/settings").json()["look"] == "tetromino"
    res = client.put(
        "/api/settings", json={"look": "poster", "das_ms": 120, "show_thinking": False}
    )
    assert res.status_code == 200 and res.json()["look"] == "poster"
    again = client.get("/api/settings").json()
    assert again["das_ms"] == 120 and again["show_thinking"] is False
    assert web_config.settings_path.exists()
    assert client.put("/api/settings", json={"look": "neon"}).status_code == 422
    assert client.put("/api/settings", json={"bindings": {"jump": ["j"]}}).status_code == 422
    res = client.put("/api/settings", json={"bindings": {"hold": ["v"]}})
    assert res.json()["bindings"]["hold"] == ["v"]


def test_lobby_remembers_last_selection(client) -> None:
    default = client.get("/api/lobby/marathon").json()
    assert default["slots"][0] is not None
    start(client, opponents=(("ga", "Hard"), ("ga", "Easy")))
    lobby = client.get("/api/lobby/marathon").json()
    assert [s and (s["agent"], s["preset"]) for s in lobby["slots"]] == [
        ("ga", "Hard"),
        ("ga", "Easy"),
        None,
    ]
    race = client.get("/api/lobby/race").json()
    assert race["race"]["time_limit"] == 120.0


def test_match_validation(client, tmp_path, monkeypatch) -> None:
    use_checkpoint_dir(monkeypatch, tmp_path / "checkpoints")

    def post(**body):
        return client.post("/api/matches", json=body).status_code

    assert post(mode="marathon", opponents=[]) == 422
    assert post(mode="marathon", opponents=[{"agent": "ga"}] * 4) == 422
    assert post(mode="marathon", opponents=[{"agent": "nope"}]) == 422
    # 422 while dt is only planned; 409 once registered without a checkpoint.
    assert post(mode="marathon", opponents=[{"agent": "dt"}]) in (409, 422)
    assert post(mode="marathon", opponents=[{"agent": "bc_pure"}]) == 422  # leaderboard only
    assert post(mode="marathon", opponents=[{"agent": "bc_you"}]) == 409  # no checkpoint
    assert post(mode="solo", opponents=[{"agent": "ga"}]) == 422
    assert post(mode="watch", opponents=[{"agent": "ga"}, {"agent": "cem"}]) == 422


def test_match_lifecycle_and_history(client, web_config) -> None:
    info = start(client, opponents=(("dellacherie", "Easy"), ("dellacherie", "Hard")), seed=48213)
    assert info["seed"] == 48213
    names = [p["name"] for p in info["players"]]
    assert names == ["You", "Dellacherie 1", "Dellacherie 2"]
    mid = info["match"]
    assert client.post(f"/api/matches/{mid}/keep_watching").status_code == 409
    ended = client.post(f"/api/matches/{mid}/end").json()
    assert ended["finished"] and len(ended["standings"]) == 3 and ended["history_id"]
    history = json.loads(web_config.history_path.read_text())
    assert history["matches"][0]["seed"] == 48213
    assert history["matches"][0]["opponents"][1] == {"agent": "dellacherie", "preset": "Hard"}
    record = agents_by_key(client)["dellacherie"]["record"]
    assert sum(record["Easy"][k] for k in ("wins", "losses", "ties")) == 1
    assert client.get("/api/history").json()["matches"][0]["id"] == ended["history_id"]
    assert client.post(f"/api/matches/{mid}/keep_watching").status_code == 200
    again = client.post(f"/api/matches/{mid}/rematch", json={"new_seed": True}).json()
    assert not again["finished"]
    same = client.post(f"/api/matches/{mid}/rematch", json={"seed": 7}).json()
    assert same["seed"] == 7
    assert client.delete(f"/api/matches/{mid}").status_code == 204
    assert client.get(f"/api/matches/{mid}").status_code == 404


def test_new_match_replaces_the_old_one(client) -> None:
    first = start(client)["match"]
    second = start(client, mode="watch", opponents=(("ga", "Unlimited"),))["match"]
    assert client.get(f"/api/matches/{first}").status_code == 404
    assert client.get(f"/api/matches/{second}").json()["players"][0]["kind"] == "agent"


def test_leaderboard_from_reports(client, tmp_path, monkeypatch) -> None:
    """The repo's reports (Phase 5), read with only the committed checkpoints: GA,
    CEM, NEAT and the baseline verify; the other agents' checkpoints are absent,
    so their results show as pending."""
    use_checkpoint_dir(monkeypatch, tmp_path / "checkpoints", COMMITTED_CHECKPOINTS)
    data = client.get("/api/leaderboard").json()
    rows = {r["key"]: r for r in data["survival"]["rows"]}
    cem = rows["cem"]
    assert cem["status"] == "valid" and cem["source"] == "evaluated cem/best.json"
    assert cem["games"] == 500 and cem["top_out_rate"] == 0.15
    assert abs(cem["lo"] - 0.1214) < 1e-3 and abs(cem["hi"] - 0.1840) < 1e-3
    assert not cem["off_scale"]
    for key in ("ppo", "dqn", "bc_pure", "bc_dagger", "bc_you"):
        row = rows[key]
        assert row["status"] == "pending_rerun", key
        assert "not on this computer" in row["reason"] and "top_out_rate" not in row
    assert not rows["dt"]["evaluated"]
    assert "bc" not in rows  # alias of bc_pure or bc_dagger
    order = [r["key"] for r in data["survival"]["rows"] if r["evaluated"]]
    assert order == ["cem", "ga", "neat", "dellacherie"]
    assert "Wilson" in data["survival"]["method"]
    note = " ".join(f"{line['label']}: {line['text']}" for line in data["survival"]["significance"])
    assert "Against the baseline (Dellacherie): CEM" in note and "GA heuristic" in note
    assert "PPO" not in note  # pending results never appear in the note
    pairs = {(c["a"], c["b"]) for c in data["survival"]["comparisons"]}
    assert ("cem", "ga") in pairs and not any("ppo" in p for p in pairs)
    marathon = {r["key"]: r for r in data["marathon"]["rows"]}
    medium = marathon["cem"]["presets"]["Medium"]
    assert medium["score"]["3"] is not None and medium["time_is_lower_bound"]
    assert marathon["cem"]["presets"]["Unlimited"] is None
    assert marathon["ppo"]["status"] == "pending_rerun"
    assert all(v is None for v in marathon["ppo"]["presets"].values())
    assert rows["neat"]["status"] == "valid"
    assert marathon["neat"]["status"] == rows["neat"]["status"]


def test_significance_note_wording() -> None:
    pairs = {
        ("a", "dellacherie"): (-0.2, -0.25, -0.15),
        ("b", "dellacherie"): (-0.1, -0.15, -0.05),
        ("e", "dellacherie"): (-0.1, -0.15, -0.05),
        ("a", "b"): (-0.02, -0.06, 0.02),
        ("a", "e"): (-0.05, -0.09, -0.01),
        ("b", "e"): (-0.01, -0.04, 0.02),
        ("e", "c"): (-0.4, -0.5, -0.3),
        ("c", "dellacherie"): (0.5, 0.4, 0.6),
    }
    note = significance_note(["a", "b", "e", "dellacherie", "c"], pairs)
    assert note == [
        {"label": "Statistically tied", "text": "a and b. b is also tied with e."},
        {
            "label": "Significant gaps",
            "text": "e tops out less often than Dellacherie; Dellacherie tops out less "
            "often than c.",
        },
        {"label": "Against the baseline (Dellacherie)", "text": "a, b, and e better; c worse."},
    ]
    assert tie_groups(["a", "b", "e", "dellacherie", "c"], pairs) == [
        ["a", "b"],
        ["e"],
        ["dellacherie"],
        ["c"],
    ]


def test_comparison_table_covers_every_ranked_pair(tmp_path) -> None:
    (tmp_path / "leaderboard_bootstrap.csv").write_text(
        "a,b,metric,mean,lo,hi\nx,y,fitness,0.1,0.05,0.15\nx,y,top_out_rate,-0.02,-0.05,0.01\n"
    )
    table = bootstrap_table(tmp_path, {})
    rows = comparisons(["y", "x", "z"], table)
    assert rows == [
        {
            "a": "y",
            "b": "x",
            "fitness": {"mean": -0.1, "lo": -0.15, "hi": -0.05, "significant": True},
            "top_out_rate": {"mean": 0.02, "lo": -0.01, "hi": 0.05, "significant": False},
        }
    ]


def test_record_mode_saves_a_recording(client, web_config, tmp_path) -> None:
    web_config.recordings_dir = tmp_path / "recordings"
    assert client.get("/api/recordings").json()["totals"]["games"] == 0
    info = start(client, mode="record", opponents=(), record_setting="garbage")
    assert info["mode"] == "record" and info["recording"]["setting"] == "garbage"
    mid = info["match"]
    session = client.app.state.arena.sessions[mid]
    assert session.match.config.eval_garbage_every == 3
    assert client.post(f"/api/matches/{mid}/recording/save").status_code == 409  # still running
    game = session.match.players["human"].game

    async def ignore(text: str) -> None:
        return None

    session.attach(ignore)
    for _ in range(500):
        if session.match.finished:
            break
        session.handle_input("hard", True, 0, 0.0)
        session.handle_input("hard", False, 0, 0.0)
        session.step(0.0)
    saved = client.post(f"/api/matches/{mid}/recording/save").json()
    assert saved["recording"]["saved"].endswith(".jsonl.gz")
    assert saved["totals"]["games"] == 1
    assert saved["recording"]["on_space"] + saved["recording"]["off_space"] == game.pieces_placed
    client.post(f"/api/matches/{mid}/rematch", json={"new_seed": True})
    assert client.get(f"/api/matches/{mid}").json()["recording"]["saved"] is None


def test_watch_speed_change(client) -> None:
    info = start(client, mode="watch", opponents=(("ga", "Easy"),))
    mid = info["match"]
    res = client.post(f"/api/matches/{mid}/difficulty", json={"player": "ai1", "preset": "Hard"})
    assert res.status_code == 200 and res.json()["players"][0]["preset"] == "Hard"
    session = client.app.state.arena.sessions[mid]
    assert session.match.players["ai1"].spec.difficulty.pps == 2.0
    bad = client.post(f"/api/matches/{mid}/difficulty", json={"player": "human", "preset": "Hard"})
    assert bad.status_code == 422


def test_index_without_a_build(client) -> None:
    res = client.get("/")
    assert res.status_code == 200 and "/debug" in res.text


class _SlowVerifier:
    """Reports ``slow`` as checking for the first ``polls`` status calls."""

    def __init__(self, real, slow: str, polls: int) -> None:
        self.real, self.slow, self.polls = real, slow, polls

    def status(self, spec):
        from tetris.agents.checkpoint_check import CHECKING, CheckResult

        if spec.key == self.slow and self.polls > 0:
            self.polls -= 1
            return CheckResult(CHECKING, "loading")
        return self.real.status(spec)

    def close(self) -> None:
        self.real.close()


def test_agents_report_checking_then_available(client) -> None:
    state = client.app.state.arena
    state.verifier = _SlowVerifier(state.verifier, "dellacherie", polls=1)
    first = {a["key"]: a for a in client.get("/api/agents").json()["agents"]}
    assert first["dellacherie"]["status"]["state"] == "checking"
    assert first["dellacherie"]["available"] is False
    second = {a["key"]: a for a in client.get("/api/agents").json()["agents"]}
    assert second["dellacherie"]["available"] is True


def test_lobby_keeps_agents_still_checking(client) -> None:
    start(client, opponents=(("dellacherie", "Hard"),))
    state = client.app.state.arena
    state.verifier = _SlowVerifier(state.verifier, "dellacherie", polls=1_000)
    lobby = client.get("/api/lobby/marathon").json()
    slot = lobby["slots"][0]
    assert (slot["agent"], slot["preset"], slot["custom"]) == ("dellacherie", "Hard", None)


def test_start_waits_for_a_running_check(client) -> None:
    state = client.app.state.arena
    # Every agent_entries() call reads each agent once, so a few polls pass
    # before the check reports OK.
    state.verifier = _SlowVerifier(state.verifier, "dellacherie", polls=3)
    start(client, opponents=(("dellacherie", "Easy"),))


def test_ending_a_match_records_it_once(client, web_config) -> None:
    import json
    import time

    info = start(client, opponents=(("dellacherie", "Easy"),))
    for _ in range(5):
        client.post(f"/api/matches/{info['match']}/end")
    time.sleep(0.2)  # let the session loop run a few ticks after the end
    data = json.loads(web_config.history_path.read_text())
    assert len(data["matches"]) == 1


def test_registering_a_planned_agent_makes_it_available(client, monkeypatch) -> None:
    """Phase 6 registers dt and llm, which this UI lists as planned identities."""
    from dataclasses import replace

    from tetris.agents import registry

    before = {a["key"]: a for a in client.get("/api/agents").json()["agents"]}
    assert not before["dt"]["available"]
    spec = replace(registry.AGENTS["dellacherie"], key="dt", name="Decision Transformer")
    monkeypatch.setitem(registry.AGENTS, "dt", spec)
    after = {a["key"]: a for a in client.get("/api/agents").json()["agents"]}
    assert after["dt"]["available"] is True
    assert after["dt"]["name"] == "Decision Transformer"
