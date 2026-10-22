"""Phase 6: Decision Transformer and the LLM agent (CPU only; no model or API calls)."""

from __future__ import annotations

import datetime as dt
import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tetris.agents.decision_transformer import collect as dt_collect
from tetris.agents.decision_transformer import train as dt_train
from tetris.agents.decision_transformer.dataset import encode_dt_files
from tetris.agents.decision_transformer.knob import target_stops
from tetris.agents.decision_transformer.model import DecisionTransformer
from tetris.agents.difficulty import DifficultyParams
from tetris.agents.evaluation import EvalSetting, GameStats, game_fitness, play_game
from tetris.agents.game_log import log_episode, read_episode
from tetris.agents.llm import backends as llm_backends
from tetris.agents.llm.agent import LLMAgent, load_llm_config
from tetris.agents.llm.backends import BackendError, BackendTimeout, OllamaBackend, Reply
from tetris.agents.llm.prompt import SYSTEM_PROMPT, options, parse_choice, user_prompt
from tetris.agents.llm.spending import BudgetExhausted, Price, SpendingLedger
from tetris.agents.registry import AGENTS, create_agent, leaderboard_agents
from tetris.engine import Game
from tetris.env import action_mask
from tetris.match.lobby import OpponentSlot
from tetris.rl.nets import MASKED_LOGIT, save_checkpoint

from .states import assert_legal

TINY = EvalSetting("garbage", max_pieces=30, garbage_every=3)
DATA = Path(__file__).with_name("fixtures")


@pytest.fixture(scope="module")
def dt_logs(tmp_path_factory) -> list[Path]:
    root = tmp_path_factory.mktemp("dt_logs")
    files = []
    for i, noise in enumerate((None, {"noise": 0.5}, None)):
        path = root / f"{i}.jsonl.gz"
        log_episode("dellacherie", None, 300 + i, TINY, path, noise)
        files.append(path)
    return files


@pytest.fixture(scope="module")
def dt_data(dt_logs):
    return encode_dt_files(dt_logs, TINY.max_pieces)


def _tiny_net(**kw) -> DecisionTransformer:
    torch.manual_seed(0)
    return DecisionTransformer(
        d_model=32, n_layers=2, n_heads=2, context=6, board_dim=16, extra_dim=8, **kw
    )


# ------------------------------------------------------- Decision Transformer


def test_dt_returns_add_up_to_the_game_fitness(dt_logs, dt_data) -> None:
    for ep, path in enumerate(dt_logs):
        result = next(r for r in read_episode(path) if r["type"] == "result")
        stats = GameStats(**{k: v for k, v in result.items() if k != "type"})
        states = np.flatnonzero(dt_data.episode == ep)
        assert len(states) == stats.pieces
        assert (dt_data.timestep[states] == np.arange(stats.pieces)).all()
        assert (dt_data.rewards[states] > 0).all()
        # Return-to-go falls by each step's reward.
        np.testing.assert_allclose(
            dt_data.rtg[states][:-1] - dt_data.rtg[states][1:],
            dt_data.rewards[states][:-1],
            rtol=1e-4,
        )
        if stats.pieces == stats.max_pieces:  # reached the cap: return == fitness
            assert dt_data.rtg[states[0]] == pytest.approx(game_fitness(stats), rel=1e-5)
    rows = dt_data.steps
    for s in range(len(dt_data)):  # the chosen row is the logged action
        assert rows.actions[dt_data.chosen[s]] == rows.teacher[s]
        assert rows.target[dt_data.chosen[s]]


def test_dt_cannot_see_the_future_or_the_padding() -> None:
    net = _tiny_net(dropout=0.0).eval()
    k, d = net.context, net.d_model
    g = torch.Generator().manual_seed(1)
    rtg = torch.rand(1, k, generator=g)
    obs_b = torch.randint(0, 256, (1, k, 30), dtype=torch.uint8, generator=g)
    obs_e = torch.rand(1, k, 62, generator=g)
    act = torch.randn(1, k, d, generator=g)
    ts = torch.arange(k).unsqueeze(0)
    valid = torch.ones(1, k, dtype=torch.bool)
    with torch.no_grad():
        base = net(rtg, obs_b, obs_e, act, ts, valid)
        for t in range(k):
            # Changing step t's action, and everything after it, leaves queries up to t unchanged.
            act2, rtg2, obs2 = act.clone(), rtg.clone(), obs_e.clone()
            act2[:, t:] = torch.randn(1, k - t, d, generator=g)
            rtg2[:, t + 1 :] += 1.0
            obs2[:, t + 1 :] += 1.0
            out = net(rtg2, obs_b, obs2, act2, ts, valid)
            torch.testing.assert_close(out[:, : t + 1], base[:, : t + 1])
        # Left padding: whatever the padded slots hold, valid outputs are the same.
        valid = torch.tensor([[False, False] + [True] * (k - 2)])
        a = net(rtg, obs_b, obs_e, act, ts, valid)
        junk = (torch.rand(1, k), torch.rand(1, k, 62), torch.randn(1, k, d))
        rtg3, obs3, act3 = rtg.clone(), obs_e.clone(), act.clone()
        rtg3[:, :2], obs3[:, :2], act3[:, :2] = junk[0][:, :2], junk[1][:, :2], junk[2][:, :2]
        b = net(rtg3, obs_b, obs3, act3, ts, valid)
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a[:, 2:], b[:, 2:])


def test_dt_loss_masks_illegal_actions_and_learns(dt_data) -> None:
    net = _tiny_net()
    ends = np.array([0, 3, 17, 29, 40, 75])
    batch = dt_train.dt_batch(dt_data, ends, net.context, torch.device("cpu"))
    logits = dt_train.dt_logits(net, batch)
    owner, actions = batch[8], batch[9]
    legal = torch.zeros_like(logits, dtype=torch.bool)
    legal[owner, actions] = True
    assert (logits[~legal] == MASKED_LOGIT).all() and (logits[legal] > MASKED_LOGIT).all()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    first = float(dt_train.dt_loss(net, batch))
    for _ in range(40):
        loss = dt_train.dt_loss(net, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < 0.5 * first


def test_dt_windows_stay_inside_their_episode(dt_data) -> None:
    starts = dt_data.episode_starts()
    for end in (0, 5, 29, 30, 31, 60):
        w = dt_train.window_states(dt_data, np.array([end]), 6)[0]
        assert w[-1] == end and len(w) <= 6
        assert (dt_data.episode[w] == dt_data.episode[end]).all()
        assert w[0] == max(starts[end], end - 5)


def test_dt_collection_is_deterministic(tmp_path) -> None:
    cfg = {
        "collect": {
            "seed_base": 700000,
            "out_dir": "unused",
            "games_per_source": 2,
            "noise_levels": [0.0, 0.3],
            "agents": ["dellacherie"],
            "extra_sources": [{"agent": "ga", "noise": 0.0, "games": 1}],
            "max_pieces": 15,
        }
    }
    names = [s.name for s in dt_collect.sources(cfg)]
    assert names == ["dellacherie_noise0", "dellacherie_noise0.3", "ga_noise0"]
    seeds = [t[2] for t in dt_collect.collection_tasks(cfg, tmp_path)]
    assert seeds == [700000, 700001, 701000, 701001, 702000]
    contents = []
    for run in ("a", "b"):
        manifest = dt_collect.collect(cfg, workers=0, root=tmp_path / run)
        assert manifest["total_games"] == 5
        files = sorted((tmp_path / run).glob("**/*.jsonl.gz"))
        contents.append([gzip.open(f, "rt").read() for f in files])
    assert contents[0] == contents[1]
    header = json.loads(contents[0][2].splitlines()[0])  # dellacherie at noise 0.3
    assert header["difficulty"] == {"noise": 0.3}


def test_dt_sources_can_name_a_training_stage(tmp_path) -> None:
    cfg = {
        "collect": {
            "seed_base": 720000,
            "out_dir": "unused",
            "max_pieces": 10,
            "extra_sources": [
                {
                    "label": "ga_best",
                    "agent": "ga",
                    "checkpoint": "checkpoints/ga/best.json",
                    "games": 1,
                },
                {"label": "ga_promoted", "agent": "ga", "games": 1},
            ],
        }
    }
    srcs = dt_collect.sources(cfg)
    assert [s.name for s in srcs] == ["ga_best", "ga_promoted"] and srcs[0].noise == 0.0
    tasks = dt_collect.collection_tasks(cfg, tmp_path)
    assert tasks[0][1] == str(dt_collect.ROOT / "checkpoints" / "ga" / "best.json")
    assert tasks[1][1] is None and [t[2] for t in tasks] == [720000, 721000]
    dt_collect.collect(cfg, workers=0, root=tmp_path)
    log = next((tmp_path / "ga_best").glob("**/*.jsonl.gz"))
    header = json.loads(gzip.open(log, "rt").readline())
    assert header["checkpoint"] == tasks[0][1] and header["difficulty"] is None
    cfg["collect"]["extra_sources"][1]["label"] = "ga_best"
    with pytest.raises(ValueError, match="duplicate"):
        dt_collect.sources(cfg)


def test_dt_v2_config_changes_only_the_data() -> None:
    v1 = dt_train.load_config(dt_train.DEFAULT_CONFIG)
    v2 = dt_train.load_config(str(Path(dt_train.DEFAULT_CONFIG).with_name("config_v2.yaml")))
    assert {k: v for k, v in v1.items() if k != "collect"} == {
        k: v for k, v in v2.items() if k != "collect"
    }
    srcs = dt_collect.sources(v2)
    assert all(s.noise == 0.0 for s in srcs) and len(srcs) == 14
    staged = [s for s in srcs if s.checkpoint]
    assert len(staged) == 12  # trained checkpoints are local, so only their paths are checked
    assert all(s.checkpoint.startswith(f"checkpoints/{s.agent}/seed0/") for s in staged)
    seeds = {t[2] for t in dt_collect.collection_tasks(v2, Path("unused"))}
    assert min(seeds) >= 720000 and max(seeds) < 800000  # below the selection seeds


@pytest.fixture
def dt_checkpoint(tmp_path) -> Path:
    path = tmp_path / "dt.pt"
    save_checkpoint(path, "dt", _tiny_net(), default_target=1.1, return_cap=500)
    return path


def test_dt_agent_plays_legally_and_tracks_its_return(dt_checkpoint) -> None:
    agent = create_agent("dt", dt_checkpoint, DifficultyParams(target_return=0.9), seed=0)
    rtgs, earned = [], []

    def policy(game: Game):
        if not action_mask(game).any():
            return None
        p = agent.select_placement(game)
        assert_legal(game, p)
        rtgs.append(agent.current.rtg)
        earned.append(agent.earned)
        return p.index

    stats = play_game(policy, 4, TINY)
    assert rtgs[0] == pytest.approx(0.9)
    # Each piece lowers the return-to-go by its reward: survival 1/500 plus the score bonus.
    assert np.all(np.diff(rtgs) <= -1 / 500 + 1e-9)
    assert rtgs[-1] == pytest.approx(0.9 - earned[-1])
    assert len(agent.history) == min(stats.pieces - 1, agent.model.context - 1)
    # A new game resets the history and the return.
    agent.select_placement(Game(9))
    assert agent.earned == 0.0 and len(agent.history) == 0
    # Without a target the checkpoint's default is used; changing it applies next piece.
    agent.set_difficulty(DifficultyParams())
    g = Game(11)
    agent.select_placement(g)
    assert agent.current.rtg == pytest.approx(1.1)


def test_dt_agent_starts_a_new_segment_every_cap_pieces(dt_checkpoint) -> None:
    agent = create_agent("dt", dt_checkpoint, seed=0)
    agent.cap = 8
    g = Game(2)
    timesteps = []
    for _ in range(18):
        p = agent.select_placement(g)
        timesteps.append(agent.current.timestep)
        g.place(p)
    assert timesteps == [*range(8), *range(8), *range(2)]


def test_dt_training_and_calibration_smoke(dt_logs, tmp_path, monkeypatch) -> None:
    import tetris.rl.training as rl_training

    for mod in (dt_train, rl_training):
        monkeypatch.setattr(mod, "CHECKPOINT_DIR", tmp_path / "ckpt")
        monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(dt_train, "linear_references", lambda: {})
    (tmp_path / "reports").mkdir()
    cfg = {
        **dt_train.load_config(dt_train.DEFAULT_CONFIG),
        "device": "cpu",
        "cache_dir": "cache",
        "return_cap": TINY.max_pieces,
        "budget_minutes": 0.05,
        "batch_windows": 4,
        "eval_workers": 1,
        "max_pieces": TINY.max_pieces,
        "val_fraction": 0.34,
        "calibration_targets": [0.5, 1.0],
        "calibration_games": 2,
        "model": {
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 2,
            "context": 4,
            "board_dim": 16,
            "extra_dim": 8,
        },
    }
    monkeypatch.setattr(
        dt_train, "cached_dt_dataset", lambda c: encode_dt_files(dt_logs, TINY.max_pieces)
    )
    summary = dt_train.train(cfg, "smoke")
    best = tmp_path / "ckpt" / "dt" / "best.pt"
    assert best.exists() and summary["selection_target"] > 0
    assert (tmp_path / "ckpt" / "dt" / "smoke" / "preflight.json").exists()
    state_path = tmp_path / "ckpt" / "dt" / "smoke" / "state.pt"
    first = torch.load(state_path, weights_only=False)
    resumed = dt_train.train({**cfg, "budget_minutes": 0.1}, "smoke", resume=True)
    second = torch.load(state_path, weights_only=False)
    assert second["steps"] > first["steps"] and second["elapsed"] > first["elapsed"]
    assert len(resumed["evaluations"]) == len(summary["evaluations"]) + 1
    result = dt_train.calibrate(cfg, best, workers=0)
    assert [r["target"] for r in result["rows"]] == [0.5, 1.0]
    assert result["leaderboard_target"] in (0.5, 1.0)
    assert (tmp_path / "reports" / "dt_calibration.md").exists()
    stops = target_stops(tmp_path / "ckpt" / "dt" / "calibration.json")
    assert [s.target for s in stops] == [0.5, 1.0] and stops[0].achieved is not None
    assert "plays at about" in stops[0].label
    # A second run, not promoted, calibrated with its own report, then the v1/v2 choice.
    v1_bytes = best.read_bytes()
    dt_train.train(cfg, "smoke_v2", promote=False)
    assert best.read_bytes() == v1_bytes
    v2 = dt_train.calibrate(cfg, workers=0, run_name="smoke_v2", report="dt_calibration_v2")
    selected = json.loads((tmp_path / "ckpt" / "dt" / "smoke_v2" / "selection.json").read_text())
    assert v2["checkpoint"] == selected["selected_checkpoint"]
    assert (tmp_path / "ckpt" / "dt" / "smoke_v2" / "calibration.json").exists()
    assert (tmp_path / "reports" / "dt_calibration_v2.md").exists()
    assert best.read_bytes() == v1_bytes  # calibrating v2 leaves the promoted model alone
    choice = dt_train.choose_version("smoke_v2")
    fits = choice["selection_fitness"]
    assert choice["chosen"] == ("v2" if fits["v2"] > fits["v1"] else "v1")
    assert dt_train.version_dir("v1").parent.parent == tmp_path / "ckpt" / "dt" / "versions"
    assert (dt_train.version_dir("v1") / "best.pt").read_bytes() == v1_bytes
    chosen = dt_train.version_dir(choice["chosen"])
    assert best.read_bytes() == (chosen / "best.pt").read_bytes()
    calibration = json.loads((tmp_path / "ckpt" / "dt" / "calibration.json").read_text())
    assert calibration == json.loads((chosen / "calibration.json").read_text())
    dt_train.choose_version("smoke_v2")  # rerun: v1 is kept, never replaced by v2
    assert (dt_train.version_dir("v1") / "best.pt").read_bytes() == v1_bytes


def test_dt_knob_falls_back_and_lobby_applies_target(tmp_path) -> None:
    stops = target_stops(tmp_path / "missing.json")
    assert stops and all(s.achieved is None for s in stops)
    slot = OpponentSlot("dt", "Hard", target_return=0.6)
    assert slot.difficulty.target_return == 0.6
    assert slot.difficulty.pps == DifficultyParams(pps=2.0).pps
    again = OpponentSlot.from_dict(slot.to_dict())
    assert again.target_return == 0.6
    assert OpponentSlot("ga", "Hard", target_return=0.6).difficulty.target_return is None
    assert AGENTS["dt"].supports_target_return and AGENTS["dt"].leaderboard


def test_dt_spearman() -> None:
    assert dt_train.spearman([1, 2, 3], [0.1, 0.5, 0.9]) == pytest.approx(1.0)
    assert dt_train.spearman([1, 2, 3], [0.9, 0.5, 0.1]) == pytest.approx(-1.0)


# ------------------------------------------------------------------- LLM agent


def _game() -> Game:
    g = Game(3)
    ga = create_agent("ga")
    for _ in range(12):
        g.place(ga.select_placement(g))
    return g


def test_llm_prompt_matches_the_saved_copy() -> None:
    g = _game()
    opts = options(g)
    for variant in ("features", "bare"):
        text = SYSTEM_PROMPT + "\n\n" + user_prompt(g, opts, variant) + "\n"
        assert text == (DATA / f"llm_prompt_{variant}.txt").read_text(), variant
    # Every distinct legal placement is listed once.
    boards = {tuple(a.board.flatten()) for a in __import__("tetris").env.enumerate_afterstates(g)}
    assert len(opts) == len(boards)
    assert "holes" not in user_prompt(g, opts, "bare")


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ('{"choice": 3}', 3),
        ('Sure! {"choice": "2"}', 2),
        ('{"choice": 0}', None),
        ('{"choice": 99}', None),
        ("place it left", None),
        ('{"pick": 1}', None),
        ('{"choice": true}', None),
    ],
)
def test_llm_parse_choice(text, ok) -> None:
    from tetris.agents.llm.prompt import InvalidAnswer

    if ok is None:
        with pytest.raises(InvalidAnswer):
            parse_choice(text, 10)
    else:
        assert parse_choice(text, 10) == ok


class FakeBackend:
    name = "fake"
    model = "fake"

    def __init__(self, reply=None, exc=None) -> None:
        self.reply, self.exc, self.calls = reply, exc, 0

    def complete(self, system, user, timeout):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return Reply(self.reply, 100, 5, 0.01, 0.001)


@pytest.mark.parametrize(
    ("backend", "outcome"),
    [
        (FakeBackend(exc=BackendTimeout("slow")), "timeout"),
        (FakeBackend(reply="no idea"), "invalid"),
        (FakeBackend(reply='{"choice": 999}'), "invalid"),
        (FakeBackend(exc=BackendError("HTTP 500")), "error"),
    ],
)
def test_llm_failures_fall_back_to_the_heuristic(backend, outcome) -> None:
    g = _game()
    agent = LLMAgent(seed=0)
    agent.configure(backend=backend)
    assert agent.select_placement(g) == create_agent("ga").select_placement(g)
    assert agent.stats.outcomes[outcome] == 1 and agent.stats.fallbacks == 1
    assert agent.stats.to_dict()["fallback_rate"] == 1.0


def test_llm_valid_answer_is_played() -> None:
    g = _game()
    opts = options(g)
    target = opts[-1]  # a placement the heuristic does not rank first
    agent = LLMAgent(seed=0)
    agent.configure(backend=FakeBackend(reply=json.dumps({"choice": target.number})))
    ranked = agent.rank_placements(g)
    assert ranked[0].placement == target.placement
    assert len({r.placement for r in ranked}) == len(ranked)
    assert agent.stats.outcomes["llm"] == 1 and agent.stats.input_tokens == 100
    assert agent.stats.cost_usd == pytest.approx(0.001)


def test_spending_cap_refuses_calls_before_sending(tmp_path, monkeypatch) -> None:
    from tetris.agents.llm.spending import estimate_tokens

    cap = 0.01
    ledger = SpendingLedger(tmp_path / "spend.json", cap_usd=cap)
    monkeypatch.setenv("FAKE_KEY", "x")
    sent = []
    system, user = "s" * 100, "u" * 2000
    price = Price(input=1.0, output=2.0)

    class Paid(llm_backends._PaidBackend):
        def _request(self, system, user, timeout):
            sent.append(1)
            return '{"choice": 1}', 700, 5  # a realistic count, well under the bound

    backend = Paid("m", "FAKE_KEY", price, ledger, 24)
    reserve = price.cost(estimate_tokens(system + user), 24)
    actual = price.cost(700, 5)
    with pytest.raises(BudgetExhausted):
        for _ in range(100):
            backend.complete(system, user, 1.0)
    # Calls run while the worst case fits; the one that could cross the cap is not sent.
    expected = int(1 + (cap - reserve) // actual)
    assert len(sent) == expected
    state = ledger.snapshot()
    assert state["spent_usd"] == pytest.approx(expected * actual)
    assert state["spent_usd"] <= cap < state["spent_usd"] + reserve
    assert state["reserved_usd"] == pytest.approx(0.0)
    assert state["by_model"]["m"]["calls"] == expected
    # A fresh ledger object sees the same total (it is on disk, shared by processes).
    assert SpendingLedger(tmp_path / "spend.json", cap).spent == pytest.approx(state["spent_usd"])
    # In an evaluation the agent stops instead of falling back.
    agent = LLMAgent(seed=0)
    agent.configure(backend=backend, stop_on_budget=True)
    with pytest.raises(BudgetExhausted):
        agent.select_placement(_game())
    agent.configure(stop_on_budget=False)
    agent.select_placement(_game())
    assert agent.stats.outcomes["budget"] == 2
    assert len(sent) == expected


def test_token_bound_covers_the_real_prompts() -> None:
    from tetris.agents.llm.spending import estimate_tokens

    g = _game()
    text = SYSTEM_PROMPT + user_prompt(g, options(g), "features")
    assert estimate_tokens(text) >= len(text)


def test_api_backend_needs_its_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MISSING_KEY", raising=False)
    ledger = SpendingLedger(tmp_path / "spend.json", 5.0)
    backend = llm_backends.OpenAIBackend("gpt-4.1-mini", "MISSING_KEY", Price(0.4, 1.6), ledger)
    with pytest.raises(BackendError, match="MISSING_KEY"):
        backend.complete("s", "u", 1.0)
    assert not (tmp_path / "spend.json").exists() or ledger.snapshot()["calls"] == 0


def test_ollama_retries_without_the_thinking_flag(monkeypatch) -> None:
    payloads = []

    def fake_post(url, payload, headers, timeout):
        payloads.append(dict(payload))
        if "think" in payload:
            raise BackendError('HTTP 400: {"error":"llama3.1:8b does not support thinking"}')
        return {"message": {"content": '{"choice": 1}'}, "prompt_eval_count": 7, "eval_count": 3}

    monkeypatch.setattr(llm_backends, "_post", fake_post)
    backend = OllamaBackend("llama3.1:8b")
    reply = backend.complete("s", "u", 5.0)
    assert reply.text == '{"choice": 1}' and (reply.input_tokens, reply.output_tokens) == (7, 3)
    backend.complete("s", "u", 5.0)
    assert ["think" in p for p in payloads] == [True, False, False]


def test_llm_is_off_by_default_and_never_on_the_leaderboard(monkeypatch) -> None:
    from tetris.agents import registry

    assert load_llm_config()["enabled"] is False
    assert not AGENTS["llm"].available
    monkeypatch.setattr(
        registry.AgentSpec, "agent_config", property(lambda self: {"enabled": True})
    )
    assert AGENTS["llm"].available
    assert "llm" not in [a.key for a in leaderboard_agents()]


def test_llm_eval_requires_recent_prices() -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "eval_llm.py"
    spec = importlib.util.spec_from_file_location("eval_llm", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = load_llm_config()
    mod.check_prices(cfg, today=dt.date(2026, 10, 25))
    with pytest.raises(SystemExit):
        mod.check_prices(cfg, today=dt.date(2026, 11, 30))
    assert set(mod.RUNS) == {
        "qwen3-features",
        "qwen3-bare",
        "llama31-features",
        "api-features",
        "api-bare",
    }
    run_cfg, label = mod.run_config(cfg, "llama31-features")
    assert run_cfg["ollama"]["model"] == "llama3.1:8b" and run_cfg["prompt"] == "features"
    assert cfg["ollama"]["model"] == "qwen3:8b"  # the default agent model is unchanged


def _eval_llm_module():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "scripts" / "eval_llm.py"
    spec = importlib.util.spec_from_file_location("eval_llm_report", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_llm_report_uses_survival_metrics_with_ga_reference(tmp_path, monkeypatch) -> None:
    mod = _eval_llm_module()
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    (tmp_path / "reports").mkdir()
    setting = EvalSetting("garbage", max_pieces=100, garbage_every=3)
    seeds = [900000, 900001, 900002]
    ga = [GameStats(s, 100, 40, 9000, 3, False, 100) for s in seeds]

    def game(seed, pieces, lines, topped):
        outcomes = {"llm": pieces - 2, "timeout": 1, "invalid": 1, "error": 0, "budget": 0}
        return {
            **GameStats(seed, pieces, lines, 100 * lines, 0, topped, 100).to_dict(),
            "timeout_seconds": 15.0,
            "seconds": 2.0 * pieces,
            "outcomes": outcomes,
            "llm_seconds": 1.9 * pieces,
            "input_tokens": 850 * pieces,
            "output_tokens": 6 * pieces,
            "cost_usd": 0.0,
        }

    runs = [
        {
            "name": "qwen3-features",
            "label": "Ollama qwen3:8b, features",
            "games": [game(s, 60 + 10 * i, 20, True) for i, s in enumerate(seeds)],
            "stopped": None,
        },
        {
            "name": "llama31-features",
            "label": "Ollama llama3.1:8b, features",
            "games": [game(s, 40, 10, True) for s in seeds],
            "stopped": None,
        },
    ]
    summaries = [mod.summarize_run(r, {s.seed: s for s in ga}) for r in runs]
    empty = {"name": "api-features", "label": "API, features", "games": [], "stopped": None}
    summaries.append(mod.summarize_run(empty, {}, "no valid API key"))
    q = summaries[0]
    assert q["metrics"]["pieces"]["mean"] == pytest.approx(70.0)
    assert q["metrics"]["top_out"]["mean"] == 1.0
    assert q["minus_ga"]["pieces"][0] == pytest.approx(-30.0)
    assert q["minus_ga"]["top_out"][0] == pytest.approx(1.0)
    assert q["fallback_rate"] == pytest.approx(6 / 210)
    comparison = mod.compare(runs[0], runs[1])
    assert comparison["difference"]["pieces"][0] == pytest.approx(30.0)
    cfg = load_llm_config()
    mod.write_report(summaries, [comparison], ga, setting, {}, mod.api_note(cfg))
    text = (tmp_path / "reports" / "llm_agent.md").read_text()
    assert "Pieces survived" in text and "Lines per piece" in text and "Top-out rate" in text
    assert "GA heuristic (reference)" in text and "reached the cap in 3 of 3 games" in text
    assert "gpt-4.1-mini, a small API model, not a frontier model" in text
    assert "3 games per run gives wide intervals" in text
    assert "Fitness [" not in text
    header = next(line for line in text.splitlines() if line.startswith("| Run "))
    sep = text.splitlines()[text.splitlines().index(header) + 1]
    assert header.count("|") == sep.count("|")
    pending = next(line for line in text.splitlines() if line.startswith("| API, features"))
    assert "Pending (no valid API key)" in pending and pending.count("|") == sep.count("|")
    assert "| 15 s |" in text
    data = json.loads((tmp_path / "reports" / "llm_agent.json").read_text())
    assert data["main_metrics"] == ["pieces", "top_out", "lines_per_piece"]
    run_cfg, label = mod.run_config(cfg, "api-features")
    assert label == "gpt-4.1-mini (small API model, not a frontier model), features"


def _paid(ledger, monkeypatch, request):
    monkeypatch.setenv("FAKE_KEY", "x")

    class Paid(llm_backends._PaidBackend):
        def _request(self, system, user, timeout):
            return request()

    return Paid("m", "FAKE_KEY", Price(input=1.0, output=2.0), ledger, 24)


def test_rejected_api_requests_cost_nothing(tmp_path, monkeypatch) -> None:
    from tetris.agents.llm.spending import estimate_tokens

    ledger = SpendingLedger(tmp_path / "spend.json", cap_usd=5.0)
    reserve = Price(1.0, 2.0).cost(estimate_tokens("s" + "u"), 24)

    def rejected():
        raise BackendError("HTTP 401: bad key", status=401)

    def no_reply():
        raise BackendTimeout("slow")

    def dropped():
        raise BackendError("connection failed: reset")

    with pytest.raises(BackendError):
        _paid(ledger, monkeypatch, rejected).complete("s", "u", 1.0)
    assert ledger.spent == 0.0 and ledger.snapshot()["reserved_usd"] == 0.0
    # No reply: the worst case is kept, since the call may have been billed.
    for request in (no_reply, dropped):
        with pytest.raises(BackendError):
            _paid(ledger, monkeypatch, request).complete("s", "u", 1.0)
    assert ledger.spent == pytest.approx(2 * reserve)


def test_openai_http_error_is_rejected_at_no_cost(tmp_path, monkeypatch) -> None:
    import io
    import urllib.error

    def urlopen(req, timeout):
        body = io.BytesIO(b'{"error": {"message": "Incorrect API key provided"}}')
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, body)

    monkeypatch.setattr(llm_backends.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("FAKE_KEY", "x")
    ledger = SpendingLedger(tmp_path / "spend.json", cap_usd=5.0)
    backend = llm_backends.OpenAIBackend("gpt-4.1-mini", "FAKE_KEY", Price(0.4, 1.6), ledger)
    with pytest.raises(BackendError) as info:
        backend.complete("s", "u", 1.0)
    assert info.value.status == 401 and "Incorrect API key" in str(info.value)
    assert (
        ledger.snapshot()["spent_usd"] == 0.0
        and ledger.snapshot()["by_model"]["gpt-4.1-mini"]["calls"] == 1
    )


def test_error_streak_resets_on_any_other_outcome() -> None:
    agent = LLMAgent(seed=0)
    agent.configure(backend=FakeBackend(exc=BackendError("down")))
    for _ in range(3):
        agent.select_placement(_game())
    assert agent.stats.error_streak == 3
    agent.configure(backend=FakeBackend(exc=BackendTimeout("slow")))
    agent.select_placement(_game())
    assert agent.stats.error_streak == 0


def _eval_setup(mod, tmp_path, monkeypatch, backend):
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path)
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "status.json")
    monkeypatch.setattr(mod, "make_backend", lambda cfg: backend)
    return load_llm_config(), EvalSetting("garbage", max_pieces=30, garbage_every=3)


def test_llm_eval_stops_after_five_errors_in_a_row(tmp_path, monkeypatch) -> None:
    mod = _eval_llm_module()
    backend = FakeBackend(exc=BackendError("connection failed: refused"))
    cfg, setting = _eval_setup(mod, tmp_path, monkeypatch, backend)
    run = mod.evaluate_run(cfg, "qwen3-features", [900000, 900001], setting, 15.0)
    assert "5 backend errors in a row" in run["stopped"] and "refused" in run["stopped"]
    assert run["games"] == [] and backend.calls == 5  # the game in progress is discarded
    assert "5 backend errors" in mod.load_status()["qwen3-features"]


def test_llm_eval_records_its_timeout(tmp_path, monkeypatch) -> None:
    mod = _eval_llm_module()
    backend = FakeBackend(reply='{"choice": 1}')
    cfg, setting = _eval_setup(mod, tmp_path, monkeypatch, backend)
    assert cfg["timeout_seconds"] == 3.0 and cfg["eval_timeout_seconds"] == 15.0
    run = mod.evaluate_run(cfg, "qwen3-features", [900000], setting, 15.0)
    assert run["stopped"] is None and run["games"][0]["timeout_seconds"] == 15.0
    assert (tmp_path / "qwen3-features_cap30_t15.jsonl").exists()
    assert mod.stored_run(cfg, "qwen3-features", 30, 3.0)["games"] == run["games"]
    assert LLMAgent(seed=0).timeout == 3.0  # live play keeps the short timeout


@pytest.mark.parametrize(
    ("exc", "reason"),
    [
        (BackendError("HTTP 401: Incorrect API key provided", status=401), "no valid API key"),
        (BackendError("OPENAI_API_KEY is not set"), "no valid API key"),
        (BackendError("HTTP 500: busy", status=500), "API test call failed: HTTP 500: busy"),
    ],
)
def test_api_run_is_skipped_when_its_test_call_fails(tmp_path, monkeypatch, exc, reason) -> None:
    mod = _eval_llm_module()
    backend = FakeBackend(exc=exc)
    backend.name = "openai"
    cfg, setting = _eval_setup(mod, tmp_path, monkeypatch, backend)
    run = mod.evaluate_run(cfg, "api-features", [900000, 900001], setting, 15.0)
    assert backend.calls == 1 and run["games"] == [] and run["stopped"] == reason
    assert mod.load_status() == {"api-features": reason}
    summary = mod.summarize_run(run, {}, mod.load_status().get("api-features"))
    assert summary["pending"] == reason


def test_api_run_plays_after_a_good_test_call(tmp_path, monkeypatch) -> None:
    mod = _eval_llm_module()
    backend = FakeBackend(reply='{"choice": 1}')
    backend.name = "openai"
    cfg, setting = _eval_setup(mod, tmp_path, monkeypatch, backend)
    mod.set_status("api-features", "no valid API key")
    run = mod.evaluate_run(cfg, "api-features", [900000], setting, 15.0)
    assert run["stopped"] is None and len(run["games"]) == 1
    assert backend.calls == 1 + run["games"][0]["pieces"]
    assert mod.load_status() == {}  # results replace the pending reason


def test_mark_pending_records_the_reason(tmp_path, monkeypatch) -> None:
    mod = _eval_llm_module()
    monkeypatch.setattr(mod, "STATUS_PATH", tmp_path / "status.json")
    mod.main(["--mark-pending", "api-features,api-bare", "--reason", "no valid API key"])
    assert mod.load_status() == {"api-features": "no valid API key", "api-bare": "no valid API key"}
