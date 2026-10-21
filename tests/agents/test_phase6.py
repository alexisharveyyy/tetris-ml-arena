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
DATA = Path(__file__).with_name("data")


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
