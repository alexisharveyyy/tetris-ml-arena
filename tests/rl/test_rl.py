from __future__ import annotations

import numpy as np
import pytest
import torch

from tetris.agents.dqn.train import double_dqn_targets
from tetris.agents.evaluation import EvalSetting
from tetris.agents.ppo.train import compute_gae
from tetris.agents.registry import create_agent
from tetris.engine import Game, Placement
from tetris.engine.placement import NUM_ACTIONS
from tetris.env import action_mask, enumerate_afterstates
from tetris.rl.encoding import (
    AFTERSTATE_EXTRA_DIM,
    OBS_EXTRA_DIM,
    PACKED_BOARD_BYTES,
    action_afterstate_arrays,
    afterstate_arrays,
    observation_arrays,
    pack_boards,
    reward_for_score,
    unpack_boards,
)
from tetris.rl.nets import (
    MASKED_LOGIT,
    ActorCritic,
    AfterstateActorCritic,
    AfterstateValueNet,
    build,
    masked_log_probs,
    save_checkpoint,
    unpack_boards_torch,
)
from tetris.rl.replay import PrioritizedReplay, SumTree
from tetris.rl.vec_env import VecEnv

from ..agents.states import assert_legal, random_states

GAMMA = 0.99


# ------------------------------------------------------- truncation (DQN)


def _targets(discount: float, partial: float = 0.0) -> float:
    k = 3
    v_online = torch.tensor([[1.0, 5.0, 2.0]])
    v_target = torch.tensor([[1.5, 4.0, 3.0]])
    reward = torch.tensor([[1.0, 1.0, 2.0]])
    top = torch.zeros(1, k, dtype=torch.bool)
    valid = torch.full((1, k), discount > 0)
    part, disc = torch.tensor([partial]), torch.tensor([discount])
    return float(double_dqn_targets(v_online, v_target, reward, top, valid, part, disc, GAMMA)[0])


def test_dqn_truncation_bootstraps_termination_does_not() -> None:
    # Online picks candidate 1 (1 + 0.99 * 5); the target net evaluates it: 1 + 0.99 * 4.
    q = 1 + GAMMA * 4.0
    assert _targets(discount=1.0) == pytest.approx(q)
    # A truncated n-step transition keeps its discount and bootstraps.
    assert _targets(discount=GAMMA**2, partial=1.99) == pytest.approx(1.99 + GAMMA**2 * q)
    # A terminated transition (discount 0, no candidates) is just its partial return.
    assert _targets(discount=0.0, partial=2.5) == pytest.approx(2.5)


def test_dqn_targets_ignore_invalid_and_zero_value_after_top_out() -> None:
    v = torch.tensor([[100.0, 1.0]])
    reward = torch.tensor([[1.0, 1.0]])
    top = torch.tensor([[True, False]])
    valid = torch.tensor([[True, True]])
    zero, one = torch.tensor([0.0]), torch.tensor([1.0])
    y = double_dqn_targets(v, v, reward, top, valid, zero, one, GAMMA)
    assert float(y[0]) == pytest.approx(1 + GAMMA * 1.0)  # top-out candidate is worth only r
    no_top = torch.zeros(1, 2, dtype=torch.bool)
    valid = torch.tensor([[False, True]])
    y = double_dqn_targets(v, v, torch.tensor([[50.0, 1.0]]), no_top, valid, zero, one, GAMMA)
    assert float(y[0]) == pytest.approx(1 + GAMMA * 1.0)


def test_vec_env_truncation_returns_final_state() -> None:
    setting = EvalSetting("t", max_pieces=5, garbage_every=None)
    for mode in ("afterstate", "obs"):
        vec = VecEnv(mode, 1, 2, setting, seed=0)
        try:
            pkgs = vec.reset()
            for _ in range(5):
                if mode == "afterstate":
                    actions = [int(p["actions"][0]) for p in pkgs]
                else:
                    actions = [int(np.flatnonzero(p["mask"])[0]) for p in pkgs]
                results = vec.step(actions)
                pkgs = [r.package for r in results]
            assert all(r.truncated and not r.terminated for r in results)
            assert all(r.final is not None and r.episode["pieces"] == 5 for r in results)
        finally:
            vec.close()


# ------------------------------------------------------- truncation (PPO)


def test_ppo_gae_truncation_bootstraps_termination_does_not() -> None:
    rewards = np.array([[1.0, 1.0]], np.float32)
    values = np.array([[0.0, 0.0]], np.float32)
    terminated = np.array([[True, False]])
    truncated = np.array([[False, True]])
    final_values = np.array([[0.0, 10.0]], np.float32)
    last_values = np.array([50.0, 50.0], np.float32)  # belongs to the next episode: must be ignored
    adv, ret = compute_gae(
        rewards, values, terminated, truncated, final_values, last_values, GAMMA, 0.95
    )
    assert ret[0, 0] == pytest.approx(1.0)
    assert ret[0, 1] == pytest.approx(1.0 + GAMMA * 10.0)


def test_ppo_gae_continues_across_normal_steps() -> None:
    rewards = np.ones((2, 1), np.float32)
    values = np.zeros((2, 1), np.float32)
    no = np.zeros((2, 1), bool)
    adv, ret = compute_gae(
        rewards,
        values,
        no,
        no,
        np.zeros((2, 1), np.float32),
        np.array([2.0], np.float32),
        GAMMA,
        1.0,
    )
    assert ret[1, 0] == pytest.approx(1 + GAMMA * 2.0)
    assert ret[0, 0] == pytest.approx(1 + GAMMA * (1 + GAMMA * 2.0))


# ----------------------------------------------------------- PPO masking


@pytest.mark.parametrize("legal", [1, 2, 80])
def test_masked_distribution_never_nan(legal: int) -> None:
    torch.manual_seed(0)
    logits_raw = (torch.randn(64, 80) * 50).requires_grad_()
    mask = torch.zeros(64, 80, dtype=torch.bool)
    for row in range(64):
        mask[row, torch.randperm(80)[:legal]] = True
    logits = logits_raw.masked_fill(~mask, MASKED_LOGIT)
    logp, entropy = masked_log_probs(logits, mask)
    probs = logp.exp()
    assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()
    assert torch.all(probs[~mask] == 0)
    assert torch.allclose(probs.sum(-1), torch.ones(64))
    if legal == 1:
        assert torch.allclose(entropy, torch.zeros(64), atol=1e-6)
    samples = torch.multinomial(probs, 200, replacement=True)
    assert mask.gather(1, samples).all()
    (logp.gather(1, samples[:, :1]).sum() - entropy.sum()).backward()
    assert torch.isfinite(logits_raw.grad).all()


def test_actor_critic_masks_illegal_actions() -> None:
    net = ActorCritic()
    game = Game(3)
    board, extras = observation_arrays(game)
    mask = action_mask(game)
    logits, value = net(
        torch.from_numpy(board[None].astype(np.float32)),
        torch.from_numpy(extras[None]),
        torch.from_numpy(mask[None]),
    )
    assert torch.all(logits[0, ~torch.from_numpy(mask)] == MASKED_LOGIT) and value.shape == (1,)


# ------------------------------------------------------------- encoding


def test_pack_round_trip_cpu_and_torch() -> None:
    b = np.random.default_rng(0).random((4, 24, 10)) < 0.5
    p = pack_boards(b)
    assert p.shape == (4, 30)
    np.testing.assert_array_equal(unpack_boards(p), b)
    np.testing.assert_array_equal(unpack_boards_torch(torch.from_numpy(p)).numpy(), b)


def test_afterstate_extras_reflect_hold() -> None:
    game = Game(5)
    afs = enumerate_afterstates(game, dedupe=True)
    boards, extras, rewards, tops, actions = afterstate_arrays(game, afs)
    assert extras.shape == (len(afs), AFTERSTATE_EXTRA_DIM)
    nf = 13
    for a, row in zip(afs, extras, strict=True):
        hold_slot = int(np.argmax(row[nf + 7 : nf + 15]))
        next_piece = int(np.argmax(row[nf : nf + 7])) + 1
        if a.placement.use_hold:
            assert hold_slot == int(game.active.piece)  # current piece goes to hold
            assert next_piece == int(game.queue[1])  # hold was empty: queue[0] is placed
        else:
            assert hold_slot == 0 and next_piece == int(game.queue[0])
    assert np.allclose(rewards, reward_for_score(np.array([a.score for a in afs])))


def test_afterstate_rewards_match_engine_through_line_clears() -> None:
    agent = create_agent("dellacherie")
    game = Game(11)
    for _ in range(300):
        afs = enumerate_afterstates(game, dedupe=True)
        _, _, rewards, _, actions = afterstate_arrays(game, afs)
        choice = agent.select_placement(game)
        i = int(np.flatnonzero(actions == choice.index)[0])
        before = game.score
        game.place(choice)
        assert reward_for_score(game.score - before) == pytest.approx(rewards[i])
    assert game.lines > 50


def test_observation_shape() -> None:
    board, extras = observation_arrays(Game(1))
    assert board.shape == (24, 10) and extras.shape == (OBS_EXTRA_DIM,)


# ---------------------------------------------------------------- replay


def test_sum_tree_sampling_is_proportional() -> None:
    tree = SumTree(8)
    tree.update(np.arange(4), np.array([1.0, 0.0, 3.0, 0.0]))
    tree.update_one(5, 4.0)
    assert tree.total == pytest.approx(8.0)
    rng = np.random.default_rng(0)
    idx = tree.find(rng.uniform(0, tree.total, 40000))
    counts = np.bincount(idx, minlength=8) / 40000
    assert counts[1] == 0 and counts[3] == 0
    assert counts[[0, 2, 5]] == pytest.approx([1 / 8, 3 / 8, 4 / 8], abs=0.01)


def test_replay_requires_candidates_unless_terminal() -> None:
    r = PrioritizedReplay(16, k=4)
    b = np.zeros(30, np.uint8)
    e = np.zeros(AFTERSTATE_EXTRA_DIM, np.float32)
    r.add(b, e, terminated=True, truncated=False)
    with pytest.raises(ValueError):
        r.add(b, e, terminated=False, truncated=True)
    r.add(
        b,
        e,
        False,
        True,
        np.zeros((2, 30), np.uint8),
        np.zeros((2, AFTERSTATE_EXTRA_DIM)),
        np.ones(2),
        np.zeros(2, bool),
    )
    assert len(r) == 2 and r.next_valid[1].tolist() == [True, True, False, False] and r.truncated[1]
    idx, w = r.sample(8, 0.4)
    assert set(idx.tolist()) <= {0, 1} and w.max() == pytest.approx(1.0)


# ---------------------------------------------------------------- agents


@pytest.fixture(scope="module")
def tiny_checkpoints(tmp_path_factory):
    d = tmp_path_factory.mktemp("ckpt")
    torch.manual_seed(0)
    save_checkpoint(d / "dqn.pt", "dqn", AfterstateValueNet(), gamma=GAMMA)
    save_checkpoint(d / "ppo.pt", "ppo", AfterstateActorCritic())
    save_checkpoint(d / "ppo_raw.pt", "ppo_raw", ActorCritic())
    return d


@pytest.mark.parametrize("key", ["dqn", "ppo", "ppo_raw"])
def test_learned_agents_return_legal_placements(tiny_checkpoints, key: str) -> None:
    agent = create_agent(key, tiny_checkpoints / f"{key}.pt")
    checked = 0
    for game in random_states(30, seed=21):
        if not action_mask(game).any():
            continue
        assert_legal(game, agent.select_placement(game))
        assert_legal(game, agent.select_placement(game.to_dict()))
        checked += 1
    assert checked > 20


@pytest.mark.parametrize("key", ["dqn", "ppo", "ppo_raw"])
def test_learned_agents_noise_picks_from_ranked_alternatives(tiny_checkpoints, key: str) -> None:
    from tetris.agents.difficulty import DifficultyParams

    agent = create_agent(key, tiny_checkpoints / f"{key}.pt", DifficultyParams(noise=1.0), seed=1)
    game = Game(2)
    ranked = [s.placement for s in agent.rank_placements(game)]
    picks = {agent.select_placement(game) for _ in range(40)}
    assert ranked[0] not in picks and picks <= set(ranked[1:6])


def test_cpu_and_mps_values_agree(tiny_checkpoints) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS")
    from tetris.rl.nets import load_checkpoint

    cpu, _ = load_checkpoint(tiny_checkpoints / "dqn.pt", "dqn", "cpu")
    mps, _ = load_checkpoint(tiny_checkpoints / "dqn.pt", "dqn", "mps")
    game = Game(4)
    boards, extras, *_ = afterstate_arrays(game, enumerate_afterstates(game, dedupe=True))
    b, e = torch.from_numpy(unpack_boards(boards)), torch.from_numpy(extras)
    with torch.no_grad():
        np.testing.assert_allclose(
            cpu(b, e).numpy(), mps(b.to("mps"), e.to("mps")).cpu().numpy(), atol=1e-4
        )
    _ = Placement


def _pkg(n: int, reward: float, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "boards": rng.integers(0, 255, (n, 30), dtype=np.uint8),
        "extras": rng.random((n, AFTERSTATE_EXTRA_DIM)).astype(np.float32),
        "rewards": np.full(n, reward, np.float32),
        "tops": np.zeros(n, bool),
        "actions": np.arange(n, dtype=np.int16),
    }


def test_n_step_tracker_returns_and_episode_ends() -> None:
    from tetris.agents.dqn.train import NStepTracker

    replay = PrioritizedReplay(64, k=4)
    tr = NStepTracker(1, n=3, gamma=GAMMA, replay=replay)
    rng = np.random.default_rng(0)
    rewards = [1.0, 2.0, 3.0, 4.0]
    for t, r in enumerate(rewards):
        p = _pkg(3, r, t)
        tr.observe(0, p, np.array([3.0, 2.0, 1.0]), False, rng)  # always picks index 0
    # The transition from state 0 completes at state 3 (n = 3):
    # R = r1 + gamma r2 (rewards chosen at states 1 and 2), bootstrap discount gamma^2.
    assert len(replay) == 1
    assert replay.partial_return[0] == pytest.approx(2.0 + GAMMA * 3.0)
    assert replay.discount[0] == pytest.approx(GAMMA**2)
    # Truncation: the three pending transitions bootstrap from the final state,
    # each with the discount it has reached.
    tr.end(0, _pkg(2, 9.0, 99), np.array([1.0, 0.0]))
    assert len(replay) == 4 and replay.truncated[1:4].all() and not replay.terminated[1:4].any()
    assert replay.partial_return[1:4] == pytest.approx([3.0 + GAMMA * 4.0, 4.0, 0.0])
    assert replay.discount[1:4] == pytest.approx([GAMMA**2, GAMMA, 1.0])
    # Termination: no bootstrap.
    tr.observe(0, _pkg(2, 1.0, 5), np.array([1.0, 0.0]), False, rng)
    tr.end(0, None, None)
    assert replay.terminated[4] and replay.discount[4] == 0.0


def test_action_afterstate_rows_match_actions() -> None:
    game = Game(6)
    boards, extras, mask = action_afterstate_arrays(game)
    np.testing.assert_array_equal(mask, action_mask(game))
    afs = enumerate_afterstates(game, dedupe=False)
    ref_boards, ref_extras, *_ = afterstate_arrays(game, afs)
    for a, b, e in zip(afs, ref_boards, ref_extras, strict=True):
        np.testing.assert_array_equal(boards[a.placement.index], b)
        np.testing.assert_allclose(extras[a.placement.index], e)
    assert not boards[~mask].any() and not extras[~mask].any()


def test_afterstate_policy_is_a_distribution_over_80_actions() -> None:
    torch.manual_seed(0)
    net = AfterstateActorCritic().eval()
    games = [Game(s) for s in range(4)]
    games[1].can_hold = False  # fewer legal actions in one row
    obs = [observation_arrays(g) for g in games]
    cands = [action_afterstate_arrays(g) for g in games]
    board = torch.from_numpy(np.stack([o[0] for o in obs]).astype(np.float32))
    extras = torch.from_numpy(np.stack([o[1] for o in obs]))
    cb = torch.from_numpy(np.stack([c[0] for c in cands]))
    ce = torch.from_numpy(np.stack([c[1] for c in cands]))
    mask = torch.from_numpy(np.stack([c[2] for c in cands]))
    with torch.no_grad():
        logits, value = net(board, extras, mask, cb, ce)
        logp, entropy = masked_log_probs(logits, mask)
    assert logits.shape == (4, 80) and value.shape == (4,)
    probs = logp.exp()
    assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()
    assert torch.allclose(probs.sum(-1), torch.ones(4)) and torch.all(probs[~mask] == 0)
    # Each legal logit is the shared scorer applied to that action's afterstate.
    a = int(np.flatnonzero(cands[0][2])[3])
    with torch.no_grad():
        direct = net.scorer(unpack_boards_torch(cb[0, a : a + 1]), ce[0, a : a + 1])
    assert float(logits[0, a]) == pytest.approx(float(direct[0]), abs=1e-5)


def test_dqn_soft_target_update_and_lr_schedule() -> None:
    from tetris.agents.dqn.train import Learner, cosine_lr

    cfg = {"gamma": GAMMA, "lr": 1e-4, "lr_final": 1e-5, "target_tau": 0.25}
    learner = Learner(cfg, torch.device("cpu"))
    with torch.no_grad():
        for p in learner.online.parameters():
            p.add_(1.0)
    before = [p.clone() for p in learner.target.parameters()]
    learner.sync_target()
    for b, t, o in zip(
        before, learner.target.parameters(), learner.online.parameters(), strict=True
    ):
        assert torch.allclose(t, b + 0.25 * (o - b))
    assert learner.set_lr(0.0) == pytest.approx(1e-4)
    assert learner.set_lr(0.5) == pytest.approx(5.5e-5)
    assert learner.set_lr(1.0) == pytest.approx(1e-5)
    assert learner.opt.param_groups[0]["lr"] == pytest.approx(1e-5)
    assert cosine_lr(2.5e-4, None, 0.7) == 2.5e-4


def test_dqn_hard_target_sync_without_tau() -> None:
    from tetris.agents.dqn.train import Learner

    learner = Learner({"gamma": GAMMA, "lr": 1e-4, "target_update": 2}, torch.device("cpu"))
    with torch.no_grad():
        for p in learner.online.parameters():
            p.add_(1.0)
    learner.updates = 1
    learner.sync_target()
    assert not all(
        torch.equal(t, o)
        for t, o in zip(learner.target.parameters(), learner.online.parameters(), strict=True)
    )
    learner.updates = 2
    learner.sync_target()
    assert all(
        torch.equal(t, o)
        for t, o in zip(learner.target.parameters(), learner.online.parameters(), strict=True)
    )


def _afterstate_batch(n_states: int, seed: int = 0):
    """Random PPO afterstate-policy inputs with about 70% of actions legal."""
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(n_states, NUM_ACTIONS, generator=g) < 0.7
    mask[:, 0] = True
    return [
        torch.randint(0, 2, (n_states, 24, 10), generator=g).float(),
        torch.rand(n_states, OBS_EXTRA_DIM, generator=g),
        mask,
        torch.randint(
            0, 256, (n_states, NUM_ACTIONS, PACKED_BOARD_BYTES), dtype=torch.uint8, generator=g
        ),
        torch.rand(n_states, NUM_ACTIONS, AFTERSTATE_EXTRA_DIM, generator=g),
    ]


def _scorer_grads(net, inputs, actions, adv):
    net.zero_grad(set_to_none=True)
    logits, _ = net(*inputs)
    logp, entropy = masked_log_probs(logits, inputs[2])
    loss = -(adv * logp.gather(1, actions[:, None]).squeeze(1)).mean() - 0.01 * entropy.mean()
    loss.backward()
    return logits.detach().cpu(), {
        n: p.grad.detach().cpu() for n, p in net.scorer.named_parameters()
    }


def test_afterstate_policy_chunked_scoring_matches_single_pass(monkeypatch) -> None:
    import tetris.rl.nets as nets

    inputs = _afterstate_batch(24)
    actions = torch.multinomial(inputs[2].float(), 1).squeeze(1)
    adv = torch.randn(24, generator=torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    net = build("ppo", {})
    monkeypatch.setattr(nets, "SCORER_CHUNK", 10**9)
    logits_one, grads_one = _scorer_grads(net, inputs, actions, adv)
    monkeypatch.setattr(nets, "SCORER_CHUNK", 37)
    logits_chunked, grads_chunked = _scorer_grads(net, inputs, actions, adv)
    assert torch.allclose(logits_one, logits_chunked, atol=1e-5)
    for name, g in grads_one.items():
        assert torch.allclose(g, grads_chunked[name], rtol=1e-4, atol=1e-6), name


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_afterstate_policy_mps_gradients_match_cpu_at_training_scale() -> None:
    """Regression: one PPO minibatch (2,048 states, about 115k legal afterstates)
    scored in a single call gave wrong convolution gradients on MPS (relative
    error about 6e8), so the afterstate policy never trained. Scoring in chunks
    of SCORER_CHUNK fixes it. Compared as one vector: the last bias has an
    analytic gradient of zero, so per-parameter relative errors are noise."""
    n = 2048
    inputs = _afterstate_batch(n)
    actions = torch.multinomial(inputs[2].float(), 1).squeeze(1)
    adv = torch.randn(n, generator=torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    cpu_net = build("ppo", {})
    mps_net = build("ppo", {})
    mps_net.load_state_dict(cpu_net.state_dict())
    mps_net.to("mps")
    _, grads_cpu = _scorer_grads(cpu_net, inputs, actions, adv)
    _, grads_mps = _scorer_grads(
        mps_net, [x.to("mps") for x in inputs], actions.to("mps"), adv.to("mps")
    )
    g_cpu = torch.cat([g.flatten() for g in grads_cpu.values()])
    g_mps = torch.cat([grads_mps[k].flatten() for k in grads_cpu])
    assert torch.isfinite(g_mps).all()
    rel = float((g_cpu - g_mps).norm() / g_cpu.norm())
    assert rel < 1e-3, f"relative gradient difference {rel:.2e}"
