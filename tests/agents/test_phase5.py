"""Phase 5: recording mode, behavior cloning (pure, DAgger, you) and NEAT."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tetris.agents.difficulty import DifficultyParams
from tetris.agents.evaluation import EvalSetting, play_game
from tetris.agents.game_log import read_episode
from tetris.agents.planner import plan_path
from tetris.agents.registry import AGENTS, create_agent
from tetris.engine import Game, Placement
from tetris.engine.pieces import Piece
from tetris.engine.placement import spawn_position
from tetris.env import action_mask, enumerate_afterstates
from tetris.match import MatchConfig, MatchController, PlayerSpec
from tetris.match.recording import GameRecorder, match_action, recording_totals
from tetris.rl.nets import AfterstateActorCritic, save_checkpoint

from .states import assert_legal, random_states

TINY = EvalSetting("tiny", max_pieces=12, garbage_every=3, garbage_rows=1)


# ----------------------------------------------------------------- recording


def _recorded_match(pieces: int, seed: int = 5, garbage: int | None = None):
    match = MatchController(
        MatchConfig(mode="solo", seed=seed, eval_garbage_every=garbage), [PlayerSpec("h", "You")]
    )
    recorder = GameRecorder("h", seed, "garbage" if garbage else "plain")
    match.lock_listeners.append(recorder.on_lock)
    game = match.players["h"].game
    recorder.start(game)
    chooser = create_agent("dellacherie")
    intended = []
    for i in range(pieces):
        if game.game_over:
            break
        before = game.copy()
        if i % 4 == 1:  # exercise hold
            holds = [a.placement for a in enumerate_afterstates(game) if a.placement.use_hold]
            placement = holds[0] if holds else chooser.select_placement(game)
        else:
            placement = chooser.select_placement(game)
        moves = plan_path(game, placement)
        assert moves is not None
        for move in moves:
            match.apply_input("h", move)
        expected = before.copy()
        expected.place(placement)
        intended.append((before, placement, expected.board.copy()))
    return match, recorder, intended


@pytest.mark.parametrize("garbage", [None, 3])
def test_recording_matches_scripted_hard_drop_placements(garbage) -> None:
    match, recorder, intended = _recorded_match(24, garbage=garbage)
    assert len(recorder.steps) == len(intended) >= 20
    assert recorder.off_space == 0
    assert any(p.use_hold for _, p, _ in intended)
    for step, (before, placement, board_after) in zip(recorder.steps, intended, strict=True):
        state = Game.from_dict(step.state)
        assert state.to_dict()["board"] == before.to_dict()["board"]
        assert action_mask(state)[step.action]
        replay = state.copy()
        replay.place(Placement.from_index(step.action))
        assert (replay.board == board_after).all()  # same locked cells as intended
        assert step.action == placement.index or step.action < placement.index


def test_match_action_marks_tuck_under_overhang_off_space() -> None:
    game = Game(3)
    board = game.board
    board[:] = 0
    board[-4, 0:4] = 8  # overhang over columns 0-3, empty underneath
    game.active = None
    game.hold = None
    pos = spawn_position(board, Piece.O)
    from tetris.engine.game import ActivePiece

    game.active = ActivePiece(Piece.O, 0, *pos)
    under = [(len(board) - 2, 0), (len(board) - 2, 1), (len(board) - 1, 0), (len(board) - 1, 1)]
    assert match_action(game, "O", under) == -1
    right = [(len(board) - 2, 8), (len(board) - 2, 9), (len(board) - 1, 8), (len(board) - 1, 9)]
    action = match_action(game, "O", right)
    assert action >= 0 and not Placement.from_index(action).use_hold


def test_recording_save_round_trip(tmp_path) -> None:
    match, recorder, _ = _recorded_match(8)
    path = recorder.save(match.players["h"].game, root=tmp_path)
    records = list(read_episode(path))
    assert records[0]["agent"] == "human" and records[0]["setting"]["name"] == "plain"
    steps = [r for r in records if r["type"] == "step"]
    assert [s["action"] for s in steps] == [s.action for s in recorder.steps]
    assert records[-1]["on_space"] == len(steps)
    assert recording_totals(tmp_path) == {"games": 1, "on_space": len(steps), "off_space": 0}


# ----------------------------------------------------------- behavior cloning


@pytest.fixture(scope="module")
def ga_log(tmp_path_factory):
    """One logged GA game (12 pieces, garbage) in the game-log format."""
    from tetris.agents.game_log import log_episode

    path = tmp_path_factory.mktemp("logs") / "ga" / "tiny" / "seed_1.jsonl.gz"
    log_episode("ga", str(AGENTS["ga"].checkpoint_path), 1, TINY, path)
    return path


def test_dataset_targets_cover_equivalent_actions() -> None:
    from tetris.agents.imitation.dataset import encode_step

    found = 0
    for seed in range(60):
        game = Game(seed)
        if game.active.piece != Piece.O:
            continue
        mask = action_mask(game)
        action = int(np.flatnonzero(mask[:40])[0])  # a non-hold O placement
        _, _, actions, target = encode_step(game.to_dict(), action)
        # Every rotation of the O at the same column gives the same afterstate.
        same = [int(a) for a in actions[target]]
        assert action in same and len(same) >= 2
        found += 1
    assert found >= 3


def test_dataset_subset_and_concat(ga_log) -> None:
    from tetris.agents.imitation.dataset import ImitationData, encode_episode

    data = encode_episode(ga_log)
    assert len(data) == 12 and data.offsets[-1] == data.n_rows
    assert all(data.target[data.offsets[s] : data.offsets[s + 1]].any() for s in range(12))
    sub = data.subset(np.array([2, 5]))
    assert len(sub) == 2
    assert (sub.boards[: sub.offsets[1]] == data.boards[data.offsets[2] : data.offsets[3]]).all()
    both = ImitationData.concat([data, data])
    assert len(both) == 24 and len(np.unique(both.episode)) == 2


def test_bc_loss_decreases_and_agreement_is_consistent(ga_log) -> None:
    from tetris.agents.imitation.dataset import encode_episode
    from tetris.agents.imitation.train import agreement, batch_tensors, bc_loss
    from tetris.rl.nets import build

    data = encode_episode(ga_log)
    states = np.arange(len(data))
    torch.manual_seed(0)
    net = build("ppo", {})
    opt = torch.optim.Adam(net.scorer.parameters(), lr=3e-3)
    batch = batch_tensors(data, states, torch.device("cpu"))
    first = float(bc_loss(net, batch).detach())
    for _ in range(40):
        loss = bc_loss(net, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(bc_loss(net, batch).detach()) < 0.5 * first
    m = agreement(net, data, states, torch.device("cpu"))
    assert 0.0 <= m["top1"] <= m["top3"] <= 1.0
    assert m["top1"] > 0.5  # fits 12 states after 40 steps


def test_dagger_episode_labels_states_with_the_ga(tmp_path) -> None:
    from tetris.agents.imitation.dagger import dagger_episode

    ckpt = tmp_path / "bc.pt"
    save_checkpoint(ckpt, "ppo", AfterstateActorCritic())
    out = dagger_episode((str(ckpt), 7, TINY.to_dict(), str(tmp_path / "d" / "seed_7.jsonl.gz")))
    ga = create_agent("ga")
    steps = [r for r in read_episode(out["path"]) if r["type"] == "step"]
    assert len(steps) == out["steps"] > 0
    for s in steps:
        assert s["action"] == ga.select_placement(s["state"]).index
        assert action_mask(Game.from_dict(s["state"]))[s["played"]]


def test_choose_version_picks_higher_selection_fitness(tmp_path, monkeypatch) -> None:
    import tetris.agents.imitation.train as bc_train

    monkeypatch.setattr(bc_train, "CHECKPOINT_DIR", tmp_path)
    for key, fit in (("bc_pure", 0.9), ("bc_dagger", 1.1)):
        (tmp_path / key).mkdir()
        (tmp_path / key / "best.pt").write_text(key)
        (tmp_path / key / "selection.json").write_text(json.dumps({"selection_fitness": fit}))
    choice = bc_train.choose_version()
    assert choice["chosen"] == "bc_dagger"
    assert (tmp_path / "bc" / "best.pt").read_text() == "bc_dagger"
    saved = json.loads((tmp_path / "bc" / "version_choice.json").read_text())
    assert saved == {"selection_fitness": {"bc_pure": 0.9, "bc_dagger": 1.1}, "chosen": "bc_dagger"}


def test_bc_you_skips_below_the_placement_minimum(tmp_path, monkeypatch) -> None:
    import tetris.agents.imitation.train as bc_train
    import tetris.match.recording as recording

    monkeypatch.setattr(recording, "RECORDINGS_DIR", tmp_path / "rec")
    monkeypatch.setattr(bc_train, "ROOT", tmp_path)
    (tmp_path / "reports").mkdir()
    out = bc_train.train_you({"you_min_placements": 800})
    assert "skipped" in out
    assert (
        "0 on-space placements"
        in json.loads((tmp_path / "reports" / "bc_you.json").read_text())["skipped"]
    )


def test_split_recordings_holds_out_the_latest_games() -> None:
    from tetris.agents.imitation.dataset import ImitationData
    from tetris.agents.imitation.train import split_recordings

    data = ImitationData.empty()
    data.episode = np.repeat(np.arange(10), 5).astype(np.int32)
    data.teacher = np.zeros(50, np.int16)
    train, val, test = split_recordings(data, 0.1, 0.2)
    assert set(data.episode[test]) == {8, 9} and set(data.episode[val]) == {7}
    assert len(train) + len(val) + len(test) == 50


@pytest.mark.parametrize("key", ["bc", "bc_pure", "bc_dagger", "bc_you"])
def test_bc_agents_return_legal_placements_and_ranked_mistakes(tmp_path, key) -> None:
    ckpt = tmp_path / "bc.pt"
    save_checkpoint(ckpt, "ppo", AfterstateActorCritic())
    agent = create_agent(key, ckpt)
    noisy = create_agent(key, ckpt, DifficultyParams(noise=1.0), seed=1)
    for game in random_states(12, seed=4):
        if not action_mask(game).any():
            continue
        assert_legal(game, agent.select_placement(game))
        ranked = [s.placement for s in agent.rank_placements(game)]
        assert noisy.select_placement(game) in ranked[1:6]  # mistakes: the next five


def test_registry_flags() -> None:
    assert not AGENTS["bc"].leaderboard and AGENTS["bc"].playable
    assert AGENTS["bc_pure"].leaderboard and not AGENTS["bc_pure"].playable
    assert AGENTS["bc_dagger"].leaderboard and not AGENTS["bc_dagger"].playable
    assert AGENTS["bc_you"].playable and AGENTS["neat"].playable


# ---------------------------------------------------------------------- NEAT


@pytest.fixture(scope="module")
def neat_genomes():
    """Mutated genomes with hidden nodes and mixed activations."""
    import neat

    from tetris.agents.neuroevolution.train import neat_config

    config = neat_config("tetris/agents/neuroevolution/neat_config.ini")
    population = neat.Population(config, seed=3)
    genomes = list(population.population.values())[:8]
    rng = np.random.default_rng(0)
    for g in genomes:
        for _ in range(12):
            g.mutate(config.genome_config)
        for node in g.nodes.values():
            node.activation = str(rng.choice(["identity", "tanh", "relu", "sigmoid"]))
    return config, genomes


def test_genome_net_matches_neat_python(neat_genomes) -> None:
    from neat.nn import FeedForwardNetwork

    from tetris.agents.neuroevolution.network import GenomeNet

    config, genomes = neat_genomes
    x = np.random.default_rng(1).normal(0, 1, (25, 13))
    assert any(GenomeNet.from_genome(g, config).hidden for g in genomes)
    for g in genomes:
        ours = GenomeNet.from_genome(g, config)
        reference = FeedForwardNetwork.create(g, config)
        expected = np.array([reference.activate(list(row))[0] for row in x])
        np.testing.assert_allclose(ours(x), expected, rtol=1e-9, atol=1e-12)
        again = GenomeNet.from_dict(json.loads(json.dumps(ours.to_dict())))
        np.testing.assert_allclose(again(x), expected, rtol=1e-9, atol=1e-12)


def test_neat_agent_legal_deterministic_and_plotted(neat_genomes, tmp_path) -> None:
    from tetris.agents.neuroevolution.agent import best_placement_index, save_genome_checkpoint
    from tetris.agents.neuroevolution.network import GenomeNet
    from tetris.agents.neuroevolution.plot import plot_topology

    config, genomes = neat_genomes
    net = GenomeNet.from_genome(genomes[0], config)
    ckpt = tmp_path / "neat.json"
    save_genome_checkpoint(ckpt, net, generation=4)
    agent = create_agent("neat", ckpt)
    noisy = create_agent("neat", ckpt, DifficultyParams(noise=1.0), seed=2)
    for game in random_states(12, seed=6):
        if not action_mask(game).any():
            continue
        assert_legal(game, agent.select_placement(game))
        ranked = [s.placement for s in agent.rank_placements(game)]
        assert noisy.select_placement(game) in ranked[1:6]  # mistakes: the next five
    a = play_game(lambda g: best_placement_index(g, net), 11, TINY)
    b = play_game(lambda g: best_placement_index(g, net), 11, TINY)
    assert a == b  # fitness evaluation is deterministic for a seed
    png = plot_topology(ckpt, tmp_path / "topology.png")
    assert png.exists() and png.stat().st_size > 10_000


def test_topology_edges_bend_around_nodes_they_skip() -> None:
    from tetris.agents.neuroevolution.plot import CLEARANCE, _gap_at, edge_path

    nodes = [(1.0, 0.0), (2.0, 0.0), (1.0, 3.0)]
    # Input on row 0 wired to the output on row 0, past two hidden nodes on the same row.
    xs, ys = edge_path((0.0, 0.0), (3.0, 0.0), nodes)
    assert (xs[0], ys[0]) == (0.0, 0.0) and (xs[-1], ys[-1]) == (3.0, 0.0)
    assert all(_gap_at(xs, ys, x, y) >= CLEARANCE for x, y in nodes)
    # Nothing in the way: a straight line.
    xs, ys = edge_path((0.0, 2.0), (1.0, 0.0), nodes)
    assert ys[len(ys) // 2] == 1.0


def test_bc_you_fine_tunes_with_game_split_and_reports_baseline(tmp_path, monkeypatch) -> None:
    import tetris.agents.imitation.train as bc_train
    import tetris.match.recording as recording
    import tetris.rl.training as rl_training

    rec_root = tmp_path / "rec"
    monkeypatch.setattr(recording, "RECORDINGS_DIR", rec_root)
    for name in ("CHECKPOINT_DIR",):
        monkeypatch.setattr(bc_train, name, tmp_path / "ckpt")
    monkeypatch.setattr(bc_train, "ROOT", tmp_path)
    monkeypatch.setattr(rl_training, "ROOT", tmp_path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "ckpt" / "bc").mkdir(parents=True)
    (tmp_path / "ckpt" / "bc_you").mkdir(parents=True)
    save_checkpoint(tmp_path / "ckpt" / "bc" / "best.pt", "ppo", AfterstateActorCritic())
    for seed in range(5):  # five recorded games
        match, recorder, _ = _recorded_match(10, seed=seed)
        recorder.save(match.players["h"].game, root=rec_root)
    cfg = {
        **bc_train.load_config(bc_train.DEFAULT_CONFIG),
        "device": "cpu",
        "cache_dir": str(tmp_path / "cache"),
        "you_min_placements": 40,
        "you_minutes": 0.15,
        "you_patience": 2,
        "batch_states": 16,
    }
    report = bc_train.train_you(cfg, "test")
    split = report["split_by_game"]
    assert split["test"]["games"] >= 1 and split["validation"]["games"] >= 1
    assert sum(v["games"] for v in split.values()) == 5
    for key in ("baseline_ga_trained_bc", "fine_tuned"):
        assert 0.0 <= report[key]["top1"] <= report[key]["top3"] <= 1.0
    diff = report["difference_fine_tuned_minus_baseline"]
    assert diff["interval"] is None
    per_game = diff["top1"]["per_test_game"]
    assert len(per_game) == split["test"]["games"]
    assert sum(g["states"] for g in per_game) == split["test"]["states"]
    weighted = sum(g["difference"] * g["states"] for g in per_game) / split["test"]["states"]
    assert abs(weighted - diff["top1"]["mean"]) < 1e-9
    assert report["validation_curve"][0]["minutes"] == 0.0  # first check: the baseline model
    assert (tmp_path / "ckpt" / "bc_you" / "best.pt").exists()
    assert json.loads((tmp_path / "reports" / "bc_you.json").read_text())["fine_tuned"]
