"""Export the trained agents and the app's static data for the browser version (site/).

Nothing here changes a checkpoint or a result: the script only reads them.

Writes:

* ``site/public/models/<name>.onnx``: the neural agents as ONNX graphs (the
  dynamo exporter with ``optimize=False``, so every initializer keeps its
  ``state_dict`` name). The browser reads the weights from these files.
  PPO and the BC agents export only the afterstate scorer (the critic is not
  used in play); the Decision Transformer exports two graphs, the afterstate
  embedder (``dt_embed``) and the sequence model (``dt_core``).
* ``site/src/data/*.json``: what the local server's REST API would return
  (agents, presets, settings defaults, leaderboard), the linear and NEAT
  weights, agent metadata (DQN gamma, DT target and context) and the gravity
  table as exact float64 bit patterns.
* ``site/public/models/manifest.json``: size and SHA-256 of every model file,
  with the SHA-256 of the checkpoint it came from.

Checks, each of which stops the export on failure:

* every ONNX initializer equals its ``state_dict`` tensor bit for bit;
* onnxruntime on the exported graph matches the PyTorch model (as the agents
  run it) on real afterstates, to within float32 rounding.

Usage: python scripts/tools/web_export.py [--skip-models]
Needs the ``export`` extra (onnx, onnxscript, onnxruntime).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import tempfile
from pathlib import Path

import numpy as np

from tetris.agents.difficulty import PRESETS
from tetris.agents.linear import DELLACHERIE_WEIGHTS, load_linear_checkpoint, weights_from_dict
from tetris.agents.registry import AGENTS, ROOT
from tetris.engine import scoring
from tetris.env.features import FEATURE_NAMES
from tetris.settings import Settings
from tetris.web import leaderboard
from tetris.web.api import PRESET_TEXT, lineup_identities

SITE = ROOT / "site"
MODELS = SITE / "public" / "models"
DATA = SITE / "src" / "data"
REPORTS = ROOT / "reports" / "leaderboard"

#: Agents offered in the browser. The LLM agent is left out of the web version.
WEB_AGENTS = ("dqn", "cem", "ga", "ppo", "bc", "bc_you", "neat", "dt", "dellacherie",
              "ppo_raw", "bc_dagger")
#: Listed on the leaderboard only (as on the local server).
LEADERBOARD_ONLY = ("bc_pure",)
#: Group changes for the web lineup: BC + DAgger becomes a playable variant.
WEB_GROUPS = {"bc_dagger": "variant"}
#: ONNX file per neural agent; the BC agents share the PPO architecture.
NEURAL = {"dqn": "dqn", "ppo": "ppo", "ppo_raw": "ppo_raw", "bc": "ppo", "bc_dagger": "ppo",
          "bc_you": "ppo"}

DQN_EXTRA = 29
OBS_EXTRA = 62


def float_bits(x: float) -> str:
    """IEEE 754 float64 bit pattern as 16 hex digits (exact in any language)."""
    return struct.pack(">d", float(x)).hex()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")


# ------------------------------------------------------------------ data


def agents_data() -> dict:
    entries = []
    for ident in lineup_identities():
        key = ident.key
        if key not in WEB_AGENTS and key not in LEADERBOARD_ONLY:
            continue
        spec = AGENTS[key]
        d = ident.to_dict()
        d["group"] = WEB_GROUPS.get(key, d["group"])
        playable = key in WEB_AGENTS
        entries.append(
            {
                **d,
                "registered": True,
                "available": True,
                "status": {"state": "ok", "detail": ""},
                "playable": playable,
                "on_leaderboard": bool(spec.leaderboard),
                "description": spec.description,
                "supports_target_return": bool(spec.supports_target_return),
            }
        )
    from tetris.agents.identity import GROUPS, HUMAN, LOOKS

    return {"agents": entries, "human": HUMAN.to_dict(), "groups": list(GROUPS),
            "looks": list(LOOKS)}


def presets_data() -> list[dict]:
    return [
        {"name": name, "description": PRESET_TEXT.get(name, ""), "params": params.to_dict()}
        for name, params in PRESETS.items()
    ]


def leaderboard_data(agent_keys: list[str]) -> dict:
    name_to_key = {spec.name: key for key, spec in AGENTS.items()}
    return {
        "survival": leaderboard.survival(REPORTS, agent_keys, name_to_key),
        "marathon": leaderboard.marathon(REPORTS, agent_keys),
    }


def weights_data() -> dict:
    linear = {"dellacherie": [float(w) for w in weights_from_dict(DELLACHERIE_WEIGHTS)]}
    for key in ("ga", "cem"):
        w, _ = load_linear_checkpoint(AGENTS[key].checkpoint_path)
        linear[key] = [float(x) for x in w]
    neat = json.loads(AGENTS["neat"].checkpoint_path.read_text())
    if neat.get("format") != "neat-genome-v1":
        raise SystemExit("unexpected NEAT checkpoint format")
    return {
        "feature_names": list(FEATURE_NAMES),
        # Bit patterns too, so the TypeScript weights are the exact float64 values.
        "linear": linear,
        "linear_bits": {k: [float_bits(x) for x in v] for k, v in linear.items()},
        "neat": neat["network"],
    }


def gravity_data() -> dict:
    return {
        "note": "scoring.gravity_rows_per_second(level) for levels 1 to 20, as float64 bits.",
        "bits": [float_bits(scoring.gravity_rows_per_second(level)) for level in range(1, 21)],
    }


# ---------------------------------------------------------------- models


def _sample_inputs(n_states: int = 40, seed: int = 7):
    """Real afterstate and observation inputs from a few Dellacherie games."""
    from tetris.agents.linear import DellacherieAgent
    from tetris.engine.game import Game
    from tetris.env.afterstates import enumerate_afterstates
    from tetris.rl.encoding import afterstate_arrays, observation_arrays, unpack_boards

    agent = DellacherieAgent(seed=seed)
    rng = np.random.default_rng(seed)
    boards, extras, obs_b, obs_e = [], [], [], []
    game = Game(seed)
    while len(obs_b) < n_states:
        if game.game_over:
            game = Game(int(rng.integers(1 << 30)))
        afs = enumerate_afterstates(game, dedupe=True)
        if not afs:
            game = Game(int(rng.integers(1 << 30)))
            continue
        b, e, _, _, _ = afterstate_arrays(game, afs)
        boards.append(unpack_boards(b))
        extras.append(e)
        ob, oe = observation_arrays(game)
        obs_b.append(ob.astype(np.float32))
        obs_e.append(oe)
        ranked = agent.rank_placements(game)
        pick = ranked[int(rng.integers(min(4, len(ranked))))].placement
        game.place(pick)
    return (np.concatenate(boards), np.concatenate(extras), np.stack(obs_b), np.stack(obs_e))


def _check_initializers(onnx_path: Path, module) -> int:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    state = {k: v.detach().cpu().numpy() for k, v in module.state_dict().items()}
    used = 0
    for name, value in state.items():
        if name not in inits:
            continue
        if inits[name].dtype != value.dtype or not np.array_equal(inits[name], value):
            raise SystemExit(f"{onnx_path.name}: initializer {name} differs from the checkpoint")
        used += 1
    if used == 0:
        raise SystemExit(f"{onnx_path.name}: no initializer matches a state_dict name")
    return used


def _export(module, args, path: Path, input_names, output_names, dynamic) -> None:
    import torch
    from torch import nn

    from tetris.rl import nets

    # Conv3x3 runs as a matmul on the CPU; the result is the same convolution.
    # Export it as a plain Conv so the graph keeps the weights in Conv2d layout.
    original = nets.Conv3x3.forward
    nets.Conv3x3.forward = nn.Conv2d.forward
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / path.name
            torch.onnx.export(
                module, args, str(out), input_names=input_names, output_names=output_names,
                dynamic_shapes=dynamic, optimize=False, external_data=False, dynamo=True,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out, path)
    finally:
        nets.Conv3x3.forward = original


def _ort(path: Path, feeds: dict) -> list[np.ndarray]:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return sess.run(None, feeds)


def export_models(manifest: dict) -> dict:
    import torch
    from torch.export import Dim

    from tetris.rl.nets import load_checkpoint

    torch.set_num_threads(1)
    boards, extras, obs_b, obs_e = _sample_inputs()
    meta: dict = {}
    done: dict[str, Path] = {}
    report = {}
    n = Dim("n", min=1, max=4096)

    for key, kind in NEURAL.items():
        ckpt = AGENTS[key].checkpoint_path
        model, data = load_checkpoint(ckpt, kind, "cpu")
        model.eval()
        path = MODELS / f"{key}.onnx"
        if kind in ("dqn", "ppo"):
            scorer = model if kind == "dqn" else model.scorer
            tb, te = torch.from_numpy(boards[:8]), torch.from_numpy(extras[:8])
            _export(scorer, (tb, te), path, ["boards", "extras"], ["score"],
                    {"boards": {0: n}, "extras": {0: n}})
            with torch.inference_mode():
                ref = scorer(torch.from_numpy(boards), torch.from_numpy(extras)).numpy()
            got = _ort(path, {"boards": boards, "extras": extras})[0]
            used = _check_initializers(path, scorer)
            if kind == "dqn":
                meta[key] = {"gamma": float(data.get("gamma", 0.99))}
        else:  # ppo_raw: policy logits from the observation
            class Policy(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.board, self.extra, self.trunk, self.policy = (
                        m.board, m.extra, m.trunk, m.policy)

                def forward(self, b, e):
                    h = self.trunk(torch.cat([self.board(b), self.extra(e)], dim=-1))
                    return self.policy(h)

            pol = Policy(model).eval()
            _export(pol, (torch.from_numpy(obs_b[:2]), torch.from_numpy(obs_e[:2])), path,
                    ["boards", "extras"], ["logits"], {"b": {0: n}, "e": {0: n}})
            with torch.inference_mode():
                ref = pol(torch.from_numpy(obs_b), torch.from_numpy(obs_e)).numpy()
            got = _ort(path, {"boards": obs_b, "extras": obs_e})[0]
            used = _check_initializers(path, pol)
        err = float(np.max(np.abs(ref - got)))
        if not err <= 1e-4 * max(1.0, float(np.max(np.abs(ref)))):
            raise SystemExit(f"{key}: onnxruntime differs from torch by {err}")
        report[key] = {"max_abs_diff_vs_torch": err, "initializers_checked": used}
        done[key] = path
        manifest[key] = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
                         "checkpoint_sha256": sha256(ckpt)}

    # Decision Transformer: embedder and sequence model.
    ckpt = AGENTS["dt"].checkpoint_path
    dt, data = load_checkpoint(ckpt, "dt", "cpu")
    dt.eval()
    cfg = dict(dt.config)
    meta["dt"] = {"default_target": float(data.get("default_target", 1.0)),
                  "return_cap": int(data.get("return_cap", 500)), "config": cfg}
    emb_path = MODELS / "dt_embed.onnx"
    _export(dt.scorer, (torch.from_numpy(boards[:8]), torch.from_numpy(extras[:8])), emb_path,
            ["boards", "extras"], ["embedding"], {"boards": {0: n}, "extras": {0: n}})
    with torch.inference_mode():
        ref = dt.scorer(torch.from_numpy(boards), torch.from_numpy(extras)).numpy()
    got = _ort(emb_path, {"boards": boards, "extras": extras})[0]
    err_e = float(np.max(np.abs(ref - got)))

    from tetris.rl.encoding import pack_boards

    class Core(torch.nn.Module):
        """Query at the last state token, from unpacked boards (batch 1)."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, rtg, obs_boards, obs_extras, action_emb, timesteps, valid):
            # Same as DecisionTransformer.forward, minus the bit unpacking.
            m = self.m
            b, k = rtg.shape
            state = m.state_proj(torch.cat([m.state_board(obs_boards.reshape(b * k, 24, 10)),
                                            m.state_extra(obs_extras.reshape(b * k, -1))], -1)
                                 ).reshape(b, k, -1)
            time = m.timestep(timesteps.clamp(0, m.max_timestep - 1))
            types = m.token_type.weight
            tokens = torch.stack([m.rtg_proj(rtg.unsqueeze(-1)) + time + types[0],
                                  state + time + types[1],
                                  m.action_proj(action_emb) + time + types[2]], 2
                                 ).reshape(b, 3 * k, -1)
            h = m.embed_norm(tokens)
            h = m.blocks(h, mask=m.attention_mask(valid))
            h = m.final_norm(h).reshape(b, k, 3, -1)[:, :, 1]
            return m.query(h)[:, -1]

    k = cfg["context"]
    d = cfg["d_model"]
    core = Core(dt).eval()
    rng = np.random.default_rng(3)
    rtg = rng.uniform(0, 1.4, (1, k)).astype(np.float32)
    ob = obs_b[:k][None].astype(np.float32)
    oe = obs_e[:k][None]
    ae = ref[:k][None].astype(np.float32)
    ts = np.arange(k, dtype=np.int64)[None] + 100
    valid = np.ones((1, k), dtype=bool)
    valid[0, :5] = False
    inputs = tuple(torch.from_numpy(x) for x in (rtg, ob, oe, ae, ts, valid))
    core_path = MODELS / "dt_core.onnx"
    _export(core, inputs, core_path,
            ["rtg", "obs_boards", "obs_extras", "action_emb", "timesteps", "valid"], ["query"],
            None)
    with torch.inference_mode():
        q_ref = core(*inputs).numpy()
        # The wrapper must match the agent's own forward (packed boards).
        packed = torch.from_numpy(pack_boards(ob.astype(bool)))
        q_agent = dt(inputs[0], packed, inputs[2], inputs[3], inputs[4], inputs[5])[:, -1].numpy()
    if float(np.max(np.abs(q_ref - q_agent))) > 1e-5:
        raise SystemExit("dt: export wrapper differs from DecisionTransformer.forward")
    q_got = _ort(core_path, dict(zip(
        ["rtg", "obs_boards", "obs_extras", "action_emb", "timesteps", "valid"],
        (rtg, ob, oe, ae, ts, valid))))[0]
    err_c = float(np.max(np.abs(q_ref - q_got)))
    for path, module in ((emb_path, dt.scorer), (core_path, core)):
        _check_initializers(path, module)
    if err_e > 1e-4 or err_c > 1e-3:
        raise SystemExit(f"dt: onnxruntime differs from torch ({err_e}, {err_c})")
    report["dt"] = {"embed_max_abs_diff": err_e, "core_max_abs_diff": err_c, "d_model": d}
    for name, path in (("dt_embed", emb_path), ("dt_core", core_path)):
        manifest[name] = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
                          "checkpoint_sha256": sha256(ckpt)}
    return {"meta": meta, "report": report}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-models", action="store_true", help="only write the JSON data")
    args = ap.parse_args()

    agents = agents_data()
    write_json(DATA / "agents.json", agents)
    write_json(DATA / "presets.json", presets_data())
    write_json(DATA / "settings_defaults.json", Settings().to_dict())
    keys = [a["key"] for a in agents["agents"] if a["on_leaderboard"]]
    write_json(DATA / "leaderboard.json", leaderboard_data(keys))
    write_json(DATA / "weights.json", weights_data())
    write_json(DATA / "gravity.json", gravity_data())
    if args.skip_models:
        return
    manifest: dict = {}
    out = export_models(manifest)
    write_json(DATA / "agent_meta.json", out["meta"])
    write_json(MODELS / "manifest.json", manifest)
    for key, r in out["report"].items():
        print(key, r)
    total = sum(m["bytes"] for m in manifest.values())
    print(f"models: {len(manifest)} files, {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
