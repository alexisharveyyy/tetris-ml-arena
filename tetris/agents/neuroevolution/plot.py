"""Topology plot of a NEAT network (matplotlib; graphviz is not required)."""

from __future__ import annotations

from pathlib import Path

from tetris.agents.neuroevolution.agent import load_genome_checkpoint
from tetris.agents.neuroevolution.network import GenomeNet
from tetris.env.features import FEATURE_NAMES

POSITIVE = "#2b6cb0"
NEGATIVE = "#c53030"


def node_depths(net: GenomeNet) -> dict[int, int]:
    """Inputs at depth 0; every other node one past its deepest input."""
    depth = {k: 0 for k in net.inputs}
    for node in net.nodes:  # evaluation order is topological
        depth[node.key] = 1 + max((depth.get(i, 0) for i, _ in node.links), default=0)
    out_depth = max([depth[k] for k in net.outputs] + [1])
    for k in net.outputs:  # outputs drawn in the last column
        depth[k] = out_depth
    return depth


def plot_topology(checkpoint: str | Path, out: str | Path) -> Path:
    """Draw inputs (feature names), hidden nodes by depth and the output; returns ``out``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    net, meta = load_genome_checkpoint(checkpoint)
    depth = node_depths(net)
    columns: dict[int, list[int]] = {}
    for k, d in depth.items():
        columns.setdefault(d, []).append(k)
    pos = {}
    for d, keys in columns.items():
        keys.sort(reverse=d == 0)  # inputs top to bottom in feature order
        for j, k in enumerate(keys):
            pos[k] = (d, (len(keys) - 1) / 2 - j)
    weights = [abs(w) for n in net.nodes for _, w in n.links] or [1.0]
    w_max = max(weights)
    fig, ax = plt.subplots(figsize=(9, 7))
    for n in net.nodes:
        for i, w in n.links:
            if i not in pos:
                continue
            (x0, y0), (x1, y1) = pos[i], pos[n.key]
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=POSITIVE if w > 0 else NEGATIVE,
                lw=0.4 + 3.0 * abs(w) / w_max,
                alpha=0.75,
                zorder=1,
            )
    activation = {n.key: n.activation for n in net.nodes}
    for k, (x, y) in pos.items():
        kind = "input" if k in net.inputs else "output" if k in net.outputs else "hidden"
        color = {"input": "#e2e8f0", "hidden": "#faf089", "output": "#9ae6b4"}[kind]
        ax.scatter([x], [y], s=260, color=color, edgecolor="#2d3748", zorder=2)
        if kind == "input":
            ax.text(x - 0.08, y, FEATURE_NAMES[-k - 1], ha="right", va="center", fontsize=9)
        else:
            ax.text(x, y - 0.32, activation.get(k, ""), ha="center", va="top", fontsize=7)
    ax.set_title(
        f"NEAT genome, generation {meta.get('generation', '?')}: "
        f"{len(net.hidden)} hidden nodes, {net.n_connections} connections\n"
        "blue = positive weight, red = negative, width = |weight|",
        fontsize=10,
    )
    ax.axis("off")
    ax.set_xlim(-1.6, max(depth.values()) + 0.4)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
