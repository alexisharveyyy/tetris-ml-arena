"""Topology plot of a NEAT network (matplotlib; graphviz is not required)."""

from __future__ import annotations

from pathlib import Path

from tetris.agents.neuroevolution.agent import load_genome_checkpoint
from tetris.agents.neuroevolution.network import GenomeNet
from tetris.env.features import FEATURE_NAMES

POSITIVE = "#2b6cb0"
NEGATIVE = "#c53030"
#: Minimum vertical gap (in node rows) between an edge and a node it passes.
CLEARANCE = 0.4


def node_depths(net: GenomeNet) -> dict[int, int]:
    """Inputs at depth 0; every other node one past its deepest input."""
    depth = {k: 0 for k in net.inputs}
    for node in net.nodes:  # evaluation order is topological
        depth[node.key] = 1 + max((depth.get(i, 0) for i, _ in node.links), default=0)
    out_depth = max([depth[k] for k in net.outputs] + [1])
    for k in net.outputs:  # outputs drawn in the last column
        depth[k] = out_depth
    return depth


def edge_path(
    start: tuple[float, float],
    end: tuple[float, float],
    obstacles: list[tuple[float, float]],
    samples: int = 60,
) -> tuple[list[float], list[float]]:
    """Points of an edge from ``start`` to ``end`` that keeps clear of ``obstacles``.

    A straight line when nothing is in the way. An edge that would pass
    through a node in a column it skips (for example an input wired straight
    to the output past a hidden node on the same row) is drawn as a quadratic
    curve, bent just enough to keep every such node :data:`CLEARANCE` away.
    """
    (x0, y0), (x1, y1) = start, end
    between = [(x, y) for x, y in obstacles if min(x0, x1) < x < max(x0, x1)]
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    for bend in [0.0] + [s * 0.5 * i for i in range(1, 25) for s in (1, -1)]:
        ts = [i / samples for i in range(samples + 1)]
        cy = my + 2 * bend  # control point; the curve's midpoint moves by ``bend``
        xs = [(1 - s) ** 2 * x0 + 2 * (1 - s) * s * mx + s**2 * x1 for s in ts]
        ys = [(1 - s) ** 2 * y0 + 2 * (1 - s) * s * cy + s**2 * y1 for s in ts]
        if all(_gap_at(xs, ys, x, y) >= CLEARANCE for x, y in between):
            return xs, ys
    return xs, ys


def _gap_at(xs: list[float], ys: list[float], x: float, y: float) -> float:
    """Vertical distance from the path to ``(x, y)`` at that column (x is monotonic)."""
    for i in range(1, len(xs)):
        if xs[i] >= x:
            f = (x - xs[i - 1]) / (xs[i] - xs[i - 1]) if xs[i] != xs[i - 1] else 0.0
            return abs(ys[i - 1] + f * (ys[i] - ys[i - 1]) - y)
    return abs(ys[-1] - y)


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
            xs, ys = edge_path(pos[i], pos[n.key], list(pos.values()))
            ax.plot(
                xs,
                ys,
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
