"""Vectorized evaluation of NEAT genomes, and their JSON form.

neat-python's ``FeedForwardNetwork`` evaluates one input row at a time in
pure Python. An agent scores every legal afterstate (about 35 to 70 rows) per
decision, so this module evaluates the same network over all rows at once
with NumPy. It reproduces neat-python's activation functions exactly
(including their input scaling and clamping); a test compares the two.

A genome is stored as plain JSON (``GenomeNet.to_dict``), so loading an agent
needs neither neat-python nor a pickle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(5.0 * z, -60.0, 60.0)))


def _tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(np.clip(2.5 * z, -60.0, 60.0))


ACTIVATIONS = {
    "identity": lambda z: z,
    "tanh": _tanh,
    "sigmoid": _sigmoid,
    "relu": lambda z: np.maximum(z, 0.0),
    "clamped": lambda z: np.clip(z, -1.0, 1.0),
}


@dataclass
class NodeEval:
    key: int
    activation: str
    bias: float
    response: float
    links: list[tuple[int, float]]


@dataclass
class GenomeNet:
    """A feed-forward network in evaluation order (inputs have negative keys)."""

    inputs: list[int]
    outputs: list[int]
    nodes: list[NodeEval]

    @classmethod
    def from_genome(cls, genome: Any, config: Any) -> GenomeNet:
        """Build from a neat-python genome, in the order ``FeedForwardNetwork.create`` uses."""
        from neat.graphs import feed_forward_layers

        gc = config.genome_config
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]
        layers, required = feed_forward_layers(gc.input_keys, gc.output_keys, connections)
        usable = required | set(gc.input_keys)
        nodes = []
        for layer in layers:
            for node in layer:
                links = [
                    (i, float(genome.connections[(i, o)].weight))
                    for i, o in connections
                    if o == node and i in usable
                ]
                ng = genome.nodes[node]
                if ng.aggregation != "sum":
                    raise ValueError(f"unsupported aggregation {ng.aggregation}")
                nodes.append(
                    NodeEval(node, ng.activation, float(ng.bias), float(ng.response), links)
                )
        return cls(list(gc.input_keys), list(gc.output_keys), nodes)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Outputs for input rows ``x`` of shape ``(n, num_inputs)``; returns ``(n,)``
        for one output, else ``(n, num_outputs)``."""
        x = np.asarray(x, dtype=np.float64)
        values = {k: x[:, i] for i, k in enumerate(self.inputs)}
        zero = np.zeros(len(x))
        for node in self.nodes:
            s = zero.copy()
            for i, w in node.links:
                s += values.get(i, zero) * w
            values[node.key] = ACTIVATIONS[node.activation](node.bias + node.response * s)
        out = np.stack([values.get(k, zero) for k in self.outputs], axis=1)
        return out[:, 0] if out.shape[1] == 1 else out

    # -------------------------------------------------------------- storage

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "nodes": [
                {
                    "key": n.key,
                    "activation": n.activation,
                    "bias": n.bias,
                    "response": n.response,
                    "links": [[i, w] for i, w in n.links],
                }
                for n in self.nodes
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GenomeNet:
        return cls(
            list(d["inputs"]),
            list(d["outputs"]),
            [
                NodeEval(
                    int(n["key"]),
                    n["activation"],
                    float(n["bias"]),
                    float(n["response"]),
                    [(int(i), float(w)) for i, w in n["links"]],
                )
                for n in d["nodes"]
            ],
        )

    @property
    def hidden(self) -> list[int]:
        return [n.key for n in self.nodes if n.key not in self.outputs]

    @property
    def n_connections(self) -> int:
        return sum(len(n.links) for n in self.nodes)
