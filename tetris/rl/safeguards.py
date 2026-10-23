"""Training safeguards against silent accelerator bugs.

The first afterstate PPO run trained for 3 hours on MPS without learning:
convolution gradients were wrong at its batch size and nothing failed (see
reports/training/ppo_mps_bug.md). Two checks now guard every training run.

* :func:`preflight` runs before training on a non-CPU device. It builds the
  run's network twice from the same weights, computes the run's real loss on
  one batch of the run's real size on the device and on the CPU, and compares
  the full gradient vectors. Beyond the tolerance it prints a loud warning,
  then tries smaller scorer chunks (for chunked models) and finally falls back
  to the CPU. The result is written to the run directory.
* :class:`ProgressWatchdog` aborts a run whose policy (KL per update) or
  parameters (relative change per check) stay at about zero for ``patience``
  consecutive updates.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

#: Largest accepted relative difference ``|g_dev - g_cpu| / |g_cpu|`` over the
#: whole gradient vector. Healthy runs measure 1e-5 to 1e-4; the MPS bug was 6e8.
PREFLIGHT_TOLERANCE = 1e-2
#: Scorer chunk sizes tried, in order, when a chunked model fails the pre-flight.
FALLBACK_CHUNKS = (2048, 1024, 512)

#: Loss for the pre-flight: ``loss_fn(net, batch_on_device) -> scalar tensor``.
LossFn = Callable[[torch.nn.Module, list[torch.Tensor]], torch.Tensor]


@dataclass
class PreflightResult:
    name: str
    device: str
    batch_description: str
    attempts: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = False
    final_device: str = "cpu"
    scorer_chunk: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _flat_grad(net: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten().cpu()
            for p in net.parameters()
        ]
    )


def gradient_difference(
    net: torch.nn.Module, batch: list[torch.Tensor], loss_fn: LossFn, device: torch.device
) -> tuple[float, float]:
    """Relative gradient difference between ``device`` and the CPU, and the seconds it took."""
    import copy

    t0 = time.perf_counter()
    cpu_net = copy.deepcopy(net).cpu()
    dev_net = copy.deepcopy(net).to(device)
    grads = []
    for model, dev in ((cpu_net, torch.device("cpu")), (dev_net, device)):
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model, [x.to(dev) for x in batch])
        loss.backward()
        grads.append(_flat_grad(model))
    g_cpu, g_dev = grads
    if not torch.isfinite(g_dev).all():
        return float("inf"), time.perf_counter() - t0
    rel = float((g_cpu - g_dev).norm() / g_cpu.norm().clamp_min(1e-30))
    return rel, time.perf_counter() - t0


def preflight(
    name: str,
    net: torch.nn.Module,
    batch: list[torch.Tensor],
    loss_fn: LossFn,
    device: torch.device,
    batch_description: str,
    run_dir: Path | None = None,
    tolerance: float = PREFLIGHT_TOLERANCE,
) -> PreflightResult:
    """Check gradients on ``device`` against the CPU; fall back to chunking or the CPU.

    Returns the device to train on in ``final_device`` (and the scorer chunk
    size for chunked models). Nothing to check on the CPU itself.
    """
    from tetris.rl import nets

    result = PreflightResult(name, str(device), batch_description)
    if device.type == "cpu":
        result.passed, result.final_device = True, "cpu"
        return _finish(result, run_dir)
    chunked = hasattr(net, "scorer")
    chunks: list[int | None] = [nets.SCORER_CHUNK if chunked else None]
    if chunked:
        chunks += [c for c in FALLBACK_CHUNKS if c < nets.SCORER_CHUNK]
    for chunk in chunks:
        if chunk is not None:
            nets.SCORER_CHUNK = chunk
        rel, seconds = gradient_difference(net, batch, loss_fn, device)
        ok = rel <= tolerance
        result.attempts.append(
            {"scorer_chunk": chunk, "relative_difference": rel, "seconds": seconds, "ok": ok}
        )
        print(
            f"pre-flight {name} on {device}: {batch_description}"
            + (f", scorer chunk {chunk}" if chunk else "")
            + f": relative gradient difference {rel:.2e} (tolerance {tolerance:g}, {seconds:.1f}s)"
            + ("" if ok else "  ** FAILED **"),
            flush=True,
        )
        if ok:
            result.passed, result.final_device, result.scorer_chunk = True, str(device), chunk
            return _finish(result, run_dir)
        _loud(f"{name}: {device} gradients disagree with the CPU (relative difference {rel:.2e})")
    _loud(f"{name}: no {device} configuration passed; TRAINING FALLS BACK TO THE CPU")
    result.final_device = "cpu"
    if chunked:
        nets.SCORER_CHUNK = chunks[0]
        result.scorer_chunk = chunks[0]
    return _finish(result, run_dir)


def _loud(message: str) -> None:
    bar = "!" * 78
    print(f"{bar}\n!! PRE-FLIGHT: {message}\n{bar}", flush=True)


def _finish(result: PreflightResult, run_dir: Path | None) -> PreflightResult:
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "preflight.json").write_text(json.dumps(result.to_dict(), indent=2))
    return result


class StalledTrainingError(RuntimeError):
    """Raised by :class:`ProgressWatchdog` when training makes no progress."""


class ProgressWatchdog:
    """Aborts when a progress signal stays at about zero for ``patience`` updates.

    Policy models report the per-update KL between the rollout policy and the
    updated policy (always >= 0 with the ``(r - 1) - log r`` estimator); other
    models report the relative parameter change since the last check.
    """

    def __init__(self, name: str, signal: str, threshold: float, patience: int) -> None:
        self.name = name
        self.signal = signal
        self.threshold = threshold
        self.patience = patience
        self.stalled = 0

    def update(self, value: float, scale: float = 1.0) -> None:
        """``scale`` multiplies the threshold (PPO passes (lr / lr0)^2, since KL per
        update shrinks roughly with the square of the step size)."""
        if value != value or abs(value) < self.threshold * scale:  # NaN counts as stalled
            self.stalled += 1
        else:
            self.stalled = 0
        if self.stalled >= self.patience:
            raise StalledTrainingError(
                f"{self.name}: {self.signal} below {self.threshold:g} for {self.patience} "
                f"consecutive updates (last {value:.3g}); the model is not training"
            )


class ParameterChange:
    """Relative parameter change ``|theta - theta_prev| / |theta_prev|`` between calls."""

    def __init__(self, net: torch.nn.Module) -> None:
        self.net = net
        self.prev = self._flat()

    def _flat(self) -> torch.Tensor:
        with torch.no_grad():
            return torch.cat([p.detach().flatten().float().cpu() for p in self.net.parameters()])

    def __call__(self) -> float:
        now = self._flat()
        change = float((now - self.prev).norm() / self.prev.norm().clamp_min(1e-30))
        self.prev = now
        return change
