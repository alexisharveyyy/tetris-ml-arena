"""Neural networks for the DQN and PPO agents."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tetris.engine.board import HEIGHT, WIDTH
from tetris.engine.placement import NUM_ACTIONS
from tetris.rl.encoding import AFTERSTATE_EXTRA_DIM, OBS_EXTRA_DIM

#: Logit for illegal actions. Large and negative but finite, so softmax gives
#: exactly zero probability without producing NaN in log-probabilities or entropy.
MASKED_LOGIT = -1e9


class Conv3x3(nn.Conv2d):
    """3x3 convolution (padding 1) that runs as one matrix multiply on the CPU.

    PyTorch builds without oneDNN (such as the macOS arm64 wheels) fall back to
    a slow generic convolution on the CPU: about 300 ms for a batch of 3,200
    boards versus about 5 ms as a matmul through Accelerate. Other devices use
    the native kernel. Weights are a normal ``nn.Conv2d``, so checkpoints are
    interchangeable.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__(in_ch, out_ch, 3, stride=stride, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cpu":
            return super().forward(x)
        n, c, h, w = x.shape
        s = self.stride[0]
        ho, wo = (h - 1) // s + 1, (w - 1) // s + 1
        xp = nn.functional.pad(x, (1, 1, 1, 1))
        cols = torch.stack(
            [
                xp[:, :, i : i + s * (ho - 1) + 1 : s, j : j + s * (wo - 1) + 1 : s]
                for i in range(3)
                for j in range(3)
            ],
            dim=2,
        )  # (n, c, 9, ho, wo)
        cols = cols.permute(0, 3, 4, 1, 2).reshape(n * ho * wo, c * 9)
        out = cols @ self.weight.reshape(self.out_channels, c * 9).t() + self.bias
        return out.reshape(n, ho, wo, self.out_channels).permute(0, 3, 1, 2)


class BoardEncoder(nn.Module):
    """Small CNN over the 24x10 occupancy grid (about 0.5 MFLOP per board)."""

    def __init__(self, out_dim: int = 128, channels: tuple[int, int, int] = (16, 32, 32)) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.conv = nn.Sequential(
            Conv3x3(1, c1),
            nn.ReLU(),
            Conv3x3(c1, c2, stride=2),  # 12 x 5
            nn.ReLU(),
            Conv3x3(c2, c3, stride=2),  # 6 x 3
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, 1, HEIGHT, WIDTH)).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat, out_dim), nn.ReLU())

    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(boards.unsqueeze(1)))


class AfterstateValueNet(nn.Module):
    """V(afterstate): CNN over the board plus an MLP over features, next piece, hold, level."""

    def __init__(self, board_dim: int = 128, extra_dim: int = 64, hidden: int = 128) -> None:
        super().__init__()
        self.config = {"board_dim": board_dim, "extra_dim": extra_dim, "hidden": hidden}
        self.board = BoardEncoder(board_dim)
        self.extra = nn.Sequential(nn.Linear(AFTERSTATE_EXTRA_DIM, extra_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(board_dim + extra_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, boards: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.board(boards), self.extra(extras)], dim=-1)
        return self.head(h).squeeze(-1)


class ActorCritic(nn.Module):
    """PPO policy over the 80 placement actions plus a state-value head."""

    def __init__(self, board_dim: int = 128, extra_dim: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.config = {"board_dim": board_dim, "extra_dim": extra_dim, "hidden": hidden}
        self.board = BoardEncoder(board_dim)
        self.extra = nn.Sequential(nn.Linear(OBS_EXTRA_DIM, extra_dim), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(board_dim + extra_dim, hidden), nn.ReLU())
        self.policy = nn.Linear(hidden, NUM_ACTIONS)
        self.value = nn.Linear(hidden, 1)

    def forward(self, boards: torch.Tensor, extras: torch.Tensor, mask: torch.Tensor):
        """Masked logits ``(B, 80)`` and values ``(B,)``."""
        h = self.trunk(torch.cat([self.board(boards), self.extra(extras)], dim=-1))
        logits = self.policy(h).masked_fill(~mask, MASKED_LOGIT)
        return logits, self.value(h).squeeze(-1)


_BIT_SHIFTS: dict[torch.device, torch.Tensor] = {}


def unpack_boards_torch(packed: torch.Tensor) -> torch.Tensor:
    """Unpack ``(..., 30)`` uint8 boards (``np.packbits`` order) to float ``(..., 24, 10)``.

    Runs on the tensor's device, so only the packed bytes cross to the GPU.
    """
    dev = packed.device
    if dev not in _BIT_SHIFTS:
        _BIT_SHIFTS[dev] = torch.arange(7, -1, -1, device=dev, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> _BIT_SHIFTS[dev]) & 1
    return bits.reshape(*packed.shape[:-1], HEIGHT, WIDTH).float()


def to_cpu(obj: Any) -> Any:
    """Recursively move tensors to the CPU (MPS tensors save very slowly)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)
    return obj


def masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probabilities and entropy over legal actions only.

    Illegal actions contribute exactly zero to the entropy (no ``0 * -inf``).
    """
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    entropy = -torch.where(mask, p * logp, torch.zeros_like(p)).sum(-1)
    return logp, entropy


def build(kind: str, config: dict[str, Any]) -> nn.Module:
    if kind == "dqn":
        return AfterstateValueNet(**config)
    if kind == "ppo":
        return ActorCritic(**config)
    raise ValueError(kind)


def select_device(preference: str = "auto") -> torch.device:
    """``"auto"`` picks CUDA, then MPS, then CPU; otherwise the named device."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path, kind: str, model: nn.Module, **meta: Any) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    tmp = path.with_suffix(".tmp")
    torch.save({"kind": kind, "config": model.config, "state_dict": state, **meta}, tmp)
    tmp.replace(path)


def load_checkpoint(path, kind: str, device: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    data = torch.load(path, map_location=device, weights_only=False)
    if data.get("kind") != kind:
        raise ValueError(f"{path} is a {data.get('kind')} checkpoint, expected {kind}")
    model = build(kind, data["config"])
    model.load_state_dict(data["state_dict"])
    model.to(device).eval()
    return model, data
