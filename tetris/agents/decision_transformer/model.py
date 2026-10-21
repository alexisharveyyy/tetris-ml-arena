"""Decision Transformer with afterstate matching.

The sequence is the standard one, three tokens per piece:
``(return-to-go, state, action)``. A causal transformer reads it, and the
output at each state token predicts that piece's action.

The action head differs from the original 80-way classifier. Raw-action PPO
showed that learning the 80 abstract actions from the board fails here
(held-out fitness 0.103). Instead, every legal placement is represented by an
embedding of its afterstate (board, features, next piece, hold, level), and an
action's logit is the scaled dot product of the state token's output with
that embedding. Action tokens in the sequence are the embedding of the
afterstate actually chosen. The policy is still a distribution over the same
80-action space (illegal actions masked), like PPO and BC.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from tetris.rl.encoding import AFTERSTATE_EXTRA_DIM, OBS_EXTRA_DIM
from tetris.rl.nets import BoardEncoder, score_afterstate_rows, unpack_boards_torch


class AfterstateEmbedder(nn.Module):
    """Embedding ``(N, d_model)`` of afterstates: board CNN plus an MLP over the extras."""

    def __init__(self, d_model: int, board_dim: int, extra_dim: int) -> None:
        super().__init__()
        self.board = BoardEncoder(board_dim)
        self.extra = nn.Sequential(nn.Linear(AFTERSTATE_EXTRA_DIM, extra_dim), nn.ReLU())
        self.out = nn.Sequential(
            nn.Linear(board_dim + extra_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

    def forward(self, boards: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        return self.out(torch.cat([self.board(boards), self.extra(extras)], dim=-1))


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        context: int = 20,
        max_timestep: int = 512,
        board_dim: int = 128,
        extra_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "context": context,
            "max_timestep": max_timestep,
            "board_dim": board_dim,
            "extra_dim": extra_dim,
            "dropout": dropout,
        }
        self.d_model = d_model
        self.n_heads = n_heads
        self.context = context
        self.max_timestep = max_timestep
        # Named "scorer" so the MPS pre-flight treats it as the chunked part.
        self.scorer = AfterstateEmbedder(d_model, board_dim, extra_dim)
        self.state_board = BoardEncoder(board_dim)
        self.state_extra = nn.Sequential(nn.Linear(OBS_EXTRA_DIM, extra_dim), nn.ReLU())
        self.state_proj = nn.Linear(board_dim + extra_dim, d_model)
        self.rtg_proj = nn.Linear(1, d_model)
        self.action_proj = nn.Linear(d_model, d_model)
        self.timestep = nn.Embedding(max_timestep, d_model)
        self.token_type = nn.Embedding(3, d_model)
        self.embed_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)
        self.query = nn.Linear(d_model, d_model)

    def embed_afterstates(self, boards_packed: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
        """``(N, d_model)`` for packed afterstate boards ``(N, 30)`` (chunked)."""
        return score_afterstate_rows(self.scorer, boards_packed, extras)

    def attention_mask(self, valid: torch.Tensor) -> torch.Tensor:
        """Float mask ``(B * heads, 3K, 3K)``: causal, padding hidden.

        A padded token may still attend to itself, so no row is fully masked
        (which would give NaN). Valid tokens never see padding.
        """
        b, k = valid.shape
        length = 3 * k
        tok_valid = valid.repeat_interleave(3, dim=1)  # (B, 3K)
        causal = torch.ones(length, length, dtype=torch.bool, device=valid.device).tril()
        allowed = causal.unsqueeze(0) & tok_valid.unsqueeze(1)
        allowed = allowed | torch.eye(length, dtype=torch.bool, device=valid.device)
        mask = torch.zeros(b, length, length, device=valid.device)
        mask = mask.masked_fill(~allowed, float("-inf"))
        return mask.repeat_interleave(self.n_heads, dim=0)

    def forward(
        self,
        rtg: torch.Tensor,
        obs_boards: torch.Tensor,
        obs_extras: torch.Tensor,
        action_emb: torch.Tensor,
        timesteps: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Action queries ``(B, K, d_model)`` at every state token.

        ``rtg`` ``(B, K)``; ``obs_boards`` packed ``(B, K, 30)``; ``obs_extras``
        ``(B, K, 62)``; ``action_emb`` ``(B, K, d_model)`` (afterstate
        embeddings of the chosen actions; the last one is never seen by the
        prediction it would answer); ``timesteps`` ``(B, K)``; ``valid``
        ``(B, K)`` bool, False for left padding.
        """
        b, k = rtg.shape
        boards = unpack_boards_torch(obs_boards.reshape(b * k, -1))
        state = self.state_proj(
            torch.cat(
                [self.state_board(boards), self.state_extra(obs_extras.reshape(b * k, -1).float())],
                dim=-1,
            )
        ).reshape(b, k, -1)
        time = self.timestep(timesteps.clamp(0, self.max_timestep - 1))
        types = self.token_type.weight
        tokens = torch.stack(
            [
                self.rtg_proj(rtg.unsqueeze(-1)) + time + types[0],
                state + time + types[1],
                self.action_proj(action_emb) + time + types[2],
            ],
            dim=2,
        ).reshape(b, 3 * k, -1)
        h = self.drop(self.embed_norm(tokens))
        h = self.blocks(h, mask=self.attention_mask(valid))
        h = self.final_norm(h).reshape(b, k, 3, -1)[:, :, 1]
        return self.query(h)

    def action_logits(self, queries: torch.Tensor, row_emb: torch.Tensor) -> torch.Tensor:
        """Logits ``(N,)`` of afterstate rows given each row's state query ``(N, d)``."""
        return (queries * row_emb).sum(-1) / math.sqrt(self.d_model)


def build_dt(config: dict[str, Any]) -> DecisionTransformer:
    return DecisionTransformer(**config)
