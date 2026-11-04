"""Prediction heads that consume per-mutation features [B, M, D]."""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor, nn


class PooledAttentionHead(nn.Module):
    """Pool mutation features with learned attention and predict one scalar."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.attn_pool = nn.Linear(input_dim, 1)
        self.final_layer = nn.Linear(input_dim, 1)

    def forward(
        self, final_embed: Float[Tensor, "batch mutations features"]
    ) -> Float[Tensor, "batch 1"]:
        # Normalize over mutations; a single mutation has weight one.
        weights = torch.softmax(self.attn_pool(final_embed), dim=1)
        pooled = torch.sum(weights * final_embed, dim=1)
        return self.final_layer(pooled)
