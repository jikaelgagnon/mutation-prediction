"""Prediction heads for mutation tensors or structured all-residue features."""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor, nn

from mutation_prediction.model.feature_extraction import ProteinFeatures


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


class AllNodeTransformerHead(nn.Module):
    """Bidirectional all-residue attention, then masked mutation-site pooling."""

    def __init__(
        self,
        input_dim: int,
        *,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
    ):
        super().__init__()
        if d_model <= 0 or nhead <= 0 or d_model % nhead:
            raise ValueError("d_model must be positive and divisible by positive nhead")
        if num_layers <= 0 or dim_feedforward <= 0:
            raise ValueError("num_layers and dim_feedforward must be positive")
        self.input_projection = nn.Linear(input_dim, d_model)
        # Construct separately: TransformerEncoder would clone identical weights.
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=norm_first,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.attn_pool = nn.Linear(d_model, 1)
        self.final_layer = nn.Linear(d_model, 1)

    def forward(self, features: ProteinFeatures) -> Float[Tensor, "batch 1"]:
        valid = features.valid_mask.bool()
        mutation_mask = valid.gather(1, features.mutation_positions)
        if features.mutation_mask is not None:
            mutation_mask = mutation_mask & features.mutation_mask.bool()
        if not mutation_mask.any(dim=1).all():
            raise ValueError("Each example requires at least one valid mutation site")

        # Zero invalid inputs before projection, including possible NaN padding.
        x = self.input_projection(features.nodes.masked_fill(~valid.unsqueeze(-1), 0))
        for block in self.blocks:
            x = block(x, src_key_padding_mask=~valid)
        x = self.final_norm(x)
        positions = features.mutation_positions.unsqueeze(-1).expand(
            -1, -1, x.shape[-1]
        )
        mutations = x.gather(1, positions).masked_fill(~mutation_mask.unsqueeze(-1), 0)
        scores = self.attn_pool(mutations).masked_fill(
            ~mutation_mask.unsqueeze(-1), -torch.inf
        )
        weights = torch.softmax(scores, dim=1)
        return self.final_layer(torch.sum(weights * mutations, dim=1))
