"""Extract per-mutation features from one wild-type ProteinMPNN pass."""

from __future__ import annotations

import torch
from jaxtyping import Float, Int, Real
from torch import Tensor, nn

from mutation_prediction.thermo_mpnn_d.thermompnn.model.modules import (
    get_protein_mpnn,
)


def _check_sequence_match(
    S: Int[Tensor, "batch length"],
    wt: Int[Tensor, "batch mutations"],
    mut: Int[Tensor, "batch mutations"],
    pos: Int[Tensor, "batch mutations"],
) -> Int[Tensor, "batch length"]:
    """Check WT identities on a clone; mut is unused in the WT encoder pass."""
    S = S.clone()
    for mut_idx in range(wt.shape[-1]):
        S_check = torch.gather(S, -1, pos[..., mut_idx, None])
        if torch.sum(S_check - wt[..., mut_idx]) != 0:
            S.scatter_(
                dim=1,
                index=pos[..., mut_idx, None],
                src=wt[..., mut_idx, None],
            )
    return S


class ProteinMPNNFeatureExtractor(nn.Module):
    """Return final_embed [B, M, D] for the M mutated residues.

    Each vector concatenates the last K decoder states with the WT amino-acid
    embedding, optionally minus the mutant embedding. D = E * (K + 1), where
    E is the ProteinMPNN embedding width (128 for the pretrained encoder).
    """

    def __init__(self, cfg, *, protein_encoder: nn.Module | None = None):
        super().__init__()
        self.num_final_layers = cfg.model.num_final_layers
        self.mutant_embedding = cfg.model.mutant_embedding
        if self.num_final_layers <= 0:
            raise ValueError("num_final_layers must be positive")
        self.prot_mpnn = (
            protein_encoder if protein_encoder is not None else get_protein_mpnn(cfg)
        )
        self.output_dim = self.prot_mpnn.W_s.embedding_dim * (self.num_final_layers + 1)

    def forward(
        self,
        X: Float[Tensor, "batch length atoms 3"],
        S: Int[Tensor, "batch length"],
        mask: Real[Tensor, "batch length"],
        chain_M: Real[Tensor, "batch length"],
        residue_idx: Int[Tensor, "batch length"],
        chain_encoding_all: Int[Tensor, "batch length"],
        mut_positions: Int[Tensor, "batch mutations"],
        mut_wildtype_AAs: Int[Tensor, "batch mutations"],
        mut_mutant_AAs: Int[Tensor, "batch mutations"],
    ) -> Float[Tensor, "batch mutations features"]:
        S = _check_sequence_match(S, mut_wildtype_AAs, mut_mutant_AAs, mut_positions)
        X = torch.nan_to_num(X, nan=0.0)
        hidden_states, wt_embed, _, _ = self.prot_mpnn(
            X, S, mask, chain_M, residue_idx, chain_encoding_all
        )
        if self.num_final_layers > len(hidden_states):
            raise ValueError("num_final_layers exceeds the encoder's decoder layers")

        # ProteinMPNN returns its decoder states last layer first.
        hidden = torch.cat(hidden_states[:self.num_final_layers], dim=-1)
        positions = mut_positions.unsqueeze(-1)
        struct_embed = hidden.gather(1, positions.expand(-1, -1, hidden.shape[-1]))
        seq_embed = wt_embed.gather(1, positions.expand(-1, -1, wt_embed.shape[-1]))
        if self.mutant_embedding:
            seq_embed = seq_embed - self.prot_mpnn.W_s(mut_mutant_AAs)

        final_embed = torch.cat([struct_embed, seq_embed], dim=-1)
        return final_embed
