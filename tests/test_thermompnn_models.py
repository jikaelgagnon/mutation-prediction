from __future__ import annotations

import unittest

import torch
from jaxtyping import Float, Int, Real
from omegaconf import OmegaConf
from torch import Tensor, nn

from mutation_prediction.thermompnn_d_vendor.thermompnn.thermompnn.model.v2_model import (
    TransferModelv2,
)


class StubProteinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence_embedding = nn.Embedding(21, 128)

    def forward(
        self,
        coordinates: Float[Tensor, "batch length atoms 3"],
        sequence: Int[Tensor, "batch length"],
        mask: Real[Tensor, "batch length"],
        chain_mask: Real[Tensor, "batch length"],
        residue_idx: Int[Tensor, "batch length"],
        chain_encoding: Int[Tensor, "batch length"],
    ) -> tuple[
        list[Float[Tensor, "batch length hidden"]],
        Float[Tensor, "batch length hidden"],
        None,
        Float[Tensor, "batch length 1 hidden"],
    ]:
        embeddings = self.sequence_embedding(sequence)
        hidden = [embeddings, embeddings, embeddings]
        edges = embeddings.new_zeros((*embeddings.shape[:2], 1, embeddings.shape[-1]))
        return hidden, embeddings, None, edges


def make_config(*, siamese=False):
    return OmegaConf.create(
        {
            "data": {"side_chains": False},
            "model": {
                "aggregation": "siamese" if siamese else None,
                "num_final_layers": 2,
                "hidden_dims": [16],
                "lightattn": siamese,
                "dist": False,
                "side_chain_module": False,
                "mutant_embedding": False,
                "edges": False,
                "dropout": None,
                "single_target": siamese,
                "subtract_mut": not siamese,
            },
        }
    )


class ThermoMPNNModelTests(unittest.TestCase):
    def setUp(self):
        self.encoder = StubProteinEncoder()
        self.coordinates = torch.randn(2, 3, 4, 3)
        self.sequence = torch.tensor([[0, 1, 2], [1, 2, 3]])
        self.mask = torch.ones(2, 3)
        self.chain_mask = torch.ones(2, 3)
        self.residue_idx = torch.arange(3).expand(2, -1)
        self.chain_encoding = torch.ones(2, 3, dtype=torch.long)
        self.atom_mask = torch.ones(2, 3, 14)
        self.ddg = torch.randn(2, 1)

    def test_single_mutant_model_allows_encoder_and_head_replacement(self):
        head = nn.Linear(384, 21)
        model = TransferModelv2(
            make_config(),
            protein_encoder=self.encoder,
            prediction_head=head,
        )
        self.assertIs(model.prot_mpnn, self.encoder)
        self.assertIs(model.ddg_out, head)

        positions = torch.tensor([[0], [1]])
        wildtype = torch.gather(self.sequence, 1, positions)
        mutant = (wildtype + 1) % 21
        predictions, _ = model(
            self.coordinates,
            self.sequence,
            self.mask,
            self.chain_mask,
            self.residue_idx,
            self.chain_encoding,
            positions,
            wildtype,
            mutant,
            self.ddg,
            self.atom_mask,
        )

        self.assertEqual(predictions.shape, (2, 1))
        predictions.mean().backward()
        self.assertIsNotNone(head.weight.grad)
        self.assertIsNotNone(self.encoder.sequence_embedding.weight.grad)


if __name__ == "__main__":
    unittest.main()
