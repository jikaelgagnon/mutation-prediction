"""Check feature semantics, interchangeable heads, and optimizer coverage."""

import unittest

import torch
from omegaconf import OmegaConf
from torch import nn

from mutation_prediction.model import (
    MutationPredictionModel,
    MutationPredictionModelPL,
    PooledAttentionHead,
    ProteinMPNNFeatureExtractor,
)


class StubProteinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_s = nn.Embedding(21, 8)
        self.calls = 0

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        self.calls += 1
        self.last_coordinates = X
        self.last_sequence = S
        embeddings = self.W_s(S)
        context = residue_idx.unsqueeze(-1) * 10
        hidden = [embeddings + context + i for i in (2, 1, 0)]
        return hidden, embeddings, None, None


def make_config(**model_options):
    return OmegaConf.create({
        "model": {
            "num_final_layers": 2,
            "mutant_embedding": True,
            "freeze_weights": False,
            **model_options,
        },
        "training": {
            "learn_rate": 0.001,
            "mpnn_learn_rate": 0.0001,
            "lr_schedule": False,
        },
    })


def make_inputs(batch_size=2, mutations=2):
    sequence = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])[:batch_size]
    positions = torch.tensor([[3, 0, 2], [1, 3, 0]])[:batch_size, :mutations]
    wt = sequence.gather(1, positions)
    return (
        torch.randn(batch_size, 4, 4, 3),
        sequence,
        torch.ones(batch_size, 4),
        torch.ones(batch_size, 4),
        torch.arange(4).expand(batch_size, -1),
        torch.ones(batch_size, 4, dtype=torch.long),
        positions,
        wt,
        (wt + 1) % 21,
        torch.ones(batch_size, 1),
        torch.zeros(batch_size, 4, 4),
    )


class MutationPredictionModelTests(unittest.TestCase):
    def test_features_preserve_layer_order_positions_and_optional_subtraction(self):
        for batch_size in (1, 2):
            for mutations in (1, 3):
                for subtract in (False, True):
                    for layers in (1, 2):
                        with self.subTest(
                            batch=batch_size, mutations=mutations,
                            subtract=subtract, layers=layers,
                        ):
                            encoder = StubProteinEncoder()
                            features = ProteinMPNNFeatureExtractor(
                                make_config(
                                    mutant_embedding=subtract, num_final_layers=layers
                                ),
                                protein_encoder=encoder,
                            )
                            inputs = make_inputs(batch_size, mutations)
                            output = features(*inputs[:9])
                            positions, wt, mutant = inputs[6:9]
                            wt_vectors = encoder.W_s(wt)
                            structural = [
                                wt_vectors + positions.unsqueeze(-1) * 10 + i
                                for i in (2, 1)[:layers]
                            ]
                            sequence = wt_vectors
                            if subtract:
                                sequence = sequence - encoder.W_s(mutant)
                            expected = torch.cat([*structural, sequence], dim=-1)
                            torch.testing.assert_close(output, expected)
                            self.assertEqual(output.shape, (
                                batch_size, mutations, features.output_dim
                            ))
                            self.assertEqual(encoder.calls, 1)

    def test_extractor_sanitizes_inputs_without_changing_callers_tensors(self):
        encoder = StubProteinEncoder()
        features = ProteinMPNNFeatureExtractor(make_config(), protein_encoder=encoder)
        inputs = list(make_inputs())
        inputs[0][0, 0, 0, 0] = float("nan")
        original_sequence = inputs[1].clone()
        inputs[7] = inputs[7] + 2
        features(*inputs[:9])
        self.assertTrue(torch.isnan(inputs[0][0, 0, 0, 0]))
        self.assertTrue(torch.isfinite(encoder.last_coordinates).all())
        torch.testing.assert_close(inputs[1], original_sequence)
        torch.testing.assert_close(
            encoder.last_sequence.gather(1, inputs[6]), inputs[7]
        )

    def test_pooling_is_permutation_invariant_and_handles_single_mutations(self):
        head = PooledAttentionHead(6)
        features = torch.randn(2, 3, 6)
        torch.testing.assert_close(head(features), head(features[:, [2, 0, 1]]))
        torch.testing.assert_close(
            head(features[:, :1]), head.final_layer(features[:, 0])
        )
        with torch.no_grad():
            head.attn_pool.weight.zero_()
            head.attn_pool.bias.zero_()
        torch.testing.assert_close(head(features), head.final_layer(features.mean(1)))

    def test_model_composes_the_extractor_and_attention_head(self):
        features = ProteinMPNNFeatureExtractor(
            make_config(), protein_encoder=StubProteinEncoder()
        )
        model = MutationPredictionModel(features, PooledAttentionHead(features.output_dim))
        inputs = make_inputs()
        self.assertIsInstance(model.feature_extractor, ProteinMPNNFeatureExtractor)
        self.assertIsInstance(model.prediction_head, PooledAttentionHead)
        torch.testing.assert_close(
            model(*inputs), model.prediction_head(model.feature_extractor(*inputs[:9]))
        )

    def test_custom_head_trains_with_frozen_or_trainable_extractor(self):
        for frozen in (False, True):
            with self.subTest(frozen=frozen):
                cfg = make_config(freeze_weights=frozen)
                encoder = StubProteinEncoder().requires_grad_(not frozen)
                features = ProteinMPNNFeatureExtractor(cfg, protein_encoder=encoder)
                head = nn.Sequential(
                    nn.Flatten(start_dim=1),
                    nn.Linear(2 * features.output_dim, 5),
                    nn.Tanh(),
                    nn.Linear(5, 1),
                )
                model = MutationPredictionModel(features, head)
                module = MutationPredictionModelPL(cfg, model=model)
                inputs = make_inputs()
                predictions = module(*inputs)
                self.assertEqual(predictions.shape, (2, 1))
                loss = nn.functional.mse_loss(predictions, inputs[9])
                loss.backward()
                self.assertTrue(all(p.grad is not None for p in head.parameters()))
                self.assertEqual(encoder.W_s.weight.grad is None, frozen)

                optimizer = module.configure_optimizers()
                optimized = {
                    id(p) for group in optimizer.param_groups for p in group["params"]
                }
                expected = {id(p) for p in model.parameters() if p.requires_grad}
                self.assertEqual(optimized, expected)
                self.assertEqual(
                    len(optimized),
                    sum(len(group["params"]) for group in optimizer.param_groups),
                )
                if not frozen:
                    self.assertEqual(optimizer.param_groups[0]["lr"], 0.0001)
                self.assertEqual(optimizer.param_groups[-1]["lr"], 0.001)
                before = head[-1].weight.detach().clone()
                optimizer.step()
                self.assertFalse(torch.equal(before, head[-1].weight))

    def test_composition_accepts_a_different_feature_extractor(self):
        class OtherFeatures(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Parameter(torch.randn(1, 2, 3))

            def forward(self, *inputs):
                return self.features.expand(inputs[0].shape[0], -1, -1)

        features = OtherFeatures()
        head = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(6, 1))
        model = MutationPredictionModel(features, head)
        # Custom extractors control freezing through requires_grad themselves.
        module = MutationPredictionModelPL(make_config(freeze_weights=True), model=model)
        inputs = make_inputs()
        prediction = module(*inputs)
        torch.testing.assert_close(prediction, head(features(*inputs[:9])))
        prediction.sum().backward()
        self.assertIsNotNone(features.features.grad)
        optimizer = module.configure_optimizers()
        self.assertIn(id(features.features), {
            id(p) for group in optimizer.param_groups for p in group["params"]
        })


if __name__ == "__main__":
    unittest.main()
