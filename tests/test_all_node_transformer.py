"""All-residue feature semantics, attention, masking, and checkpoint round trips."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from test_pooled_attention_model import StubProteinEncoder, make_config, make_inputs
from torch import nn

from mutation_prediction.evaluation_models import load_checkpoint
from mutation_prediction.model import (
    AllNodeProteinMPNNFeatureExtractor,
    AllNodeTransformerHead,
    MutationPredictionModelPL,
    ProteinFeatures,
    build_model,
)
from mutation_prediction.model.train import parse_cfg


def transformer_config(**options):
    return parse_cfg(
        make_config(
            head="all_node_transformer",
            target_name="epsilon",
            transformer={
                "d_model": 16,
                "nhead": 4,
                "num_layers": 2,
                "dim_feedforward": 32,
                "dropout": 0.0,
                **options,
            },
        )
    )


class AllNodeTransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_features_keep_all_wt_nodes_and_separate_mutation_information(self):
        encoder = StubProteinEncoder()
        # The legacy mutant_embedding switch must not disable variant identity.
        cfg = transformer_config()
        cfg.model.mutant_embedding = False
        extractor = AllNodeProteinMPNNFeatureExtractor(cfg, protein_encoder=encoder)
        inputs = list(make_inputs())
        original = inputs[1].clone()
        inputs[1] = original.scatter(1, inputs[6], inputs[8])
        mutant_sequence = inputs[1].clone()
        inputs[0][0, 0, 0, 0] = float("nan")
        features = extractor(*inputs[:9])
        wt = encoder.W_s(original)
        context = inputs[4].unsqueeze(-1) * 10
        torch.testing.assert_close(
            features.nodes[..., :24],
            torch.cat([wt + context + 2, wt + context + 1, wt], dim=-1),
        )
        expected_delta = torch.zeros_like(wt).scatter(
            1,
            inputs[6].unsqueeze(-1).expand(-1, -1, 8),
            encoder.W_s(inputs[7]) - encoder.W_s(inputs[8]),
        )
        torch.testing.assert_close(features.nodes[..., 24:32], expected_delta)
        expected_indicator = torch.zeros_like(original).scatter(1, inputs[6], 1)
        torch.testing.assert_close(features.nodes[..., -1], expected_indicator.float())
        self.assertEqual(features.nodes.shape, (2, 4, extractor.output_dim))
        self.assertEqual(extractor.output_dim, 33)
        self.assertEqual(encoder.calls, 1)
        torch.testing.assert_close(encoder.last_sequence, original)
        torch.testing.assert_close(inputs[1], mutant_sequence)
        self.assertTrue(torch.isnan(inputs[0][0, 0, 0, 0]))
        self.assertTrue(torch.isfinite(encoder.last_coordinates).all())
        self.assertTrue(features.valid_mask.all())
        self.assertTrue(features.mutation_mask.all())

    def test_predictions_depend_on_mutant_identity_and_not_mutation_order(self):
        model = build_model(
            transformer_config(), protein_encoder=StubProteinEncoder()
        ).eval()
        for batch_size in (1, 2):
            for mutations in (1, 3):
                inputs = list(make_inputs(batch_size, mutations))
                prediction = model(*inputs)
                self.assertEqual(prediction.shape, (batch_size, 1))
                reordered = inputs.copy()
                for index in (6, 7, 8):
                    reordered[index] = inputs[index].flip(1)
                torch.testing.assert_close(prediction, model(*reordered))
                inputs[8] = (inputs[8] + 3) % 21
                self.assertFalse(torch.allclose(prediction, model(*inputs)))

    def test_nonmutation_nodes_influence_output_and_receive_gradients(self):
        head = AllNodeTransformerHead(
            7, d_model=16, nhead=4, num_layers=2, dim_feedforward=32, dropout=0.0
        ).eval()
        nodes = torch.randn(1, 5, 7, requires_grad=True)
        features = ProteinFeatures(
            nodes, torch.ones(1, 5, dtype=torch.bool), torch.tensor([[0, 1]])
        )
        prediction = head(features)
        prediction.sum().backward()
        # Later unmutated residues influence earlier mutation sites: no causal mask.
        self.assertTrue((nodes.grad[:, 2:].abs().sum(-1) > 0).all())
        changed = nodes.detach().clone()
        changed[:, -1] += torch.arange(7)
        self.assertFalse(
            torch.allclose(prediction, head(replace(features, nodes=changed)))
        )

    def test_padding_is_excluded_from_attention_and_readout(self):
        head = AllNodeTransformerHead(
            7, d_model=16, nhead=4, num_layers=2, dim_feedforward=32, dropout=0.0
        ).eval()
        nodes = torch.randn(2, 5, 7, requires_grad=True)
        valid = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, False]]
        )
        positions = torch.tensor([[0, 1, 4], [0, 1, 4]])
        features = ProteinFeatures(nodes, valid, positions)
        # Exercise both autograd and the inference fast path of PyTorch's blocks.
        for context in (torch.enable_grad, torch.inference_mode):
            with context():
                prediction = head(features)
                corrupted = (
                    nodes.detach()
                    .clone()
                    .masked_fill(~valid.unsqueeze(-1), float("nan"))
                )
                torch.testing.assert_close(
                    prediction, head(replace(features, nodes=corrupted))
                )
                for row, length in enumerate((3, 4)):
                    unpadded = ProteinFeatures(
                        nodes[row : row + 1, :length],
                        valid[row : row + 1, :length],
                        positions[row : row + 1, :2],
                    )
                    torch.testing.assert_close(
                        prediction[row : row + 1], head(unpadded)
                    )
        head(features).sum().backward()
        self.assertTrue(torch.isfinite(nodes.grad).all())
        self.assertEqual(nodes.grad[~valid].abs().sum().item(), 0)

    def test_padded_mutation_slots_do_not_overwrite_real_residue_zero(self):
        encoder = StubProteinEncoder()
        model = build_model(transformer_config(), protein_encoder=encoder).eval()
        inputs = list(make_inputs(batch_size=1, mutations=1))
        inputs[1] = torch.tensor([[5, 1, 2, 3]])
        inputs[6] = torch.tensor([[0]])
        inputs[7] = torch.tensor([[5]])
        inputs[8] = torch.tensor([[6]])
        expected = model(*inputs)
        for index in (6, 7, 8):
            inputs[index] = torch.cat(
                [inputs[index], torch.zeros(1, 1, dtype=torch.long)], 1
            )
        torch.testing.assert_close(expected, model(*inputs))
        torch.testing.assert_close(encoder.last_sequence, inputs[1])
        features = model.feature_extractor(*inputs[:9])
        torch.testing.assert_close(
            features.mutation_mask, torch.tensor([[True, False]])
        )
        for index in (6, 7, 8):
            inputs[index] = inputs[index].flip(1)
        torch.testing.assert_close(expected, model(*inputs))

    def test_examples_without_valid_mutations_fail_explicitly(self):
        model = build_model(transformer_config(), protein_encoder=StubProteinEncoder())
        for all_masked in (False, True):
            inputs = list(make_inputs())
            if all_masked:
                inputs[2].zero_()
            else:
                inputs[2].scatter_(1, inputs[6], 0)
            with self.assertRaisesRegex(ValueError, "at least one valid mutation"):
                model(*inputs)

    def test_layers_are_independently_initialized_and_dimensions_are_validated(self):
        model = build_model(transformer_config(), protein_encoder=StubProteinEncoder())
        a, b = model.prediction_head.blocks
        self.assertIsInstance(a, nn.TransformerEncoderLayer)
        self.assertIsNot(a, b)
        self.assertFalse(
            torch.equal(a.self_attn.in_proj_weight, b.self_attn.in_proj_weight)
        )
        for options in (
            {"d_model": 15},
            {"nhead": 0},
            {"num_layers": 0},
            {"dim_feedforward": 0},
        ):
            with self.assertRaises(ValueError):
                AllNodeTransformerHead(7, **options)

    def test_frozen_and_unfrozen_encoder_gradients_and_optimizer_coverage(self):
        for frozen in (False, True):
            cfg = transformer_config()
            # Injected encoder, not cfg.freeze_weights, owns requires_grad.
            cfg.model.freeze_weights = not frozen
            encoder = StubProteinEncoder().requires_grad_(not frozen)
            model = build_model(cfg, protein_encoder=encoder)
            module = MutationPredictionModelPL(cfg, model)
            inputs = make_inputs()
            nn.functional.mse_loss(module(*inputs), inputs[9]).backward()
            self.assertEqual(encoder.W_s.weight.grad is None, frozen)
            for name, parameter in model.prediction_head.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            optimizer = module.configure_optimizers()
            parameters = [
                p for group in optimizer.param_groups for p in group["params"]
            ]
            self.assertEqual(
                {id(p) for p in parameters},
                {id(p) for p in model.parameters() if p.requires_grad},
            )
            self.assertEqual(len(parameters), len({id(p) for p in parameters}))
            self.assertEqual(optimizer.param_groups[-1]["lr"], cfg.training.learn_rate)
            if not frozen:
                self.assertEqual(
                    optimizer.param_groups[0]["lr"], cfg.training.mpnn_learn_rate
                )
            before = model.prediction_head.input_projection.weight.detach().clone()
            optimizer.step()
            self.assertFalse(
                torch.equal(before, model.prediction_head.input_projection.weight)
            )

    def test_evaluator_reconstructs_both_pairings_and_saved_transformer_settings(self):
        encoder_path = (
            "mutation_prediction.thermompnn_d_vendor.thermompnn.thermompnn."
            "protein_mpnn_utils.ProteinMPNN"
        )
        for selection in ("pooled_attention", "all_node_transformer"):
            cfg = transformer_config(
                d_model=24,
                nhead=3,
                num_layers=3,
                dim_feedforward=40,
                activation="relu",
                norm_first=False,
            )
            cfg.model.head = selection
            model = build_model(cfg, protein_encoder=StubProteinEncoder()).eval()
            module = MutationPredictionModelPL(cfg, model)
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "model.ckpt"
                torch.save(
                    {
                        "hyper_parameters": dict(module.hparams),
                        "state_dict": module.state_dict(),
                    },
                    checkpoint,
                )
                with patch(
                    encoder_path, side_effect=lambda **kwargs: StubProteinEncoder()
                ):
                    restored = load_checkpoint({"checkpoint": str(checkpoint)}, "cpu")
            inputs = make_inputs()
            with torch.inference_mode():
                torch.testing.assert_close(model(*inputs), restored.model(*inputs))
            self.assertFalse(any(p.requires_grad for p in restored.model.parameters()))
            self.assertEqual(set(model.state_dict()), set(restored.model.state_dict()))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.model.state_dict()[name])
            if selection == "all_node_transformer":
                self.assertIsInstance(
                    restored.model.feature_extractor, AllNodeProteinMPNNFeatureExtractor
                )
                self.assertEqual(len(restored.model.prediction_head.blocks), 3)
                self.assertEqual(
                    restored.model.prediction_head.blocks[0].self_attn.num_heads, 3
                )

    def test_experiment_configs_preserve_default_and_save_transformer_defaults(self):
        root = Path(__file__).resolve().parents[1]
        original = parse_cfg(OmegaConf.load(root / "configs/model_train_epsilon.yaml"))
        cfg = parse_cfg(
            OmegaConf.load(root / "configs/model_train_epsilon_transformer.yaml")
        )
        self.assertEqual(original.model.head, "pooled_attention")
        self.assertEqual(cfg.model.head, "all_node_transformer")
        self.assertEqual(cfg.data.target_column, "epsilon_target")
        self.assertEqual(cfg.model.target_name, "epsilon")
        self.assertEqual(cfg.training.checkpoint_metric, "mse")
        self.assertLess(cfg.training.batch_size, original.training.batch_size)
        minimal = parse_cfg(make_config(head="all_node_transformer"))
        self.assertEqual(
            OmegaConf.to_container(minimal.model.transformer),
            OmegaConf.to_container(cfg.model.transformer),
        )


if __name__ == "__main__":
    unittest.main()
