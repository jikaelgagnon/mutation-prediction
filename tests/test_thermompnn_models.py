import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torch import nn

from mutation_prediction.thermo_mpnn_d.thermompnn.model.v2_model import (
    TransferModelv2,
    TransferModelv2Siamese,
)
from mutation_prediction.thermo_mpnn_d.thermompnn.inference.inference_utils import (
    get_metrics_full,
)
from mutation_prediction.thermo_mpnn_d.thermompnn.inference.run_inference import (
    inference,
)
from mutation_prediction.thermo_mpnn_d.thermompnn.trainer.v2_trainer import (
    TransferModelPLv2,
    TransferModelPLv2Siamese,
)


class StubProteinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequence_embedding = nn.Embedding(21, 128)

    def forward(self, coordinates, sequence, mask, chain_mask, residue_idx, chain_encoding):
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

    def test_siamese_model_allows_encoder_and_head_replacement(self):
        head = nn.Linear(256, 1)
        model = TransferModelv2Siamese(
            make_config(siamese=True),
            protein_encoder=self.encoder,
            prediction_head=head,
        )
        self.assertIs(model.prot_mpnn, self.encoder)
        self.assertIs(model.ddg_out, head)

        positions = torch.tensor([[0, 1], [1, 2]])
        wildtype = torch.gather(self.sequence, 1, positions)
        mutant = (wildtype + 1) % 21
        prediction_a, prediction_b = model(
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

        self.assertEqual(prediction_a.shape, (2, 1))
        self.assertEqual(prediction_b.shape, (2, 1))
        (prediction_a.mean() + prediction_b.mean()).backward()
        self.assertIsNotNone(head.weight.grad)
        self.assertIsNotNone(self.encoder.sequence_embedding.weight.grad)

    def test_double_mutant_subtraction_training_step_computes_loss(self):
        cfg = make_config()
        cfg.data.mut_types = ["double"]
        cfg.model.aggregation = "mean"
        cfg.model.lightattn = True
        cfg.model.freeze_weights = True
        cfg.training = {
            "learn_rate": 0.0001,
            "mpnn_learn_rate": None,
            "lr_schedule": True,
        }
        head = nn.Linear(128, 21)
        module = TransferModelPLv2(
            cfg,
            protein_encoder=self.encoder,
            prediction_head=head,
        )

        positions = torch.tensor([[0, 1], [1, 2]])
        wildtype = torch.gather(self.sequence, 1, positions)
        mutant = (wildtype + 1) % 21
        batch = (
            self.coordinates,
            self.sequence,
            self.mask,
            torch.tensor([3, 3]),
            self.chain_mask,
            self.chain_encoding,
            self.residue_idx,
            positions,
            wildtype,
            mutant,
            self.ddg,
            self.atom_mask,
        )
        loss = module.training_step(batch, 0)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(head.weight.grad)
        self.assertIn("lr_scheduler", module.configure_optimizers())

    def test_inference_metrics_default_to_cpu(self):
        metrics = get_metrics_full()
        self.assertTrue(all(metric.device.type == "cpu" for metric in metrics.values()))

    def test_inference_loads_the_matching_checkpoint_wrapper(self):
        for aggregation, wrapper in (
            (None, TransferModelPLv2),
            ("siamese", TransferModelPLv2Siamese),
        ):
            with self.subTest(aggregation=aggregation):
                cfg = OmegaConf.create(
                    {"data": {"dataset": "fixture"}, "model": {"aggregation": aggregation}}
                )
                args = SimpleNamespace(model="reference.ckpt", keep_preds=True)
                loaded = SimpleNamespace(model=nn.Identity())
                with (
                    patch(
                        "mutation_prediction.thermo_mpnn_d.thermompnn.inference.run_inference.load_v2_dataset",
                        return_value=object(),
                    ),
                    patch.object(wrapper, "load_from_checkpoint", return_value=loaded) as load,
                    patch(
                        "mutation_prediction.thermo_mpnn_d.thermompnn.inference.run_inference.run_prediction_batched",
                        return_value=[{"ddG_pred": 1.0}],
                    ),
                    patch("pandas.DataFrame.to_csv"),
                ):
                    result = inference(cfg, args)

                self.assertEqual(list(result.columns), ["ddG_pred"])
                load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
