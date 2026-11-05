"""Verify checkpoint loading and single-pass single-mutant inference offline."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mutation_prediction.data.mutations import Mutation
from mutation_prediction.data.collate import (
    tied_featurize_mut,
)
from mutation_prediction.vendor.thermompnn.thermompnn.model.v2_model import TransferModelv2
from mutation_prediction.vendor.thermompnn.thermompnn.protein_mpnn_utils import ProteinMPNN
from mutation_prediction.thermompnn import (
    ALPHABET,
    ThermoMPNNPredictor,
    _single_mutant_config,
)


def make_protein():
    coordinates = np.random.default_rng(42).normal(size=(4, 4, 3))
    return {
        "name": "test_protein",
        "seq": "ACDE",
        "seq_chain_A": "ACDE",
        "num_of_chains": 1,
        "coords_chain_A": {
            f"{atom}_chain_A": coordinates[:, index].tolist()
            for index, atom in enumerate(("N", "CA", "C", "O"))
        },
    }


class ThermoMPNNInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.directory.name) / "single.ckpt"
        # Construct a complete checkpoint offline; no network or pretrained
        # weights are required for unit tests.
        encoder = ProteinMPNN(
            num_letters=21, node_features=128, edge_features=128, hidden_dim=128,
            num_encoder_layers=3, num_decoder_layers=3, k_neighbors=48,
            augment_eps=0, dropout=0,
        )
        cls.reference = TransferModelv2(
            _single_mutant_config(), protein_encoder=encoder
        ).eval()
        torch.save({"state_dict": {
            f"model.{key}": value for key, value in cls.reference.state_dict().items()
        }}, cls.checkpoint)
        cls.predictor = ThermoMPNNPredictor(cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_saturation_predictions_match_the_standard_single_mutant_forward(self):
        protein = make_protein()
        protein["mutation"] = Mutation([3], ["E"], ["V"], 4.0)
        annotation = protein["mutation"]
        predictions = self.predictor.predict(protein)
        self.assertEqual(predictions.shape, (4, 20))
        self.assertEqual(predictions.device.type, "cpu")
        self.assertFalse(predictions.requires_grad)
        self.assertTrue(torch.isfinite(predictions).all())
        self.assertIs(protein["mutation"], annotation)
        self.assertTrue(all(not p.requires_grad for p in self.predictor.model.parameters()))

        for position, mutant in ((0, "W"), (2, "G")):
            with self.subTest(position=position, mutant=mutant):
                record = dict(protein)
                record["mutation"] = Mutation(
                    [position], [protein["seq"][position]], [mutant], 0.0
                )
                batch = tied_featurize_mut([record])
                X, S, mask, _, chain_M, chain_ids, residue_idx, pos, wt, mut, ddg, atom_mask = batch
                with torch.inference_mode():
                    expected, _ = self.reference(
                        X, S, mask, chain_M, residue_idx, chain_ids,
                        pos, wt, mut, ddg, atom_mask,
                    )
                torch.testing.assert_close(
                    predictions[position, ALPHABET.index(mutant)], expected[0, 0]
                )
        torch.testing.assert_close(
            predictions[torch.arange(4), torch.tensor([0, 1, 2, 3])], torch.zeros(4)
        )
        torch.testing.assert_close(self.predictor.predict(protein), predictions, rtol=0, atol=0)

    def test_unknown_and_missing_backbone_rows_are_nan(self):
        protein = make_protein()
        protein["seq"] = protein["seq_chain_A"] = "ACXE"
        protein["coords_chain_A"]["CA_chain_A"][1] = [float("nan")] * 3
        predictions = self.predictor.predict(protein)
        self.assertTrue(torch.isnan(predictions[1:3]).all())
        self.assertTrue(torch.isfinite(predictions[[0, 3]]).all())
        self.assertNotIn("mutation", protein)

    def test_empty_or_mismatched_sequences_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonempty sequence"):
            self.predictor.predict({"seq": ""})
        protein = make_protein()
        protein["seq"] = "EDCA"
        with self.assertRaisesRegex(ValueError, "concatenated chain sequences"):
            self.predictor.predict(protein)

    def test_incompatible_checkpoint_is_rejected(self):
        path = Path(self.directory.name) / "wrong.ckpt"
        torch.save({"state_dict": {"model.wrong.weight": torch.ones(1)}}, path)
        with self.assertRaisesRegex(ValueError, "single-mutant ThermoMPNN"):
            ThermoMPNNPredictor(path)

    def test_pdb_output_includes_all_nonidentity_substitutions_and_sequence_indices(self):
        path = Path(self.directory.name) / "numbered.pdb"
        coordinates = make_protein()["coords_chain_A"]
        lines = []
        serial = 1
        for index, name in enumerate(("ALA", "CYS", "ASP", "GLU")):
            for atom in ("N", "CA", "C", "O"):
                x, y, z = coordinates[f"{atom}_chain_A"][index]
                lines.append(
                    f"ATOM  {serial:5d} {atom:^4s} {name:>3s} A{index + 10:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}          {atom[0]:>2s}\n"
                )
                serial += 1
        path.write_text("".join(lines) + "END\n")
        predictions = self.predictor.predict_pdb(path, chain="A")
        self.assertEqual(len(predictions), 4 * 19)
        self.assertEqual(set(predictions.position), {0, 1, 2, 3})
        self.assertIn("A1C", set(predictions.mutation))
        self.assertTrue((predictions.wildtype != predictions.mutant).all())
        self.assertTrue(np.isfinite(predictions.ddg).all())
        with self.assertRaisesRegex(ValueError, "chain 'B'"):
            self.predictor.predict_pdb(path, chain="B")


if __name__ == "__main__":
    unittest.main()
