"""Cluster-only regression checks through the public evaluation interface."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from mutation_prediction.evaluate import evaluate


def label_free_predictor(spec, device):
    """Example plugin: assert labels are hidden, then return a constant."""

    def predict(batch):
        if torch.count_nonzero(batch[10]).item():
            raise AssertionError("Experimental targets were passed into inference")
        return torch.full((len(batch[0]), 1), spec.get("value", 0.0), device=device)

    return predict


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = self.root / "test.csv"
        self.cache = self.root / "epsilon.csv"
        self.predictions = self.root / "predictions.csv"
        self.frame = pd.DataFrame(
            {
                "WT_name": ["first.pdb", "first.pdb", "second.pdb"],
                "mut_type": ["A1V:D3G", "C2A:E4Y", "A1G:E4K"],
                "aa_seq": ["VCGE", "AADY", "GCDK"],
                "ddG_ML": [0.0, 1.0, 2.0],
                "split": ["test"] * 3,
                "baseline_ddg": [10.0, 20.0, 30.0],
            }
        )
        self.frame.to_csv(self.reference, index=False)
        extra = self.frame.iloc[:1].assign(WT_name="train_only.pdb", split="train")
        pd.concat([self.frame, extra]).to_csv(self.cache, index=False)
        self.frame[["WT_name", "mut_type"]].assign(
            predicted_epsilon=[-10.0, -18.0, -29.0]
        ).iloc[::-1].to_csv(self.predictions, index=False)
        self.config = {
            "batch_size": 2,
            "datasets": {
                "megascale_test": {
                    "kind": "megascale",
                    "csv": str(self.cache),
                    "reference_csv": str(self.reference),
                    "pdb_dir": str(self.root),
                    "expected_rows": 3,
                    "expected_proteins": 2,
                },
            },
            "models": {
                "epsilon": {
                    "kind": "csv",
                    "output": "epsilon",
                    "column": "predicted_epsilon",
                    "predictions": {"megascale_test": str(self.predictions)},
                },
            },
        }

    def output(self, name="result"):
        return self.root / name

    def write_structures(self):
        for name in ("first", "second"):
            lines = []
            for residue, aa in enumerate(("ALA", "CYS", "ASP", "GLU"), 1):
                for atom_index, atom in enumerate(("N", "CA", "C", "O")):
                    serial = (residue - 1) * 4 + atom_index + 1
                    x, y, z = residue * 3.8 + atom_index * 0.4, atom_index * 0.7, 0.3
                    lines.append(
                        f"ATOM  {serial:5d} {atom:^4s} {aa:>3s} A{residue:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}          {atom[0]:>2s}\n"
                    )
            (self.root / f"{name}.pdb").write_text("".join(lines) + "END\n")

    def test_shuffled_epsilon_predictions_are_aligned_and_scored_as_ddg(self):
        summary = evaluate(self.config, self.output(), device="cpu")
        row = summary.iloc[0]
        self.assertEqual(row.n_mutations, 3)
        self.assertAlmostEqual(row.mse, 2 / 3)
        self.assertAlmostEqual(row.rmse, np.sqrt(2 / 3))
        self.assertAlmostEqual(row.pearson, 0.5)
        self.assertAlmostEqual(row.spearman, 0.5)
        result = pd.read_csv(self.output() / "megascale_test/epsilon.predictions.csv")
        self.assertEqual(result.predicted_ddg.tolist(), [0.0, 2.0, 1.0])
        self.assertEqual(result.predicted_epsilon.tolist(), [-10.0, -18.0, -29.0])
        manifest = json.loads((self.output() / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["datasets"]["megascale_test"]["n_mutations"], 3)

    def test_incomplete_duplicate_extra_and_nonfinite_predictions_are_rejected(self):
        original = pd.read_csv(self.predictions)
        for index, invalid in enumerate(
            (
                original.iloc[:2],
                pd.concat([original, original.iloc[:1]]),
                pd.concat([original, original.iloc[:1].assign(WT_name="unknown.pdb")]),
                original.assign(predicted_epsilon=np.nan),
            )
        ):
            with self.subTest(case=index):
                invalid.to_csv(self.predictions, index=False)
                with self.assertRaises(ValueError):
                    evaluate(self.config, self.output(f"invalid-{index}"), device="cpu")

    def test_missing_published_examples_changed_targets_and_wrong_splits_fail(self):
        for index, invalid in enumerate(
            (
                self.frame.iloc[:2],
                self.frame.assign(ddG_ML=[0.0, 1.0, -2.0]),
                self.frame.assign(split="train"),
            )
        ):
            with self.subTest(case=index):
                invalid.to_csv(self.cache, index=False)
                with self.assertRaises(ValueError):
                    evaluate(
                        self.config, self.output(f"membership-{index}"), device="cpu"
                    )

    def test_a_cached_baseline_cannot_be_used_with_different_checkpoint_weights(self):
        checkpoint = self.root / "different.ckpt"
        checkpoint.write_bytes(b"different baseline weights")
        self.cache.with_suffix(".metadata.json").write_text(
            json.dumps(
                {"checkpoint": {"sha256": "original-label-generation-checkpoint"}}
            )
        )
        self.config["baseline_checkpoint"] = str(checkpoint)
        self.config["models"] = {"additive": {"kind": "additive"}}
        with self.assertRaisesRegex(ValueError, "Baseline checkpoint differs"):
            evaluate(self.config, self.output(), device="cpu")

    def test_cached_additive_and_column_models_need_no_structure_files(self):
        self.config["models"] = {
            "additive": {"kind": "additive"},
            "saved": {"kind": "column", "column": "baseline_ddg"},
        }
        summary = evaluate(self.config, self.output(), device="cpu")
        self.assertEqual(summary.mse.tolist(), [415.0, 415.0])
        self.assertEqual(summary.n_mutations.tolist(), [3, 3])

    def test_ptmul_signs_and_double_only_selection_use_the_same_scoring(self):
        source = self.root / "ptmul.csv"
        pd.DataFrame(
            {
                "PDB": ["1ABCA", "1ABCA", "2ABCA", "3ABCA"],
                "MUTS": ["A1V;D3G", "C2A;E4Y", "A1G;E4K", "A1V;C2A;D3G"],
                "DDG": [0.0, 1.0, 2.0, 99.0],
            }
        ).to_csv(source, index=False)
        self.config["datasets"] = {
            "ptmul_d": {
                "kind": "ptmul",
                "csv": str(source),
                "expected_rows": 3,
                "expected_proteins": 2,
            }
        }
        pd.DataFrame(
            {
                "WT_name": ["2ABCA", "1ABCA", "1ABCA"],
                "mut_type": ["A1G:E4K", "C2A:E4Y", "A1V:D3G"],
                "prediction": [-1.0, -2.0, 0.0],
            }
        ).to_csv(self.predictions, index=False)
        self.config["models"] = {
            "direct": {
                "kind": "csv",
                "output_sign": -1,
                "column": "prediction",
                "predictions": {"ptmul_d": str(self.predictions)},
            }
        }
        summary = evaluate(self.config, self.output(), device="cpu")
        self.assertAlmostEqual(summary.iloc[0].mse, 2 / 3)

    def test_custom_predictors_get_zero_targets_and_ensembles_average_predictions(self):
        self.write_structures()
        self.config["models"] = {
            "seed1": {
                "kind": "factory",
                "factory": f"{__name__}:label_free_predictor",
                "value": 0.0,
                "group": "custom",
            },
            "seed2": {
                "kind": "factory",
                "factory": f"{__name__}:label_free_predictor",
                "value": 2.0,
                "group": "custom",
            },
            "ensemble": {"kind": "ensemble", "members": ["seed1", "seed2"]},
        }
        summary = evaluate(self.config, self.output(), device="cpu")
        self.assertAlmostEqual(
            summary.loc[summary.model.eq("ensemble"), "mse"].item(), 2 / 3
        )
        predictions = pd.read_csv(
            self.output() / "megascale_test/ensemble.predictions.csv"
        )
        self.assertEqual(predictions.predicted_ddg.tolist(), [1.0, 1.0, 1.0])
        groups = pd.read_csv(self.output() / "replicates.csv")
        self.assertEqual(groups.loc[groups.group.eq("custom"), "n_runs"].item(), 2)
        manifest = json.loads((self.output() / "manifest.json").read_text())
        self.assertIsNone(manifest["metrics"][0]["pearson"])

    def test_ptmul_inference_maps_pdb_numbers_gaps_and_insertion_codes(self):
        self.write_structures()
        lines = []
        numbers = [(10, " "), (11, "A"), (13, " "), (14, " ")]
        for line in (self.root / "first.pdb").read_text().splitlines():
            if line.startswith("ATOM"):
                residue = int(line[22:26]) - 1
                number, insertion = numbers[residue]
                line = line[:22] + f"{number:4d}{insertion}" + line[27:]
            lines.append(line)
        (self.root / "1ABC.pdb").write_text("\n".join(lines) + "\n")
        ptmul = self.root / "ptmul.csv"
        pd.DataFrame(
            {
                "PDB": ["1ABCA", "1ABCA"],
                "MUTS": ["A10V;D13G", "C11AA;E14Y"],
                "DDG": [0.0, 2.0],
            }
        ).to_csv(ptmul, index=False)
        self.config["datasets"] = {
            "ptmul_d": {
                "kind": "ptmul",
                "csv": str(ptmul),
                "pdb_dir": str(self.root),
                "expected_rows": 2,
                "expected_proteins": 1,
            }
        }
        self.config["models"] = {
            "custom": {
                "kind": "factory",
                "factory": f"{__name__}:label_free_predictor",
                "value": 1.0,
            }
        }
        summary = evaluate(self.config, self.output(), device="cpu")
        self.assertEqual(summary.iloc[0].mse, 1.0)

    def test_real_project_checkpoint_restores_without_a_vanilla_weight_file(self):
        from mutation_prediction.model import (
            MutationPredictionModel,
            PooledAttentionHead,
            ProteinMPNNFeatureExtractor,
        )
        from mutation_prediction.thermompnn_d_vendor.thermompnn.thermompnn.protein_mpnn_utils import (
            ProteinMPNN,
        )

        self.write_structures()
        cfg = OmegaConf.create(
            {
                "model": {
                    "head": "pooled_attention",
                    "num_final_layers": 2,
                    "mutant_embedding": True,
                }
            }
        )
        encoder = ProteinMPNN(
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=48,
            augment_eps=0,
            dropout=0,
        )
        features = ProteinMPNNFeatureExtractor(cfg, protein_encoder=encoder)
        head = PooledAttentionHead(features.output_dim)
        with torch.no_grad():
            head.final_layer.weight.zero_()
            head.final_layer.bias.zero_()
        model = MutationPredictionModel(features, head)
        checkpoint = self.root / "model.ckpt"
        torch.save(
            {
                "hyper_parameters": {"cfg": OmegaConf.to_container(cfg)},
                "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
            },
            checkpoint,
        )
        self.config["models"] = {
            "checkpoint": {
                "kind": "checkpoint",
                "checkpoint": str(checkpoint),
                "output": "epsilon",
            }
        }
        summary = evaluate(self.config, self.output(), device="cpu")
        # Squared errors are 10^2, 19^2, and 28^2: mean = 415.
        self.assertEqual(summary.iloc[0].mse, 415.0)
        result = pd.read_csv(
            self.output() / "megascale_test/checkpoint.predictions.csv"
        )
        self.assertEqual(result.predicted_ddg.tolist(), [10.0, 20.0, 30.0])


if __name__ == "__main__":
    unittest.main()
