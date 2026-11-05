"""Check pair mapping, reused inference, and preservation of source targets."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from mutation_prediction.prepare_targets import (
    add_epsilon_targets,
    main,
    parse_pair,
    precompute_single_effects,
)


class StubPredictor:
    def __init__(self):
        self.calls = []

    def predict(self, protein):
        self.calls.append(protein["name"])
        offset = 100 if protein["name"] == "second" else 0
        return torch.arange(80).reshape(4, 20).float() + offset


class PrecomputeMegaScaleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.pdb_dir = Path(self.directory.name)
        for name in ("first", "second"):
            (self.pdb_dir / f"{name}.pdb").write_text("fixture\n")
        self.parser = patch(
            "mutation_prediction.prepare_targets.parse_PDB",
            side_effect=lambda path, **kwargs: [
                {"name": Path(path).stem, "seq": "ACDE"}
            ],
        )
        self.parser.start()
        self.addCleanup(self.parser.stop)
        self.predictor = StubPredictor()
        self.frame = pd.DataFrame(
            {
                "WT_name": ["first.pdb", "second.pdb", "first.pdb"],
                "mut_type": ["A1V:D3G", "C2A:E4Y", "D3G:A1V"],
                "aa_seq": ["VCGE", "AADY", "VCGE"],
                "ddG_ML": [2.5, -0.7, 2.5],
                "split": ["train", "val", "train"],
            },
            index=[9, 2, 15],
        )

    def run_export(self, frame=None):
        return precompute_single_effects(
            self.frame if frame is None else frame,
            self.predictor,
            self.pdb_dir,
            progress=False,
        )

    def test_predictions_reuse_wt_passes_and_preserve_pair_order_and_targets(self):
        original = self.frame.copy(deep=True)
        result, metadata = self.run_export()
        pd.testing.assert_frame_equal(self.frame, original)
        pd.testing.assert_frame_equal(
            result[self.frame.columns], self.frame.reset_index(drop=True)
        )
        self.assertEqual(self.predictor.calls, ["first", "second"])
        self.assertEqual(result.mutation_1.tolist(), ["A1V", "C2A", "D3G"])
        self.assertEqual(result.mutation_2.tolist(), ["D3G", "E4Y", "A1V"])
        self.assertEqual(result.pred_ddg_1.tolist(), [17, 120, 45])
        self.assertEqual(result.pred_ddg_2.tolist(), [45, 179, 17])
        self.assertEqual(result.baseline_ddg.tolist(), [62, 299, 62])
        self.assertEqual(metadata["pair_count"], 3)
        self.assertEqual(metadata["protein_count"], 2)
        self.assertEqual(metadata["unique_single_mutation_count"], 4)
        self.assertEqual(len(metadata["structures"]["first.pdb"]["sha256"]), 64)

    def test_pair_validation_rejects_non_double_or_non_substitution_records(self):
        for value in ("A1V", "A1V:C2G:D3W", "A1A:D3G", "A1V:C1G", "A0V:D3G", "A1X:D3G"):
            with self.subTest(mutation=value), self.assertRaises(ValueError):
                parse_pair(value)

    def test_bad_mutant_sequence_and_wrong_wt_mapping_fail_before_inference(self):
        for sequence in ("ACDE", "VFGE", "VCD"):
            with self.subTest(sequence=sequence):
                frame = self.frame.iloc[:1].copy()
                frame["aa_seq"] = sequence
                with self.assertRaisesRegex(ValueError, "CSV row 2, first.pdb"):
                    self.run_export(frame)
        self.assertEqual(self.predictor.calls, [])

    def test_missing_structure_is_reported(self):
        (self.pdb_dir / "first.pdb").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "first.pdb"):
            self.run_export()

    def test_nonfinite_predictions_are_reported_instead_of_dropping_pairs(self):
        table = torch.zeros(4, 20)
        table[0, 17] = torch.nan
        with patch.object(self.predictor, "predict", return_value=table):
            with self.assertRaisesRegex(
                ValueError, "Nonfinite prediction for first.pdb, A1V"
            ):
                self.run_export()

    def test_missing_columns_empty_input_and_previous_exports_are_rejected(self):
        for frame in (
            self.frame.drop(columns="WT_name"),
            self.frame.iloc[:0],
            self.frame.assign(pred_ddg_1=0.0),
            self.frame.assign(aa_seq=None),
        ):
            with (
                self.subTest(columns=frame.columns.tolist()),
                self.assertRaises(ValueError),
            ):
                self.run_export(frame)

    def test_epsilon_labels_use_model_baseline_and_preserve_experimental_signs(self):
        frame = pd.DataFrame(
            {
                "ddG_ML": [1.5, -1.5, 0.0],
                "pred_ddg_1": [0.4, -0.4, 0.2],
                "pred_ddg_2": [0.6, -0.6, -0.2],
                "baseline_ddg": [1.0, -1.0, 0.0],
                # Experimental epistasis must not be substituted for model residuals.
                "ddG_ML_epi_term": [9.0, 8.0, 7.0],
            }
        )
        result = add_epsilon_targets(frame)
        self.assertEqual(result.true_ddg_12.tolist(), [1.5, -1.5, 0.0])
        self.assertEqual(result.epsilon_target.tolist(), [0.5, -0.5, 0.0])
        pd.testing.assert_frame_equal(result[frame.columns], frame)
        self.assertNotIn("epsilon_target", frame)

    def test_invalid_labels_and_inconsistent_baselines_are_rejected(self):
        frame = pd.DataFrame(
            {
                "ddG_ML": [1.5],
                "pred_ddg_1": [0.4],
                "pred_ddg_2": [0.6],
                "baseline_ddg": [1.0],
            }
        )
        for invalid in (
            frame.drop(columns="ddG_ML"),
            frame.assign(ddG_ML=float("nan")),
            frame.assign(ddG_ML="-"),
            frame.assign(baseline_ddg=5.0),
            frame.assign(pred_ddg_1=float("inf")),
            frame.iloc[:0],
        ):
            with self.subTest(values=invalid.to_dict()), self.assertRaises(ValueError):
                add_epsilon_targets(invalid)

    def test_cached_export_adds_targets_without_loading_thermompnn(self):
        frame, _ = self.run_export()
        predictions = self.pdb_dir / "predictions.csv"
        output = self.pdb_dir / "epsilon.csv"
        frame.to_csv(predictions, index=False)
        predictions.with_suffix(".metadata.json").write_text(
            json.dumps(
                {
                    "created_at_utc": "2026-11-04T00:00:00+00:00",
                    "prediction_sign": "positive = destabilizing",
                    "checkpoint": {"sha256": "original-checkpoint-hash"},
                }
            )
        )
        with (
            patch(
                "sys.argv",
                [
                    "precompute",
                    "--predictions-csv",
                    str(predictions),
                    "--output",
                    str(output),
                ],
            ),
            patch(
                "mutation_prediction.prepare_targets.ThermoMPNNPredictor"
            ) as predictor,
            patch("builtins.print"),
        ):
            main()
        predictor.assert_not_called()
        result = pd.read_csv(output)
        self.assertEqual(result.epsilon_target.tolist(), [-59.5, -299.7, -59.5])
        metadata = json.loads(output.with_suffix(".metadata.json").read_text())
        self.assertEqual(metadata["checkpoint"]["sha256"], "original-checkpoint-hash")
        self.assertEqual(metadata["training_target"], "epsilon_target")
        self.assertEqual(
            metadata["prediction_created_at_utc"], "2026-11-04T00:00:00+00:00"
        )


if __name__ == "__main__":
    unittest.main()
