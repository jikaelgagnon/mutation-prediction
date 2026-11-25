"""Check residual-label signs, split isolation, loss, and checkpoint selection."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers.logger import Logger
from torch import nn

from mutation_prediction.data.collate import tied_featurize_mut
from mutation_prediction.data.megascale import MegaScaleDatasetv2
from mutation_prediction.model import (
    MutationPredictionModel,
    MutationPredictionModelPL,
    PooledAttentionHead,
    ProteinMPNNFeatureExtractor,
)
from mutation_prediction.model.train import HEADS, parse_cfg, train


class RecordingRun:
    def __init__(self):
        self.summary = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class RecordingWandbLogger(Logger):
    """Exercise the real Trainer/callback lifecycle without a W&B connection."""

    def __init__(self, **kwargs):
        super().__init__()
        self.settings = kwargs
        self.run = RecordingRun()

    @property
    def name(self):
        return self.settings["name"]

    @property
    def version(self):
        return self.settings["id"]

    @property
    def experiment(self):
        return self.run

    def log_hyperparams(self, params):
        pass

    def log_metrics(self, metrics, step=None):
        pass


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_s = nn.Embedding(21, 8)
        self.requires_grad_(False)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        embedding = self.W_s(S)
        return [embedding, embedding], embedding, None, None


def make_model(cfg):
    features = ProteinMPNNFeatureExtractor(cfg, protein_encoder=TinyEncoder())
    return MutationPredictionModel(features, PooledAttentionHead(features.output_dim))


def structure(path, **kwargs):
    coords = np.random.default_rng(42).normal(size=(4, 4, 3))
    return [
        {
            "name": Path(path).stem,
            "seq": "ACDE",
            "seq_chain_A": "ACDE",
            "coords_chain_A": {
                f"{atom}_chain_A": coords[:, i].tolist()
                for i, atom in enumerate(("N", "CA", "C", "O"))
            },
        }
    ]


class EpsilonTrainingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.csv = self.root / "epsilon.csv"
        self.frame = pd.DataFrame(
            {
                "WT_name": ["train.pdb"] * 2 + ["val.pdb"] * 3,
                "mut_type": ["A1V:D3G", "A1G:E4K", "A1V:D3G", "A1G:E4K", "C2V:D3W"],
                "aa_seq": ["VCGE", "GCDK", "VCGE", "GCDK", "AVWE"],
                "ddG_ML": [99.0] * 5,
                "epsilon_target": [0.5, -1.25, 0.0, 2.0, -0.75],
            }
        )
        self.frame.to_csv(self.csv, index=False)
        with (self.root / "splits.pkl").open("wb") as stream:
            pickle.dump({"train": ["train.pdb"], "val": ["val.pdb"]}, stream)
        self.cfg = parse_cfg(
            OmegaConf.create(
                {
                    "name": "epsilon_test",
                    "data": {
                        "dataset": "megascale",
                        "mut_types": ["double"],
                        "splits": ["train", "val"],
                        "target_column": "epsilon_target",
                    },
                    "data_loc": {
                        "megascale_csv": str(self.csv),
                        "megascale_splits": str(self.root / "splits.pkl"),
                        "megascale_pdbs": str(self.root),
                    },
                    "model": {"mutant_embedding": True},
                    "training": {
                        "epochs": 1,
                        "batch_size": 2,
                        "learn_rate": 0.001,
                        "lr_schedule": True,
                        "checkpoint_metric": "mse",
                        "output_dir": str(self.root / "checkpoints"),
                    },
                    "platform": {"accel": "cpu"},
                }
            )
        )
        parser = patch(
            "mutation_prediction.data.megascale.parse_PDB",
            side_effect=structure,
        )
        parser.start()
        self.addCleanup(parser.stop)

    def test_epsilon_survives_loading_sorting_and_collation_without_sign_flip(self):
        for split in ("train", "val"):
            dataset = MegaScaleDatasetv2(self.cfg, split)
            self.assertEqual(set(dataset.df.WT_name), {f"{split}.pdb"})
            expected = dataset.df.epsilon_target.to_numpy(dtype=np.float32)
            batch = tied_featurize_mut([dataset[i] for i in range(len(dataset))])
            torch.testing.assert_close(batch[10], torch.from_numpy(expected[:, None]))
            self.assertEqual(batch[7].shape[1], 2)

    def test_legacy_ddg_training_still_negates_original_labels(self):
        del self.cfg.data.target_column
        dataset = MegaScaleDatasetv2(self.cfg, "train")
        self.assertEqual(dataset[0]["mutation"].ddG, -99.0)

    def test_invalid_epsilon_labels_and_augmentation_fail(self):
        for value in (float("nan"), float("inf"), "invalid"):
            with self.subTest(value=value):
                frame = self.frame.copy()
                frame["epsilon_target"] = frame["epsilon_target"].astype(object)
                frame.loc[0, "epsilon_target"] = value
                frame.to_csv(self.csv, index=False)
                with self.assertRaises(ValueError):
                    MegaScaleDatasetv2(self.cfg, "train")
        self.cfg.data.mut_types = ["double", "double-aug"]
        with self.assertRaisesRegex(ValueError, "unmodified double mutants"):
            MegaScaleDatasetv2(self.cfg, "train")

    def test_identity_substitution_in_either_site_is_excluded(self):
        extra = self.frame.iloc[[0, 0]].copy()
        extra["mut_type"] = ["A1A:D3G", "A1V:D3D"]
        pd.concat([self.frame, extra]).to_csv(self.csv, index=False)
        dataset = MegaScaleDatasetv2(self.cfg, "train")
        self.assertEqual(len(dataset), 2)

    def test_structure_key_normalization_matches_on_load_and_lookup(self):
        frame = self.frame.replace({"WT_name": {"train.pdb": "train|part.pdb"}})
        frame.to_csv(self.csv, index=False)
        with (self.root / "splits.pkl").open("wb") as stream:
            pickle.dump({"train": ["train|part.pdb"]}, stream)
        dataset = MegaScaleDatasetv2(self.cfg, "train")
        self.assertEqual(dataset[0]["mutation"].pdb, "train|part.pdb")

    def test_loss_is_mse_against_epsilon_and_singleton_batches_update_metrics(self):
        module = MutationPredictionModelPL(self.cfg, make_model(self.cfg))
        dataset = MegaScaleDatasetv2(self.cfg, "train")
        for size in (1, 2):
            batch = tied_featurize_mut([dataset[i] for i in range(size)])
            X, S, mask, _, chain_M, chains, indices, pos, wt, mut, target, atom_mask = (
                batch
            )
            expected = nn.functional.mse_loss(
                module(
                    X,
                    S,
                    mask,
                    chain_M,
                    indices,
                    chains,
                    pos,
                    wt,
                    mut,
                    target,
                    atom_mask,
                ),
                target,
            )
            with patch.object(module, "log"):
                loss = module.training_step(batch, 0)
            torch.testing.assert_close(loss, expected)
            loss.backward()
            self.assertIsNotNone(module.model.prediction_head.final_layer.weight.grad)
        self.assertEqual(module.configure_optimizers()["monitor"], "val_epsilon_mse")

    def test_unknown_head_fails_before_loading_data(self):
        self.cfg.model.head = "unknown"
        with patch("mutation_prediction.model.train.get_v2_dataset") as load_data:
            with self.assertRaisesRegex(
                ValueError, "Unknown model.head.*Available heads"
            ):
                train(self.cfg)
        load_data.assert_not_called()

    def test_training_saves_checkpoint_with_correct_validation_mse(self):
        self.assertEqual(self.cfg.model.head, "pooled_attention")
        self.cfg.model.head = "registered_attention"
        head_constructor = Mock(wraps=PooledAttentionHead)
        with (
            patch.dict(HEADS, registered_attention=head_constructor),
            patch(
                "mutation_prediction.model.feature_extraction.get_protein_mpnn",
                side_effect=lambda cfg: TinyEncoder(),
            ),
        ):
            trainer = train(self.cfg)
        head_constructor.assert_called_once_with(input_dim=24)
        self.assertEqual(trainer.global_step, 1)
        self.assertTrue(Path(trainer.checkpoint_callback.best_model_path).is_file())
        run_directory = Path(trainer.checkpoint_callback.dirpath)
        self.assertEqual(run_directory.parent, self.root / "checkpoints")
        self.assertEqual(run_directory.name, self.cfg.run.run_name)
        last_checkpoint = run_directory / "last.ckpt"
        self.assertTrue(last_checkpoint.is_file())
        original_last = last_checkpoint.read_bytes()
        run_metadata = json.loads((run_directory / "run.json").read_text())
        self.assertEqual(
            run_metadata["best_checkpoint_path"],
            trainer.checkpoint_callback.best_model_path,
        )
        self.assertEqual(run_metadata["last_checkpoint_path"], str(last_checkpoint))
        self.assertEqual(trainer.checkpoint_callback.monitor, "val_epsilon_mse")
        self.assertEqual(trainer.checkpoint_callback.mode, "min")
        # Validation's final batch has one sample. All three must contribute.
        dataset = MegaScaleDatasetv2(self.cfg, "val")
        batch = tied_featurize_mut([dataset[i] for i in range(len(dataset))])
        with torch.no_grad(), patch.object(trainer.lightning_module, "log"):
            expected = trainer.lightning_module.validation_step(batch, 0)
        torch.testing.assert_close(
            trainer.callback_metrics["val_epsilon_mse"], expected
        )
        checkpoint_path = trainer.checkpoint_callback.best_model_path
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.assertEqual(
            checkpoint["hyper_parameters"]["cfg"]["data"]["target_column"],
            "epsilon_target",
        )
        self.assertEqual(
            checkpoint["hyper_parameters"]["cfg"]["run"]["run_id"],
            self.cfg.run.run_id,
        )
        saved_config = OmegaConf.load(run_directory / "config.yaml")
        self.assertEqual(saved_config.run.checkpoint_dir, str(run_directory))
        restored = MutationPredictionModelPL.load_from_checkpoint(
            checkpoint_path,
            model=make_model(self.cfg),
            weights_only=True,
        )
        for name, value in trainer.lightning_module.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value)

        self.cfg.training.ckpt = str(last_checkpoint)
        self.cfg.training.epochs = 2
        with (
            patch.dict(HEADS, registered_attention=head_constructor),
            patch(
                "mutation_prediction.model.feature_extraction.get_protein_mpnn",
                side_effect=lambda cfg: TinyEncoder(),
            ),
        ):
            resumed = train(self.cfg)
        self.assertEqual(resumed.global_step, 2)
        resumed_directory = Path(resumed.checkpoint_callback.dirpath)
        self.assertNotEqual(resumed_directory, run_directory)
        self.assertEqual(resumed_directory.parent, run_directory.parent)
        self.assertEqual(last_checkpoint.read_bytes(), original_last)
        self.assertEqual(self.cfg.run.resumed_from, str(last_checkpoint))
        self.assertEqual(
            Path(resumed.checkpoint_callback.best_model_path).parent, resumed_directory
        )
        self.assertTrue((resumed_directory / "last.ckpt").is_file())

    def test_wandb_summary_tracks_the_saved_checkpoints_for_each_run(self):
        self.cfg.project = "test-project"
        trainers = []
        with (
            patch.dict("os.environ", {"SLURM_JOB_ID": "12345"}),
            patch("pytorch_lightning.loggers.WandbLogger", RecordingWandbLogger),
            patch(
                "mutation_prediction.model.feature_extraction.get_protein_mpnn",
                side_effect=lambda cfg: TinyEncoder(),
            ),
        ):
            for _ in range(2):
                trainers.append(train(self.cfg))

        directories = set()
        for trainer in trainers:
            logger = trainer.logger
            callback = trainer.checkpoint_callback
            summary = logger.experiment.summary
            directory = Path(summary["checkpoint_dir"])
            directories.add(directory)
            self.assertEqual(directory.name, logger.name)
            self.assertIn("-slurm-12345-", directory.name)
            self.assertTrue(directory.name.endswith(logger.version))
            self.assertEqual(summary["run_id"], logger.version)
            self.assertEqual(summary["slurm_job_id"], "12345")
            self.assertEqual(summary["best_checkpoint_path"], callback.best_model_path)
            self.assertEqual(summary["last_checkpoint_path"], callback.last_model_path)
            self.assertEqual(
                summary["best_checkpoint_score"], callback.best_model_score.item()
            )
            self.assertEqual(summary["checkpoint_monitor"], "val_epsilon_mse")
            for key in ("best_checkpoint_path", "last_checkpoint_path"):
                self.assertTrue(Path(summary[key]).is_file())
                self.assertEqual(Path(summary[key]).parent, directory)
            self.assertEqual(json.loads((directory / "run.json").read_text()), summary)
            self.assertEqual(
                logger.settings["config"]["checkpoint_dir"], str(directory)
            )
            self.assertFalse(logger.settings["log_model"])
            self.assertIsNone(summary["resumed_from"])
        self.assertEqual(len(directories), 2)


if __name__ == "__main__":
    unittest.main()
