"""Exercise prepare-targets through its CLI with real parsing and inference.

Run on the cluster. A tiny, randomly initialized checkpoint keeps these tests
offline; they verify dataset membership, not pretrained prediction accuracy.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

import mutation_prediction
from mutation_prediction.thermompnn import _single_mutant_config
from mutation_prediction.thermompnn_d_vendor.thermompnn.thermompnn.model.v2_model import (
    TransferModelv2,
)
from mutation_prediction.thermompnn_d_vendor.thermompnn.thermompnn.protein_mpnn_utils import (
    ProteinMPNN,
)


class PublishedSplitPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint_directory = tempfile.TemporaryDirectory()
        cls.checkpoint = Path(cls.checkpoint_directory.name) / "single.ckpt"
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
        model = TransferModelv2(_single_mutant_config(), protein_encoder=encoder)
        torch.save(
            {"state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}},
            cls.checkpoint,
        )

    @classmethod
    def tearDownClass(cls):
        cls.checkpoint_directory.cleanup()

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.csv_dir = self.root / "data/raw/megascale/csv"
        self.pdb_dir = self.root / "data/raw/megascale/pdb"
        self.csv_dir.mkdir(parents=True)
        self.pdb_dir.mkdir(parents=True)
        self.output = self.root / "epsilon.csv"
        # The legacy full CSV deliberately omits a published pair and adds a
        # protein excluded from the paper's ablation training set.
        for split, name, mutation, sequence, target in (
            ("train", "train", "A1V:D3G", "VCGE", 1.5),
            ("val", "val", "C2A:E4Y", "AADY", -0.75),
            ("test", "test", "A1G:E4K", "GCDK", 0.0),
        ):
            pd.DataFrame(
                [
                    {
                        "WT_name": f"{name}.pdb",
                        "mut_type": mutation,
                        "aa_seq": sequence,
                        "ddG_ML": target,
                    }
                ]
            ).to_csv(self.csv_dir / f"Megascale-D-{split}.csv", index=False)
        pd.DataFrame(
            [
                {
                    "WT_name": "excluded.pdb",
                    "mut_type": "A1V:D3G",
                    "aa_seq": "VCGE",
                    "ddG_ML": 9.0,
                }
            ]
        ).to_csv(self.csv_dir / "Megascale-D-full.csv", index=False)
        for name in ("train", "val", "test", "excluded"):
            lines = []
            for residue, aa in enumerate(("ALA", "CYS", "ASP", "GLU"), 1):
                for atom_index, atom in enumerate(("N", "CA", "C", "O")):
                    serial = (residue - 1) * 4 + atom_index + 1
                    x, y, z = residue * 3.8 + atom_index * 0.4, atom_index * 0.7, 0.3
                    lines.append(
                        f"ATOM  {serial:5d} {atom:^4s} {aa:>3s} A{residue:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}          {atom[0]:>2s}\n"
                    )
            (self.pdb_dir / f"{name}.pdb").write_text("".join(lines) + "END\n")

    def run_prepare(self, *arguments):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(
            Path(mutation_prediction.__file__).parent.parent
        )
        environment["OMP_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "mutation_prediction.prepare_targets",
                "--checkpoint",
                str(self.checkpoint),
                "--device",
                "cpu",
                "--output",
                str(self.output),
                *arguments,
            ],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_default_export_keeps_exact_published_mutation_memberships(self):
        command = self.run_prepare()
        self.assertEqual(command.returncode, 0, command.stdout + command.stderr)
        result = pd.read_csv(self.output)
        self.assertEqual(
            list(zip(result.WT_name, result.mut_type)),
            [
                ("train.pdb", "A1V:D3G"),
                ("val.pdb", "C2A:E4Y"),
                ("test.pdb", "A1G:E4K"),
            ],
            "PUBLISHED_SPLIT_MEMBERSHIP: default export must use the published split CSVs",
        )
        self.assertEqual(result.split.tolist(), ["train", "val", "test"])
        self.assertEqual(result.true_ddg_12.tolist(), [1.5, -0.75, 0.0])
        for row in result.itertuples():
            self.assertAlmostEqual(row.epsilon_target, row.ddG_ML - row.baseline_ddg)
        metadata = json.loads(self.output.with_suffix(".metadata.json").read_text())
        self.assertEqual(metadata["split_counts"], {"train": 1, "val": 1, "test": 1})
        self.assertEqual(set(metadata["input_csvs"]), {"train", "val", "test"})
        self.assertTrue(
            all(len(v["sha256"]) == 64 for v in metadata["input_csvs"].values())
        )


if __name__ == "__main__":
    unittest.main()
