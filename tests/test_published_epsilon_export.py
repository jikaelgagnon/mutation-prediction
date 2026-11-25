"""Audit the actual prepare-targets output after generation on the cluster.

MEGASCALE_EPSILON_CSV selects an existing export; this test never runs inference.
"""

import os
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


@unittest.skipUnless(
    os.environ.get("MEGASCALE_EPSILON_CSV"),
    "Set MEGASCALE_EPSILON_CSV to audit the full cluster export",
)
class PublishedEpsilonExportTests(unittest.TestCase):
    def test_export_matches_the_published_ablation_splits(self):
        root = Path(__file__).resolve().parents[1]
        result = pd.read_csv(os.environ["MEGASCALE_EPSILON_CSV"])
        expected = pd.concat(
            [
                pd.read_csv(
                    root / f"data/raw/megascale/csv/Megascale-D-{split}.csv"
                ).assign(split=split)
                for split in ("train", "val", "test")
            ],
            ignore_index=True,
        )
        columns = ["WT_name", "mut_type", "split"]
        self.assertEqual(
            Counter(result[columns].itertuples(index=False, name=None)),
            Counter(expected[columns].itertuples(index=False, name=None)),
            "Export must retain every published pair in its original split, exactly once",
        )
        self.assertEqual(
            result.groupby("split").size().to_dict(),
            {"train": 85253, "val": 10282, "test": 18574},
        )
        self.assertEqual(
            result.groupby("split").WT_name.nunique().to_dict(),
            {"train": 90, "val": 17, "test": 20},
        )
        index = ["WT_name", "mut_type"]
        expected = expected.set_index(index).sort_index()
        result = result.set_index(index).sort_index()
        # CSV serialization and parsing can shift the last few floating-point
        # bits. Membership/count assertions above remain exact; numeric labels
        # allow only round-trip noise, far below the experimental precision.
        np.testing.assert_allclose(
            result.ddG_ML, expected.ddG_ML, rtol=1e-14, atol=1e-14
        )
        self.assertTrue(
            np.isfinite(result[["baseline_ddg", "epsilon_target"]]).all().all()
        )
        np.testing.assert_allclose(
            result.true_ddg_12, expected.ddG_ML, rtol=1e-14, atol=1e-14
        )
        np.testing.assert_allclose(
            result.baseline_ddg, result.pred_ddg_1 + result.pred_ddg_2
        )
        np.testing.assert_allclose(
            result.epsilon_target,
            expected.ddG_ML - result.baseline_ddg,
            rtol=1e-10,
            atol=1e-10,
        )


if __name__ == "__main__":
    unittest.main()
