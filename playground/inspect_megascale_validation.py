# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.25.1",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inspect the MegaScale validation split

    Load `MegaScaleDatasetv2` with the validation split, browse its rows, and inspect an individual sample. Set `index` in the final cell to view another example.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import numpy as np
    from omegaconf import OmegaConf

    REPO_ROOT = Path.cwd()
    if not (REPO_ROOT / "configs").is_dir():
        REPO_ROOT = Path.cwd().parent
    assert (REPO_ROOT / "configs").is_dir(), "Open this notebook from the repository or examples directory."

    SRC_ROOT = REPO_ROOT / "src"
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))

    from mutation_prediction.thermo_mpnn_d.thermompnn.datasets.v2_datasets import MegaScaleDatasetv2

    return MegaScaleDatasetv2, OmegaConf, Path, REPO_ROOT


@app.cell
def _(MegaScaleDatasetv2, OmegaConf, Path, REPO_ROOT):
    paths_cfg = OmegaConf.load(REPO_ROOT / "configs" / "thermompnn_paths.example.yaml")
    dataset_cfg = OmegaConf.load(REPO_ROOT / "configs" / "thermompnn_train_single.yaml")
    cfg = OmegaConf.merge(paths_cfg, dataset_cfg)

    # Resolve paths relative to the repository, not the notebook's working directory.
    for key in ("megascale_splits", "megascale_pdbs", "megascale_csv"):
        path = Path(cfg.data_loc[key])
        if not path.is_absolute():
            cfg.data_loc[key] = str(REPO_ROOT / path)

    dataset = MegaScaleDatasetv2(cfg, split="val")
    print(f"Validation samples: {len(dataset):,}")
    dataset.df.head()
    return (dataset,)


@app.cell
def _(dataset):
    index = 0  # Change this to inspect another validation sample.
    sample = dataset[index]

    sample.keys()
    return (sample,)


@app.cell
def _(sample):
    from pprint import pprint

    pprint(sample)
    return


if __name__ == "__main__":
    app.run()
