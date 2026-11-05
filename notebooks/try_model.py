import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch

    return (torch,)


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import pytorch_lightning as pl
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    repo_root = Path.cwd()
    if not (repo_root / "configs").is_dir():
        repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "configs").is_dir():
        raise FileNotFoundError("Open this notebook from within the repository.")

    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from mutation_prediction.data.megascale import (
        MegaScaleDatasetv2,
        tied_featurize_mut,
    )
    from mutation_prediction.model.train import (
        parse_cfg,
    )

    return (
        DataLoader,
        MegaScaleDatasetv2,
        OmegaConf,
        Path,
        mo,
        parse_cfg,
        pl,
        repo_root,
        tied_featurize_mut,
    )


@app.cell
def _(OmegaConf, Path, repo_root):
    paths_cfg = OmegaConf.load(repo_root / "configs/proteinmpnn_paths.example.yaml")
    model_cfg = OmegaConf.load(repo_root / "configs/model_train_epistatic.yaml")
    cfg = OmegaConf.merge(paths_cfg, model_cfg)

    for key in ("megascale_splits", "megascale_pdbs", "megascale_csv"):
        path = Path(cfg.data_loc[key])
        if not path.is_absolute():
            cfg.data_loc[key] = str(repo_root / path)

    cfg.model.freeze_weights = True
    cfg.model.load_pretrained = True
    cfg.data.mut_types = ["double"]
    cfg.data.splits = ["train", "val"]
    cfg.training.num_workers = 0
    return (cfg,)


@app.cell
def _(cfg):
    dict(cfg)


@app.cell
def _(Path, cfg, repo_root):
    if not Path(cfg.data_loc.megascale_csv).is_file():
        raise FileNotFoundError(f"MegaScale CSV not found: {cfg.data_loc.megascale_csv}")
    if not Path(cfg.data_loc.megascale_splits).is_file():
        raise FileNotFoundError(
            f"MegaScale splits not found: {cfg.data_loc.megascale_splits}"
        )
    if not Path(cfg.data_loc.megascale_pdbs).is_dir():
        raise FileNotFoundError(
            f"MegaScale structure directory not found: {cfg.data_loc.megascale_pdbs}"
        )
    checkpoint_dir = Path(cfg.platform.thermompnn_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = repo_root / checkpoint_dir
    cfg.platform.thermompnn_dir = str(checkpoint_dir)
    checkpoint_paths = sorted(checkpoint_dir.glob("v_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No pretrained ProteinMPNN checkpoints found in {checkpoint_dir}."
        )

    checkpoint_names = [path.name for path in checkpoint_paths]
    default_checkpoint = (
        "v_48_020.pt" if "v_48_020.pt" in checkpoint_names else checkpoint_names[0]
    )


@app.cell
def _(MegaScaleDatasetv2, cfg):
    train_dataset = MegaScaleDatasetv2(cfg, split="train")
    val_dataset = MegaScaleDatasetv2(cfg, split="val")
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("MegaScale train and validation splits must both be non-empty.")
    return train_dataset, val_dataset


@app.cell(hide_code=True)
def _(mo, train_dataset, val_dataset):
    mo.md(f"""
    **MegaScale examples with structures:** {len(train_dataset):,} train ·
    {len(val_dataset):,} validation. The data and structure files are read from
    `data/raw/megascale/`.
    """)


@app.cell
def _(train_dataset):
    train_dataset.df[["WT_name", "mut_type", "ddG_ML", "aa_seq"]].head(12)


@app.cell
def _(active_cfg):
    active_cfg.training.batch_size = 1


@app.cell
def _(active_cfg):
    active_cfg.training.batch_size


@app.cell
def _(
    DataLoader,
    cfg,
    parse_cfg,
    pl,
    tied_featurize_mut,
    train_dataset,
    val_dataset,
):
    pl.seed_everything(42, workers=True)

    active_cfg = parse_cfg(cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=active_cfg.training.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: tied_featurize_mut(batch, side_chains=False),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=active_cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: tied_featurize_mut(batch, side_chains=False),
    )
    return active_cfg, train_loader


@app.cell
def _(train_loader):
    single_batch = next(iter(train_loader))
    single_batch
    return (single_batch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can unpack the tuple:
    """)


@app.cell
def _(single_batch):
    (X,
        S,
        mask,
        lengths,
        chain_M,
        chain_encoding_all,
        residue_idx,
        mut_positions,
        mut_wildtype_AAs,
        mut_mutant_AAs,
        mut_ddGs,
        atom_mask,
        ) = single_batch
    return (
        S,
        X,
        atom_mask,
        chain_M,
        chain_encoding_all,
        mask,
        mut_ddGs,
        mut_mutant_AAs,
        mut_positions,
        mut_wildtype_AAs,
        residue_idx,
    )


@app.cell
def _(
    S,
    X,
    atom_mask,
    chain_M,
    chain_encoding_all,
    mask,
    mut_ddGs,
    mut_mutant_AAs,
    mut_positions,
    mut_wildtype_AAs,
    residue_idx,
):
    model_inputs = (
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        mut_positions,
        mut_wildtype_AAs,
        mut_mutant_AAs,
        mut_ddGs,
        atom_mask,
        )
    return (model_inputs,)


@app.cell
def _(X):
    X.shape


@app.cell
def _(torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return (device,)


@app.cell
def _(device, model_inputs):
    inputs = tuple(tensor.to(device) for tensor in model_inputs)
    return (inputs,)


@app.cell
def _():
    from mutation_prediction.model import (
        MutationPredictionModel,
        PooledAttentionHead,
        ProteinMPNNFeatureExtractor,
    )

    return MutationPredictionModel, PooledAttentionHead, ProteinMPNNFeatureExtractor


@app.cell
def _(MutationPredictionModel, PooledAttentionHead, ProteinMPNNFeatureExtractor, cfg, device):
    feature_extractor = ProteinMPNNFeatureExtractor(cfg)
    prediction_head = PooledAttentionHead(input_dim=feature_extractor.output_dim)
    model = MutationPredictionModel(feature_extractor, prediction_head).to(device)
    model
    return (model,)


@app.cell
def _(inputs, model):
    model(*inputs)


if __name__ == "__main__":
    app.run()
