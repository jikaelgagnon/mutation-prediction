# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "bio==1.8.4",
#     "marimo>=0.25.1",
#     "omegaconf>=2.3.0",
#     "pandas==3.0.6",
#     "pytorch-lightning>=2.6.6",
#     "torch>=2.14.0",
# ]
# ///

"""Train the ThermoMPNN-D Siamese model on the repository's MegaScale data."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Train ThermoMPNN-D on MegaScale double mutants

    This notebook trains the repository's Siamese double-mutant model on the
    MegaScale-D train and validation splits. It uses the pretrained
    `v_48_020.pt` ProteinMPNN encoder as a **frozen feature extractor** and
    trains the ThermoMPNN-D projection and prediction layers.

    The default run follows the paper's model/training settings (Adam,
    learning rate `1e-5`, batch size `256`, up to `100` epochs). Training starts
    only when you click **Train model**. The paper's optional over-and-back data
    augmentation is not enabled: it additionally requires the separate
    multi-gigabyte MegaScale Rosetta structure archive.
    """)
    return


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import pytorch_lightning as pl
    import torch
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from torch.utils.data import DataLoader

    repo_root = Path.cwd()
    if not (repo_root / "configs").is_dir():
        repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "configs").is_dir():
        raise FileNotFoundError("Open this notebook from within the repository.")

    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from mutation_prediction.thermo_mpnn_d.thermompnn.datasets.v2_datasets import (
        MegaScaleDatasetv2,
        tied_featurize_mut,
    )
    from mutation_prediction.thermo_mpnn_d.thermompnn.train_thermompnn import (
        parse_cfg,
    )
    from mutation_prediction.thermo_mpnn_d.thermompnn.trainer.v2_trainer import (
        TransferModelPLv2Siamese,
    )

    return (
        DataLoader,
        EarlyStopping,
        MegaScaleDatasetv2,
        ModelCheckpoint,
        OmegaConf,
        Path,
        TransferModelPLv2Siamese,
        mo,
        parse_cfg,
        pd,
        pl,
        repo_root,
        tied_featurize_mut,
        torch,
    )


@app.cell
def _(OmegaConf, Path, parse_cfg, repo_root):
    paths_cfg = OmegaConf.load(repo_root / "configs/thermompnn_paths.example.yaml")
    model_cfg = OmegaConf.load(repo_root / "configs/thermompnn_train_epistatic.yaml")
    cfg = parse_cfg(OmegaConf.merge(paths_cfg, model_cfg))

    for key in ("megascale_splits", "megascale_pdbs", "megascale_csv"):
        path = Path(cfg.data_loc[key])
        if not path.is_absolute():
            cfg.data_loc[key] = str(repo_root / path)

    checkpoint_dir = Path(cfg.platform.thermompnn_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = repo_root / checkpoint_dir
    cfg.platform.thermompnn_dir = str(checkpoint_dir)

    cfg.model.freeze_weights = True
    cfg.model.load_pretrained = True
    cfg.data.mut_types = ["double"]
    cfg.data.splits = ["train", "val"]
    cfg.training.num_workers = 0

    if cfg.model.aggregation != "siamese":
        raise ValueError("Expected the Siamese ThermoMPNN-D model configuration.")
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
    checkpoint_paths = sorted(checkpoint_dir.glob("v_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No pretrained ProteinMPNN checkpoints found in {checkpoint_dir}."
        )

    checkpoint_names = [path.name for path in checkpoint_paths]
    default_checkpoint = (
        "v_48_020.pt" if "v_48_020.pt" in checkpoint_names else checkpoint_names[0]
    )
    return cfg, checkpoint_names, default_checkpoint


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
    return


@app.cell
def _(train_dataset):
    train_dataset.df[["WT_name", "mut_type", "ddG_ML", "aa_seq"]].head(12)
    return


@app.cell(hide_code=True)
def _(checkpoint_names, default_checkpoint, mo, torch):
    checkpoint_picker = mo.ui.dropdown(
        options=checkpoint_names,
        value=default_checkpoint,
        label="Frozen ProteinMPNN checkpoint",
    )
    device_options = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    device_picker = mo.ui.dropdown(
        options=device_options,
        value=device_options[0],
        label="Training device",
    )
    epochs_picker = mo.ui.number(
        start=1, stop=100, value=100, step=1, label="Maximum epochs"
    )
    batch_size_picker = mo.ui.number(
        start=1, stop=1024, value=256, step=1, label="Batch size"
    )
    learning_rate_picker = mo.ui.number(
        start=0.000001,
        stop=0.001,
        value=0.00001,
        step=0.000001,
        label="Learning rate",
    )
    train_button = mo.ui.run_button(label="Train model")
    mo.vstack(
        [
            mo.hstack([checkpoint_picker, device_picker], justify="start"),
            mo.hstack(
                [epochs_picker, batch_size_picker, learning_rate_picker],
                justify="start",
            ),
            train_button,
        ]
    )
    return (
        batch_size_picker,
        checkpoint_picker,
        device_picker,
        epochs_picker,
        learning_rate_picker,
        train_button,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The model uses both mutation orders (AB and BA) and averages their
    predictions. Its training loss combines prediction MSE with the symmetry
    penalty, as implemented by `TransferModelPLv2Siamese`. The pretrained
    ProteinMPNN is frozen; the Siamese projection/head remains trainable.
    """)
    return


@app.cell
def _(
    DataLoader,
    EarlyStopping,
    ModelCheckpoint,
    OmegaConf,
    TransferModelPLv2Siamese,
    batch_size_picker,
    cfg,
    checkpoint_picker,
    device_picker,
    epochs_picker,
    learning_rate_picker,
    mo,
    parse_cfg,
    pl,
    repo_root,
    tied_featurize_mut,
    train_button,
    train_dataset,
    val_dataset,
):
    trained_model = None
    training_result = None

    if not train_button.value:
        mo.md("Set the training options above, then click **Train model**.")
    else:
        pl.seed_everything(42, workers=True)
        active_cfg = OmegaConf.merge(
            cfg,
            {
                "model": {"version": checkpoint_picker.value},
                "training": {
                    "epochs": int(epochs_picker.value),
                    "batch_size": int(batch_size_picker.value),
                    "learn_rate": float(learning_rate_picker.value),
                    "num_workers": 0,
                    "shuffle": True,
                    "lr_schedule": True,
                },
            },
        )
        active_cfg = parse_cfg(active_cfg)

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

        trained_model = TransferModelPLv2Siamese(active_cfg)
        encoder_parameters = list(trained_model.model.prot_mpnn.parameters())
        if any(parameter.requires_grad for parameter in encoder_parameters):
            raise RuntimeError("ProteinMPNN encoder must be frozen during training.")
        trainable_parameters = [
            parameter
            for parameter in trained_model.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("No trainable ThermoMPNN-D parameters were found.")

        output_dir = repo_root / "checkpoints/thermompnn-d-marimo"
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename="thermompnn-d-{epoch:02d}-{val_ddG_spearman:.3f}",
            monitor="val_ddG_spearman",
            mode="max",
            save_top_k=1,
        )
        early_stopping = EarlyStopping(
            monitor="val_ddG_mse",
            mode="min",
            patience=10,
        )
        trainer = pl.Trainer(
            max_epochs=active_cfg.training.epochs,
            accelerator="gpu" if device_picker.value == "cuda" else "cpu",
            devices=1,
            callbacks=[checkpoint_callback, early_stopping],
            logger=False,
            log_every_n_steps=50,
            enable_checkpointing=True,
        )
        trainer.fit(trained_model, train_loader, val_loader)

        best_checkpoint = checkpoint_callback.best_model_path
        if not best_checkpoint:
            raise RuntimeError(
                "Training finished without a best checkpoint. Check validation "
                "metrics and the Lightning output above."
            )
        training_result = {
            "best checkpoint": best_checkpoint,
            "best validation Spearman": float(checkpoint_callback.best_model_score),
            "epochs completed": trainer.current_epoch + 1,
            "frozen ProteinMPNN parameters": sum(
                parameter.numel() for parameter in encoder_parameters
            ),
            "trainable ThermoMPNN-D parameters": sum(
                parameter.numel() for parameter in trainable_parameters
            ),
        }

    training_result
    return (trained_model,)


@app.cell
def _(
    device_picker,
    pd,
    tied_featurize_mut,
    torch,
    trained_model,
    val_dataset,
):
    if trained_model is None:
        prediction_summary = pd.DataFrame(
            columns=["mutation", "measured ddG", "predicted ddG", "AB prediction", "BA prediction"]
        )
    else:
        trained_model.eval()
        device = torch.device(device_picker.value)
        sample_index = 0
        validation_sample = val_dataset[sample_index]
        batch = tied_featurize_mut([validation_sample], side_chains=False)
        (
            X,
            S,
            mask,
            _lengths,
            chain_M,
            chain_encoding_all,
            residue_idx,
            mut_positions,
            mut_wildtype_AAs,
            mut_mutant_AAs,
            mut_ddGs,
            atom_mask,
        ) = batch
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
        model_inputs = tuple(tensor.to(device) for tensor in model_inputs)
        model_for_prediction = trained_model.to(device)
        with torch.inference_mode():
            pred_ab, pred_ba = model_for_prediction(*model_inputs)
        mutation = validation_sample["mutation"]
        mutation_name = ":".join(
            f"{wt}{position + 1}{mut}"
            for position, wt, mut in zip(
                mutation.position, mutation.wildtype, mutation.mutation
            )
        )
        prediction_summary = pd.DataFrame(
            [
                {
                    "mutation": mutation_name,
                    "measured ddG": float(mut_ddGs.item()),
                    "predicted ddG": float(((pred_ab + pred_ba) / 2).item()),
                    "AB prediction": float(pred_ab.item()),
                    "BA prediction": float(pred_ba.item()),
                }
            ]
        )
    prediction_summary
    return


if __name__ == "__main__":
    app.run()
