"""Trace one MegaScale double mutant through the ThermoMPNN-D model.

Run from the repository root:
    uv run python playground/trace_megascale_double.py

This follows the validation inspector's config/data setup, but uses the
epistatic config and installs temporary hooks to summarize ProteinMPNN and
Siamese-model intermediate tensors without dumping full arrays.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


def summarize(name: str, value: Any) -> None:
    """Print tensor shape and small numeric summary, without the full tensor."""
    if isinstance(value, torch.Tensor):
        detached = value.detach()
        if detached.numel() and detached.is_floating_point():
            print(
                f"  {name}: shape={tuple(detached.shape)}, "
                f"mean={detached.float().mean().item():.4g}, "
                f"std={detached.float().std(unbiased=False).item():.4g}, "
                f"range=[{detached.min().item():.4g}, {detached.max().item():.4g}]"
            )
        else:
            print(f"  {name}: shape={tuple(detached.shape)}, dtype={detached.dtype}")
    else:
        print(f"  {name}: {type(value).__name__}")


def resolve_repo_path(repo_root: Path, value: str) -> str:
    """Resolve a configured path relative to the repository root."""
    path = Path(value)
    return str(path if path.is_absolute() else repo_root / path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace one MegaScale double mutation through ProteinMPNN."
    )
    parser.add_argument("--index", type=int, default=0, help="Dataset row index (default: 0)")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Model device (default: CUDA if available, otherwise CPU)",
    )
    parser.add_argument(
        "--paths-config",
        default="configs/thermompnn_paths.example.yaml",
        help="Data and pretrained-weight paths config",
    )
    parser.add_argument(
        "--training-config",
        default="configs/thermompnn_train_epistatic.yaml",
        help="Double-mutant model config",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    if not (repo_root / "configs").is_dir():
        repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "configs").is_dir():
        raise FileNotFoundError("Could not locate the repository configs/ directory.")

    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from mutation_prediction.thermo_mpnn_d.thermompnn.datasets.dataset_utils import ALPHABET
    from mutation_prediction.thermo_mpnn_d.thermompnn.datasets.v2_datasets import (
        MegaScaleDatasetv2,
        tied_featurize_mut,
    )
    from mutation_prediction.thermo_mpnn_d.thermompnn.model.v2_model import (
        TransferModelv2Siamese,
    )
    from mutation_prediction.thermo_mpnn_d.thermompnn.train_thermompnn import (
        parse_cfg,
    )

    paths_cfg = OmegaConf.load(repo_root / args.paths_config)
    model_cfg = OmegaConf.load(repo_root / args.training_config)
    cfg = parse_cfg(OmegaConf.merge(paths_cfg, model_cfg))

    if cfg.model.aggregation != "siamese":
        raise ValueError("This trace expects model.aggregation: siamese.")
    if "double" not in cfg.data.mut_types:
        raise ValueError("This trace expects data.mut_types to include 'double'.")

    for key in ("megascale_splits", "megascale_pdbs", "megascale_csv"):
        cfg.data_loc[key] = resolve_repo_path(repo_root, cfg.data_loc[key])
    cfg.platform.thermompnn_dir = resolve_repo_path(
        repo_root, cfg.platform.thermompnn_dir
    )
    if cfg.model.load_pretrained:
        model_version = cfg.model.get("version", "v_48_020.pt")
        checkpoint = Path(cfg.platform.thermompnn_dir) / model_version
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"ProteinMPNN checkpoint not found: {checkpoint}. "
                "Download the configured checkpoint into the directory set by "
                "platform.thermompnn_dir."
            )

    print("Loading MegaScale dataset and its configured protein structures...")
    dataset = MegaScaleDatasetv2(cfg, split=args.split)
    if not 0 <= args.index < len(dataset):
        raise IndexError(
            f"Index {args.index} is outside the {args.split} split "
            f"(contains {len(dataset)} examples)."
        )

    row = dataset.df.iloc[args.index]
    sample = dataset[args.index]
    mutation = sample["mutation"]
    batch = tied_featurize_mut([sample], side_chains=cfg.data.side_chains)
    if batch is None:
        raise ValueError(f"Could not featurize dataset example {args.index}.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")

    batch = tuple(value.to(device) if isinstance(value, torch.Tensor) else value for value in batch)
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

    print("\n1. MegaScale example")
    print(f"  split/index: {args.split}/{args.index}")
    print(f"  protein: {row.WT_name}")
    print(f"  CSV mutation: {row.mut_type}")
    print(f"  sequence length: {len(sample['seq'])}")
    for i, (position, wt, mutant) in enumerate(
        zip(mutation.position, mutation.wildtype, mutation.mutation)
    ):
        print(
            f"  mutation {i + 1}: {wt}{position + 1}{mutant} "
            f"(zero-based index {position})"
        )
    print(f"  model target (dataset ddG): {mutation.ddG:.6g}")

    print("\n2. Batched model inputs")
    summarize("X (backbone coordinates)", X)
    summarize("S (sequence token IDs)", S)
    summarize("mask (valid structured residues)", mask)
    summarize("mut_positions", mut_positions)
    summarize("mut_wildtype_AAs", mut_wildtype_AAs)
    summarize("mut_mutant_AAs", mut_mutant_AAs)
    summarize("mut_ddGs (regression target)", mut_ddGs)
    for i, (position, wt_id, mut_id) in enumerate(
        zip(
            mut_positions[0, : len(mutation.position)].tolist(),
            mut_wildtype_AAs[0, : len(mutation.position)].tolist(),
            mut_mutant_AAs[0, : len(mutation.position)].tolist(),
        )
    ):
        print(
            f"  decoded mutation {i + 1}: {ALPHABET[wt_id]}{position + 1}"
            f"{ALPHABET[mut_id]}"
        )

    model = TransferModelv2Siamese(cfg).to(device).eval()
    encoder = model.prot_mpnn
    hooks = []

    def feature_hook(_module, _inputs, output):
        edge_features, neighbor_indices = output
        print("\n3. ProteinFeatures: geometric graph passed to ProteinMPNN")
        summarize("E (embedded residue-neighbor features)", edge_features)
        summarize("E_idx (neighbor residue indices)", neighbor_indices)
        for i, position in enumerate(mutation.position):
            neighbors = neighbor_indices[0, position].tolist()
            print(f"  neighbors for mutation {i + 1} (0-based indices): {neighbors}")

    def layer_hook(stage: str, layer_index: int):
        def hook(_module, _inputs, output):
            print(f"  {stage} layer {layer_index + 1}:")
            if isinstance(output, tuple):
                for output_index, tensor in enumerate(output):
                    summarize(f"output[{output_index}]", tensor)
            else:
                summarize("residue representation", output)

        return hook

    def mpnn_hook(_module, _inputs, output):
        hidden_layers, sequence_embedding, log_probs, edge_embedding = output
        print("\n4. ProteinMPNN outputs")
        for i, hidden in enumerate(hidden_layers):
            summarize(f"decoder hidden layer {i + 1}", hidden)
        summarize("wild-type sequence embedding (h_S)", sequence_embedding)
        summarize("amino-acid log probabilities", log_probs)
        summarize("learned edge embeddings (h_E)", edge_embedding)

    def projection_hook(_module, inputs):
        print("\n5. Siamese mutation-site features before projection")
        summarize("two concatenated site vectors [sites, batch, features]", inputs[0])

    head_call = 0

    def head_hook(_module, inputs):
        nonlocal head_call
        order = "AB" if head_call == 0 else "BA"
        head_call += 1
        print(f"\n6. Prediction-head input ({order} order)")
        summarize("combined site vector", inputs[0])

    hooks.append(encoder.features.register_forward_hook(feature_hook))
    for i, layer in enumerate(encoder.encoder_layers):
        hooks.append(layer.register_forward_hook(layer_hook("encoder", i)))
    for i, layer in enumerate(encoder.decoder_layers):
        hooks.append(layer.register_forward_hook(layer_hook("decoder", i)))
    hooks.append(encoder.register_forward_hook(mpnn_hook))
    hooks.append(model.light_attention.register_forward_pre_hook(projection_hook))
    hooks.append(model.ddg_out.register_forward_pre_hook(head_hook))

    print(f"\nRunning model on {device} (evaluation mode, no gradients)...")
    try:
        with torch.inference_mode():
            pred_ab, pred_ba = model(
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
    finally:
        for hook in hooks:
            hook.remove()

    average = (pred_ab + pred_ba) / 2
    disagreement = (pred_ab - pred_ba).abs() / 2
    print("\n7. Siamese predictions and target")
    print(f"  prediction AB: {pred_ab.item():.6g}")
    print(f"  prediction BA: {pred_ba.item():.6g}")
    print(f"  averaged prediction: {average.item():.6g}")
    print(f"  half absolute order disagreement: {disagreement.item():.6g}")
    print(f"  MegaScale target: {mut_ddGs.item():.6g}")


if __name__ == "__main__":
    main()
