"""Prediction adapters for checkpoint, additive, CSV, and experimental models."""

from __future__ import annotations

import importlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from mutation_prediction.data.collate import tied_featurize_mut
from mutation_prediction.evaluation_data import KEYS, finite_values, require_unique
from mutation_prediction.prepare_targets import sha256


class DatasetPredictions:
    """Share ordered predictions and a baseline across models on one dataset."""

    def __init__(self, dataset, config, device, seed):
        self.dataset = dataset
        self.config = config
        self.device = device
        self.seed = seed
        self.baseline = None
        self._predictions = {}
        self._visiting = set()

    def get_baseline(self):
        if self.baseline is None:
            checkpoint = self.config.get("baseline_checkpoint")
            self.baseline = additive_baseline(self.dataset, checkpoint, self.device)
            if checkpoint:
                self.dataset.files.append(Path(checkpoint))
        return self.baseline

    def predict(self, name):
        if name in self._predictions:
            return self._predictions[name]
        if name in self._visiting:
            raise ValueError("Ensemble members contain a cycle")
        if name not in self.config["models"]:
            raise ValueError(f"Unknown ensemble member {name!r}")
        self._visiting.add(name)
        spec = self.config["models"][name]
        if spec["kind"] == "ensemble":
            members = spec["members"]
            if not members or len(set(members)) != len(members):
                raise ValueError("Ensembles need distinct, nonempty members")
            result = np.mean([self.predict(member) for member in members], axis=0)
        else:
            sign = spec.get("output_sign", 1)
            if sign not in (-1, 1):
                raise ValueError("output_sign must be 1 or -1")
            output = spec.get("output", "ddg")
            if output not in ("ddg", "epsilon"):
                raise ValueError("Model output must be ddg or epsilon")
            if output == "epsilon":
                self.get_baseline()
            result = self._predict_raw(spec) * sign
            if output == "epsilon":
                result = self.get_baseline() + result
        result = finite_values(result, name)
        if len(result) != len(self.dataset):
            raise ValueError(f"{name}: expected one prediction per evaluation row")
        self._predictions[name] = result
        self._visiting.remove(name)
        return result

    def _predict_raw(self, spec):
        # Baseline generation can consume RNG state. Reset here so a model's
        # predictions do not depend on which adapters were evaluated first.
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        kind = spec["kind"]
        if kind == "additive":
            return self.get_baseline()
        if kind == "column":
            return finite_values(self.dataset.frame[spec["column"]], spec["column"])
        if kind == "csv":
            return align_predictions(
                self.dataset,
                spec["predictions"][self.dataset.name],
                spec.get("column", "predicted_ddg"),
            )
        if kind not in ("checkpoint", "factory"):
            raise ValueError(f"Unknown model kind {kind!r}")
        predictor = (
            load_checkpoint(spec, self.device)
            if kind == "checkpoint"
            else import_factory(spec["factory"])(spec, self.device)
        )
        result = predict_batches(
            predictor,
            self.dataset,
            self.device,
            self.dataset.spec.get("batch_size", self.config.get("batch_size", 64)),
            self.config.get("num_workers", 0),
        )
        if spec.get("checkpoint"):
            self.dataset.files.append(Path(spec["checkpoint"]))
        return result


def import_factory(reference):
    """Import a user-provided ``package.module:function``."""
    module, separator, name = reference.partition(":")
    if not separator:
        raise ValueError("Factories must be written as package.module:function")
    return getattr(importlib.import_module(module), name)


def align_predictions(dataset, path, column):
    """Align external predictions by keys, never by CSV row order."""
    predictions = pd.read_csv(path)
    keys = ["example_id"] if "example_id" in predictions else KEYS
    require_unique(predictions, keys, str(path))
    require_unique(dataset.frame, keys, dataset.name)
    if column in keys:
        raise ValueError("Prediction column cannot be an example key")
    joined = dataset.frame[keys].merge(
        predictions[keys + [column]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError(
            f"{path}: predictions must cover exactly the evaluation examples"
        )
    # Merge back explicitly: outer joins may sort keys in newer pandas versions.
    ordered = dataset.frame[keys].merge(
        predictions[keys + [column]], on=keys, how="left", validate="one_to_one"
    )
    dataset.files.append(Path(path))
    return finite_values(ordered[column], str(path))


class CheckpointPredictor:
    def __init__(self, model):
        self.model = model

    def __call__(self, batch):
        X, S, mask, _, chain_M, chains, indices, pos, wt, mut, target, atom_mask = batch
        return self.model(
            X, S, mask, chain_M, indices, chains, pos, wt, mut, target, atom_mask
        )


def load_checkpoint(spec, device):
    """Restore a complete project Lightning checkpoint without vanilla weights.

    Custom architectures can specify model_factory(cfg) -> nn.Module; otherwise
    the standard ProteinMPNN extractor and configured training head are rebuilt.
    The complete state is loaded strictly, with no randomly retained weights.
    """
    from mutation_prediction.model import (
        MutationPredictionModel,
        ProteinMPNNFeatureExtractor,
    )
    from mutation_prediction.model.train import HEADS, parse_cfg
    from mutation_prediction.thermompnn_d_vendor.thermompnn.thermompnn.protein_mpnn_utils import (
        ProteinMPNN,
    )

    if not spec.get("checkpoint"):
        raise ValueError(
            "Supply a validation-selected checkpoint with --checkpoint MODEL=PATH"
        )
    payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    cfg = parse_cfg(OmegaConf.create(payload["hyper_parameters"]["cfg"]))
    if spec.get("model_factory"):
        model = import_factory(spec["model_factory"])(cfg)
    else:
        version = cfg.model.get("version", "v_48_020.pt")
        if version != "v_48_020.pt" and "encoder_neighbors" not in spec:
            raise ValueError(
                "Nonstandard encoders require encoder_neighbors or model_factory"
            )
        encoder = ProteinMPNN(
            use_ipmp="IPMP" in version,
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=spec.get("encoder_neighbors", 48),
            augment_eps=0.0,
            dropout=cfg.model.get("proteinmpnn_dropout", 0.0),
        )
        extractor = ProteinMPNNFeatureExtractor(cfg, protein_encoder=encoder)
        if cfg.model.head not in HEADS:
            raise ValueError(f"Unknown head {cfg.model.head!r}; supply model_factory")
        model = MutationPredictionModel(
            extractor, HEADS[cfg.model.head](extractor.output_dim)
        )
    state = payload["state_dict"]
    if not all(key.startswith("model.") for key in state):
        raise ValueError(
            "Expected a project Lightning checkpoint with model.* state keys"
        )
    model.load_state_dict(
        {key.removeprefix("model."): value for key, value in state.items()}, strict=True
    )
    model.to(device).eval().requires_grad_(False)
    return CheckpointPredictor(model)


def predict_batches(predictor, dataset, device, batch_size, num_workers):
    """Predict in dataset order from label-free MutationBatch tensors.

    Custom factories return a callable(batch) -> Tensor[B] or Tensor[B, 1].
    batch[10] is always zero; outputs must contain exactly one scalar per row.
    """
    if isinstance(predictor, torch.nn.Module):
        predictor.to(device).eval().requires_grad_(False)
    if num_workers:
        # Parse/validate in the parent so structure hashes are recorded even
        # when DataLoader workers subsequently read independent dataset copies.
        for index in tqdm(range(len(dataset)), desc=f"{dataset.name} structures"):
            dataset[index]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=tied_featurize_mut,
    )
    outputs = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=dataset.name):
            batch = tuple(
                value.to(device) if isinstance(value, torch.Tensor) else value
                for value in batch
            )
            result = predictor(batch)
            if not isinstance(result, torch.Tensor) or tuple(result.shape) not in (
                (len(batch[0]),),
                (len(batch[0]), 1),
            ):
                raise ValueError(
                    "Predictors must return a Tensor shaped [batch] or [batch, 1]"
                )
            outputs.append(result.detach().cpu().reshape(-1).double().numpy())
    return finite_values(np.concatenate(outputs), f"{dataset.name} predictions")


def additive_baseline(dataset, checkpoint, device):
    """Use a cache when present, otherwise run one WT pass per structure."""
    if "baseline_ddg" in dataset.frame:
        metadata_path = Path(dataset.spec["csv"]).with_suffix(".metadata.json")
        if metadata_path.is_file() and checkpoint:
            metadata = json.loads(metadata_path.read_text())
            recorded_hash = metadata.get("checkpoint", {}).get("sha256")
            if recorded_hash and recorded_hash != sha256(Path(checkpoint)):
                raise ValueError(
                    "Baseline checkpoint differs from the cached label-generation checkpoint"
                )
        return finite_values(dataset.frame.baseline_ddg, "cached additive baseline")
    if not checkpoint:
        raise ValueError(
            "baseline_checkpoint is required when baseline_ddg is not cached"
        )
    from mutation_prediction.thermompnn import ALPHABET, ThermoMPNNPredictor

    predictor = ThermoMPNNPredictor(checkpoint, device=device)
    tables = {}
    values = []
    for index in tqdm(range(len(dataset)), desc=f"{dataset.name} additive baseline"):
        record = dataset[index]
        key = dataset.frame.iloc[index].WT_name
        if key not in tables:
            tables[key] = predictor.predict(record).numpy()
        mutation = record["mutation"]
        values.append(
            sum(
                tables[key][p, ALPHABET.index(aa)]
                for p, aa in zip(mutation.position, mutation.mutation)
            )
        )
    return finite_values(values, "computed additive baseline")
