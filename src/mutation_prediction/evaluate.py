"""Compare models on fixed MegaScale-D/PTMUL-D benchmarks in physical ΔΔG units.

Run ``python -m mutation_prediction.evaluate configs/model_evaluate.yaml --help``.
New models can supply a Python factory or a keyed CSV of predictions; neither
requires changing the scoring or dataset-selection implementation.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.stats import pearsonr, spearmanr

from mutation_prediction.evaluation_data import finite_values, load_dataset
from mutation_prediction.evaluation_models import DatasetPredictions
from mutation_prediction.prepare_targets import sha256


def regression_metrics(truth, prediction):
    """Pooled per-mutation metrics; undefined correlations are JSON null.

    Compute on complete arrays, never average batch RMSEs or correlations.
    """
    truth = finite_values(truth, "experimental targets")
    prediction = finite_values(prediction, "predictions")
    if truth.shape != prediction.shape or truth.ndim != 1 or not truth.size:
        raise ValueError("Metrics require matching, nonempty one-dimensional arrays")
    mse = float(np.mean(np.square(prediction - truth)))
    correlations_defined = (
        len(truth) > 1 and np.ptp(truth) > 0 and np.ptp(prediction) > 0
    )
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "pearson": float(pearsonr(truth, prediction).statistic)
        if correlations_defined
        else None,
        "spearman": float(spearmanr(truth, prediction).statistic)
        if correlations_defined
        else None,
    }


def evaluate(config, output_dir, *, datasets=None, models=None, device="auto"):
    """Evaluate configured models and write aligned predictions and metrics.

    Paths are relative to the calling working directory. All models see exactly
    the same examples. Unknown, missing, duplicate, or nonfinite predictions are
    errors, and targets are never passed into batch predictors. A factory is
    ``factory(model_spec, torch.device) -> callable(MutationBatch) -> Tensor``.
    """
    config = (
        OmegaConf.to_container(config, resolve=True)
        if OmegaConf.is_config(config)
        else config
    )
    dataset_names = (
        datasets
        if datasets is not None
        else [
            name
            for name, spec in config["datasets"].items()
            if spec.get("enabled", True)
        ]
    )
    model_names = models if models is not None else list(config["models"])
    if not dataset_names or not model_names:
        raise ValueError("Select at least one dataset and one model")
    for names, specs in (
        (dataset_names, config["datasets"]),
        (model_names, config["models"]),
    ):
        if len(names) != len(set(names)):
            raise ValueError("Dataset and model selections must not contain duplicates")
        for name in names:
            if name not in specs or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                raise ValueError(f"Unknown or invalid evaluation name: {name!r}")
    if config.get("batch_size", 64) < 1 or config.get("num_workers", 0) < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(config), output_dir / "config.yaml")
    seed = config.get("seed", 42)
    report, provenance = [], {}
    for dataset_name in dataset_names:
        dataset = load_dataset(dataset_name, config["datasets"][dataset_name])
        target = dataset.frame.true_ddg.to_numpy(dtype=np.float64)
        directory = output_dir / dataset_name
        directory.mkdir()
        dataset.frame.to_csv(directory / "examples.csv", index=False)
        predictions = DatasetPredictions(dataset, config, device, seed)

        for model_name in model_names:
            spec = config["models"][model_name]
            prediction = predictions.predict(model_name)
            metrics = regression_metrics(target, prediction)
            record = {
                "dataset": dataset_name,
                "model": model_name,
                "group": spec.get("group", model_name),
                "n_mutations": len(dataset),
                "n_proteins": int(dataset.frame.WT_name.nunique()),
                **metrics,
            }
            reference = config.get("paper_no_da", {}).get(dataset_name)
            if reference:
                for metric in ("rmse", "pearson", "spearman"):
                    record[f"paper_no_da_{metric}_mean"] = reference[f"{metric}_mean"]
                    record[f"paper_no_da_{metric}_sd"] = reference[f"{metric}_sd"]
                    record[f"{metric}_minus_paper_mean"] = (
                        metrics[metric] - reference[f"{metric}_mean"]
                        if metrics[metric] is not None
                        else None
                    )
            report.append(record)
            frame = dataset.frame[
                ["example_id", "WT_name", "mut_type", "true_ddg"]
            ].copy()
            frame["predicted_ddg"] = prediction
            frame["error"] = prediction - target
            frame["squared_error"] = np.square(frame.error)
            if predictions.baseline is not None:
                frame["baseline_ddg"] = predictions.baseline
                frame["predicted_epsilon"] = prediction - predictions.baseline
            frame.to_csv(directory / f"{model_name}.predictions.csv", index=False)
            per_protein = [
                {
                    "WT_name": protein,
                    "n_mutations": len(group),
                    **regression_metrics(group.true_ddg, group.predicted_ddg),
                }
                for protein, group in frame.groupby("WT_name", sort=False)
            ]
            pd.DataFrame(per_protein).to_csv(
                directory / f"{model_name}.per_protein.csv", index=False
            )
            print(
                f"{dataset_name} / {model_name}: RMSE={metrics['rmse']:.6f}, MSE={metrics['mse']:.6f}",
                flush=True,
            )
        provenance[dataset_name] = {
            "n_mutations": len(dataset),
            "n_proteins": int(dataset.frame.WT_name.nunique()),
            "examples_csv_sha256": sha256(directory / "examples.csv"),
            "files": {
                str(path.resolve()): sha256(path) for path in sorted(set(dataset.files))
            },
        }
    summary = pd.DataFrame(report)
    summary.to_csv(output_dir / "metrics.csv", index=False)
    replicates = []
    for (dataset, group), rows in summary.groupby(["dataset", "group"], sort=False):
        entry = {"dataset": dataset, "group": group, "n_runs": len(rows)}
        for metric in ("mse", "rmse", "pearson", "spearman"):
            values = rows[metric].dropna()
            entry[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            entry[f"{metric}_sd"] = (
                float(values.std(ddof=1)) if len(values) > 1 else None
            )
        replicates.append(entry)
    pd.DataFrame(replicates).to_csv(output_dir / "replicates.csv", index=False)
    manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "device": str(device),
        "seed": seed,
        "target_units": "kcal/mol",
        "target_sign": "positive = destabilizing",
        "aggregation": "pooled per mutation; not an average over batches or proteins",
        "datasets": provenance,
        "metrics": report,
        "paper_reference": config.get("paper_reference"),
        "paper_no_da": config.get("paper_no_da", {}),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Benchmark YAML listing datasets and models")
    parser.add_argument(
        "--datasets", nargs="+", help="Dataset names; default: all enabled datasets"
    )
    parser.add_argument(
        "--models", nargs="+", help="Model names; default: all configured models"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="New directory for predictions, metrics and provenance",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="Override a model checkpoint; repeat for multiple models",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    for override in args.checkpoint:
        name, separator, path = override.partition("=")
        if not separator or not path or name not in cfg.models:
            parser.error("--checkpoint must be MODEL=PATH for a configured model")
        cfg.models[name].checkpoint = path
    output = args.output_dir or str(
        Path("results/evaluation") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    )
    summary = evaluate(
        cfg, output, datasets=args.datasets, models=args.models, device=args.device
    )
    print(summary.to_string(index=False))
    print(f"Evaluation saved to {output}")


if __name__ == "__main__":
    main()
