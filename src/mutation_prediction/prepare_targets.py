"""Cache ThermoMPNN single-mutant effects and residual targets for MegaScale-D.

One encoder pass per WT structure supplies both constituent predictions for
every pair belonging to that protein. This script does not generate augmentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mutation_prediction.thermompnn import ALPHABET, ThermoMPNNPredictor
from mutation_prediction.vendor.thermompnn.thermompnn.protein_mpnn_utils import (
    parse_PDB,
)

REPO_ROOT = Path.cwd()
SUBSTITUTION = re.compile(r"([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])")
PREDICTION_COLUMNS = ("mutation_1", "mutation_2", "pred_ddg_1", "pred_ddg_2", "baseline_ddg")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_pair(value: str) -> list[tuple[str, int, str]]:
    """Return (WT, zero-based position, mutant) for two distinct substitutions."""
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected exactly two substitutions, got {value!r}")
    parsed = []
    for part in parts:
        match = SUBSTITUTION.fullmatch(part)
        if match is None:
            raise ValueError(f"Invalid substitution {part!r}")
        wt, position, mutant = match.groups()
        if wt == mutant:
            raise ValueError(f"Expected a nonidentity substitution, got {part!r}")
        parsed.append((wt, int(position) - 1, mutant))
    if parsed[0][1] == parsed[1][1]:
        raise ValueError(f"The two substitutions must use distinct positions: {value!r}")
    return parsed


def add_epsilon_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Use published MegaScale-D CSV labels (positive = destabilizing).

    epsilon_target is the correction to the cached ThermoMPNN baseline. It
    includes single-mutant prediction errors as well as biological epistasis.
    The existing experimental ddG_ML_epi_term column is a different quantity.
    """
    columns = ["ddG_ML", "pred_ddg_1", "pred_ddg_2", "baseline_ddg"]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for epsilon targets: {sorted(missing)}")
    values = frame[columns].apply(pd.to_numeric, errors="raise")
    if frame.empty or not np.isfinite(values.to_numpy()).all():
        raise ValueError("Epsilon targets require finite experimental labels and predictions")
    if not np.allclose(
        values.baseline_ddg, values.pred_ddg_1 + values.pred_ddg_2,
        rtol=1e-10, atol=1e-10,
    ):
        raise ValueError("baseline_ddg does not match the sum of its single-mutant predictions")
    result = frame.copy()
    result["true_ddg_12"] = values.ddG_ML
    result["epsilon_target"] = values.ddG_ML - values.baseline_ddg
    return result


def precompute_single_effects(
    frame: pd.DataFrame,
    predictor: ThermoMPNNPredictor,
    pdb_dir: Path,
    *,
    chain: str = "A",
    progress: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Append predictions, retaining every input row and original target value.

    aa_seq in MegaScale-D is the double-mutant sequence. Reverting the two
    substitutions must reproduce the WT structure sequence exactly. Unmapped
    residues, missing structures/backbones, and malformed pairs raise errors;
    rows are never silently dropped or assigned zero predictions.
    """
    required = ["WT_name", "mut_type", "aa_seq"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    if frame.empty or frame[required].isna().any().any():
        raise ValueError("Input must contain pairs with nonmissing WT_name, mut_type, and aa_seq")
    if set(PREDICTION_COLUMNS) & set(frame.columns):
        raise ValueError("Input already contains prediction columns; use the original MegaScale-D CSV")

    result = frame.copy().reset_index(drop=True)
    pairs = result.mut_type.str.split(":")
    if not pairs.map(len).eq(2).all():
        raise ValueError("Input must contain only measured double-substitution pairs")
    result["mutation_1"] = pairs.str[0]
    result["mutation_2"] = pairs.str[1]
    predictions = np.empty((len(result), 2), dtype=np.float64)
    structures = {}
    unique_mutations = 0

    groups = result.groupby("WT_name", sort=False)
    for name, group in tqdm(groups, total=groups.ngroups, desc="WT proteins", disable=not progress):
        pdb_path = pdb_dir / (name.removesuffix(".pdb").replace("|", ":") + ".pdb")
        if not pdb_path.is_file():
            raise FileNotFoundError(f"Missing WT structure for {name}: {pdb_path}")
        protein = parse_PDB(str(pdb_path), input_chain_list=[chain])[0]
        reference = protein["seq"]
        if not reference:
            raise ValueError(f"No residues found for {name}, chain {chain}")

        # Validate every input row before computing this reference's predictions.
        parsed_rows = []
        for row in group.itertuples():
            try:
                substitutions = parse_pair(row.mut_type)
                wt_sequence = list(row.aa_seq)
                for wt, position, mutant in substitutions:
                    if position >= len(wt_sequence) or wt_sequence[position] != mutant:
                        raise ValueError("aa_seq does not match the mutant identity/position")
                    wt_sequence[position] = wt
                if "".join(wt_sequence) != reference:
                    raise ValueError("reconstructed WT sequence does not match the structure")
            except ValueError as error:
                raise ValueError(
                    f"CSV row {row.Index + 2}, {name}, {row.mut_type}: {error}"
                ) from error
            parsed_rows.append((row.Index, substitutions))

        table = predictor.predict(protein)  # [L, 20], shared by every pair in this group
        if table.shape != (len(reference), 20):
            raise ValueError(f"Unexpected prediction table shape for {name}: {tuple(table.shape)}")
        single_effects = {}
        for index, substitutions in parsed_rows:
            for column, substitution in enumerate(substitutions):
                if substitution not in single_effects:
                    wt, position, mutant = substitution
                    value = float(table[position, ALPHABET.index(mutant)])
                    if not np.isfinite(value):
                        raise ValueError(
                            f"Nonfinite prediction for {name}, {wt}{position + 1}{mutant}; "
                            "check for a missing backbone at the mutation site"
                        )
                    single_effects[substitution] = value
                predictions[index, column] = single_effects[substitution]
        unique_mutations += len(single_effects)
        structures[name] = {"path": str(pdb_path.resolve()), "sha256": sha256(pdb_path)}

    result["pred_ddg_1"] = predictions[:, 0]
    result["pred_ddg_2"] = predictions[:, 1]
    result["baseline_ddg"] = predictions.sum(axis=1)
    metadata = {
        "pair_count": len(result),
        "protein_count": len(structures),
        "unique_single_mutation_count": unique_mutations,
        "chain": chain,
        "structures": structures,
    }
    return result, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input-csv", type=Path, default=REPO_ROOT / "data/raw/megascale/csv/Megascale-D-full.csv")
    inputs.add_argument("--predictions-csv", type=Path, help="Add epsilon targets to a previous prediction export without running inference")
    parser.add_argument("--pdb-dir", type=Path, default=REPO_ROOT / "data/raw/megascale/pdb")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints/thermompnn/ThermoMPNN-ens1.ckpt")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/processed/thermompnn/Megascale-D-single-predictions.csv")
    parser.add_argument("--chain", default="A")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    args = parser.parse_args()

    if args.predictions_csv is not None:
        result = pd.read_csv(args.predictions_csv)
        source_metadata = args.predictions_csv.with_suffix(".metadata.json")
        metadata = json.loads(source_metadata.read_text()) if source_metadata.exists() else {}
        if metadata.get("prediction_sign", "positive = destabilizing") != "positive = destabilizing":
            raise ValueError("Cached predictions must use positive = destabilizing")
        if "created_at_utc" in metadata:
            metadata["prediction_created_at_utc"] = metadata["created_at_utc"]
        metadata["predictions_csv"] = {
            "path": str(args.predictions_csv.resolve()),
            "sha256": sha256(args.predictions_csv),
        }
    else:
        frame = pd.read_csv(args.input_csv)
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = ThermoMPNNPredictor(args.checkpoint, device=device)
        result, metadata = precompute_single_effects(frame, predictor, args.pdb_dir, chain=args.chain)
        metadata.update({
            "model": "ThermoMPNN-v2-single",
            "device": str(predictor.device),
            "input_csv": {"path": str(args.input_csv.resolve()), "sha256": sha256(args.input_csv)},
            "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)},
        })
    result = add_epsilon_targets(result)
    metadata.update({
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pair_count": len(result),
        "prediction_units": "kcal/mol",
        "prediction_sign": "positive = destabilizing",
        "source_columns": "preserved without target sign changes",
        "augmentation": False,
        "experimental_ddg_column": "ddG_ML",
        "experimental_ddg_sign": "positive = destabilizing",
        "training_target": "epsilon_target",
        "target_formula": "epsilon_target = true_ddg_12 - (pred_ddg_1 + pred_ddg_2)",
        "target_kind": "correction to frozen ThermoMPNN predictions",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Saved {len(result):,} pairs with baseline predictions and epsilon targets to {args.output}"
    )
    print(f"Provenance: {metadata_path}")


if __name__ == "__main__":
    main()
