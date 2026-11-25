"""Load benchmark examples without changing their published memberships."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mutation_prediction.data.mutations import Mutation
from mutation_prediction.prepare_targets import parse_pair
from mutation_prediction.thermompnn_d_vendor.thermompnn.thermompnn.protein_mpnn_utils import (
    parse_PDB,
)

KEYS = ["WT_name", "mut_type"]


def require_unique(frame, columns, description):
    if frame[columns].isna().any().any() or frame.duplicated(columns).any():
        raise ValueError(f"{description} has missing or duplicate keys: {columns}")


def finite_values(values, description):
    array = np.asarray(pd.to_numeric(values, errors="raise"), dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{description} contains nonfinite values")
    return array


@dataclass
class EvaluationDataset:
    """An ordered benchmark table and lazily parsed structure records.

    ``frame`` uses positive-destabilizing kcal/mol targets. Records handed to
    predictors have zero targets, so experimental labels never enter inference.
    """

    name: str
    spec: dict
    frame: pd.DataFrame
    files: list[Path]
    _structures: dict = field(default_factory=dict, repr=False)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        return self._make_record(index)

    def _make_record(self, index):
        row = self.frame.iloc[index]
        if self.spec["kind"] == "ptmul":
            pdb, chain = str(row.WT_name)[:-1], str(row.WT_name)[-1]
        else:
            pdb = str(row.WT_name).removesuffix(".pdb").replace("|", ":")
            chain = self.spec.get("chain", "A")
        path = Path(self.spec["pdb_dir"]) / self.spec.get(
            "pdb_template", "{pdb}.pdb"
        ).format(pdb=pdb, chain=chain, WT_name=row.WT_name)
        key = (path, chain)
        if key not in self._structures:
            if not path.is_file():
                raise FileNotFoundError(path)
            protein = parse_PDB(str(path), input_chain_list=[chain])[0]
            if not protein["seq"]:
                raise ValueError(f"No residues in {path}, chain {chain}")
            if self.spec["kind"] == "ptmul":
                protein["_pdb_positions"] = pdb_position_map(path, chain)
            self._structures[key] = protein
            self.files.append(path)
        protein = deepcopy(self._structures[key])
        if self.spec["kind"] == "ptmul":
            substitutions = []
            position_map = protein["_pdb_positions"]
            overrides = self.spec.get("position_map_csv")
            if overrides:
                mapping_path = Path(overrides)
                mapping = pd.read_csv(mapping_path)
                require_unique(mapping, ["WT_name", "mutation"], str(mapping_path))
                selected = mapping.loc[mapping.WT_name.eq(row.WT_name)]
                mapped_positions = dict(zip(selected.mutation, selected.position))
                if mapping_path not in self.files:
                    self.files.append(mapping_path)
            else:
                mapped_positions = {}
            for mutation in str(row.mut_type).split(":"):
                match = re.fullmatch(
                    r"([ACDEFGHIKLMNPQRSTVWY])(-?\d+[A-Za-z]?)([ACDEFGHIKLMNPQRSTVWY])",
                    mutation,
                )
                if match is None or match[1] == match[3]:
                    raise ValueError(f"Invalid PTMUL substitution: {mutation}")
                if mutation in mapped_positions:
                    position = mapped_positions[mutation]
                    if (
                        not np.isfinite(position)
                        or int(position) != position
                        or position < 0
                    ):
                        raise ValueError(f"Invalid zero-based position for {mutation}")
                    position = int(position)
                elif match[2] in position_map:
                    position = position_map[match[2]]
                else:
                    raise ValueError(
                        f"{row.example_id}: residue {match[2]} absent from {path}"
                    )
                substitutions.append((match[1], position, match[3]))
        else:
            substitutions = parse_pair(str(row.mut_type))
        if len({p for _, p, _ in substitutions}) != 2:
            raise ValueError(f"{row.example_id}: expected two distinct mutation sites")
        for wt, position, _ in substitutions:
            if position >= len(protein["seq"]) or protein["seq"][position] != wt:
                raise ValueError(
                    f"{row.example_id}: WT identity does not match {path} at index {position}; "
                    "check dataset/PDB residue numbering"
                )
            for atom in ("N", "CA", "C", "O"):
                coords = protein[f"coords_chain_{chain}"][f"{atom}_chain_{chain}"][
                    position
                ]
                if not np.isfinite(coords).all():
                    raise ValueError(
                        f"{row.example_id}: missing backbone at index {position}"
                    )
        protein["mutation"] = Mutation(
            [p for _, p, _ in substitutions],
            [wt for wt, _, _ in substitutions],
            [mut for _, _, mut in substitutions],
            0.0,
            str(row.WT_name),
        )
        return protein


def pdb_position_map(path, chain):
    """Map original PDB numbers, including insertion codes, to padded indices.

    Match parse_PDB's ordering: numeric gaps occupy one slot, and insertion
    codes at a residue are sorted. Do not guess offsets from residue identities.
    """
    residues = set()
    for line in path.read_text().splitlines():
        if (
            line.startswith("ATOM  ")
            or (line.startswith("HETATM") and line[17:20] == "MSE")
        ) and line[21:22] == chain:
            residues.add((int(line[22:26]), line[26:27].strip()))
    if not residues:
        raise ValueError(f"No residues in {path}, chain {chain}")
    mapping, offset = {}, 0
    by_number = {}
    for number, insertion in residues:
        by_number.setdefault(number, []).append(insertion)
    for number in range(min(by_number), max(by_number) + 1):
        for insertion in sorted(by_number.get(number, [None])):
            if insertion is not None:
                mapping[f"{number}{insertion}"] = offset
            offset += 1
    return mapping


def load_dataset(name, spec):
    """Load all examples, validating counts and any canonical reference CSV.

    MegaScale reference CSVs define membership; extra cached train/val rows are
    ignored, while missing test rows or changed labels are errors. PTMUL uses
    its curated CSV after explicitly selecting double substitutions.
    """
    kind = spec["kind"]
    source = Path(spec["csv"])
    frame = pd.read_csv(source)
    files = [source]
    if kind == "megascale":
        require_unique(frame, KEYS, name)
        reference_path = Path(spec["reference_csv"])
        reference = pd.read_csv(reference_path)
        require_unique(reference, KEYS, f"{name} reference")
        targets = finite_values(reference.ddG_ML, f"{name} reference targets")
        reference = reference[KEYS].assign(_true_ddg=targets)
        frame = reference.merge(
            frame, on=KEYS, how="left", validate="one_to_one", indicator=True
        )
        if not frame["_merge"].eq("both").all():
            raise ValueError(f"{name}: cached CSV is missing published mutations")
        observed = finite_values(frame.ddG_ML, f"{name} targets")
        if not np.allclose(observed, frame["_true_ddg"], rtol=1e-12, atol=1e-12):
            raise ValueError(f"{name}: cached targets differ from published targets")
        if "split" in frame and not frame.split.eq(spec.get("split", "test")).all():
            raise ValueError(f"{name}: rows have the wrong split label")
        frame = frame.drop(columns="_merge")
        files.append(reference_path)
    elif kind == "ptmul":
        frame = frame.rename(
            columns={
                spec.get("protein_column", "PDB"): "WT_name",
                spec.get("mutation_column", "MUTS"): "mut_type",
            }
        )
        frame["mut_type"] = frame.mut_type.str.replace(";", ":", regex=False)
        # The upstream PTMUL collection may include higher-order mutations.
        frame = frame.loc[frame.mut_type.str.count(":").eq(1)].copy()
        sign = spec.get("target_sign", 1)
        if sign not in (-1, 1):
            raise ValueError("target_sign must be 1 or -1")
        frame["_true_ddg"] = (
            finite_values(frame[spec.get("target_column", "DDG")], f"{name} targets")
            * sign
        )
        if frame[KEYS].isna().any().any():
            raise ValueError(f"{name} has missing protein or mutation keys")
    else:
        raise ValueError(f"Unknown dataset kind {kind!r}; choose megascale or ptmul")
    if frame.empty:
        raise ValueError(f"{name} is empty")
    if len(frame) != spec["expected_rows"]:
        raise ValueError(
            f"{name}: expected {spec['expected_rows']} rows, got {len(frame)}"
        )
    if frame.WT_name.nunique() != spec["expected_proteins"]:
        raise ValueError(f"{name}: unexpected protein count {frame.WT_name.nunique()}")
    frame["example_id"] = frame.WT_name.astype(str) + "/" + frame.mut_type.astype(str)
    # Literature datasets can contain distinct measurements for the same pair.
    # Keep them all and disambiguate with their original source CSV row index.
    duplicates = frame.example_id.duplicated(keep=False)
    if duplicates.any():
        frame.loc[duplicates, "example_id"] += "#" + frame.index[duplicates].astype(str)
    frame["true_ddg"] = frame.pop("_true_ddg")
    require_unique(frame, ["example_id"], name)
    metadata = source.with_suffix(".metadata.json")
    if metadata.is_file():
        files.append(metadata)
    return EvaluationDataset(name, spec, frame.reset_index(drop=True), files)
