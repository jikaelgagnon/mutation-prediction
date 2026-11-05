"""Measured MegaScale splits and residual targets (adapted; see vendor license)."""

from __future__ import annotations

import os
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mutation_prediction.data.mutations import (
    Mutation,
)
from mutation_prediction.tensor_types import (
    ProteinRecord,
)
from mutation_prediction.thermompnn_d_vendor.thermompnn.thermompnn.protein_mpnn_utils import (
    parse_PDB,
)


class MegaScaleDatasetv2(torch.utils.data.Dataset):
    """Rewritten Megascale dataset doing truly batched mutation generation (getitem returns 1 sample)"""

    def __init__(self, cfg, split, flip_sign=False):

        self.cfg = cfg
        self.split = split  # which split to retrieve
        self.target_column = cfg.data.get("target_column", "ddG_ML")
        if self.target_column not in ("ddG_ML", "epsilon_target"):
            raise ValueError("data.target_column must be ddG_ML or epsilon_target")
        if self.target_column == "epsilon_target":
            if (
                list(cfg.data.mut_types) != ["double"]
                or cfg.data.get("epi", False)
                or cfg.data.get("pick") is not None
                or flip_sign
                or split not in ("train", "val", "test")
            ):
                raise ValueError(
                    "Epsilon targets require unmodified double mutants and train/val/test splits"
                )

        if split not in ("train", "val", "test") or set(cfg.data.mut_types) - {
            "single",
            "double",
        }:
            raise ValueError(
                "Only measured single/double mutants and train/val/test splits are supported"
            )
        if cfg.data.get("epi", False) or cfg.data.get("pick") is not None:
            raise ValueError(
                "Experimental epistasis and mutation-picking modes have been retired"
            )
        fname = self.cfg.data_loc.megascale_csv
        # only load rows needed to save memory
        columns = list(
            dict.fromkeys(
                ["ddG_ML", "mut_type", "WT_name", "aa_seq", self.target_column]
            )
        )
        df = pd.read_csv(fname, usecols=columns)
        if self.target_column == "epsilon_target":
            df["epsilon_target"] = pd.to_numeric(df["epsilon_target"], errors="raise")
            if not np.isfinite(df["epsilon_target"].to_numpy()).all():
                raise ValueError(
                    "epsilon_target must contain only finite numeric values"
                )
        # remove unreliable data and insertion/deletion mutations
        df = df.loc[df.ddG_ML != "-", :].reset_index(drop=True)
        df = df.loc[
            ~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del"), :
        ].reset_index(drop=True)

        mut_list = []
        if "single" in self.cfg.data.mut_types:
            mut_list.append(
                df.loc[
                    ~df.mut_type.str.contains(":") & ~df.mut_type.str.contains("wt"), :
                ].reset_index(drop=True)
            )

        if "double" in self.cfg.data.mut_types:
            tmp = df.loc[
                (df.mut_type.str.count(":") == 1) & (~df.mut_type.str.contains("wt")), :
            ].reset_index(drop=True)

            # Remove hidden single mutations
            mut = tmp.mut_type.values
            mut1 = [m.split(":")[0] for m in mut]
            mut2 = [m.split(":")[-1] for m in mut]
            flag = [a[0] == a[-1] or b[0] == b[-1] for a, b in zip(mut1, mut2)]
            flag = ~np.array(flag, dtype=bool)
            tmp = tmp.loc[flag]
            tmp["dupe"] = tmp["WT_name"] + "_" + tmp["mut_type"]
            tmp = tmp.drop_duplicates(subset=["dupe"]).reset_index(drop=True)
            mut_list.append(tmp)

        self.df = pd.concat(mut_list, axis=0).reset_index(
            drop=True
        )  # this includes points missing structure data

        # load splits produced by mmseqs clustering

        with open(self.cfg.data_loc.megascale_splits, "rb") as f:
            splits = pickle.load(f)

        """ex:
        {
            "train": ["protein_A.pdb", "protein_B.pdb"],
            "val":   ["protein_C.pdb"],
            "test":  ["protein_D.pdb"]
        }
        """
        self.wt_names = splits[
            self.split
        ]  # specific list from splits. eg. ["protein_A.pdb", "protein_B.pdb"]

        # pre-loading wildtype structures - can avoid later file I/O for 50% of data points
        self.side_chains = self.cfg.data.get("side_chains", False)
        self.pdb_data = {}
        for wt_name in tqdm(self.wt_names):
            wt_name = wt_name.removesuffix(".pdb").replace("|", ":")
            pdb_file = os.path.join(self.cfg.data_loc.megascale_pdbs, f"{wt_name}.pdb")
            pdb = parse_PDB(pdb_file, side_chains=self.side_chains)
            self.pdb_data[wt_name] = pdb[0]

        # filter df for only data with structural data
        self.df = self.df.loc[self.df.WT_name.isin(self.wt_names)].reset_index(
            drop=True
        )

        if flip_sign:
            self.df["ddG_ML"] = self.df["ddG_ML"].astype(float) * -1
        self.df = (
            self.df.assign(length=self.df.aa_seq.str.len())
            .sort_values("length")
            .drop(columns="length")
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> ProteinRecord:
        row = self.df.iloc[index]
        substitutions = row.mut_type.split(":")
        positions = [int(m[1:-1]) - 1 for m in substitutions]
        wildtype = [m[0] for m in substitutions]
        mutant = [m[-1] for m in substitutions]
        key = row.WT_name.removesuffix(".pdb").replace("|", ":")
        record = deepcopy(self.pdb_data[key])
        # Preserve the established target conventions, including legacy ddG training.
        sign = -1 if self.target_column == "ddG_ML" else 1
        target = sign * float(row[self.target_column])
        record["mutation"] = Mutation(positions, wildtype, mutant, target, row.WT_name)
        return record


# Compatibility for existing dataset consumers.
