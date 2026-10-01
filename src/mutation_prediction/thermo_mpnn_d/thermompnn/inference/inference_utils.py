"""Inference metrics and structural-context diagnostics.

Paper map: evaluates predicted versus measured stability changes and provides
a residue-neighborhood count used to describe mutation-site context.
"""

import torch
from torch import nn
from torchmetrics import R2Score, MeanSquaredError, SpearmanCorrCoef, PearsonCorrCoef

import argparse
import omegaconf as OmegaConf
import pandas as pd

from mutation_prediction.thermo_mpnn_d.thermompnn.model.modules import get_protein_mpnn
from mutation_prediction.thermo_mpnn_d.thermompnn.protein_mpnn_utils import tied_featurize


ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def get_metrics_full(device=None):
    return {
        "r2": R2Score().to(device),
        "mse": MeanSquaredError(squared=True).to(device),
        "rmse": MeanSquaredError(squared=False).to(device),
        "spearman": SpearmanCorrCoef().to(device),
        "pearson": PearsonCorrCoef().to(device),
    }


def compute_centrality(xyz, basis_atom: str = "CA", radius: float = 10.0, core_threshold: int = 20, surface_threshold: int = 15, backup_atom: str = "C", chain: str = 'A') -> torch.Tensor:
    """Expects ProteinMPNN feature dict and returns torch tensor of num_neighbors integers."""
    coords = xyz[basis_atom + f'_chain_{chain}']
    coords = torch.tensor(coords)
    # Compute distances and number of neighbors.
    pairwise_dists = torch.cdist(coords, coords)
    pairwise_dists = torch.nan_to_num(pairwise_dists, nan=2 * radius)
    num_neighbors = torch.sum(pairwise_dists < radius, dim=-1) - 1

    return num_neighbors
