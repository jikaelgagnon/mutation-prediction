"""Shared tensor contracts for protein batches and model interfaces.

Axes: batch = examples, length = padded residues, atoms = 4 backbone or 14
side-chain atoms, mutations = padded mutation sites, neighbors = graph degree.
Coordinates may omit the atom axis in CA-only mode. Floating-point precision
is deliberately unspecified to support mixed precision. Numeric masks use Real to accept integer and floating-point
representations.
These annotations document shapes; they do not enable runtime checking.
"""

from typing import Any, NotRequired, TypeAlias, TypedDict

import numpy as np
from jaxtyping import Float, Int
from torch import Tensor


class AssemblyRecord(TypedDict, total=False):
    """One structure assembly, before per-chain records and padding.

    A failed load returns only seq as a NumPy sentinel; valid records have all
    fields, with idx assigning each residue to a chain and masked listing the
    chains whose sequences should be predicted.
    """

    seq: str | Float[np.ndarray, "sentinel"]
    xyz: Float[Tensor, "length atoms 3"]
    idx: Int[Tensor, "length"]
    masked: Int[Tensor, "masked_chains"]
    label: str


# Dynamic chain keys (seq_chain_A, coords_chain_A, etc.) coexist with metadata.
ProteinRecord: TypeAlias = dict[str, Any]
Coordinates: TypeAlias = (
    Float[Tensor, "batch length atoms 3"] | Float[Tensor, "batch length 3"]
)

MutationBatch: TypeAlias = tuple[
    Coordinates,  # X: coordinates
    Int[Tensor, "batch length"],  # S: amino-acid token IDs
    Float[Tensor, "batch length"],  # mask: valid residues
    Int[np.ndarray, "batch"],  # lengths: unpadded sequence lengths
    Float[Tensor, "batch length"],  # chain_M: design mask
    Int[Tensor, "batch length"],  # chain_encoding_all: chain IDs
    Int[Tensor, "batch length"],  # residue_idx: residue numbering
    Int[Tensor, "batch mutations"],  # mut_positions: zero-based sites
    Int[Tensor, "batch mutations"],  # mut_wildtype_AAs: WT token IDs
    Int[Tensor, "batch mutations"],  # mut_mutant_AAs: mutant token IDs
    Float[Tensor, "batch 1"],  # mut_ddGs: joint stability target
    Int[Tensor, "batch length atoms"],  # atom_mask: missing atoms (1 = missing)
]

ProteinBatch: TypeAlias = tuple[
    Float[Tensor, "batch length atoms 3"],  # X
    Int[Tensor, "batch length"],  # S
    Float[Tensor, "batch length"],  # mask
    Int[np.ndarray, "batch"],  # lengths
    Float[Tensor, "batch length"],  # chain_M
    Int[Tensor, "batch length"],  # residue_idx
    Float[Tensor, "batch length length"],  # mask_self
    Int[Tensor, "batch length"],  # chain_encoding_all
]

TiedProteinBatch: TypeAlias = tuple[
    Coordinates,  # X_out
    Int[Tensor, "batch length"],  # S
    Float[Tensor, "batch length"],  # mask
    Int[np.ndarray, "batch"],  # lengths
    Float[Tensor, "batch length"],  # chain_M
    Int[Tensor, "batch length"],  # chain_encoding_all
    list[list[str]],  # letter_list_list
    list[list[str]],  # visible_list_list
    list[list[str]],  # masked_list_list
    list[list[int]],  # masked_chain_length_list_list
    Float[Tensor, "batch length"],  # chain_M_pos
    Float[Tensor, "batch length 21"],  # omit_AA_mask
    Int[Tensor, "batch length"],  # residue_idx
    Float[Tensor, "batch length 3"],  # dihedral_mask
    list[list[list[int]]],  # tied_pos_list_of_lists_list
    Float[Tensor, "batch length"],  # pssm_coef_all
    Float[Tensor, "batch length 21"],  # pssm_bias_all
    Float[Tensor, "batch length 21"],  # pssm_log_odds_all
    Float[Tensor, "batch length 21"],  # bias_by_res_all
    Float[Tensor, "length"],  # tied_beta (last example)
]


class SamplingOutput(TypedDict):
    """Autoregressive sequence samples and the associated residue scores."""

    S: Int[Tensor, "batch length"]
    probs: Float[Tensor, "batch length 21"]
    decoding_order: Int[Tensor, "batch length"]
    log_probs: NotRequired[Float[Tensor, "batch length 21"]]
    logits: NotRequired[Float[Tensor, "batch length 21"]]
