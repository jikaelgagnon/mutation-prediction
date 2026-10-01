"""Mutation records and residue-index mapping helpers.

Paper map: keeps each experimental substitution (site, wild-type residue,
mutant residue, and measured stability change) aligned to the corresponding
structure sequence before model featurization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# The 20 standard amino acids, plus "-" to represent a gap in an alignment.
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY-'


@dataclass
class Mutation:
    """Describe one or more amino-acid changes for a protein.

    The lists correspond by index: ``position[i]`` changes from
    ``wildtype[i]`` to ``mutation[i]``. Using lists lets one object describe
    both single and multiple mutations.
    """
    position: list[int] # position in sequence
    wildtype: list[str] # original AA
    mutation: list[str] # mutated AA
    # Optional measured stability change and the associated structure name.
    ddG: Optional[float] = None
    pdb: Optional[str] = ''


"""
- **ProteinMPNN receives the wild-type structure**, 
including its 3D coordinates and the amino-acid 
sequence read from that structure. The structure 
sequence may not line up position-for-position with 
the dataset sequence—for example, the structure may omit residues.

- **The dataset gives the mutation’s position and identities** 
(wild type → mutant) in its sequence’s numbering. The code aligns the 
dataset sequence with the structure-derived sequence when needed, then 
checks that the mapped structure position contains the expected wild-type amino acid.

- **ThermoMPNN-D uses the ProteinMPNN representation at that mapped position** 
and combines it with mutation information, including the mutant amino-acid identity, 
to predict the mutation’s effect.
"""

def seq1_index_to_seq2_index(align, index):
    """Map a zero-based residue index from aligned ``seqA`` to ``seqB``.

    Alignment strings include "-" characters where one sequence has a gap.
    This returns the corresponding zero-based index in the ungapped ``seqB``,
    or ``None`` when that aligned position is a gap in ``seqB``.
    """
    # Walk across the alignment until we reach the requested residue in seqA.
    # Gap characters in seqA do not represent residues and therefore do not
    # advance its residue index.
    cur_seq1_index = 0

    for aln_idx, char in enumerate(align.seqA):
        if char != '-':
            cur_seq1_index += 1
        if cur_seq1_index > index:
            break

    # There is no corresponding residue in seqB if it has a gap at this
    # alignment column.
    if align.seqB[aln_idx] == '-':
        return None

    # Convert the alignment-column index to a residue index in seqB by
    # subtracting the gaps that occur up to and including this column.
    seq2_to_idx = align.seqB[:aln_idx+1]
    seq2_idx = aln_idx
    for char in seq2_to_idx:
        if char == '-':
            seq2_idx -= 1
    
    if seq2_idx < 0:
        return None

    return seq2_idx
