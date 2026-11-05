"""Mutation records and residue-index mapping helpers.

Paper map: keeps each experimental substitution (site, wild-type residue,
mutant residue, and measured stability change) aligned to the corresponding
structure sequence before model featurization.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    ddG: float | None = None
    pdb: str | None = ''

