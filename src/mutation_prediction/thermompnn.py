"""Inference-only access to the single-mutant ThermoMPNN v2 baseline.

Uses the single-mutant architecture and checkpoints from ThermoMPNN-D, not its
double-mutant model. See docs/thermompnn_inference.md for provenance and weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from jaxtyping import Float
from omegaconf import OmegaConf
from torch import Tensor

from mutation_prediction.data.collate import (
    tied_featurize_mut,
)
from mutation_prediction.data.mutations import Mutation
from mutation_prediction.tensor_types import ProteinRecord
from mutation_prediction.thermompnn_d_vendor.thermompnn.thermompnn.model.v2_model import (
    TransferModelv2,
)
from mutation_prediction.thermompnn_d_vendor.thermompnn.thermompnn.protein_mpnn_utils import (
    ProteinMPNN,
    parse_PDB,
)

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _single_mutant_config():
    """Upstream examples/configs/single.yaml plus its inference defaults."""
    return OmegaConf.create(
        {
            "model": {
                "aggregation": None,
                "hidden_dims": [64, 32],
                "subtract_mut": True,
                "mutant_embedding": False,
                "single_target": False,
                "num_final_layers": 2,
                "lightattn": True,
                "dist": False,
                "edges": False,
                "side_chain_module": False,
                "dropout": None,
            },
        }
    )


class ThermoMPNNPredictor:
    """Frozen single-mutant predictor using matching ThermoMPNN-ens*.ckpt weights.

    predict() returns an [L, 20] CPU tensor in ALPHABET order. Entries are ΔΔG in
    kcal/mol, positive for destabilization. Positions use zero-based indices in
    the supplied structure record's sequence, including any gaps.
    """

    def __init__(self, checkpoint: str | Path, *, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.checkpoint = Path(checkpoint)
        # The transfer checkpoint contains the complete encoder. No separate
        # vanilla ProteinMPNN checkpoint or training configuration is required.
        encoder = ProteinMPNN(
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=48,
            augment_eps=0.0,
            dropout=0.0,
        )
        self.model = TransferModelv2(_single_mutant_config(), protein_encoder=encoder)
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("state_dict", payload)
        state = {key.removeprefix("model."): value for key, value in state.items()}
        try:
            self.model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError(
                "Expected a single-mutant ThermoMPNN-ens*.ckpt checkpoint from "
                "ThermoMPNN-D. Original ThermoMPNN and ThermoMPNN-D-ens* weights "
                "use different architectures."
            ) from error
        self.model.requires_grad_(False).eval().to(self.device)

    @torch.inference_mode()
    def predict(self, protein: ProteinRecord) -> Float[Tensor, "length 20"]:
        """Predict all single substitutions with one WT encoder pass.

        Missing-backbone and unknown-WT positions return NaN for every amino
        acid. A WT-to-WT substitution returns zero. The input record is not
        modified; its experimental mutation annotation, if any, is ignored.
        """
        if not protein.get("seq"):
            raise ValueError("The protein record must contain a nonempty sequence")
        record = dict(protein)
        # Only needed by the shared collator; it is never used as a model target.
        record["mutation"] = Mutation([0], ["A"], ["A"], 0.0)
        batch = tied_featurize_mut([record], device=self.device)
        X, S, mask, _, chain_M, chain_ids, residue_idx, *_ = batch
        expected_sequence = "".join(
            "ACDEFGHIKLMNPQRSTVWYX"[token] for token in S[0].tolist()
        )
        if expected_sequence != protein["seq"].replace("-", "X"):
            raise ValueError(
                "The record's sequence must match its concatenated chain sequences"
            )

        self.model.eval()
        hidden, sequence_embed, _, _ = self.model.prot_mpnn(
            torch.nan_to_num(X, nan=0.0), S, mask, chain_M, residue_idx, chain_ids
        )
        features = torch.cat(
            [*hidden[: self.model.cfg.model.num_final_layers], sequence_embed], dim=-1
        )[0]
        features = self.model.light_attention(features.unsqueeze(-1))
        scores = self.model.ddg_out(features)  # [L, 21]
        wt_scores = scores.gather(1, S[0].unsqueeze(-1))
        ddg = (scores - wt_scores)[:, :20]
        valid = mask[0].bool() & (S[0] < 20)
        ddg[~valid] = torch.nan
        return ddg.cpu()

    def predict_pdb(self, pdb: str | Path, *, chain: str = "A") -> pd.DataFrame:
        """Return all 19 nonidentity substitutions per valid residue in one chain.

        position is a zero-based parsed-sequence index; mutation uses a one-based
        sequence position. Neither column represents original PDB numbering.
        """
        pdb = Path(pdb)
        if not pdb.is_file():
            raise FileNotFoundError(pdb)
        protein = parse_PDB(str(pdb), input_chain_list=[chain])[0]
        if not protein["seq"]:
            raise ValueError(f"No protein residues found in chain {chain!r} of {pdb}")
        predictions = self.predict(protein)
        rows = []
        for position, wildtype in enumerate(protein["seq"]):
            for aa_index, mutant in enumerate(ALPHABET):
                value = predictions[position, aa_index].item()
                if wildtype == mutant or not torch.isfinite(
                    predictions[position, aa_index]
                ):
                    continue
                rows.append(
                    {
                        "protein": protein["name"],
                        "chain": chain,
                        "position": position,
                        "wildtype": wildtype,
                        "mutant": mutant,
                        "mutation": f"{wildtype}{position + 1}{mutant}",
                        "ddg": value,
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "protein",
                "chain",
                "position",
                "wildtype",
                "mutant",
                "mutation",
                "ddg",
            ],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictor = ThermoMPNNPredictor(args.checkpoint, device=args.device)
    predictions = predictor.predict_pdb(args.pdb, chain=args.chain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Saved {len(predictions)} single-mutant predictions to {args.output}")


if __name__ == "__main__":
    main()
