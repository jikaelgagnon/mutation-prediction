import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn as nn
    from itertools import permutations
    import numpy as np

    from mutation_prediction.thermo_mpnn_d.thermompnn.model.modules import (
        get_protein_mpnn,
        LightAttention,
        MPNNLayer,
        SideChainModule,
    )
    from mutation_prediction.thermo_mpnn_d.thermompnn.model.side_chain_model import get_protein_mpnn_sca

    def batched_index_select(input, dim, index):
        for ii in range(1, len(input.shape)):
            if ii != dim:
                index = index.unsqueeze(ii)
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.expand(expanse)
        return torch.gather(input, dim, index)

    def dist(X, mask, eps=1E-6, top_k=48):
        """ProteinMPNN distance calculation"""
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def check_sequence_match(S, wt, mut, pos):
        """
        Checks if S matches wt amino acids at the specified positions. 
        If not matching, adjusts S to match wt.
        """
        S = S.clone()
        for mut_idx in range(wt.shape[-1]): # check each mutation separately
            S_check = torch.gather(S, -1, pos[..., mut_idx, None]) # selects all amino acids in seq at pos locations

            # check against wt array (one-hot)
            if torch.sum(S_check - wt[..., mut_idx]) != 0: # if matched, keep S values
                S_check = torch.gather(S, -1, pos[..., mut_idx, None]) # selects all amino acids in seq at pos locations
                S.scatter_(dim=1, index=pos[..., mut_idx][..., None], src=wt[..., mut_idx][..., None]) # scatter wt values into S at specified positions
                S_check = torch.gather(S, -1, pos[..., mut_idx, None]) # selects all amino acids in seq at pos locations
        return S

    return check_sequence_match, get_protein_mpnn, nn, torch


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import pytorch_lightning as pl
    from omegaconf import OmegaConf
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from torch.utils.data import DataLoader

    repo_root = Path.cwd()
    if not (repo_root / "configs").is_dir():
        repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "configs").is_dir():
        raise FileNotFoundError("Open this notebook from within the repository.")

    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from mutation_prediction.thermo_mpnn_d.thermompnn.datasets.v2_datasets import (
        MegaScaleDatasetv2,
        tied_featurize_mut,
    )
    from mutation_prediction.thermo_mpnn_d.thermompnn.train_thermompnn import (
        parse_cfg,
    )

    return (
        DataLoader,
        MegaScaleDatasetv2,
        OmegaConf,
        Path,
        mo,
        parse_cfg,
        pl,
        repo_root,
        tied_featurize_mut,
    )


@app.cell
def _(OmegaConf, Path, repo_root):
    paths_cfg = OmegaConf.load(repo_root / "configs/thermompnn_paths.example.yaml")
    model_cfg = OmegaConf.load(repo_root / "configs/model_train_epistatic.yaml")
    cfg = OmegaConf.merge(paths_cfg, model_cfg)

    for key in ("megascale_splits", "megascale_pdbs", "megascale_csv"):
        path = Path(cfg.data_loc[key])
        if not path.is_absolute():
            cfg.data_loc[key] = str(repo_root / path)

    cfg.model.freeze_weights = True
    cfg.model.load_pretrained = True
    cfg.data.mut_types = ["double"]
    cfg.data.splits = ["train", "val"]
    cfg.training.num_workers = 0
    return (cfg,)


@app.cell
def _(cfg):
    dict(cfg)
    return


@app.cell
def _(Path, cfg, repo_root):
    if not Path(cfg.data_loc.megascale_csv).is_file():
        raise FileNotFoundError(f"MegaScale CSV not found: {cfg.data_loc.megascale_csv}")
    if not Path(cfg.data_loc.megascale_splits).is_file():
        raise FileNotFoundError(
            f"MegaScale splits not found: {cfg.data_loc.megascale_splits}"
        )
    if not Path(cfg.data_loc.megascale_pdbs).is_dir():
        raise FileNotFoundError(
            f"MegaScale structure directory not found: {cfg.data_loc.megascale_pdbs}"
        )
    checkpoint_dir = Path(cfg.platform.thermompnn_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = repo_root / checkpoint_dir
    cfg.platform.thermompnn_dir = str(checkpoint_dir)
    checkpoint_paths = sorted(checkpoint_dir.glob("v_*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No pretrained ProteinMPNN checkpoints found in {checkpoint_dir}."
        )

    checkpoint_names = [path.name for path in checkpoint_paths]
    default_checkpoint = (
        "v_48_020.pt" if "v_48_020.pt" in checkpoint_names else checkpoint_names[0]
    )
    return


@app.cell
def _(MegaScaleDatasetv2, cfg):
    train_dataset = MegaScaleDatasetv2(cfg, split="train")
    val_dataset = MegaScaleDatasetv2(cfg, split="val")
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("MegaScale train and validation splits must both be non-empty.")
    return train_dataset, val_dataset


@app.cell(hide_code=True)
def _(mo, train_dataset, val_dataset):
    mo.md(f"""
    **MegaScale examples with structures:** {len(train_dataset):,} train ·
    {len(val_dataset):,} validation. The data and structure files are read from
    `data/raw/megascale/`.
    """)
    return


@app.cell
def _(train_dataset):
    train_dataset.df[["WT_name", "mut_type", "ddG_ML", "aa_seq"]].head(12)
    return


@app.cell
def _(active_cfg):
    active_cfg.training.batch_size = 1
    return


@app.cell
def _(active_cfg):
    active_cfg.training.batch_size
    return


@app.cell
def _(
    DataLoader,
    cfg,
    parse_cfg,
    pl,
    tied_featurize_mut,
    train_dataset,
    val_dataset,
):
    pl.seed_everything(42, workers=True)

    active_cfg = parse_cfg(cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=active_cfg.training.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: tied_featurize_mut(batch, side_chains=False),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=active_cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: tied_featurize_mut(batch, side_chains=False),
    )
    return active_cfg, train_loader


@app.cell
def _(train_loader):
    single_batch = next(iter(train_loader))
    single_batch
    return (single_batch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can unpack the tuple:
    """)
    return


@app.cell
def _(single_batch):
    (X,
        S,
        mask,
        lengths,
        chain_M,
        chain_encoding_all,
        residue_idx,
        mut_positions,
        mut_wildtype_AAs,
        mut_mutant_AAs,
        mut_ddGs,
        atom_mask,
        ) = single_batch
    return (
        S,
        X,
        atom_mask,
        chain_M,
        chain_encoding_all,
        mask,
        mut_ddGs,
        mut_mutant_AAs,
        mut_positions,
        mut_wildtype_AAs,
        residue_idx,
    )


@app.cell
def _(
    S,
    X,
    atom_mask,
    chain_M,
    chain_encoding_all,
    mask,
    mut_ddGs,
    mut_mutant_AAs,
    mut_positions,
    mut_wildtype_AAs,
    residue_idx,
):
    model_inputs = (
        X,
        S,
        mask,
        chain_M,
        residue_idx,
        chain_encoding_all,
        mut_positions,
        mut_wildtype_AAs,
        mut_mutant_AAs,
        mut_ddGs,
        atom_mask,
        )
    return (model_inputs,)


@app.cell
def _(X):
    X.shape
    return


@app.cell
def _(torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return (device,)


@app.cell
def _(device, model_inputs):
    inputs = tuple(tensor.to(device) for tensor in model_inputs)
    return (inputs,)


@app.cell
def _(check_sequence_match, get_protein_mpnn, nn, torch):
    class PooledAttentionModel(nn.Module):

        def __init__(self, cfg, *, protein_encoder=None, prediction_head=None):
            super().__init__()
            self.cfg = cfg

            self.prot_mpnn = protein_encoder if protein_encoder is not None else get_protein_mpnn(cfg)

            HIDDEN_DIM, EMBED_DIM, VOCAB_DIM = self._set_model_dims()
            print(f"EMBED DIM: {EMBED_DIM}")
            self.attn_pool = nn.Linear(EMBED_DIM * (self.cfg.model.num_final_layers + 1), 1)
            self.final_layer = nn.Linear(EMBED_DIM * (self.cfg.model.num_final_layers + 1), 1)

        def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask, esm_emb=None):
            """Vectorized fwd function for arbitrary batches of mutations"""

            # check if S matches mut_wildtype_AAs - if not, overwrite it
            S = check_sequence_match(S, mut_wildtype_AAs, mut_mutant_AAs, mut_positions)

            X = torch.nan_to_num(X, nan=0.0) # [B, L, # atoms, 3]

            # get MPNN embeddings
            all_mpnn_hid, wt_embed, _, _ = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all)

            assert self.cfg.model.num_final_layers > 0

            # concatenate together the _last_ self.cfg.model.num_final_layers hidden reps from ProteinMPNN
            all_mpnn_hid = torch.cat(all_mpnn_hid[:self.cfg.model.num_final_layers], -1) # [B, L, Embed * nfl]

            # ENABLED IN THE PAPER
            # if enabled, get sequence embedding for mutant aa
            if self.cfg.model.mutant_embedding:
                # there are actually N sets of mutant sequences, so we need to run this N times
                # mut_embed list contains the embeddings of each mutation
                mut_embed_list = []
                for m in range(mut_mutant_AAs.shape[-1]):
                    # embed each mutation identity using embedding matrix from 
                    # ProteinMPNN
                    mut_embed = self.prot_mpnn.W_s(mut_mutant_AAs[:, m])
                    mut_embed_list.append(mut_embed)
                mut_embed = torch.cat([m.unsqueeze(-1) for m in mut_embed_list], -1) # shape: (Batch, Embed, N_muts)

             # gather final representation from seq and structure embeddings
            final_embed = [] 
            for i in range(mut_mutant_AAs.shape[-1]):
                # gather embedding for first mutation
                current_positions = mut_positions[:, i:i+1] # [B, 1]
    
                # gathers together the hidden states for the i-th mutation across all batches 
                # into a single tensor
                # Shape: B x 1 X E * nfl
                g_struct_embed = torch.gather(all_mpnn_hid, 1, 
                                              current_positions.unsqueeze(-1).expand(current_positions.size(0), 
                                                                                     current_positions.size(1), all_mpnn_hid.size(2)))
                # squeeze to clean things up
                g_struct_embed = torch.squeeze(g_struct_embed, 1) # [B, E * nfl] (nfl = number final layers)
    
                # get embeddings of WT at each mutated position
                g_seq_embed = torch.gather(wt_embed, 1, 
                                           current_positions.unsqueeze(-1).expand(current_positions.size(0), 
                                                                                  current_positions.size(1), wt_embed.size(2)))
                # squeeze to clean things up
                g_seq_embed = torch.squeeze(g_seq_embed, 1) # [B, E]
    
                # if mut embed enabled, subtract it from the wt embed directly to keep dims low
                if self.cfg.model.mutant_embedding:
                    # subtract mutatnt embeddings
                    g_seq_embed = g_seq_embed - mut_embed[:, :, i] # [B, E]
                # concatenated hidden reps + mutation stuff => E * nfl + E = E * (nfl + 1)
                g_embed = torch.cat([g_struct_embed, g_seq_embed], -1) # [B, E * (nfl + 1)]

                final_embed.append(g_embed)  # list with length N_mutations - used to make permutations

            final_embed = torch.stack(final_embed, dim=0) # [num_mutations, B, (nfl + 1)]
            final_embed = final_embed.permute(1,0,2)
            z = self.attn_pool(final_embed)
            weights = torch.softmax(z, dim = 1)
            pooled = torch.sum(weights * final_embed, dim = 1)
            y = self.final_layer(pooled)
            return y
        
        def _set_model_dims(self):
            """
            Parse various config options to properly set input, output, and vocab dimensions
            """
            EMBED_DIM = 128 # mpnn default seq embed size
    
            if self.cfg.model.side_chain_module:
                print('Enabling side chains!')
                EMBED_DIM += 128

            HIDDEN_DIM = 128 # mpnn default hidden dim size
            VOCAB_DIM = 441 if not self.cfg.model.single_target else 1

            return HIDDEN_DIM, EMBED_DIM, VOCAB_DIM

    return (PooledAttentionModel,)


@app.cell
def _(PooledAttentionModel, cfg, device):
    model = PooledAttentionModel(cfg).to(device)
    model
    return (model,)


@app.cell
def _(inputs, model):
    model(*inputs)
    return


if __name__ == "__main__":
    app.run()
