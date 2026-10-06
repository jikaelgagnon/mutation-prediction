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

def _dist(X, mask, eps=1E-6, top_k=48):
    """ProteinMPNN distance calculation"""
    mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
    dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
    D = mask_2D * torch.sqrt(torch.sum(dX ** 2, 3) + eps)
    D_max, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1. - mask_2D) * D_max
    D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(top_k, X.shape[1]), dim=-1, largest=False)
    return D_neighbors, E_idx

def _check_sequence_match(S, wt, mut, pos):
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

class PooledAttentionModel(nn.Module):
    """Rewritten TransferModel class using Batched datasets for faster training"""

    def __init__(self, cfg, *, protein_encoder=None, prediction_head=None):
        super().__init__()
        self.cfg = cfg

        self.prot_mpnn = protein_encoder if protein_encoder is not None else get_protein_mpnn(cfg)

        HIDDEN_DIM, EMBED_DIM, VOCAB_DIM = self._set_model_dims()

        self.attn_pool = nn.Linear(EMBED_DIM * (self.cfg.num_final_layers + 1), 1)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask, esm_emb=None):
        """Vectorized fwd function for arbitrary batches of mutations"""

        # check if S matches mut_wildtype_AAs - if not, overwrite it
        S = _check_sequence_match(S, mut_wildtype_AAs, mut_mutant_AAs, mut_positions)
        
        X = torch.nan_to_num(X, nan=0.0) # [B, L, # atoms, 3]
        
        # get MPNN embeddings
        all_mpnn_hid, wt_embed, _, _ = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all)

        assert self.cfg.model.num_final_layers > 0
        
        # concatenate together the _last_ :self.cfg.model.num_final_layers hidden reps from ProteinMPNN
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
        
        # TODO: Add the actual new code...

        return None

    def _set_model_dims(self):
        """
        Parse various config options to properly set input, output, and vocab dimensions
        """
        EMBED_DIM = 128 # mpnn default seq embed size

        if self.cfg.model.edges:  # add edge input size
            EMBED_DIM += 128
        elif self.cfg.model.dist:
            EMBED_DIM += 25
            
        if self.cfg.model.side_chain_module:
            print('Enabling side chains!')
            EMBED_DIM += 128

        HIDDEN_DIM = 128 # mpnn default hidden dim size
        VOCAB_DIM = 441 if not self.cfg.model.single_target else 1

        return HIDDEN_DIM, EMBED_DIM, VOCAB_DIM