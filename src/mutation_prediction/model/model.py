import torch
import torch.nn as nn

from mutation_prediction.thermo_mpnn_d.thermompnn.model.modules import (
    get_protein_mpnn,
)

import pytorch_lightning as pl

import torch.nn.functional as F

from mutation_prediction.thermo_mpnn_d.thermompnn.trainer.trainer_utils import get_metrics, get_metrics_functional


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

    def __init__(self, cfg, *, protein_encoder=None, prediction_head=None):
        super().__init__()
        self.cfg = cfg

        self.prot_mpnn = protein_encoder if protein_encoder is not None else get_protein_mpnn(cfg)

        EMBED_DIM = 128

        self.attn_pool = nn.Linear(EMBED_DIM * (self.cfg.model.num_final_layers + 1), 1)
        self.final_layer = nn.Linear(EMBED_DIM * (self.cfg.model.num_final_layers + 1), 1)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask, esm_emb=None):
        """Vectorized fwd function for arbitrary batches of mutations"""

        # check if S matches mut_wildtype_AAs - if not, overwrite it
        S = _check_sequence_match(S, mut_wildtype_AAs, mut_mutant_AAs, mut_positions)

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


class PooledAttentionModelPL(pl.LightningModule):
    """Batched trainer module"""
    def __init__(self, cfg, *, protein_encoder=None, prediction_head=None):
        super().__init__()
        print('Multi-mutant siamese network enabled!')
        self.model = PooledAttentionModel(
            cfg,
            protein_encoder=protein_encoder,
            prediction_head=prediction_head,
        )
        self.cfg = cfg
        self.dev = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.out = ['ddG']
        self.metrics = nn.ModuleDict()
        
        for split in ("train_metrics", "val_metrics", "test_metrics"):
            self.metrics[split] = nn.ModuleDict()
            
            for out in self.out:
                self.metrics[split][out] = nn.ModuleDict()
                for name, metric in get_metrics().items():
                    self.metrics[split][out][name] = metric

    def forward(self, *args):
        return self.model(*args)

    def shared_eval(self, batch, batch_idx, prefix):
        
        X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask = batch


        pred_ddG = self(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask)

        mse = F.mse_loss(pred_ddG, mut_ddGs) 

        for out in self.out:
            for name, metric in self.metrics[f"{prefix}_metrics"][out].items():
                try:
                    metric.update(torch.squeeze(pred_ddG), torch.squeeze(mut_ddGs))
                except IndexError:
                    continue

            for name, metric in self.metrics[f"{prefix}_metrics"][out].items():
                try:
                    metric.compute()
                except ValueError:
                    continue
                self.log(f"{prefix}_{out}_{name}", metric, prog_bar=True, on_step=False, on_epoch=True,
                            batch_size=len(batch))
            
        if mse == 0.0:
            return None
        return mse

    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')

    def configure_optimizers(self):
        
        if not self.cfg.model.freeze_weights: # fully unfrozen ProteinMPNN
            mpnn_lr = self.cfg.training.mpnn_learn_rate or self.cfg.training.learn_rate
            param_list = [{"params": self.model.prot_mpnn.parameters(), "lr": mpnn_lr}]
        else: # fully frozen MPNN
            param_list = []

        model_params = [
            {"params": self.model.attn_pool.parameters()},
            {"params": self.model.final_layer.parameters()}
            ]

        
        param_list = param_list + model_params
        opt = torch.optim.AdamW(param_list, lr=self.cfg.training.learn_rate)

        if self.cfg.training.lr_schedule: # enable additional lr scheduler conditioned on val ddG mse
            lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', factor=0.5)
            print('Enabled LR Schedule!')
            return {
                'optimizer': opt,
                'lr_scheduler': lr_sched,
                'monitor': f'val_ddG_mse'
            }
        else:
            return opt