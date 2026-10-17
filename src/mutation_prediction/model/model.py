"""Predict one ΔΔG per example by pooling features at its mutated residues.

ProteinMPNN encodes the wild-type structure and sequence once. Each mutation
gets a structural feature plus a sequence feature (optionally WT minus mutant).
Learned attention combines these mutation features into a single prediction.

Shape notation: B = batch size, L = padded sequence length, M = mutations per
example, E = encoder width (128), K = selected decoder layers, D = E * (K + 1).
"""

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
    Check sequence tokens against the supplied wild-type mutation identities.

    S has shape [B, L]; wt and pos have shape [B, M] and contain integer
    amino-acid IDs and residue indices, respectively. Work on a clone so that
    in-place writes do not change the caller's batch. The mut argument is unused.
    """
    S = S.clone()
    for mut_idx in range(wt.shape[-1]): # check each mutation separately
        S_check = torch.gather(S, -1, pos[..., mut_idx, None]) # selects all amino acids in seq at pos locations
        
        # Compare integer amino-acid IDs, not one-hot encodings.
        if torch.sum(S_check - wt[..., mut_idx]) != 0: # if matched, keep S values
            S_check = torch.gather(S, -1, pos[..., mut_idx, None]) # selects all amino acids in seq at pos locations
            S.scatter_(dim=1, index=pos[..., mut_idx][..., None], src=wt[..., mut_idx][..., None]) # scatter wt values into S at specified positions
            S_check = torch.gather(S, -1, pos[..., mut_idx, None]) # selects all amino acids in seq at pos locations
    return S

class PooledAttentionModel(nn.Module):
    """ProteinMPNN features → mutation attention → scalar ΔΔG regression."""

    def __init__(self, cfg, *, protein_encoder=None):
        super().__init__()
        self.cfg = cfg

        # A supplied encoder supports experiments with the same output interface.
        # Otherwise, the shared loader applies checkpoint and freezing settings.
        self.prot_mpnn = protein_encoder if protein_encoder is not None else get_protein_mpnn(cfg)

        EMBED_DIM = 128

        # Both layers consume K structural embeddings plus one sequence embedding.
        # Attention assigns a score to each mutation; the final layer predicts ΔΔG.
        self.attn_pool = nn.Linear(EMBED_DIM * (self.cfg.model.num_final_layers + 1), 1)
        self.final_layer = nn.Linear(EMBED_DIM * (self.cfg.model.num_final_layers + 1), 1)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask, esm_emb=None):
        """Return [B, 1] predictions for a batch with M mutations per example.

        X: backbone coordinates [B, L, atoms, 3]; S: sequence token IDs [B, L].
        mask and chain_M describe valid/design residues; residue_idx and
        chain_encoding_all give residue numbering and chain membership.
        Mutation positions and WT/mutant token IDs each have shape [B, M].
        mut_ddGs, atom_mask, and esm_emb are accepted for interface compatibility
        but are not used to compute predictions here.
        """

        # check if S matches mut_wildtype_AAs - if not, overwrite it
        S = _check_sequence_match(S, mut_wildtype_AAs, mut_mutant_AAs, mut_positions)

        # Replace missing coordinate NaNs before passing coordinates to the encoder.
        X = torch.nan_to_num(X, nan=0.0) # [B, L, # atoms, 3]

        # One WT encoder pass supplies structural states and sequence embeddings
        # for every residue. Mutant identities below use only its embedding table.
        all_mpnn_hid, wt_embed, _, _ = self.prot_mpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all)

        assert self.cfg.model.num_final_layers > 0

        # The encoder returns decoder states in reverse order (last layer first).
        # Concatenate the first K entries along features: [B, L, E * K].
        all_mpnn_hid = torch.cat(all_mpnn_hid[:self.cfg.model.num_final_layers], -1) # [B, L, Embed * nfl]

        # Optionally encode the replacement amino acid at each mutation site.
        if self.cfg.model.mutant_embedding:
            # Each column contains one mutant token per batch example. This is a
            # lookup, not a separate structure/sequence encoder pass per mutation.
            mut_embed_list = []
            for m in range(mut_mutant_AAs.shape[-1]):
                # embed each mutation identity using embedding matrix from 
                # ProteinMPNN
                mut_embed = self.prot_mpnn.W_s(mut_mutant_AAs[:, m])
                mut_embed_list.append(mut_embed)
            mut_embed = torch.cat([m.unsqueeze(-1) for m in mut_embed_list], -1) # shape: (Batch, Embed, N_muts)

        # Build one feature vector per mutation by selecting its residue features.
        final_embed = [] 
        for i in range(mut_mutant_AAs.shape[-1]):
            # Select the i-th mutation position independently for each example.
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
                # Represent the substitution as WT identity minus mutant identity.
                g_seq_embed = g_seq_embed - mut_embed[:, :, i] # [B, E]
            # concatenated hidden reps + mutation stuff => E * nfl + E = E * (nfl + 1)
            g_embed = torch.cat([g_struct_embed, g_seq_embed], -1) # [B, E * (nfl + 1)]

            final_embed.append(g_embed)  # M tensors, each [B, D]

        final_embed = torch.stack(final_embed, dim=0) # [M, B, D]
        final_embed = final_embed.permute(1,0,2) # [B, M, D]

        # Normalize over mutations, not residues or batch examples. Each example's
        # weights sum to one; for a single mutation its weight is exactly one.
        # The weighted sum is invariant to reordering the mutation feature vectors.
        z = self.attn_pool(final_embed) # [B, M, 1]
        weights = torch.softmax(z, dim = 1) # [B, M, 1]
        pooled = torch.sum(weights * final_embed, dim = 1) # [B, D]
        y = self.final_layer(pooled) # [B, 1], one joint ΔΔG for all mutations
        return y


# LightningModule defines the hooks that Trainer calls during training/evaluation.
# Guide overview: https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/lightning_module.rst
class PooledAttentionModelPL(pl.LightningModule):
    """Lightning wrapper for loss, split-specific metrics, and optimization.

    train.py builds the data loaders and Trainer; this class defines what happens
    for each batch and which parameters the optimizer updates.
    """
    def __init__(self, cfg, *, protein_encoder=None):
        super().__init__()
        print('Pooled Attention Model Enabled')
        self.model = PooledAttentionModel(
            cfg,
            protein_encoder=protein_encoder
        )
        self.cfg = cfg
        self.dev = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.out = ['ddG']
        self.metrics = nn.ModuleDict()
        
        # Separate metric instances prevent train/validation/test state mixing.
        # Logging Metric objects lets Lightning manage epoch computation/reset.
        # Guide: https://github.com/Lightning-AI/torchmetrics/blob/master/docs/source/pages/lightning.rst
        for split in ("train_metrics", "val_metrics", "test_metrics"):
            self.metrics[split] = nn.ModuleDict()
            
            for out in self.out:
                self.metrics[split][out] = nn.ModuleDict()
                for name, metric in get_metrics().items():
                    self.metrics[split][out][name] = metric

    # Like nn.Module.forward, this defines inference; it does not start training.
    # Docs: https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/lightning_module.rst#inference-in-research
    def forward(self, *args):
        return self.model(*args)

    def shared_eval(self, batch, batch_idx, prefix):
        """Run the common prediction/loss/logging path for each dataset split."""
        # Keep the 12-field tuple order produced by tied_featurize_mut. lengths
        # is unpacked for that interface but is not consumed by the model.
        
        X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask = batch


        pred_ddG = self(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask)

        # Supervise the joint prediction with the measured ΔΔG, both [B, 1].
        mse = F.mse_loss(pred_ddG, mut_ddGs) 

        # Update stateful metrics and log epoch-level values (including val_ddG_mse).
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
                # Record epoch metrics and display them in the progress bar.
                # self.log also exposes val_ddG_mse to callbacks and the scheduler.
                # Docs: https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/lightning_module.rst#train-epoch-level-metrics
                self.log(f"{prefix}_{out}_{name}", metric, prog_bar=True, on_step=False, on_epoch=True,
                            batch_size=len(batch))
            
        # Preserve the existing convention of skipping a zero-loss batch.
        if mse == 0.0:
            return None
        return mse

    # Trainer calls this for each training batch. Returning the loss enables
    # automatic gradient computation and optimizer updates; no backward() here.
    # Guide: https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/optimization.rst
    # Hook: https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/lightning_module.rst#training-loop
    def training_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'train')

    # Trainer runs validation in evaluation mode with gradients disabled.
    # This measures performance without updating model weights.
    # Docs: https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/lightning_module.rst#validation-loop
    def validation_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'val')

    # Called by Trainer.test(), rather than the Trainer.fit() used in train.py.
    # Docs: https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/lightning_module.rst#test-loop
    def test_step(self, batch, batch_idx):
        return self.shared_eval(batch, batch_idx, 'test')

    # Trainer calls this hook to obtain the optimizer and optional scheduler.
    # See the configure_optimizers docstring for return formats and scheduler monitoring:
    # https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/core/module.py
    def configure_optimizers(self):
        """Train the attention/readout layers and optionally fine-tune ProteinMPNN."""
        # When unfrozen, the encoder can use its own learning rate. Otherwise it
        # is excluded from the optimizer; the loader handles requires_grad flags.
        
        if not self.cfg.model.freeze_weights: # fully unfrozen ProteinMPNN
            mpnn_lr = self.cfg.training.mpnn_learn_rate or self.cfg.training.learn_rate
            param_list = [{"params": self.model.prot_mpnn.parameters(), "lr": mpnn_lr}]
        else: # fully frozen MPNN
            param_list = []

        # Both learned head layers always use training.learn_rate.
        model_params = [
            {"params": self.model.attn_pool.parameters()},
            {"params": self.model.final_layer.parameters()}
            ]

        
        param_list = param_list + model_params
        opt = torch.optim.AdamW(param_list, lr=self.cfg.training.learn_rate)

        if self.cfg.training.lr_schedule: # enable additional lr scheduler conditioned on val ddG mse
            # Halve the learning rate when validation MSE stops improving.
            # Lightning steps the scheduler using the logged metric named by
            # monitor below; that name must match the validation self.log call.
            lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', factor=0.5)
            print('Enabled LR Schedule!')
            return {
                'optimizer': opt,
                'lr_scheduler': lr_sched,
                'monitor': f'val_ddG_mse'
            }
        else:
            return opt