"""Compose mutation feature extractors and heads for ΔΔG or epsilon regression."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Real
from omegaconf import OmegaConf
from torch import Tensor, nn

from mutation_prediction.tensor_types import MutationBatch
from mutation_prediction.model.metrics import (
    get_metrics,
)


class MutationPredictionModel(nn.Module):
    """Connect a feature extractor [B, M, D] to any compatible prediction head.

    The extractor receives the nine protein/mutation tensors below. The head
    receives only final_embed and should return one scalar per example [B, 1].
    """

    def __init__(self, feature_extractor: nn.Module, prediction_head: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.prediction_head = prediction_head

    def forward(
        self,
        X: Float[Tensor, "batch length atoms 3"],
        S: Int[Tensor, "batch length"],
        mask: Real[Tensor, "batch length"],
        chain_M: Real[Tensor, "batch length"],
        residue_idx: Int[Tensor, "batch length"],
        chain_encoding_all: Int[Tensor, "batch length"],
        mut_positions: Int[Tensor, "batch mutations"],
        mut_wildtype_AAs: Int[Tensor, "batch mutations"],
        mut_mutant_AAs: Int[Tensor, "batch mutations"],
        mut_ddGs: Float[Tensor, "batch 1"],
        atom_mask: Real[Tensor, "batch length atoms"],
        esm_emb: Float[Tensor, "batch length esm_features"] | None = None,
    ) -> Float[Tensor, "batch 1"]:
        """Predict the configured scalar; targets and auxiliary masks are interface-only.

        Args:
            X: Protein atom coordinates, shaped [B, L, A, 3].
                The last axis contains x, y, z coordinates.
            S: Amino-acid token IDs, shaped [B, L]. Mutation sites are
                restored to their wild-type identities before encoding.
            mask: Valid-residue mask, shaped [B, L]. Values are 1 for
                usable residues and 0 for padding or missing coordinates.
            chain_M: ProteinMPNN design mask, shaped [B, L]. Values are
                1 for residues in chains whose sequences should be predicted.
            residue_idx: Residue indices, shaped [B, L], including offsets
                at chain boundaries. Used to encode relative positions.
            chain_encoding_all: Integer chain IDs for each residue,
                shaped [B, L].
            mut_positions: Zero-based mutation positions in the concatenated
                protein sequence, shaped [B, M].
            mut_wildtype_AAs: Wild-type amino-acid token IDs at the mutation
                sites, shaped [B, M].
            mut_mutant_AAs: Replacement amino-acid token IDs at the mutation
                sites, shaped [B, M].
            mut_ddGs: Training targets, shaped [B, 1]. Contains joint ΔΔG
                or epsilon, depending on configuration. Accepted for batch
                compatibility; used by the training loss, not by prediction.
            atom_mask: Missing-atom mask, shaped [B, L, A], where 1 means
                missing. Accepted for compatibility; currently unused.
            esm_emb: Optional per-residue ESM embeddings, shaped [B, L, D_ESM].
                Currently unused. Defaults to None.

        Returns:
            Predicted ΔΔG or epsilon, shaped [B, 1].
        """
        
        final_embed = self.feature_extractor(
            X,
            S,
            mask,
            chain_M,
            residue_idx,
            chain_encoding_all,
            mut_positions,
            mut_wildtype_AAs,
            mut_mutant_AAs,
        )
        return self.prediction_head(final_embed)


class MutationPredictionModelPL(pl.LightningModule):
    """Train a feature-extractor/head model against a scalar target.

    train.py builds the data loaders and Trainer; this class defines what happens
    for each batch and which parameters the optimizer updates. The configured
    target_name labels metrics for ΔΔG, epsilon, or another scalar target.
    Supply the composed model again when loading this generic wrapper from a
    checkpoint; only cfg is saved as a hyperparameter.
    """
    def __init__(
        self,
        cfg,
        model: MutationPredictionModel,
    ):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters({'cfg': OmegaConf.to_container(cfg, resolve=True)})
        self.dev = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.out = [cfg.model.get('target_name', 'ddG')]
        self.metrics = nn.ModuleDict()
        
        # Separate metric instances prevent train/validation/test state mixing.
        # Logging Metric objects lets Lightning manage epoch computation/reset.
        for split in ("train_metrics", "val_metrics", "test_metrics"):
            self.metrics[split] = nn.ModuleDict()
            
            for out in self.out:
                self.metrics[split][out] = nn.ModuleDict()
                for name, metric in get_metrics().items():
                    self.metrics[split][out][name] = metric

    # Like nn.Module.forward, this defines inference; it does not start training.
    # Docs: https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/lightning_module.rst#inference-in-research
    def forward(
        self,
        X: Float[Tensor, "batch length atoms 3"],
        S: Int[Tensor, "batch length"],
        mask: Real[Tensor, "batch length"],
        chain_M: Real[Tensor, "batch length"],
        residue_idx: Int[Tensor, "batch length"],
        chain_encoding_all: Int[Tensor, "batch length"],
        mut_positions: Int[Tensor, "batch mutations"],
        mut_wildtype_AAs: Int[Tensor, "batch mutations"],
        mut_mutant_AAs: Int[Tensor, "batch mutations"],
        mut_ddGs: Float[Tensor, "batch 1"],
        atom_mask: Real[Tensor, "batch length atoms"],
        esm_emb: Float[Tensor, "batch length esm_features"] | None = None,
    ) -> Float[Tensor, "batch 1"]:
        return self.model(
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
            esm_emb,
        )

    def shared_eval(
        self,
        batch: MutationBatch,
        batch_idx: int,
        prefix: str,
    ) -> Float[Tensor, ""] | None:
        """Run the common prediction/loss/logging path for each dataset split."""
        # Keep the 12-field tuple order produced by tied_featurize_mut. lengths
        # is unpacked for that interface but is not consumed by the model.
        
        X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, targets, atom_mask = batch


        predictions = self(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, targets, atom_mask)

        # The target slot holds either measured ΔΔG or cached epsilon, both [B, 1].
        mse = F.mse_loss(predictions, targets)

        # Update split-specific metrics and log epoch-level ddG/epsilon scores.
        for out in self.out:
            for name, metric in self.metrics[f"{prefix}_metrics"][out].items():
                metric.update(predictions.reshape(-1), targets.reshape(-1))
                # Record epoch metrics and display them in the progress bar.
                # self.log exposes the target's MSE to callbacks and the scheduler.
                # Docs: https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/lightning_module.rst#train-epoch-level-metrics
                self.log(f"{prefix}_{out}_{name}", metric, prog_bar=True, on_step=False, on_epoch=True,
                            batch_size=targets.shape[0])
            
        return mse

    # Trainer calls this for each training batch. Returning the loss enables
    # automatic gradient computation and optimizer updates; no backward() here
    def training_step(
        self, batch: MutationBatch, batch_idx: int
    ) -> Float[Tensor, ""] | None:
        return self.shared_eval(batch, batch_idx, 'train')

    # Trainer runs validation in evaluation mode with gradients disabled.
    # This measures performance without updating model weights.
    # Docs: https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/lightning_module.rst#validation-loop
    def validation_step(
        self, batch: MutationBatch, batch_idx: int
    ) -> Float[Tensor, ""] | None:
        return self.shared_eval(batch, batch_idx, 'val')

    # Called by Trainer.test(), rather than the Trainer.fit() used in train.py.
    # Docs: https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/lightning_module.rst#test-loop
    def test_step(self, batch: MutationBatch, batch_idx: int) -> Float[Tensor, ""] | None:
        return self.shared_eval(batch, batch_idx, 'test')

    # Trainer calls this hook to obtain the optimizer and optional scheduler.
    # See the configure_optimizers docstring for return formats and scheduler monitoring:
    # https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/core/module.py
    def configure_optimizers(self) -> torch.optim.Optimizer | dict[str, object]:
        """Train the supplied head and any unfrozen feature-extractor parameters."""
        # The ProteinMPNN loader sets requires_grad from model.freeze_weights.
        # Custom extractors/heads control their own trainable parameters.
        param_list = []
        feature_params = [
            p for p in self.model.feature_extractor.parameters() if p.requires_grad
        ]
        if feature_params:
            feature_lr = (
                self.cfg.training.get('mpnn_learn_rate') or self.cfg.training.learn_rate
            )
            param_list.append({"params": feature_params, "lr": feature_lr})
        head_params = [
            p for p in self.model.prediction_head.parameters() if p.requires_grad
        ]
        if head_params:
            param_list.append({"params": head_params})
        opt = torch.optim.AdamW(param_list, lr=self.cfg.training.learn_rate)

        if self.cfg.training.lr_schedule:
            # Halve the learning rate when validation MSE stops improving.
            # Lightning steps the scheduler using the logged metric named by
            # monitor below; that name must match the target's validation log.
            lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', factor=0.5)
            print('Enabled LR Schedule!')
            return {
                'optimizer': opt,
                'lr_scheduler': lr_sched,
                'monitor': f'val_{self.out[0]}_mse'
            }
        else:
            return opt
