"""Compose mutation feature extractors and heads for ΔΔG or epsilon regression."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int, Real
from omegaconf import OmegaConf
from torch import Tensor

from mutation_prediction.model.features import ProteinMPNNFeatureExtractor
from mutation_prediction.model.heads import PooledAttentionHead
from mutation_prediction.tensor_types import MutationBatch
from mutation_prediction.thermo_mpnn_d.thermompnn.trainer.trainer_utils import get_metrics


class MutationPredictionModel(nn.Module):
    """Connect a feature extractor [B, M, D] to any compatible prediction head.

    The extractor receives the nine protein/mutation tensors below. The head
    receives only final_embed and should return [B, 1] for ΔΔG training.
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
        """Predict the configured scalar; targets and auxiliary masks are interface-only."""
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


class PooledAttentionModel(MutationPredictionModel):
    """Default ProteinMPNN extractor and pooled-attention head."""

    def __init__(self, cfg, *, protein_encoder: nn.Module | None = None):
        feature_extractor = ProteinMPNNFeatureExtractor(cfg, protein_encoder=protein_encoder)
        super().__init__(
            feature_extractor,
            PooledAttentionHead(feature_extractor.output_dim),
        )


# LightningModule defines the hooks that Trainer calls during training/evaluation.
# Guide overview: https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/lightning_module.rst
class PooledAttentionModelPL(pl.LightningModule):
    """Lightning wrapper for loss, split-specific metrics, and optimization.

    train.py builds the data loaders and Trainer; this class defines what happens
    for each batch and which parameters the optimizer updates.
    """
    def __init__(
        self,
        cfg,
        *,
        protein_encoder: nn.Module | None = None,
        model: MutationPredictionModel | None = None,
    ):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        if model is not None and protein_encoder is not None:
            raise ValueError("Supply protein_encoder inside model, or use the default model")
        self.model = model if model is not None else PooledAttentionModel(
            cfg, protein_encoder=protein_encoder
        )
        self.cfg = cfg
        self.save_hyperparameters({'cfg': OmegaConf.to_container(cfg, resolve=True)})
        self.dev = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.out = [cfg.model.get('target_name', 'ddG')]
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
        
        X, S, mask, lengths, chain_M, chain_encoding_all, residue_idx, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask = batch


        pred_ddG = self(X, S, mask, chain_M, residue_idx, chain_encoding_all, mut_positions, mut_wildtype_AAs, mut_mutant_AAs, mut_ddGs, atom_mask)

        # The target slot holds either measured ΔΔG or cached epsilon, both [B, 1].
        mse = F.mse_loss(pred_ddG, mut_ddGs) 

        # Update split-specific metrics and log epoch-level ddG/epsilon scores.
        for out in self.out:
            for name, metric in self.metrics[f"{prefix}_metrics"][out].items():
                metric.update(pred_ddG.reshape(-1), mut_ddGs.reshape(-1))
                # Record epoch metrics and display them in the progress bar.
                # self.log exposes the target's MSE to callbacks and the scheduler.
                # Docs: https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/lightning_module.rst#train-epoch-level-metrics
                self.log(f"{prefix}_{out}_{name}", metric, prog_bar=True, on_step=False, on_epoch=True,
                            batch_size=mut_ddGs.shape[0])
            
        return mse

    # Trainer calls this for each training batch. Returning the loss enables
    # automatic gradient computation and optimizer updates; no backward() here.
    # Guide: https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/optimization.rst
    # Hook: https://github.com/Lightning-AI/pytorch-lightning/blob/master/docs/source-pytorch/common/lightning_module.rst#training-loop
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
