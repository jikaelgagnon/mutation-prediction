"""Construct the same extractor/head pairing for training and checkpoint loading."""

from torch import nn

from mutation_prediction.model.feature_extraction import (
    AllNodeProteinMPNNFeatureExtractor,
    ProteinMPNNFeatureExtractor,
)
from mutation_prediction.model.heads import AllNodeTransformerHead, PooledAttentionHead
from mutation_prediction.model.model import MutationPredictionModel

HEADS = {
    "pooled_attention": PooledAttentionHead,
    "all_node_transformer": AllNodeTransformerHead,
}

TRANSFORMER_DEFAULTS = {
    "d_model": 128,
    "nhead": 4,
    "num_layers": 3,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "activation": "gelu",
    "norm_first": True,
}


def get_head_constructor(cfg):
    name = cfg.model.get("head", "pooled_attention")
    if name not in HEADS:
        raise ValueError(
            f"Unknown model.head {name!r}. Available heads: {', '.join(HEADS)}"
        )
    return HEADS[name]


def build_model(
    cfg, *, protein_encoder: nn.Module | None = None
) -> MutationPredictionModel:
    """Build a configured model; injected encoders retain their freezing policy.

    Registered custom heads keep the mutation-tensor interface. The built-in
    all_node_transformer selects ProteinFeatures and model.transformer settings.
    """
    constructor = get_head_constructor(cfg)
    options = {}
    extractor_type = ProteinMPNNFeatureExtractor
    if cfg.model.get("head", "pooled_attention") == "all_node_transformer":
        extractor_type = AllNodeProteinMPNNFeatureExtractor
        options = {**TRANSFORMER_DEFAULTS, **cfg.model.get("transformer", {})}
    extractor = extractor_type(cfg, protein_encoder=protein_encoder)
    head = constructor(input_dim=extractor.output_dim, **options)
    return MutationPredictionModel(extractor, head)
