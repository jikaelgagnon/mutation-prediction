"""Exercise declared shapes against real model, collation, and geometry outputs."""

import importlib
import inspect
import types
import unittest
from typing import get_args, get_origin, get_type_hints

import numpy as np
import torch
from jaxtyping import jaxtyped
from omegaconf import OmegaConf

from mutation_prediction.data.collate import (
    tied_featurize_mut,
)
from mutation_prediction.data.mutations import Mutation
from mutation_prediction.model import (
    MutationPredictionModel,
    PooledAttentionHead,
    ProteinMPNNFeatureExtractor,
)
from mutation_prediction.vendor.thermompnn.proteinmpnn import model_utils, rigid_utils
from mutation_prediction.vendor.thermompnn.thermompnn import protein_mpnn_utils
from mutation_prediction.vendor.thermompnn.thermompnn.model import (
    modules,
    side_chain_model,
    v2_model,
)


class TensorShapeTests(unittest.TestCase):
    def assert_contract(self, value, annotation):
        """Check containers recursively and share jaxtyping's named axis bindings."""
        origin = get_origin(annotation)
        if origin is tuple:
            fields = get_args(annotation)
            self.assertEqual(len(value), len(fields))
            for item, field in zip(value, fields):
                self.assert_contract(item, field)
        elif origin is list:
            for item in value:
                self.assert_contract(item, get_args(annotation)[0])
        elif origin is dict:
            self.assertIsInstance(value, dict)
        elif origin is types.UnionType:
            for variant in get_args(annotation):
                try:
                    self.assert_contract(value, variant)
                    break
                except (AssertionError, TypeError):
                    continue
            else:
                self.fail(f"{type(value)} does not match {annotation}")
        else:
            self.assertIsInstance(value, annotation)

    def check_call(self, function, *args, **kwargs):
        signature = inspect.signature(function)
        hints = get_type_hints(function)
        bound = signature.bind(*args, **kwargs)
        with jaxtyped("context"):
            for name, value in bound.arguments.items():
                if name in hints:
                    self.assert_contract(value, hints[name])
            output = function(*args, **kwargs)
            self.assert_contract(output, hints["return"])
        return output

    @staticmethod
    def record(length, mutations=1, atoms=4):
        atom_names = ["N", "CA", "C", "O"] + [f"SC{i}" for i in range(1, 11)]
        coords = np.random.default_rng(length).normal(size=(length, atoms, 3))
        return {
            "name": f"protein_{length}",
            "seq": "ACDEFGHIK"[:length],
            "seq_chain_A": "ACDEFGHIK"[:length],
            "coords_chain_A": {
                f"{atom}_chain_A": coords[:, i].tolist()
                for i, atom in enumerate(atom_names[:atoms])
            },
            "masked_list": ["A"],
            "visible_list": [],
            "num_of_chains": 1,
            "mutation": Mutation(
                list(range(mutations)), list("AC"[:mutations]), ["G"] * mutations, 0.5
            ),
        }

    def test_annotation_names_resolve_for_all_tensor_modules(self):
        names = (
            "mutation_prediction.tensor_types",
            "mutation_prediction.thermompnn",
            "mutation_prediction.model.feature_extraction",
            "mutation_prediction.model.heads",
            "mutation_prediction.model.model",
            "mutation_prediction.vendor.thermompnn.proteinmpnn.model_utils",
            "mutation_prediction.vendor.thermompnn.proteinmpnn.rigid_utils",
            "mutation_prediction.vendor.thermompnn.thermompnn.protein_mpnn_utils",
            "mutation_prediction.vendor.thermompnn.thermompnn.model.modules",
            "mutation_prediction.vendor.thermompnn.thermompnn.model.v2_model",
            "mutation_prediction.vendor.thermompnn.thermompnn.model.side_chain_model",
            "mutation_prediction.data.megascale",
        )
        for name in names:
            module = importlib.import_module(name)
            for _, member in inspect.getmembers(module):
                if getattr(member, "__module__", None) != name:
                    continue
                if inspect.isfunction(member):
                    with self.subTest(function=member.__qualname__):
                        get_type_hints(member)
                elif inspect.isclass(member):
                    get_type_hints(member)
                    for _, method in inspect.getmembers(member, inspect.isfunction):
                        if method.__module__ != name:
                            continue
                        with self.subTest(function=method.__qualname__):
                            get_type_hints(method)

    def test_collators_and_pooled_model_cover_variable_batch_shapes(self):
        encoder = protein_mpnn_utils.ProteinMPNN(
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=1,
            num_decoder_layers=1,
            k_neighbors=3,
            augment_eps=0,
            dropout=0,
        ).eval()
        cfg = OmegaConf.create(
            {"model": {"num_final_layers": 1, "mutant_embedding": True}}
        )
        features = ProteinMPNNFeatureExtractor(cfg, protein_encoder=encoder)
        model = MutationPredictionModel(
            features, PooledAttentionHead(features.output_dim)
        ).eval()
        for batch_size, atoms in ((1, 4), (2, 4), (2, 14)):
            with self.subTest(batch_size=batch_size, atoms=atoms):
                records = [
                    self.record(5 + i, mutations=i + 1, atoms=atoms)
                    for i in range(batch_size)
                ]
                batch = self.check_call(
                    tied_featurize_mut, records, side_chains=atoms == 14
                )
                (
                    X,
                    S,
                    mask,
                    lengths,
                    chain_M,
                    chain_ids,
                    residue_idx,
                    positions,
                    wt,
                    mutant,
                    ddg,
                    atom_mask,
                ) = batch
                # This encoder consumes the backbone; the pooled model accepts the full coordinate tensor.
                with torch.no_grad():
                    output = self.check_call(
                        model.forward,
                        X,
                        S,
                        mask,
                        chain_M,
                        residue_idx,
                        chain_ids,
                        positions,
                        wt,
                        mutant,
                        ddg,
                        atom_mask,
                    )
                self.assertEqual(output.shape, (batch_size, 1))
        self.check_call(tied_featurize_mut, [self.record(5)], ca_only=True)
        self.check_call(model_utils.featurize, [self.record(5), self.record(6)], "cpu")
        self.check_call(
            protein_mpnn_utils.tied_featurize, [self.record(5)], "cpu", None
        )

    def test_gather_contracts_preserve_integer_and_float_features(self):
        neighbors = torch.tensor([[[0, 2], [1, 0], [2, 1]]])
        for module in (model_utils, protein_mpnn_utils):
            for dtype in (torch.int64, torch.float32):
                nodes = torch.arange(12).reshape(1, 3, 4).to(dtype)
                self.check_call(module.gather_nodes, nodes, neighbors)
                self.check_call(module.gather_nodes_t, nodes, neighbors[:, 0])
                self.check_call(
                    module.gather_edges,
                    nodes.unsqueeze(1).expand(1, 3, 3, 4),
                    neighbors,
                )

    def test_loss_and_zero_shot_contracts_distinguish_scalars_from_vectors(self):
        sequence = torch.tensor([[0, 1, 2], [2, 3, 4]])
        log_probs = torch.randn(2, 3, 21).log_softmax(-1)
        mask = torch.ones(2, 3)
        for module in (model_utils, protein_mpnn_utils):
            self.check_call(module.loss_nll, sequence, log_probs, mask)
            self.check_call(module.loss_smoothed, sequence, log_probs, mask)

    def test_encoder_variants_and_side_chain_features(self):
        coordinates = torch.randn(2, 6, 4, 3)
        sequence = torch.randint(0, 21, (2, 6))
        mask = torch.ones(2, 6)
        residue_idx = torch.arange(6).expand(2, -1)
        chain_ids = torch.ones(2, 6, dtype=torch.long)
        for module in (model_utils, side_chain_model):
            with self.subTest(encoder=module.__name__):
                encoder = module.ProteinMPNN(
                    num_letters=21,
                    node_features=16,
                    edge_features=16,
                    hidden_dim=16,
                    num_encoder_layers=1,
                    num_decoder_layers=1,
                    k_neighbors=3,
                    augment_eps=0,
                    dropout=0,
                ).eval()
                self.check_call(
                    encoder.forward,
                    coordinates,
                    sequence,
                    mask,
                    mask,
                    residue_idx,
                    chain_ids,
                )
                self.check_call(
                    encoder.features.forward, coordinates, mask, residue_idx, chain_ids
                )
        features = modules.SideChainProteinFeatures(
            16, 16, top_k=3, action_centers="com"
        )
        self.check_call(
            features.forward,
            torch.randn(2, 6, 14, 3),
            mask,
            residue_idx,
            chain_ids,
            torch.zeros(2, 6, 14, dtype=torch.long),
            sequence,
        )

    def test_attention_contract_covers_squeezed_and_unsqueezed_outputs(self):
        attention = modules.LightAttention(embeddings_dim=8, kernel_size=1).eval()
        self.check_call(attention.forward, torch.randn(2, 8, 1))
        self.check_call(attention.forward, torch.randn(2, 8, 3))

    def test_geometry_contracts_cover_broadcasting_and_coordinate_formats(self):
        quaternions = torch.randn(2, 5, 4)
        rotations = self.check_call(rigid_utils.quat_to_rot, quaternions)
        self.check_call(rigid_utils.rot_to_quat, rotations)
        self.check_call(rigid_utils.rot_vec_mul, rotations, torch.randn(1, 5, 3))
        self.check_call(rigid_utils.rot_matmul, rotations, rotations[:1])
        self.check_call(rigid_utils.quat_multiply, quaternions, quaternions[:1])
        self.check_call(
            rigid_utils.quat_multiply_by_vec, quaternions, torch.randn(1, 5, 3)
        )
        transform = rigid_utils.Rigid.from_3_points(
            *(torch.randn(2, 5, 3) for _ in range(3))
        )
        tensor = self.check_call(transform.to_tensor_4x4)
        self.check_call(rigid_utils.Rigid.from_tensor_4x4, tensor)
        self.check_call(transform.apply, torch.randn(1, 5, 3))
        self.check_call(v2_model._get_cbeta, torch.randn(2, 5, 4, 3))

    def test_named_axis_bindings_detect_inconsistent_inputs(self):
        hints = get_type_hints(model_utils.gather_nodes)
        with jaxtyped("context"):
            self.assert_contract(torch.randn(2, 5, 8), hints["nodes"])
            self.assertNotIsInstance(
                torch.zeros(3, 5, 2, dtype=torch.long), hints["neighbor_idx"]
            )


if __name__ == "__main__":
    unittest.main()
