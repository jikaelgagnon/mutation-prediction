# Tensor shapes


Tensor and NumPy APIs use [jaxtyping](https://docs.kidger.site/jaxtyping/api/array/)
to show input/output shapes directly in function signatures. For example,
`Float[Tensor, "batch length atoms 3"]` means floating-point coordinates with
one axis each for examples, padded residues, atoms, and xyz coordinates.

| Axis | Meaning |
| --- | --- |
| `batch` | Number of examples. |
| `length` | Padded number of residues, including all chains. |
| `atoms` | Usually 4 backbone atoms or 14 atoms with side chains. CA-only coordinates omit this axis. |
| `mutations` | Maximum number of mutation sites in a batch; shorter records are zero-padded. |
| `neighbors` | Number of neighbors retained in the residue graph. |
| `hidden`, `embedding`, `edge_features` | Feature widths determined by the encoder/layer configuration. |
| `vocab` | Amino-acid vocabulary size (usually 21). |
| `*batch` | Any number of leading batch axes in geometry helpers. |

`Int` marks token IDs and indices; `Real` accepts integer or floating-point
masks; `Shaped` preserves arbitrary dtypes in indexing helpers. An empty shape
`""` denotes a scalar loss. Fixed axes such as `3`, `21`, and `1` are explicit;
pooled stability predictions have shape `"batch 1"`.

The shared [batch contracts](../src/mutation_prediction/tensor_types.py) document
every field of the collator tuples, including NumPy sequence lengths, mutation
indices, targets, and atom masks. `atom_mask` from `tied_featurize_mut` uses
1 for **missing** atoms. Model and Lightning `forward` signatures expose their
inputs individually, in model argument order.

Annotations cover the pooled model, ThermoMPNN heads, the retained ProteinMPNN implementations, graph and side-chain features, geometry,
parsing, collation, and computational notebook helpers. Marimo's generated cell
wrappers retain their normal format. Annotations are descriptive and add no
runtime checking to training. Shape tests verify representative inputs and
outputs, including named-axis consistency:

```bash
uv run python -m unittest discover -s tests -v
```

