# Model architecture and training


Our model has three independent PyTorch modules:

- [`ProteinMPNNFeatureExtractor`](../src/mutation_prediction/model/feature_extraction.py)
  returns `final_embed` with shape `[batch, mutations, features]`. Each mutation
  gets the last K ProteinMPNN decoder states concatenated with its WT amino-acid
  embedding, optionally minus the mutant embedding. `output_dim` gives the
  feature width (384 with K = 2 and the pretrained encoder).
- [`PooledAttentionHead`](../src/mutation_prediction/model/heads.py) consumes that
  tensor, pools across mutations, and predicts one scalar per example, `[batch, 1]`.
- [`MutationPredictionModel`](../src/mutation_prediction/model/model.py) connects
  any compatible feature extractor and prediction head.

```python
from mutation_prediction.model import (
    MutationPredictionModel,
    PooledAttentionHead,
    MutationPredictionModelPL,
    ProteinMPNNFeatureExtractor,
)

features = ProteinMPNNFeatureExtractor(cfg)
head = PooledAttentionHead(features.output_dim)
model = MutationPredictionModel(features, head)
training_module = MutationPredictionModelPL(cfg, model=model)
```

Replace `head` with any `nn.Module` that maps `[batch, mutations, features]` to
`[batch, 1]` to try another architecture. The training wrapper optimizes the
supplied head and any feature-extractor parameters with `requires_grad=True`.
The ProteinMPNN loader applies `model.freeze_weights`; custom extractors and
injected encoders control their own parameter freezing. Unfrozen extractor
parameters use `training.mpnn_learn_rate` when set; head parameters use
`training.learn_rate`.

`MutationPredictionModelPL(cfg, model)` supplies the training loss, metrics,
optimizer, and learning-rate scheduler for the composed model. The training
script and checkpoint evaluator both use `build_model(cfg)` from
[`model/build.py`](../src/mutation_prediction/model/build.py). `model.head`
selects `pooled_attention` (the unchanged default) or `all_node_transformer`,
including the matching extractor. `HEADS` remains available from `model.train`;
additional registered heads receive the legacy mutation tensor interface.
When using `MutationPredictionModelPL.load_from_checkpoint`, supply the composed
model explicitly with `model=model`; the checkpoint saves its weights and cfg,
but does not reconstruct custom components.
Parameter paths now start with `feature_extractor.prot_mpnn` and
`prediction_head`; checkpoints from the earlier layout require key remapping.
The all-node addition preserves these paths for existing pooled-attention models.

### All-node transformer

`AllNodeProteinMPNNFeatureExtractor` shares the WT ProteinMPNN pass with the
mutation-only extractor, but retains every residue. Its `ProteinFeatures` result
contains `nodes` `[B, L, E * (K + 2) + 1]`, `valid_mask` `[B, L]`,
`mutation_positions` `[B, M]`, and `mutation_mask` `[B, M]`. Each node concatenates
the last K decoder states, the WT amino-acid embedding, a separate WT-minus-mutant
embedding, and a mutation indicator. The last two fields are zero at unchanged
sites. WT identity and mutation identity are always retained; the legacy
`mutant_embedding` switch does not affect this model. With E = 128 and K = 2,
the node width is 513. Identity substitutions, including zero-filled mutation
slots from collation, are excluded from mutation encoding and readout.

`AllNodeTransformerHead` projects nodes to `d_model`, runs independently
initialized [`nn.TransformerEncoderLayer`](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)
blocks in an `nn.ModuleList`, and applies a final layer norm. The default is
three layers, width 128, four attention heads, feed-forward width 512, GELU,
dropout 0.1, `batch_first=True`, and `norm_first=True`. The settings live under
`model.transformer` and are saved in checkpoints. There is no causal mask:
attention is bidirectional across usable residues. The key-padding mask is
`~valid_mask`; `True` excludes a key, following
[PyTorch mask semantics](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html).
Structural and residue-position context comes from ProteinMPNN; no additional
positional encoding is added.

Only after all-node attention does the head gather updated mutation-site vectors,
apply learned scalar-weight pooling, and predict `[B, 1]`. Missing-coordinate
residues and padding are excluded from attention keys and readout. Examples with
no usable mutation site raise `ValueError`. Epsilon loss, signs, optimizer groups,
checkpoint selection, and `predicted_ddg_12 = baseline_ddg + predicted_epsilon`
are unchanged. Inject an encoder with `build_model(cfg, protein_encoder=encoder)`
for focused experiments; its existing `requires_grad` settings are respected.

Launch the separate experiment with:

```bash
uv run python -m mutation_prediction.model.train \
  configs/proteinmpnn_paths.example.yaml \
  configs/model_train_epsilon_transformer.yaml
```

Full attention has quadratic cost in the padded protein length. The new config
starts with `training.batch_size: 8`; tune it for the cluster GPU and protein
lengths. Evaluation has its own batch-size setting, which should also be reduced
for this model. Checkpoints go to `checkpoints/model-epsilon-transformer/RUN_NAME/`.
The existing checkpoint evaluator reconstructs this pairing and its saved
transformer settings automatically; no `model_factory` is needed.

Focused tests use injected lightweight encoders. Run these on the cluster:

```bash
uv run python -m unittest discover -s tests -p 'test_all_node_transformer.py'
uv run python -m unittest discover -s tests -p 'test_pooled_attention_model.py'
uv run python -m unittest discover -s tests -p 'test_epsilon_training.py'
uv run python -m unittest discover -s tests -p 'test_evaluate.py'
```

[`src/mutation_prediction/model/train.py`](../src/mutation_prediction/model/train.py)
loads the YAML configuration, builds the data loaders, and launches training.

Obtain the published data and weights using the [setup guide](setup.md).
Inside a compute allocation, run [the training entry point](../src/mutation_prediction/model/train.py)
from the repository root:

```bash
uv run python -m mutation_prediction.model.train \
  configs/proteinmpnn_paths.example.yaml \
  configs/model_train_epsilon.yaml
```

This trains the configured pooled-attention head on epsilon targets from the
published MegaScale double-mutant training and validation splits. By default,
the pretrained ProteinMPNN encoder is frozen and
the pooled-attention prediction layers are trained for 100 epochs with a batch
size of 256 and a learning rate of `0.00001`.

The command takes two positional YAML files. Settings in the second file override
those in the first:

- [Paths and hardware](../configs/proteinmpnn_paths.example.yaml): dataset split and
  PDB paths, the ProteinMPNN checkpoint directory (`checkpoints/proteinmpnn`), and
  `platform.accel` (`auto` by default; set `cpu` or `gpu` to choose explicitly).
  The internal `platform.thermompnn_dir` key is retained for loader compatibility;
  it points to ProteinMPNN checkpoints.
- [Model and training options](../configs/model_train_epsilon.yaml): the dataset
  CSV path, mutation types, model settings, batch size, learning rate, epoch count,
  data-loader workers, and output directory. Reduce `training.batch_size` if you
  run out of memory. Set `model.freeze_weights: false` to fine-tune the encoder.

Place the pretrained ProteinMPNN checkpoint in `checkpoints/proteinmpnn/`
(see the [setup guide](setup.md#3-download-the-two-pretrained-weight-files)).
If your data or checkpoints are elsewhere, update the corresponding YAML paths.
Relative paths are resolved from the directory where you run training.

The best checkpoint, selected by the lowest validation epsilon MSE
(`val_epsilon_mse`), is saved in `checkpoints/model-epsilon/RUN_NAME/`. Each launch
gets a unique run name shared with W&B, such as
`pooled_attention_epsilon-slurm-12345-a1b2c3d4e5f6`; without Slurm, the job suffix
is omitted. `training.output_dir` sets the parent directory, and `name` sets the
run-name prefix. The checkpoint filename includes the model name, epoch, and
validation score. The learning-rate scheduler continues to monitor validation
MSE (`val_epsilon_mse`).

The default epsilon YAML enables Weights & Biases logging via `project`;
install the tracking extra as shown in setup. The Slurm jobs record offline,
with authentication only required for later uploads. With `project` omitted
or set to null, W&B logging is disabled.

To find the checkpoint for a W&B run, open its Overview/Summary and copy
`best_checkpoint_path` into the evaluation command's `--checkpoint epsilon=...`
argument. `checkpoint_dir`, `last_checkpoint_path`, `best_checkpoint_score`,
`checkpoint_monitor`, and `checkpoint_mode` are also recorded there. Paths are
absolute paths on the training machine or cluster; checkpoint files stay on
that filesystem. The summary updates after checkpoint saves, including during
offline recording, and is uploaded by the usual `wandb sync` command.

Every run directory contains its resolved `config.yaml` and a `run.json` with
the same checkpoint paths, scores, run ID, run name, Slurm job ID, and resume
source. The saved checkpoint's `hyper_parameters.cfg.run` also records its run
identity. Runs without W&B use the same directory layout and local metadata.
Existing checkpoints from earlier runs are left in place.

### Epsilon targets

This reads `data/processed/thermompnn/Megascale-D-epsilon.csv` and minimizes
`mean((epsilon_hat - epsilon_target) ** 2)`, where
`epsilon_target = true_ddg_12 - baseline_ddg`. Epsilon labels keep their CSV
sign; the legacy ddG label negation is not applied. The model learns the
correction to the cached ThermoMPNN single-mutant baseline, including its
prediction errors. To reconstruct the double-mutant prediction in the CSV's
sign convention, use `predicted_ddg_12 = baseline_ddg + epsilon_hat`.

Training uses the existing protein-level train/validation split without
augmentation. Prepare the epsilon CSV with the default `prepare-targets` command
to retain the published ablation memberships (85,253 training and 10,282
validation mutations); the combined CSV also retains all 18,574 test mutations.
The old `Megascale-D-full.csv` source uses different memberships. After switching
sources, start a fresh training run rather than resuming a model trained on the
old dataset. The ProteinMPNN encoder is frozen by default. Checkpoints are
selected by the lowest `val_epsilon_mse` and saved alongside `last.ckpt`,
`config.yaml`, and `run.json` in `checkpoints/model-epsilon/RUN_NAME/`. The learning-rate
scheduler also monitors `val_epsilon_mse`.

Append `--fast-dev-run` for one training and validation batch. To resume a full
run, append `--resume checkpoints/model-epsilon/RUN_NAME/last.ckpt`. A resumed
launch gets a new W&B run and directory, with `resumed_from` pointing to its
source checkpoint. Optimizer, scheduler, and epoch state resume normally; the
new run selects its best checkpoint from its own validation results and leaves
the source run's files intact. Adjust the epsilon YAML to change batch size,
learning rate, epochs, or data-loader workers.

The optional `configs/model_train_epistatic.yaml` retains a legacy measured-ddG
comparison on `Megascale-D-full.csv`; it uses different input memberships and
the opposite target sign. The setup guide and default training job use the
epsilon configuration above for the published no-augmentation splits.

