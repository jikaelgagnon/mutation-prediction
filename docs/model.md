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
script selects a head constructor from its `HEADS` dictionary using `model.head`
(default: `pooled_attention`) and combines it with `ProteinMPNNFeatureExtractor`.
Add another compatible head constructor to that dictionary to select it by name.
When using `MutationPredictionModelPL.load_from_checkpoint`, supply the composed
model explicitly with `model=model`; the checkpoint saves its weights and cfg,
but does not reconstruct custom components.
Parameter paths now start with `feature_extractor.prot_mpnn` and
`prediction_head`; checkpoints from the earlier layout require key remapping.

[`src/mutation_prediction/model/train.py`](../src/mutation_prediction/model/train.py)
loads the YAML configuration, builds the data loaders, and launches training.

Run [the training entry point](../src/mutation_prediction/model/train.py) from the
repository root:

```bash
uv run python -m mutation_prediction.model.train \
  configs/proteinmpnn_paths.example.yaml \
  configs/model_train_epistatic.yaml
```

This trains the configured pooled-attention head on the MegaScale double-mutant training and
validation splits. By default, the pretrained ProteinMPNN encoder is frozen and
the pooled-attention prediction layers are trained for 100 epochs with a batch
size of 128 and a learning rate of `0.00001`.

The command takes two positional YAML files. Settings in the second file override
those in the first:

- [Paths and hardware](../configs/proteinmpnn_paths.example.yaml): dataset split and
  PDB paths, the ProteinMPNN checkpoint directory (`checkpoints/proteinmpnn`), and
  `platform.accel` (`auto` by default; set `cpu` or `gpu` to choose explicitly).
  The internal `platform.thermompnn_dir` key is retained for loader compatibility;
  it points to ProteinMPNN checkpoints.
- [Model and training options](../configs/model_train_epistatic.yaml): the dataset
  CSV path, mutation types, model settings, batch size, learning rate, epoch count,
  data-loader workers, and output directory. Reduce `training.batch_size` if you
  run out of memory. Set `model.freeze_weights: false` to fine-tune the encoder.

Place the pretrained ProteinMPNN checkpoint in `checkpoints/proteinmpnn/` (see the README).
If your data or checkpoints are elsewhere, update the corresponding YAML paths.
Relative paths are resolved from the directory where you run training.

The best checkpoint, selected by the highest validation Spearman correlation
(`val_ddG_spearman`), is saved in `checkpoints/model-no-aug/`. Its filename includes the run
name, epoch, and validation score. Change `training.output_dir` or `name` in the
training configuration to customize these outputs. The learning-rate scheduler
continues to monitor validation MSE (`val_ddG_mse`).

To enable Weights & Biases logging, uncomment the `project` setting in
`configs/model_train_epistatic.yaml` and authenticate with W&B before training.
With `project` omitted, W&B logging is disabled.

### Train epsilon predictions

To train `MutationPredictionModelPL` to output epsilon hat directly, run:

```bash
uv run python -m mutation_prediction.model.train \
  configs/proteinmpnn_paths.example.yaml \
  configs/model_train_epsilon.yaml
```

This reads `data/processed/thermompnn/Megascale-D-epsilon.csv` and minimizes
`mean((epsilon_hat - epsilon_target) ** 2)`, where
`epsilon_target = true_ddg_12 - baseline_ddg`. Epsilon labels keep their CSV
sign; the legacy ddG label negation is not applied. The model learns the
correction to the cached ThermoMPNN single-mutant baseline, including its
prediction errors. To reconstruct the double-mutant prediction in the CSV's
sign convention, use `predicted_ddg_12 = baseline_ddg + epsilon_hat`.

Training uses the existing protein-level train/validation split without
augmentation. The ProteinMPNN encoder is frozen by default. Checkpoints are
selected by the lowest `val_epsilon_mse` and saved alongside `last.ckpt` and
the resolved `config.yaml` in `checkpoints/model-epsilon/`. The learning-rate
scheduler also monitors `val_epsilon_mse`.

Append `--fast-dev-run` for one training and validation batch. To resume a full
run, append `--resume checkpoints/model-epsilon/last.ckpt`. Adjust the epsilon
YAML to change batch size, learning rate, epochs, or data-loader workers.

