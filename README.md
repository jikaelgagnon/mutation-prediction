# Mutation Prediction

Predict protein stability changes from mutations using our pooled-attention model
with a pretrained ProteinMPNN encoder.

## Tensor shapes

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
pooled and Siamese stability predictions have shape `"batch 1"`.

The shared [batch contracts](src/mutation_prediction/tensor_types.py) document
every field of the collator tuples, including NumPy sequence lengths, mutation
indices, targets, and atom masks. `atom_mask` from `tied_featurize_mut` uses
1 for **missing** atoms. Model and Lightning `forward` signatures expose their
inputs individually, in model argument order.

Annotations cover the pooled model, ThermoMPNN heads, both ProteinMPNN
implementations, graph and side-chain features, geometry, parsing, collation,
inference helpers, and computational notebook helpers. Marimo's generated cell
wrappers retain their normal format. Annotations are descriptive and add no
runtime checking to training. Shape tests verify representative inputs and
outputs, including named-axis consistency:

```bash
uv run python -m unittest discover -s tests -v
```

## Setup

### Dependencies

#### Using `uv`

This project uses [uv](https://docs.astral.sh/uv/) for Python dependency management.

1. **Install uv** if you don't already have it:

   ```bash
   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for other platforms.

2. **Clone the repository and enter the project directory:**

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

3. **Install the project dependencies:**

   ```bash
   uv sync
   ```

   This creates a virtual environment in `.venv` and installs the dependencies specified in `uv.lock`.

4. **Run commands using the project environment:**

   ```bash
   uv run python <script.py>
   ```

   Or activate the environment manually:

   ```bash
   source .venv/bin/activate
   ```

   On Windows:

   ```powershell
   .venv\Scripts\activate
   ```
5. **Adding new dependencies**
   If you add new dependencies, use `uv` instead of `pip`. For example, to install numpy, use

   ```bash
   uv add numpy
   ```
   This updates `pyproject.toml` and `uv.lock`; other users can install the dependency
   with `uv sync`.

#### Using Marimo

As an experiment, I wanted to try using [Marimo](https://docs.marimo.io/) notebooks for this project instead of typical Jupyter Notebooks, since they have some really cool visualization features (see this [example notebook](https://molab.marimo.io/notebooks/nb_jJiFFtznAy4BxkrrZA1o9b/app)).

To run a Marimo notebook, simply run

```bash
uv run --with marimo marimo edit path/to/python/file
```

When prompted with `Run in a sandboxed venv containing this notebook's dependencies? [Y/n]` select `Y`.

## Download the data


[playground/download_data.py](playground/download_data.py) downloads the MegaScale archive
from Zenodo record `13345274` into `data/raw/megascale/`, and Rosetta
single-mutant structures into `data/raw/megascale-rosetta/`. Each archive is
deleted after extraction. Paths are resolved relative to the script.
Rosetta structures are needed only for augmented training. The Rosetta archive is huge, so don't download unless neccesary.
For the default, unaugmented setup, download only MegaScale with:

```bash
uv run python -c "from playground.download_data import get_megascale; get_megascale()"
```

For the the full dataset, run

```bash
uv run python playground/download_data.py
```

Training expects the following extracted files and directories:

```text
data/raw/megascale/
├── csv/Megascale-D-full.csv
├── pdb/
└── splits.pkl
```

Data is gitignored. The script skips downloading if `data/raw/megascale/` already
exists and is nonempty; it does not verify that an earlier download is complete.

## Pretrained single-mutant baseline

The ThermoMPNN-D code already includes the updated single-mutant ThermoMPNN.
[`mutation_prediction.thermompnn`](src/mutation_prediction/thermompnn.py) provides
an inference-only wrapper for its matching pretrained weights, returning all
single-substitution ΔΔGs in one encoder pass per reference protein. It runs on
CPU or GPU without loading a Lightning trainer or a training dataset.

See [checkpoint download, CLI, and Python usage](docs/thermompnn_inference.md).
This uses the single-mutant model, not the ThermoMPNN-D double-mutant predictor.

To precompute the individual single-mutant predictions and their additive sum
for every measured MegaScale-D pair:

```bash
uv run python playground/precompute_megascale_ddg.py
```

This writes the original rows plus `pred_ddg_1`, `pred_ddg_2`, `baseline_ddg`,
`true_ddg_12`, and `epsilon_target = true_ddg_12 - baseline_ddg`
to `data/processed/thermompnn/Megascale-D-single-predictions.csv`, with a
provenance JSON beside it. It reuses one encoder pass per WT protein and does
not require Rosetta or generate augmentation.

To add epsilon training targets to the previously cached predictions:

```bash
uv run python playground/precompute_megascale_ddg.py \
  --predictions-csv data/processed/thermompnn/Megascale-D-single-predictions.csv \
  --output data/processed/thermompnn/Megascale-D-epsilon.csv
```

This conversion reuses the cached predictions without running inference.

## Train our model

Our model has three independent PyTorch modules:

- [`ProteinMPNNFeatureExtractor`](src/mutation_prediction/model/feature_extraction.py)
  returns `final_embed` with shape `[batch, mutations, features]`. Each mutation
  gets the last K ProteinMPNN decoder states concatenated with its WT amino-acid
  embedding, optionally minus the mutant embedding. `output_dim` gives the
  feature width (384 with K = 2 and the pretrained encoder).
- [`PooledAttentionHead`](src/mutation_prediction/model/heads.py) consumes that
  tensor, pools across mutations, and predicts one scalar per example, `[batch, 1]`.
- [`MutationPredictionModel`](src/mutation_prediction/model/model.py) connects
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

[`src/mutation_prediction/model/train.py`](src/mutation_prediction/model/train.py)
loads the YAML configuration, builds the data loaders, and launches training.

Run [the training entry point](src/mutation_prediction/model/train.py) from the
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

- [Paths and hardware](configs/proteinmpnn_paths.example.yaml): dataset split and
  PDB paths, the ProteinMPNN checkpoint directory (`checkpoints/proteinmpnn`), and
  `platform.accel` (`auto` by default; set `cpu` or `gpu` to choose explicitly).
  The internal `platform.thermompnn_dir` key is retained for loader compatibility;
  it points to ProteinMPNN checkpoints.
- [Model and training options](configs/model_train_epistatic.yaml): the dataset
  CSV path, mutation types, model settings, batch size, learning rate, epoch count,
  data-loader workers, and output directory. Reduce `training.batch_size` if you
  run out of memory. Set `model.freeze_weights: false` to fine-tune the encoder.

The pretrained ProteinMPNN checkpoints are included in `checkpoints/proteinmpnn/`.
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

## Training configurations

The model uses these configs:

| Config | Purpose |
| --- | --- |
| `configs/proteinmpnn_paths.example.yaml` | Shared data paths, ProteinMPNN checkpoints, and device selection. |
| `configs/model_train_epistatic.yaml` | Measured double mutants only, without augmentation. |
| `configs/model_train_epistatic_augmented.yaml` | Measured doubles plus biased augmentation (`range: 3`). |
| `configs/model_train_epsilon.yaml` | Cached epsilon targets for direct residual prediction. |

For augmented training, prepare `data/raw/megascale/csv/Megascale-SD-full.csv`
by combining `Megascale-S-full.csv` and `Megascale-D-full.csv`, preserving their
columns and original ΔΔG signs. The downloader does not create this combined CSV.
Also extract the Rosetta archive using the full download command above. Then run:

```bash
uv run python -m mutation_prediction.model.train \
  configs/proteinmpnn_paths.example.yaml \
  configs/model_train_epistatic_augmented.yaml
```

Augmented checkpoints are saved to `checkpoints/model-biased-aug/`. The loader
uses separate protein splits and augments both training and validation with
this configuration. Use `mut_types: [double]` for final evaluation on measured
double mutants. `data.seed: 42` controls augmentation sampling only; it is an
arbitrary reproducibility seed, not a seed specified in the paper.

The older ThermoMPNN training configs were removed; the playground notebooks
keep their legacy model settings locally.
