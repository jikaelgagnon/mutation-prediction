# Mutation Prediction

Predict protein stability changes from mutations using our pooled-attention model
with a pretrained ProteinMPNN encoder.

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

## Train our model

Our model code lives in [`src/mutation_prediction/model/model.py`](src/mutation_prediction/model/model.py):

- `PooledAttentionModel` extracts ProteinMPNN features at mutated residues, pools
  them with learned attention, and predicts one ΔΔG per example.
- `PooledAttentionModelPL` wraps the model with the training loss, evaluation
  metrics, optimizer, and learning-rate scheduler.

[`src/mutation_prediction/model/train.py`](src/mutation_prediction/model/train.py)
loads the YAML configuration, builds the data loaders, and launches training.

Run [the training entry point](src/mutation_prediction/model/train.py) from the
repository root:

```bash
uv run python -m mutation_prediction.model.train \
  configs/proteinmpnn_paths.example.yaml \
  configs/model_train_epistatic.yaml
```

This trains `PooledAttentionModel` on the MegaScale double-mutant training and
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

The best checkpoint, selected by the lowest validation ΔΔG mean squared error
(`val_ddG_mse`), is saved in `checkpoints/model-no-aug/`. Its filename includes the run
name, epoch, and validation score. Change `training.output_dir` or `name` in the
training configuration to customize these outputs.

To enable Weights & Biases logging, uncomment the `project` setting in
`configs/model_train_epistatic.yaml` and authenticate with W&B before training.
With `project` omitted, W&B logging is disabled.

## Training configurations

Only three configs are needed for our model:

| Config | Purpose |
| --- | --- |
| `configs/proteinmpnn_paths.example.yaml` | Shared data paths, ProteinMPNN checkpoints, and device selection. |
| `configs/model_train_epistatic.yaml` | Measured double mutants only, without augmentation. |
| `configs/model_train_epistatic_augmented.yaml` | Measured doubles plus biased augmentation (`range: 3`). |

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
