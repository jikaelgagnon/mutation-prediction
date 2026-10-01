# Mutation Prediction

## Setup

### Dependencies


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
   This will automatically add the dependency for in `pyproject.toml`, and other users can install
   via `uv sync`

### Data

The dataset was downloaded from Zenodo; specifically, we use the MegaScale dataset. The data is gitignored. To download the data and explore it, run [ThermoMPNN_D_Data_Exploration.ipynb](playground/ThermoMPNN_D_Data_Exploration.ipynb).


## Training ThermoMPNN

The ProteinMPNN encoder, ThermoMPNN mutation head, dataset featurization, and
Lightning training modules are in
`mutation_prediction.thermo_mpnn_d.thermompnn`. Training uses the copied
implementation from [Kuhlman-Lab/ThermoMPNN-D](https://github.com/Kuhlman-Lab/ThermoMPNN-D)
and its pretrained ProteinMPNN weights; the included ThermoMPNN-D license is
in `src/mutation_prediction/thermo_mpnn_d/LICENSE`. The starter configs point
to the included MegaScale files in `data/raw/megascale` and the copied
ProteinMPNN weights in `tmp/ThermoMPNN-D`; edit
`configs/thermompnn_paths.example.yaml` only if those locations differ. Start
with:

```bash
uv run thermompnn-train configs/thermompnn_paths.example.yaml configs/thermompnn_train_single.yaml
```

For the double-mutant Siamese architecture, use
`configs/thermompnn_train_epistatic.yaml` as the second config. Both
training configs can be edited without changing model code. Add `--extra
tracking` when syncing dependencies if you set a W&B `project` in the YAML.
The starter config freezes the pretrained ProteinMPNN encoder and learns the
ThermoMPNN prediction layers; set `model.freeze_weights: false` to fine-tune
the encoder too, optionally setting `training.mpnn_learn_rate` separately.
The double-mutant config uses the direct measured double-mutant examples
available in the included CSV. The paper's optional `double-aug` training also
needs mutation-specific modeled structures from a separate Rosetta dataset;
those structures are not in `data/raw/megascale`.

The packaged implementation contains the reference ProteinMPNN encoder,
standard ThermoMPNN head (`TransferModelv2`), Siamese/epistatic head
(`TransferModelv2Siamese`), and side-chain-aware encoder. The reference
single-mutant and epistatic checkpoints remain in `tmp/ThermoMPNN-D/model_weights`;
the vanilla ProteinMPNN weights are in `tmp/ThermoMPNN-D/vanilla_model_weights`.
This avoids duplicating large binary weights while letting the package load and
benchmark the reference models.

To evaluate a checkpoint on a configured dataset and save per-mutation
predictions, run:

```bash
uv run thermompnn-infer \
  --model tmp/ThermoMPNN-D/model_weights/ThermoMPNN-ens1.ckpt \
  --config configs/thermompnn_train_single.yaml \
  --local configs/thermompnn_paths.example.yaml \
  --keep_preds
```

Use `ThermoMPNN-D-ens1.ckpt` with `configs/thermompnn_train_epistatic.yaml`
for the Siamese/epistatic model. The CSV contains `ddG_pred`, `ddG_true`,
`batch`, `mut_type`, and `WT_name`, allowing predictions from a new model to
be compared against the same dataset and reference checkpoint.

The model components are ordinary PyTorch modules: `TransferModelv2` exposes
`prot_mpnn`, `light_attention`, `side_chain_features` (when enabled), and
`ddg_out`. Replace a component with a shape-compatible `nn.Module`, or pass a
custom `protein_encoder` or `prediction_head` when constructing
`TransferModelv2(cfg, ...)` or its Lightning wrapper
`TransferModelPLv2(cfg, ...)`. The Siamese implementations offer the same
injection points. Dataset batches have the 12-tensor layout built by
`tied_featurize_mut`; retain that interface when swapping model components if
using the supplied Trainer.