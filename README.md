# Mutation prediction

Predict double-mutant stability changes using a pretrained single-mutant
ThermoMPNN baseline and our ProteinMPNN-based epsilon model.

```text
WT structure + substitutions
  ├─ ThermoMPNN → ΔΔG₁ + ΔΔG₂ = additive baseline
  └─ ProteinMPNN → mutation features → pooled-attention head → ε̂

Predicted double-mutant ΔΔG = additive baseline + ε̂
```

The training target is `epsilon_target = true_ddg_12 - baseline_ddg`.
It includes errors in the predicted single-mutant baseline as well as epistasis.
Targets and final predictions use kcal/mol, positive for destabilization.

## Quick start

Run commands from the repository root. Install Python 3.13+ and
[uv](https://docs.astral.sh/uv/), then:

```bash
uv sync
uv run download-megascale
```

Data and checkpoints are gitignored. The download creates
`data/raw/megascale/{csv,pdb,splits.pkl}`; no Rosetta structures are needed.
A nonempty existing MegaScale directory is skipped, so remove an incomplete
download before retrying.

Download the matching single-mutant ThermoMPNN v2 checkpoint using the
[inference guide](docs/thermompnn_inference.md#get-the-weights).
For the epsilon feature extractor, place `v_48_020.pt` from the pinned
[upstream ProteinMPNN weights](https://github.com/Kuhlman-Lab/ThermoMPNN-D/tree/df9a75aaddb674a7c4c193005031fc0536d325fb/vanilla_model_weights)
in `checkpoints/proteinmpnn/`. These are two different checkpoints.

Prepare the residual labels and train:

```bash
uv run prepare-targets --output data/processed/thermompnn/Megascale-D-epsilon.csv
uv run mutation-prediction configs/proteinmpnn_paths.example.yaml configs/model_train_epsilon.yaml
```

For an existing cached export, use
`prepare-targets --predictions-csv <cached.csv> --output data/processed/thermompnn/Megascale-D-epsilon.csv`.
This computes labels without running the baseline again.

Edit the paths/device in `configs/proteinmpnn_paths.example.yaml` and training
options in `configs/model_train_epsilon.yaml`. Paths resolve from the working
directory. Append `--fast-dev-run` for one training/validation batch, or
`--resume checkpoints/model-epsilon/last.ckpt` to resume training.
Checkpoints are selected by validation epsilon MSE.

To predict single-mutant effects for a structure:

```bash
uv run thermompnn-infer --checkpoint checkpoints/thermompnn/ThermoMPNN-ens1.ckpt --pdb protein.pdb --chain A --device cpu --output predictions.csv
```

The epsilon model currently exposes a Python model API and training command;
there is no combined double-mutant prediction CLI. At inference, add its output
to the baseline generated with the same ThermoMPNN checkpoint used for training.
See the [model API](docs/model.md) for composition and checkpoint loading.

## Where to start reading

| Location | Purpose |
| --- | --- |
| [thermompnn.py](src/mutation_prediction/thermompnn.py) | Frozen single-mutant baseline API and CLI. |
| [prepare_targets.py](src/mutation_prediction/prepare_targets.py) | Cached single predictions, additive baseline, epsilon labels, and provenance. |
| [data/](src/mutation_prediction/data/) | MegaScale dataset, collator, mutation records, and download. |
| [model/](src/mutation_prediction/model/) | ProteinMPNN feature extractor, your head, composition, and training. |
| [configs/](configs/) | Epsilon training plus optional measured-ddG comparison. |
| [vendor/](src/mutation_prediction/vendor/) | Retained upstream architecture; start elsewhere when learning this project. |
| [notebooks/](notebooks/) | Optional exploration, outside the supported pipeline. |
| [tests/](tests/) | Inference, label conventions, training/resume, and tensor contracts. |

Read `model/feature_extraction.py` → `model/heads.py` → `model/model.py` →
`model/train.py` to understand your model. Detailed [tensor shapes](docs/tensor_shapes.md)
and [upstream provenance](src/mutation_prediction/vendor/README.md) are separate references.

## Optional tools

```bash
uv sync --extra notebooks
uv run marimo edit notebooks/try_model.py
uv sync --extra tracking
```

Run notebooks in the project environment, not an isolated notebook sandbox.
Set `project` in the training YAML to enable W&B logging.
`configs/model_train_epistatic.yaml` retains the measured-ddG comparison;
its legacy training targets use the opposite sign from the epsilon workflow.
Rosetta augmentation, upstream ThermoMPNN training, the Siamese model, and
legacy benchmark inference have been retired.

Validation: `uv run python -m unittest discover -s tests -v`.
