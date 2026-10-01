# ThermoMPNN-D

This folder contains the code for training and using ThermoMPNN-D to predict
the effect of amino-acid mutations on protein stability. The main idea is to
start with ProteinMPNN, a model that represents a protein using its sequence
and 3D structure, then train a prediction layer using measured mutation data.

The code is split between `thermompnn/`, which contains the ThermoMPNN-D
training and prediction steps, and `proteinmpnn/`, which contains additional
ProteinMPNN code. The datasets and pretrained model files are kept elsewhere;
their locations are set in config files in the repository's `configs/` folder.

## Read the code in this order

1. **`thermompnn/train_thermompnn.py`** — The starting point for training.
   It reads the settings, loads the data, selects the model, and starts the
   training run.
2. **`thermompnn/parsers.py`** — Shows which dataset the training code uses.
   The current training setup uses MegaScale data.
3. **`thermompnn/datasets/v2_datasets.py`** — Shows how mutation measurements
   and protein structures are read and prepared for the model. In particular,
   `tied_featurize_mut` turns examples into batches of arrays with consistent
   sizes so they can be processed together.
   **`thermompnn/datasets/dataset_utils.py`** defines the fields stored for a
   mutation, including its position, original amino acid, replacement amino
   acid, and measured value.
4. **`thermompnn/model/v2_model.py`** — The main prediction model. Start with
   `TransferModelv2` for the standard model, then read
   `TransferModelv2Siamese` for the version that handles double mutations by
   considering both mutation orders.
5. **`thermompnn/model/modules.py`** — Shows how pretrained ProteinMPNN is
   loaded and describes optional model parts, including attention and
   side-chain features.
   **`thermompnn/protein_mpnn_utils.py`** contains the ProteinMPNN model and
   code that turns protein structures into information the model can use.
6. **`thermompnn/trainer/v2_trainer.py`** — Explains how predictions are
   compared with measured values during training, which scores are reported,
   and how model parameters are updated. The helper scores are in
   **`thermompnn/trainer/trainer_utils.py`**.
7. **`thermompnn/inference/run_inference.py`** and
   **`thermompnn/inference/v2_inference.py`** — Show how to load a trained
   model and predict mutation effects for a dataset. The first starts
   prediction; the second prepares data and runs it through the model.

## How a prediction is made

For each example, the input includes a protein structure, its amino-acid
sequence, the mutation position(s), the original and replacement amino acids,
and a measured stability change. The batch-preparation code pads sequences to
the same length and turns this information into arrays. In code, each batch
contains:

`X` (structure coordinates), `S` (sequence), masks and residue positions,
mutation positions, original and replacement amino acids, measured values, and
an atom mask.

ProteinMPNN uses the sequence and 3D structure to create a learned
representation of each residue. ThermoMPNN-D uses the representation at each
mutation site to predict the mutation's effect. Depending on the settings, it
can also use information about the replacement amino acid, nearby residues,
distances, or side-chain atoms.

For double mutations, the standard model can combine the information from
each mutation site. The Siamese version makes a prediction in both possible
orders and trains the two predictions to agree. During training, predictions
are compared with measured values, and the model is updated to reduce the
difference. The saved model with the best validation Spearman correlation is
used as the selected checkpoint.

During prediction, the code loads a saved checkpoint and processes the chosen
dataset in batches. Use `--keep_preds` to save individual predictions in the
output CSV.

## Files at a glance

### `thermompnn/`

- **`train_thermompnn.py`** — Starts training and sets default options.
- **`parsers.py`** — Selects the dataset used for training.
- **`protein_mpnn_utils.py`** — Reads protein structures and contains the
  ProteinMPNN model used by the standard ThermoMPNN-D model.
- **`datasets/v2_datasets.py`** — Reads mutation datasets and prepares
  structure, sequence, and mutation information for the model. Includes
  MegaScale, FireProt, ddgBench, and ProteinGym dataset support.
- **`datasets/dataset_utils.py`** — Defines a mutation and helps match
  positions between aligned sequences.
- **`model/v2_model.py`** — Standard and Siamese mutation-effect models.
- **`model/modules.py`** — Loads pretrained ProteinMPNN and defines optional
  attention, side-chain, and mutation-combination parts.
- **`model/side_chain_model.py`** — A ProteinMPNN version that can use
  side-chain atom information.
- **`trainer/v2_trainer.py`** — Training steps, losses, scores, and parameter
  updates.
- **`trainer/trainer_utils.py`** — Defines scores used during training.
- **`inference/run_inference.py`** — Starts prediction and loads a saved model.
- **`inference/v2_inference.py`** — Prepares prediction data and runs the
  model.
- **`inference/inference_utils.py`** — Prediction scores and a helper for
  measuring how many residues surround each position.
- **`inference/zero_shot_inference.py`** — Prediction using ProteinMPNN
  sequence probabilities without a ThermoMPNN-D model trained on mutation
  measurements.
- **`inference/infer.yaml`** — Example prediction settings.
- **`ssm_utils.py`** — Helpers for workflows that predict many possible
  mutations in a protein.

### `proteinmpnn/`

This folder contains supporting ProteinMPNN code, including protein geometry,
structure processing, and standalone ProteinMPNN programs.

- **`model_utils.py`** — The ProteinMPNN model and structure processing.
- **`rigid_utils.py`** — Math for representing protein rotations and movement.
- **`utils.py`** — ProteinMPNN data-loading and training helpers.
- **`training.py`**, **`testing.py`**, **`show_scores.py`** — Standalone
  ProteinMPNN training, evaluation, and score display programs.
- **`default_mpnn_020.sh`**, **`inf.sh`** — Example commands for standalone
  ProteinMPNN workflows.

**`LICENSE`** contains the license for the ThermoMPNN-D code in this folder.

## Config files and data

Run commands from the repository root. The example file
`configs/thermompnn_paths.example.yaml` specifies where to find the MegaScale
data and pretrained ProteinMPNN files. Training options are in
`configs/thermompnn_train_single.yaml` and
`configs/thermompnn_train_epistatic.yaml`. The repository's main README has
the commands to start training or prediction.

By default, training keeps the pretrained ProteinMPNN part fixed and learns
the mutation-prediction part. To also update ProteinMPNN during training, set
`model.freeze_weights: false` in the training config. The data and weight files
are not included in this folder, so make sure the configured paths point to
files available on your machine.
