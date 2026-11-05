# Single-mutant ThermoMPNN inference

`mutation_prediction.thermompnn` loads the **single-mutant ThermoMPNN v2** model
already included in our ThermoMPNN-D code. It does not use the double-mutant
Siamese predictor. Upstream describes this as an updated single-mutant
ThermoMPNN with fewer parameters and batched inference; it is not the original
2024 ThermoMPNN architecture/checkpoint.

Only a matching `ThermoMPNN-ens*.ckpt` is needed. It contains the complete
ProteinMPNN encoder and prediction head, so there is no separate encoder
checkpoint, training dataset, Lightning trainer, W&B setup, or retraining step.
No additional dependencies are required beyond this project's environment.

## Get the weights

The downloaded weights belong under `checkpoints/thermompnn/`, which is
gitignored. From the repository root:

```bash
mkdir -p checkpoints/thermompnn
curl --fail --location \
  https://raw.githubusercontent.com/Kuhlman-Lab/ThermoMPNN-D/df9a75aaddb674a7c4c193005031fc0536d325fb/model_weights/ThermoMPNN-ens1.ckpt \
  --output checkpoints/thermompnn/ThermoMPNN-ens1.ckpt
```

The file is 10,579,907 bytes, SHA-256:

```text
38248de157190e1f9efe4378239f8db9ec9d8155a2942bfa4aed2f2042dd2277
```

`ThermoMPNN-D-ens*.ckpt` and the original ThermoMPNN repository's
`thermoMPNN_default.pt` are incompatible with this single-mutant v2 loader.
Checkpoint state dictionaries are loaded with `weights_only=True` and checked
strictly against the single-mutant architecture.

## Predict from a PDB

```bash
uv run python -m mutation_prediction.thermompnn \
  --checkpoint checkpoints/thermompnn/ThermoMPNN-ens1.ckpt \
  --pdb data/raw/megascale/pdb/2LCL.pdb \
  --chain A \
  --device cpu \
  --output data/processed/thermompnn/2LCL.csv
```

Use `--device cuda` for GPU inference. The CSV contains all 19 nonidentity
substitutions at each valid residue, including destabilizing mutations. There
is no score cutoff, distance filter, or disulfide penalty. Columns are
`protein`, `chain`, `position`, `wildtype`, `mutant`, `mutation`, and `ddg`.

`position` is a **zero-based parsed-sequence index**; `mutation` uses one-based
sequence positions. These are not original PDB residue numbers or insertion
codes. The selected chain supplies the structural context; other chains are
excluded. Backbone gaps retain their sequence indices and are skipped in the
CSV. Align dataset positions to this sequence before joining predictions.

Predictions use kcal/mol with positive values indicating destabilization.

## Use with existing dataset records

```python
from mutation_prediction.thermompnn import ALPHABET, ThermoMPNNPredictor

predictor = ThermoMPNNPredictor(
    "checkpoints/thermompnn/ThermoMPNN-ens1.ckpt", device="cpu"
)

# protein_record has the seq_chain_* and coords_chain_* fields already
# returned by our dataset. Its experimental mutation annotation is ignored.
table = predictor.predict(protein_record)  # CPU tensor [sequence_length, 20]
ddg1 = table[position1, ALPHABET.index(mutant1)]
ddg2 = table[position2, ALPHABET.index(mutant2)]
baseline_ddg = ddg1 + ddg2
```

Positions are zero-based in the supplied reference record's sequence. Unknown
WT residues and missing-backbone positions have NaN rows; WT-to-WT effects are
zero. The caller's record is not modified. The model is frozen and runs in
evaluation/inference mode, with one encoder pass per reference record.

When caching, key predictions by the actual reference sequence/structure,
chains, residue mapping, and checkpoint. Augmented examples using a modeled
single-mutant reference need predictions on that reference, not the original
WT. Retain experimental double-mutant targets alongside the baseline values.

## Precompute predictions for measured MegaScale-D pairs

From the repository root:

```bash
uv run prepare-targets
```

The defaults use `Megascale-D-full.csv`, the existing WT PDB directory, and
`checkpoints/thermompnn/ThermoMPNN-ens1.ckpt`. Device selection defaults to CUDA
when available, otherwise CPU; override with `--device cpu` or `--device cuda`.
This export uses measured double mutants only and requires no Rosetta structures.

Output:

```text
data/processed/thermompnn/Megascale-D-single-predictions.csv
data/processed/thermompnn/Megascale-D-single-predictions.metadata.json
```

Each input pair keeps its original columns and row order, with these additions:

| Column | Meaning |
| --- | --- |
| `mutation_1`, `mutation_2` | Constituent substitutions, in the original pair's order. |
| `pred_ddg_1`, `pred_ddg_2` | ThermoMPNN single-mutant ΔΔGs on the same WT background. |
| `baseline_ddg` | Sum of the two single-mutant predictions. |
| `true_ddg_12` | Experimental `ddG_ML`, positive for destabilization. |
| `epsilon_target` | `true_ddg_12 - baseline_ddg`, the scalar residual training target. |

The exporter validates both mutation identities and reconstructs the WT sequence
from each row's double-mutant `aa_seq`. It requires that sequence to match the
selected PDB chain exactly. Missing files, sequence mismatches, malformed pairs,
and nonfinite predictions stop the export with an error. It does not silently
drop rows. One encoder pass per WT protein supplies predictions reused across
all its pairs. The metadata records the input, checkpoint, and structure SHA-256
hashes, model, device, chain, and row/protein/unique-mutation counts.

Input labels, including `ddG_ML`, are copied without sign changes. The published
MegaScale-D CSVs and the predictions use kcal/mol with positive values indicating
destabilization. Both signs therefore agree when calculating `epsilon_target`.
The project MegaScale loader selects `epsilon_target` without negation when
configured by `configs/model_train_epsilon.yaml`; legacy `ddG_ML` training
still negates its labels.

To export a particular existing split or use other paths:

```bash
uv run prepare-targets \
  --input-csv data/raw/megascale/csv/Megascale-D-train.csv \
  --pdb-dir data/raw/megascale/pdb \
  --checkpoint checkpoints/thermompnn/ThermoMPNN-ens1.ckpt \
  --device cpu \
  --output data/processed/thermompnn/Megascale-D-train-predictions.csv
```

The script does not create or alter train/validation/test splits.

### Add epsilon targets to an existing prediction export

To reuse predictions already computed, without loading ThermoMPNN or rerunning
inference:

```bash
uv run prepare-targets \
  --predictions-csv data/processed/thermompnn/Megascale-D-single-predictions.csv \
  --output data/processed/thermompnn/Megascale-D-epsilon.csv
```

This creates `Megascale-D-epsilon.csv` and a corresponding metadata JSON. It
retains the prediction checkpoint/structure provenance from the input's metadata
when present, adds a hash of the cached prediction CSV, and records the target
formula and sign convention. Nonfinite labels and inconsistent baseline sums
raise errors. The residual is tied to that specific frozen checkpoint and
includes its single-mutant prediction errors. The existing `ddG_ML_epi_term`
column describes experimental epistasis and is not used as this training target.

The training loss can then be simply:

```python
loss = torch.nn.functional.mse_loss(predicted_epsilon, epsilon_target)
```

At inference, add the predicted correction to the same checkpoint's additive
baseline to obtain the final double-mutant ΔΔG.

## Provenance

Architecture defaults and checkpoint are pinned to
[Kuhlman-Lab/ThermoMPNN-D, df9a75a](https://github.com/Kuhlman-Lab/ThermoMPNN-D/tree/df9a75aaddb674a7c4c193005031fc0536d325fb).
The vectorized single-pass prediction follows upstream's `run_single_ssm` in
`v2_ssm.py` and `examples/configs/single.yaml`. The wrapper adapts it to our
existing modules, supports CPU/GPU, and loads the model without the training
wrapper. The upstream MIT license is retained in
`src/mutation_prediction/thermompnn_d_vendor/thermompnn/LICENSE`.

References: [original ThermoMPNN](https://doi.org/10.1073/pnas.2314853121) and
[ThermoMPNN-D / updated single-mutant model](https://doi.org/10.1002/pro.70003).
