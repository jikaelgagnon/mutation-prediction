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

## Provenance

Architecture defaults and checkpoint are pinned to
[Kuhlman-Lab/ThermoMPNN-D, df9a75a](https://github.com/Kuhlman-Lab/ThermoMPNN-D/tree/df9a75aaddb674a7c4c193005031fc0536d325fb).
The vectorized single-pass prediction follows upstream's `run_single_ssm` in
`v2_ssm.py` and `examples/configs/single.yaml`. The wrapper adapts it to our
existing modules, supports CPU/GPU, and loads the model without the training
wrapper. The upstream MIT license is retained in
`src/mutation_prediction/thermo_mpnn_d/LICENSE`.

References: [original ThermoMPNN](https://doi.org/10.1073/pnas.2314853121) and
[ThermoMPNN-D / updated single-mutant model](https://doi.org/10.1002/pro.70003).
