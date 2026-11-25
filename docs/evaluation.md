# Compare models with ThermoMPNN-D

`mutation-evaluate` evaluates chosen models on fixed examples and reports MSE,
RMSE, Pearson correlation (PCC), and Spearman correlation (SCC). All predictions
and targets use kcal/mol, positive for destabilization. Metrics are pooled over
mutations, rather than averaged over batches or proteins.

The defaults in [`model_evaluate.yaml`](../configs/model_evaluate.yaml) compare
our cached additive baseline and a trained epsilon model on the published
MegaScale-D test split and PTMUL-D. The configured PTMUL-D paths match the
authors' archive extracted under `data/raw/ptmul/`. Both datasets must be
downloaded on the cluster. The paper's [Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC11659742/)
reports the following no-augmentation results, as mean ± SD of three runs:

| Dataset | Mutations / proteins | RMSE | PCC | SCC |
| --- | --- | --- | --- | --- |
| MegaScale-D test | 18,574 / 20 | 0.96 ± 0.01 | 0.54 ± 0.02 | 0.49 ± 0.01 |
| PTMUL-D | 536 / 83 | 2.11 ± 0.02 | 0.36 ± 0.04 | 0.35 ± 0.03 |

These are published reference scores, not results reproduced by running this
repository. This command does not reconstruct the upstream ThermoMPNN-D Siamese
architecture. Its predictions can be compared by supplying a CSV or a factory
that wraps the upstream model.

## Run on the cluster

Sync the code and copy `best_checkpoint_path` from your training run's W&B
Overview/Summary. Each training launch has its own checkpoint directory named
after its W&B run; its `run.json` records the same path if tracking is disabled
or the offline run has not been synced. From the repository root, submit:

```bash
mkdir -p logs
sbatch slurm/sbatch_evaluate_models.slrm \
  --checkpoint epsilon=checkpoints/model-epsilon/RUN_NAME/YOUR_SELECTED_CHECKPOINT.ckpt
```

Use the actual checkpoint filename written by training. The evaluator never
chooses checkpoints based on test performance. Evaluation does not restore an
optimizer, resume training, or update weights. Keep model development and
checkpoint selection on training/validation data; reserve these test results
for the final comparisons.

The equivalent command **inside a compute allocation** is:

```bash
.venv/bin/python -m mutation_prediction.evaluate configs/model_evaluate.yaml \
  --checkpoint epsilon=checkpoints/model-epsilon/RUN_NAME/YOUR_SELECTED_CHECKPOINT.ckpt \
  --device cuda
```

After updating the installed package with `uv sync`, the shorter entry point
`mutation-evaluate configs/model_evaluate.yaml ...` is also available. Paths in
the YAML resolve from the working directory; run from the repository root.

Options:

- `--models additive epsilon`: evaluate only named models.
- `--datasets megascale_test ptmul_d`: select datasets explicitly, including
  datasets marked `enabled: false` in the YAML.
- `--checkpoint NAME=PATH`: override a configured model checkpoint; repeat for
  multiple model names.
- `--output-dir PATH`: use a new output directory. Existing directories are
  rejected so a previous evaluation is not overwritten.
- `--device cpu`: run without a GPU. Cached columns and external prediction
  CSVs do not require a GPU or any structures.

MegaScale-only additive evaluation, inside a CPU allocation:

```bash
.venv/bin/python -m mutation_prediction.evaluate configs/model_evaluate.yaml \
  --models additive --datasets megascale_test --device cpu
```

## Dataset selection and baseline

For MegaScale-D, `reference_csv` defines the exact test mutation membership and
measured targets. These keys are joined to the prepared epsilon CSV. Training
and validation rows in that cache are excluded; missing test rows, duplicate
keys, changed targets, or wrong split labels cause an error. Expected mutation
and protein counts are also enforced. Evaluation uses the same published CSV
that the cluster export audit checked.

An epsilon prediction is converted before scoring:

```text
predicted_ddg = baseline_ddg + predicted_epsilon
```

The resulting MSE equals epsilon-target MSE on the same examples. PCC and SCC
are computed on reconstructed ΔΔG, which is the quantity reported in the paper.
When a cached baseline's metadata has a checkpoint hash, the evaluator verifies
that it matches `baseline_checkpoint`. Use the same frozen single-mutant
checkpoint used to prepare the training labels. Without a cached baseline, the
evaluator generates it with one WT inference pass per structure.

### Configure PTMUL-D

Follow the [setup guide](setup.md#2-download-the-published-data) to obtain the
authors' curated `PTMUL-D.tar.gz` and extract it under `data/raw/ptmul/`.
The defaults are
`data/raw/ptmul/PTMUL-D/csv/PTMUL-D.csv` and
`data/raw/ptmul/PTMUL-D/pdb`. If extracted elsewhere, set `ptmul_d.csv` and
`ptmul_d.pdb_dir` to the actual paths. This repository does not download those
files automatically. Adjust
`protein_column`, `mutation_column`, and `target_column` if necessary:

- Protein IDs contain the PDB code followed by a chain, such as `1VIIA`.
- Mutation strings use original PDB residue numbers, separated by `;` or `:`.
- `pdb_template: "{pdb}.pdb"` resolves `1VIIA` to `1VII.pdb`, chain A. The
  placeholders `{chain}` and `{WT_name}` are also available.
- Set `target_sign` explicitly to 1 for positive-destabilizing source ΔΔG, or
  -1 for the opposite convention.

Higher-order mutations in a broader PTMUL CSV are explicitly excluded; all
remaining double substitutions must total 536 mutations across 83 protein IDs.
Distinct measurements of the same pair are retained, with source row indices
appended to their `example_id` values.

Original PDB numbering, gaps, and insertion codes are mapped to parsed indices.
Mutation-site WT identities and backbone atoms are checked before inference.
Missing structures or failed mappings stop evaluation; examples are never
silently discarded to improve coverage or scores. If a curated dataset uses
numbering that differs from its PDB templates, supply `position_map_csv`:

```csv
WT_name,mutation,position
1VIIA,A10V,0
1VIIA,D25G,15
```

`position` is a zero-based index into the parsed chain sequence, including gaps.
Only listed sites override PDB-number mapping; WT identity checks still apply.
Keep any such mapping with the experiment. Its SHA-256 is included in the
manifest. The fixture tests cover mapping mechanics; validation against all
real PTMUL structures must be performed on the cluster.

Both datasets are enabled by default. To select them explicitly:

```bash
sbatch slurm/sbatch_evaluate_models.slrm \
  --checkpoint epsilon=checkpoints/model-epsilon/RUN_NAME/YOUR_SELECTED_CHECKPOINT.ckpt \
  --datasets megascale_test ptmul_d
```

## Add models without editing the evaluator

Add entries under `models` in a copied YAML. There are five prediction adapters
plus an ensemble option.

### Project checkpoints and new heads

`kind: checkpoint` restores a complete `MutationPredictionModelPL` checkpoint.
The standard ProteinMPNN encoder is reconstructed from the known architecture,
then every saved model parameter is loaded strictly. A separate vanilla weight
file is unnecessary for evaluation. Nonstandard encoders can set
`encoder_neighbors`; a custom architecture can supply
`model_factory: your_package.models:build_model`, where
`build_model(cfg)` returns the composed model used during training. It must
accept the same forward arguments as `MutationPredictionModel`.

Set `output: epsilon` for correction models. For a direct ΔΔG model use
`output: ddg`. If a legacy model emits positive-stabilizing predictions, set
`output_sign: -1` to convert them. The default is positive-destabilizing.

### Existing prediction CSVs

```yaml
models:
  other_model:
    kind: csv
    output: ddg
    output_sign: 1
    column: predicted_ddg
    predictions:
      megascale_test: results/other_model/megascale.csv
      ptmul_d: results/other_model/ptmul.csv
```

Prediction CSVs need `example_id` (as exported in `examples.csv`) or both
`WT_name` and `mut_type`, plus the chosen prediction column. They must cover
exactly the selected dataset; shuffled rows are supported. Missing, extra,
duplicate, and nonfinite predictions are rejected. Repeated PTMUL measurements
require the disambiguated `example_id`. No row-order alignment is assumed.

`kind: column` reads the configured `column` directly from the selected source
table. `kind: additive` uses the cached baseline or generates it with
`baseline_checkpoint`.

### A new Python predictor

```yaml
models:
  experiment:
    kind: factory
    factory: my_package.inference:load_predictor
    checkpoint: checkpoints/my_model/selected.pt
    output: epsilon
```

The factory receives the model specification as a plain dictionary and a
`torch.device`. It returns a callable receiving a
[`MutationBatch`](../src/mutation_prediction/tensor_types.py) and returning one
Tensor shaped `[batch]` or `[batch, 1]`:

```python
def load_predictor(spec, device):
    model = MyModel()
    model.load_state_dict(torch.load(spec["checkpoint"], map_location="cpu", weights_only=True))
    model.to(device).eval()

    def predict(batch):
        X, S, mask, lengths, chain_M, chains, residue_idx, pos, wt, mut, _, atom_mask = batch
        return model(X, S, mask, pos, wt, mut)

    return predict
```

The evaluator moves tensors to the selected device, disables gradients, and
sets the target slot to zero. If returning a plain callable that owns a model,
set that model to evaluation mode in the factory as shown. Returned `nn.Module`
predictors are automatically put into evaluation mode. Custom predictors can
adapt these tensors to a different model representation without changing the
dataset selection or scoring code.

### Independent runs and ensembles

Give independent checkpoints distinct model names and the same `group`:

```yaml
models:
  seed1:
    kind: checkpoint
    checkpoint: checkpoints/seed1/selected.ckpt
    output: epsilon
    group: epsilon
  seed2:
    kind: checkpoint
    checkpoint: checkpoints/seed2/selected.ckpt
    output: epsilon
    group: epsilon
  seed3:
    kind: checkpoint
    checkpoint: checkpoints/seed3/selected.ckpt
    output: epsilon
    group: epsilon
  ensemble:
    kind: ensemble
    members: [seed1, seed2, seed3]
```

`replicates.csv` reports the mean and sample SD of the individual run metrics.
An ensemble separately averages the reconstructed ΔΔG predictions before
scoring; it does not average RMSE values. These are different comparisons.

## Output files

Default output directories are unique UTC timestamps under `results/evaluation`.
The Slurm job uses `results/evaluation/slurm-JOBID`. Files include:

- `metrics.csv`: one row per dataset/model, MSE/RMSE/PCC/SCC, coverage counts,
  and differences from the configured paper reference means.
- `replicates.csv`: grouped run means and sample standard deviations.
- `config.yaml`: the resolved experiment settings.
- `manifest.json`: completion status, units/sign, seed/device, reference scores,
  and hashes of dataset, cache metadata, checkpoint, mapping, and structure files.
- `<dataset>/examples.csv`: exact selected examples in evaluation order.
- `<dataset>/<model>.predictions.csv`: keys, truth, reconstructed ΔΔG, errors,
  and baseline/correction values when a baseline is used.
- `<dataset>/<model>.per_protein.csv`: diagnostic scores by protein, separate
  from the pooled scores used for comparison.

Undefined correlations, for example constant predictions or a one-mutation
protein, are written as null in JSON and empty cells in CSV. Partial output
from a failed run has no completed manifest.

## Run tests on the cluster

No local execution is required. From the repository root on the cluster:

```bash
salloc --account=def-sushant --cpus-per-task=1 --mem=8G --time=00:15:00
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 srun .venv/bin/python -m unittest discover -s tests -p test_evaluate.py -v
exit
```

The tests use tiny CSV/PDB fixtures and a real randomly initialized checkpoint;
they need no GPU, downloaded dataset, or pretrained weights. They cover physical
target reconstruction, complete example coverage, keyed alignment, signs,
custom predictors without label inputs, ensemble/replicate handling, checkpoint
restoration, and PTMUL residue numbering. Real-data evaluation remains a cluster
run using the selected trained checkpoints.
