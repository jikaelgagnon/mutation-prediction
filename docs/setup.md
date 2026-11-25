# Set up the ThermoMPNN-D no-augmentation data

Run these commands from the repository root. Download the authors' published
archives and pretrained weights with `curl`; no notebook or split-generation
step is required. Downloads and extraction can be done on a cluster login node.
Run prediction, label generation, tests, training, and evaluation in compute
allocations or the supplied Slurm jobs.

This workflow uses the **Table 1 no-data-augmentation ablation splits**, as
described in [ThermoMPNN-D Methods §4.4](https://pmc.ncbi.nlm.nih.gov/articles/PMC11659742/).
The paper's full-dataset cross-validation benchmark is a separate experiment.

## 1. Install the environment

Clone this repository with its Git history intact, then enter its directory.
Install Python 3.13+ and [uv](https://docs.astral.sh/uv/) if needed:

```bash
uv sync --extra tracking
mkdir -p data/raw/archives data/raw/megascale data/raw/ptmul
mkdir -p checkpoints/thermompnn checkpoints/proteinmpnn logs
```

The tracking extra supplies W&B, which the default training config enables.
The Slurm jobs record W&B offline; logging in is only needed when uploading
runs later. Data, weights, generated labels, and evaluation results are
gitignored.

## 2. Download the published data

Both files come from the authors' versioned
[Zenodo release, record 13345274](https://zenodo.org/records/13345274).
They total approximately 44 MB compressed.

```bash
curl --fail --location --retry 3 \
  'https://zenodo.org/records/13345274/files/Megascale.tar.gz?download=1' \
  --output data/raw/archives/Megascale.tar.gz

curl --fail --location --retry 3 \
  'https://zenodo.org/records/13345274/files/PTMUL-D.tar.gz?download=1' \
  --output data/raw/archives/PTMUL-D.tar.gz
```

Verify the archive checksums published on Zenodo:

```bash
md5sum --check <<'EOF'
935d9c165f80431d3f5fdacbe01b95da  data/raw/archives/Megascale.tar.gz
4fe760d0a29b1e107a17243efa6a348a  data/raw/archives/PTMUL-D.tar.gz
EOF
```

Both checks should say `OK`. Then extract:

```bash
tar -xzf data/raw/archives/Megascale.tar.gz \
  -C data/raw/megascale --strip-components=1

tar -xzf data/raw/archives/PTMUL-D.tar.gz -C data/raw/ptmul
```

The MegaScale archive's outer `Megascale/` directory is stripped; PTMUL-D's
outer directory is kept. This gives the paths already used by the configs:

```text
data/raw/
├── archives/
│   ├── Megascale.tar.gz
│   └── PTMUL-D.tar.gz
├── megascale/
│   ├── splits.pkl
│   ├── csv/
│   │   ├── Megascale-D-train.csv
│   │   ├── Megascale-D-val.csv
│   │   └── Megascale-D-test.csv
│   └── pdb/
└── ptmul/
    └── PTMUL-D/
        ├── NOTE.txt
        ├── csv/PTMUL-D.csv
        └── pdb/
```

There are additional files in the MegaScale archive. The no-augmentation
workflow uses the three split CSVs above and preserves their published
memberships:

| Role | Dataset | Mutations | Proteins |
| --- | --- | ---: | ---: |
| Train | MegaScale-D train | 85,253 | 90 |
| Validation / checkpoint selection | MegaScale-D validation | 10,282 | 17 |
| Held-out test | MegaScale-D test | 18,574 | 20 |
| External test only | PTMUL-D | 536 | 83 |

`Megascale-D-full.csv` and the archive's `compile.py` belong to other workflows;
they are not inputs to the default no-augmentation preparation. Use the
released PTMUL-D curation directly, rather than assembling another ProTherm
dataset. No Rosetta augmentation archive is needed for this experiment.

If you already extracted the two archives to these paths, continue to the
weights step. There is no need to download or extract them again.

## 3. Download the two pretrained weight files

The two files serve different purposes. Both URLs are pinned to upstream
revision `df9a75aaddb674a7c4c193005031fc0536d325fb`.

**Single-mutant ThermoMPNN v2:** generates the additive baseline and epsilon
training targets. This checkpoint contains its own encoder and regression head.

```bash
curl --fail --location --retry 3 \
  'https://raw.githubusercontent.com/Kuhlman-Lab/ThermoMPNN-D/df9a75aaddb674a7c4c193005031fc0536d325fb/model_weights/ThermoMPNN-ens1.ckpt' \
  --output checkpoints/thermompnn/ThermoMPNN-ens1.ckpt

sha256sum --check <<'EOF'
38248de157190e1f9efe4378239f8db9ec9d8155a2942bfa4aed2f2042dd2277  checkpoints/thermompnn/ThermoMPNN-ens1.ckpt
EOF
```

**Vanilla ProteinMPNN:** supplies the frozen feature extractor for our epsilon
model. Download `v_48_020.pt`:

```bash
curl --fail --location --retry 3 \
  'https://raw.githubusercontent.com/Kuhlman-Lab/ThermoMPNN-D/df9a75aaddb674a7c4c193005031fc0536d325fb/vanilla_model_weights/v_48_020.pt' \
  --output checkpoints/proteinmpnn/v_48_020.pt
```

These are the two filenames the default configs expect. Upstream's
`ThermoMPNN-D-ens*.ckpt` files use its double-mutant Siamese architecture and
are not the single-mutant baseline checkpoint used here.

## 4. Prepare and audit epsilon labels on a compute node

From the repository root on the Narval login node:

```bash
sbatch slurm/sbatch_prepare_epsilon_published.slrm
```

The main Slurm jobs run in the directory where you submit them. Their default
account is `def-sushant`; if your allocation differs, override it before the
script name, for example
`sbatch --account=def-YOUR_PI slurm/sbatch_prepare_epsilon_published.slrm`.

This job runs the preparation regression tests, generates baseline predictions
and epsilon targets from the three published MegaScale split CSVs, then audits
the complete output. The red/green regression reads a historical Git commit,
so keep the repository history available. The job needs no notebooks.

After successful completion, these files are ready for training:

```text
data/processed/thermompnn/Megascale-D-epsilon.csv
data/processed/thermompnn/Megascale-D-epsilon.metadata.json
```

The CSV contains **114,109 rows**, with original train/validation/test
membership recorded in `split`. Targets are
`epsilon_target = ddG_ML - baseline_ddg`. PTMUL-D remains external test data and
is not included in epsilon training-label preparation. The evaluator computes
its additive predictions when needed.

Logs are `logs/epsilon_prepare_JOBID.out` and `.err`. Check completion with:

```bash
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
```

Continue after the job has completed successfully and the export audit reports
`OK`. Existing copies of the generated epsilon files are replaced by this job.
An old cache made from `Megascale-D-full.csv` must be regenerated for these
published splits.

## 5. Train, then evaluate the validation-selected checkpoint

Start a fresh training run after preparing the audited data:

```bash
sbatch slurm/sbatch_train_epsilon_no_aug.slrm
```

The job uses the published MegaScale training/validation memberships and keeps
the test data held out. The default encoder is frozen. Checkpoints and the
resolved training configuration are written to `checkpoints/model-epsilon/`.
Choose a checkpoint by validation performance, then evaluate it:

```bash
sbatch slurm/sbatch_evaluate_models.slrm \
  --checkpoint epsilon=checkpoints/model-epsilon/YOUR_SELECTED_CHECKPOINT.ckpt
```

Both test datasets are enabled in `configs/model_evaluate.yaml`, with the paths
shown above. The job reports pooled ΔΔG MSE, RMSE, Pearson, and Spearman scores
for the additive baseline and the selected model. Results are saved under
`results/evaluation/slurm-JOBID/`. See the [evaluation guide](evaluation.md) for
new models, independent training seeds, ensembles, and cluster test commands.

No setup notebook needs to be run at any stage. The [notebooks](../notebooks/README.md)
are optional exploration after the input files are in place.
