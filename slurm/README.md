### Submit a Slurm job

Run from the project directory on the Narval login node. Set up the environment
and download all data/weights using the [setup guide](../docs/setup.md) before
submitting jobs. The main preparation, training, and evaluation jobs use
`SLURM_SUBMIT_DIR`, so submit from the repository root. Their default account
is `def-sushant`; override with `sbatch --account=def-YOUR_PI ...` if needed.

```bash
mkdir -p logs
sbatch slurm/sbatch_train_epsilon_no_aug.slrm
```

Submission prints a job ID. Replace JOB_ID below with that number.

### Prepare the published no-augmentation splits

Run these checks and data preparation on the cluster. Install dependencies and
download the MegaScale data and single-mutant ThermoMPNN checkpoint first (see
the [setup instructions](../docs/setup.md)). Then submit from the repository root:

```bash
mkdir -p logs
sbatch slurm/sbatch_prepare_epsilon_published.slrm
```

The job performs three steps, stopping on any unexpected result:

1. Run the new CLI membership test against the source at commit
   `fbb33032f53267c749c74f61b054c36a28acb9e8`, requiring the old default to fail
   specifically on mutation membership. A missing dependency or other error
   does not count as a red result. Run the same test and existing preparation
   tests against the working source, requiring them to pass.
2. Generate `data/processed/thermompnn/Megascale-D-epsilon.csv` from the three
   published split CSVs using the GPU. This replaces an existing file at that path.
3. Audit the actual export against every published mutation and split, the
   paper's mutation/protein counts, unchanged measured targets, and epsilon labels.

The red phase reads the baseline into a temporary directory without changing
the working tree. Keep that commit in the cluster clone's Git history; shallow
clones must fetch it before running. The CLI test uses real PDB parsing and a
tiny random checkpoint, so the red/green checks need no downloaded weights or
dataset; full generation and its audit do need the downloaded files.

To run each step manually inside a cluster allocation:

```bash
source .venv/bin/activate
python -u scripts/check_published_splits_red_green.py
python -u -m mutation_prediction.prepare_targets --device cuda --output data/processed/thermompnn/Megascale-D-epsilon.csv
MEGASCALE_EPSILON_CSV=data/processed/thermompnn/Megascale-D-epsilon.csv python -m unittest discover -s tests -p test_published_epsilon_export.py -v
```

Expected full-export counts are 85,253 / 10,282 / 18,574 mutations and 90 / 17 /
20 proteins for train / validation / test. The actual-export test skips during
ordinary test discovery unless `MEGASCALE_EPSILON_CSV` is set.

Only after this job succeeds, submit `slurm/sbatch_train_epsilon_no_aug.slrm`
for a fresh run without `--resume`. The training configuration and architecture
do not need changes.

### Evaluate trained models

Copy `best_checkpoint_path` from the training run's W&B Overview/Summary, then
submit it for evaluation. Each training run now has its own directory named
after its W&B run; the same paths are saved locally in that directory's `run.json`.

```bash
sbatch slurm/sbatch_evaluate_models.slrm --checkpoint epsilon=checkpoints/model-epsilon/RUN_NAME/YOUR_SELECTED_CHECKPOINT.ckpt
```

This evaluates the cached additive baseline and the selected epsilon model on
the exact published MegaScale-D test set and the curated PTMUL-D set. Results go to
`results/evaluation/slurm-JOBID/`; logs go to `logs/evaluate_JOBID.out` and `.err`.
The PTMUL-D paths in `configs/model_evaluate.yaml` match the authors' archive
extracted under `data/raw/ptmul/`. To evaluate only MegaScale-D, add
`--datasets megascale_test`.
See the [evaluation guide](../docs/evaluation.md) for adding models, grouped seed
results, ensembles, output details, and the CPU test command.

### Sync W&B runs after the job

All training scripts save W&B runs offline under `wandb/` in the project
directory. The job log prints the exact directory for its run. After the job
exits, use the login node to preview and upload pending runs:

```bash
uv run --extra tracking wandb sync --dry-run
uv run --extra tracking wandb sync
```

To test logging and sync with one training batch and one validation batch,
submit `slurm/sbatch_train_epsilon_wandb_test.slrm`. See the [Narval W&B guide](../README.md#record-wb-runs-offline-on-narval)
for login and single-run sync steps.

### Check job status

# List your queued and running jobs
squeue -u "$USER"

watch -n 5 "squeue -u $USER"

# Show details, including why a job is pending
scontrol show job JOB_ID

# Check a completed job’s state, runtime, memory use, and exit code
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode

Common states: PD = pending, R = running, COMPLETED = finished successfully, FAILED = failed.

### Inspect output

For scripts using logs/slurm_%j.out and logs/slurm_%j.err:

# Follow stdout and stderr live; Ctrl+C stops watching
tail -f logs/slurm_JOB_ID.out logs/slurm_JOB_ID.err

# Show the last 100 lines
tail -n 100 logs/slurm_JOB_ID.out
tail -n 100 logs/slurm_JOB_ID.err

# Browse a full log; press q to quit
less logs/slurm_JOB_ID.out

# List recent logs first
ls -lt logs/

### Cancel a job

scancel JOB_ID

sbatch jobs keep running after you disconnect from the cluster
