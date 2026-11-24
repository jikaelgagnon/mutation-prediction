### Submit a Slurm job

Run from the project directory on the Narval login node. Set up the environment
with `uv sync --extra tracking` before submitting jobs.

```bash
cd ~/projects/def-sushant/jikael/mutation-prediction
mkdir -p logs
sbatch slurm/sbatch_train_epsilon_no_aug.slrm
```

Submission prints a job ID. Replace JOB_ID below with that number.

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
