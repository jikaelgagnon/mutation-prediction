### Submit a Slurm job

Run from the project directory so relative log paths resolve correctly:

cd ~/projects/def-sushant/jikael/mutation-prediction
mkdir -p logs
sbatch slurm/<script>

Submission prints a job ID. Replace JOB_ID below with that number.

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