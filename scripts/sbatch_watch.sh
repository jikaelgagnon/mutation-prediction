#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
    echo "Usage: $0 [sbatch options] job.slrm [job arguments]" >&2
    exit 1
fi

mkdir -p logs
submission=$(sbatch --parsable --output='logs/slurm_%j.out' --error='logs/slurm_%j.err' "$@")
job_id=${submission%%;*}
echo "Submitted job $job_id (Ctrl+C stops watching; the job keeps running)."

while true; do
    queue=$(squeue --noheader --jobs="$job_id" --format='%i %T %M %D %R')
    printf '\nJOBID STATE TIME NODES NODELIST(REASON)\n%s\n' "$queue"
    # A short job may finish between polls; follow its logs in that case too.
    [[ -z "$queue" ]] && break
    read -r _ state _ <<< "$queue"
    case "$state" in
        RUNNING|COMPLETING) break ;;
        PENDING|CONFIGURING) sleep 5 ;;
        *) echo "Job $job_id is $state; check with scontrol show job $job_id." >&2; exit 1 ;;
    esac
done

# Retry if allocation is visible before Slurm creates the log files.
exec tail -f --retry "logs/slurm_${job_id}.out" "logs/slurm_${job_id}.err"
