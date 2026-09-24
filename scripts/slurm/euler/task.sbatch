#!/usr/bin/env bash
# scripts/slurm/euler/task.sbatch — generic array worker: task i runs line i of $TASKS.
# Each line is "<logname>|<hydra overrides>" (written by submit.sh). Partition, GPU, time and
# array range are passed on the sbatch command line by submit.sh.

#SBATCH --cpus-per-task=4
#SBATCH --mem=8G                        # measured peak RSS ~2.7 GB (d=1024, 3 seeds); more blocks packing
#SBATCH --exclude=euler16               # 2x RTX 2080 Ti (11 GB): too small for the big eval cells
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm logs
export ATLAS_ROOT="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=${PY:-$HOME/workspace/iclr_code/venv/bin/python}
DEVICE=${DEVICE:-cuda}

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKS:?}")
log="${line%%|*}"; args="${line#*|}"

echo "[$(date)] task $SLURM_ARRAY_TASK_ID  host=$(hostname)  partition=$SLURM_JOB_PARTITION  gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "  $args"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

"$PY" code/main.py $args device="$DEVICE" 2>&1 | tee "logs/$log"
echo "[$(date)] done $log"
