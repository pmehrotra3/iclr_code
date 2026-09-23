#!/usr/bin/env bash
# scripts/slurm/train.sbatch — phase 1 (train) of the UNWEIGHTED sweep as a SLURM job array.
# One array task per (process, d, K) cell, training all 3 seeds of the cell together, exactly
# like scripts/main.sh phase 1. Mirrors: DIMS="2 4 8 16 32 64" KS="2 4 8 16" TTRAIN=500 NSEEDS=3.
#
# Submit with scripts/slurm/submit.sh (which chains eval.sbatch after this array), or directly:
#   sbatch scripts/slurm/train.sbatch
# Edit the four "<-- site" lines for your cluster (see scripts/slurm/README.md).

#SBATCH --job-name=atlas-train
#SBATCH --partition=gpu                 # <-- site: sinfo -s
#SBATCH --gres=gpu:1                    # <-- site: e.g. gpu:a100:1  (sinfo -o "%P %G")
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00                 # <-- site: hard kill; d=64,K=16 x 3 seeds took ~25 min on an L40S
#SBATCH --array=0-47%8                  # 2 processes x 6 dims x 4 Ks = 48 cells, at most 8 at once
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"                  # repo root (submit from there)
mkdir -p slurm logs
export ATLAS_ROOT="$PWD"                # data/ and output/ live here

# ---- environment (site specific) ----
# module load cuda/12.1                 # <-- site: module avail cuda   (only if python/torch need it)
source venv/bin/activate                # <-- site: or `conda activate <env>`
PY=${PY:-python3}

# ---- sweep definition: keep in sync with scripts/main.sh defaults ----
PROCESSES=(ddim flow)
DS=(2 4 8 16 32 64)
KS=(2 4 8 16)
TTRAIN=500
SEED=0
NSEEDS=3
RADBASE=2.0
RUN_ID=${RUN_ID:-abc123}   # one shared id for train + eval (submit.sh exports it)

# array index -> (process, d, K); Ks vary fastest so small cells come first
i=$SLURM_ARRAY_TASK_ID
nK=${#KS[@]}; nD=${#DS[@]}
P=${PROCESSES[$(( i / (nD * nK) ))]}
D=${DS[$(( (i / nK) % nD ))]}
K=${KS[$(( i % nK ))]}

echo "[$(date)] task $i : process=$P d=$D K=$K  host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-?}  run_id=$RUN_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PY" code/main.py \
    process=$P data=gmm data.weighted=false \
    process.T_train=$TTRAIN \
    sweep.d="[$D]" sweep.K="[$K]" \
    seed=$SEED n_seeds=$NSEEDS data.radius=$RADBASE \
    stages=[train] train.force_retrain=false train.graph_streams=true \
    run_id=$RUN_ID device=cuda \
    2>&1 | tee "logs/${RUN_ID}_${P}_unweighted_train_d${D}_K${K}.log"

echo "[$(date)] done process=$P d=$D K=$K"
