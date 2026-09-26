#!/usr/bin/env bash
# rank_experiment/polar_net_rank/euler.sbatch -- the full grid on a SLURM cluster, one GPU per cell.
# Task i runs one (process, d, K) cell into its own folder, cells/<process>_d<d>_K<K>/, so no two
# tasks write the same file; merge.sh then gathers the rows into abc/ and builds the tables.
#
#   sbatch -p research rank_experiment/polar_net_rank/euler.sbatch      (from the repo root)
#
# i = process_index * 27 + d_index * 3 + K_index: 0..26 ddim, 27..53 flow, K varies fastest.

#SBATCH --job-name=polar-rank
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-53
#SBATCH --exclude=euler16
#SBATCH --output=rank_experiment/polar_net_rank/slurm/%x_%A_%a.out
#SBATCH --error=rank_experiment/polar_net_rank/slurm/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=${PY:-$HOME/workspace/iclr_code/venv/bin/python}

PROCESSES=(ddim flow)
DS=(2 4 8 16 32 64 128 256 512)
KS=(2 4 8)
i=$SLURM_ARRAY_TASK_ID
P=${PROCESSES[$(( i / (${#DS[@]} * ${#KS[@]}) ))]}
D=${DS[$(( i / ${#KS[@]} % ${#DS[@]} ))]}
K=${KS[$(( i % ${#KS[@]} ))]}

echo "[$(date)] task $i: process=$P d=$D K=$K  host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
"$PY" rank_experiment/polar_net_rank/polar_net_rank.py --config rank_experiment/polar_net_rank/config.yaml \
    --process "$P" --d "$D" --K "$K" --device cuda \
    --out-dir "rank_experiment/polar_net_rank/cells/${P}_d${D}_K${K}"
echo "[$(date)] done"
