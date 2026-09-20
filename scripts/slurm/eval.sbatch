#!/usr/bin/env bash
# scripts/slurm/eval.sbatch — phase 2 (evaluate + visualize) of the UNWEIGHTED sweep.
# One array task per (process, T_true), reusing the checkpoints + gt caches train.sbatch wrote.
# Runs after the whole train array succeeds when submitted via scripts/slurm/submit.sh.

#SBATCH --job-name=atlas-eval
#SBATCH --partition=gpu                 # <-- site
#SBATCH --gres=gpu:1                    # <-- site
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                       # eval holds up to 100k anchors/mode x K modes in RAM
#SBATCH --time=12:00:00                 # <-- site: one task evaluates the whole d x K grid
#SBATCH --array=0-5                     # 2 processes x 3 T_true values
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm logs
export ATLAS_ROOT="$PWD"

# module load cuda/12.1                 # <-- site
source venv/bin/activate                # <-- site
PY=${PY:-python3}

PROCESSES=(ddim flow)
TS=(100 200 500)
DLIST="[2,4,8,16,32,64]"
KLIST="[2,4,8,16]"
ANCHORS="[20000,50000,100000]"
TTRAIN=500
SEED=0
NSEEDS=3
RADBASE=2.0
RUN_ID=${RUN_ID:?set RUN_ID to the train run id (submit.sh does this)}

i=$SLURM_ARRAY_TASK_ID
P=${PROCESSES[$(( i / ${#TS[@]} ))]}
T=${TS[$(( i % ${#TS[@]} ))]}

echo "[$(date)] task $i : process=$P T_true=$T  host=$(hostname)  run_id=$RUN_ID"

"$PY" code/main.py \
    process=$P data=gmm data.weighted=false \
    process.T_train=$TTRAIN process.T_true=$T \
    sweep.d=$DLIST sweep.K=$KLIST sweep.anchors=$ANCHORS \
    seed=$SEED n_seeds=$NSEEDS data.radius=$RADBASE \
    stages=[evaluate,visualize] train.force_retrain=false \
    run_id=$RUN_ID device=cuda \
    2>&1 | tee "logs/${RUN_ID}_${P}_unweighted_T${T}.log"

echo "[$(date)] done process=$P T_true=$T -> output/$RUN_ID/$P/unweighted/T$T/"
