#!/usr/bin/env bash
# scripts/slurm/submit.sh — submit the unweighted sweep: train array, then eval array after it.
#   ./scripts/slurm/submit.sh            # from the repo root
# Both arrays share RUN_ID so results land in output/<RUN_ID>/<process>/unweighted/T<T>/.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p slurm logs
export RUN_ID=${RUN_ID:-abc123}

train=$(sbatch --parsable --export=ALL,RUN_ID="$RUN_ID" scripts/slurm/train.sbatch)
echo "train array : $train   (run_id=$RUN_ID)"
eval_=$(sbatch --parsable --export=ALL,RUN_ID="$RUN_ID" --dependency=afterok:"$train" scripts/slurm/eval.sbatch)
echo "eval array  : $eval_   (starts after every train task succeeds)"
echo
echo "watch:   squeue -u \$USER        tail -f slurm/atlas-train_${train}_0.out"
echo "cancel:  scancel $train $eval_"
