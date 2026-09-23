#!/usr/bin/env bash
# scripts/slurm/euler/submit.sh — the scripts/main.sh sweep as three chained SLURM arrays on euler.
#   1) train : one task per (process, variant, d, K)   -> 1 GPU each
#   2) eval  : one task per (process, variant, T, d)   -> 1 GPU each, after ALL of train succeeds
#   3) merge : one task per (process, variant, T)      -> CPU only, after all of eval succeeds
# Every task may run in any partition of $PARTITIONS on any GPU type; SLURM starts it wherever
# a GPU frees up first. No concurrency cap: as many tasks run at once as the cluster allows.
#
#   ./scripts/slurm/euler/submit.sh                   # from the repo root, on the euler login node
#   DIMS="2 4" KS="2" WEIGHTED=false ./scripts/slurm/euler/submit.sh     # small test
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p slurm logs

PARTITIONS=${PARTITIONS:-research}          # covers the chrysos nodes euler05-08 too; chrysoslab itself needs group euler-chrysos
PROCESSES=${PROCESSES:-"ddim flow"}
DIMS=${DIMS:-"2 4 8 16 32 64 128 256"}
KS=${KS:-"2 4 8 16 32"}
TS=${TS:-"100 200 500"}
ANCHORS=${ANCHORS:-"[20000,50000,100000]"}
TTRAIN=${TTRAIN:-500}
RADBASE=${RADBASE:-2.0}
SEED=${SEED:-0}
NSEEDS=${NSEEDS:-3}
DATASET=${DATASET:-gmm}
WEIGHTED=${WEIGHTED:-"false true"}
TRAIN_TIME=${TRAIN_TIME:-12:00:00}
EVAL_TIME=${EVAL_TIME:-12:00:00}
EXTRA=${EXTRA:-}
RUN_ID=${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}

vname() { [ "$1" = true ] && echo weighted || echo unweighted; }
DLIST="[$(echo $DIMS | tr ' ' ',')]"
KLIST="[$(echo $KS | tr ' ' ',')]"
COMMON="data=$DATASET process.T_train=$TTRAIN seed=$SEED n_seeds=$NSEEDS data.radius=$RADBASE run_id=$RUN_ID $EXTRA"
TDIR=slurm/tasks_$RUN_ID; mkdir -p "$TDIR"

# biggest cells first so the long poles start early
: > "$TDIR/train.txt"
for D in $(echo $DIMS | tr ' ' '\n' | sort -rn); do
  for K in $(echo $KS | tr ' ' '\n' | sort -rn); do
    for P in $PROCESSES; do for W in $WEIGHTED; do V=$(vname $W)
      echo "${RUN_ID}_${P}_${V}_train_d${D}_K${K}.log|process=$P data.weighted=$W sweep.d=[$D] sweep.K=[$K] stages=[train] train.force_retrain=false train.graph_streams=true $COMMON" >> "$TDIR/train.txt"
    done; done
  done
done
: > "$TDIR/eval.txt"
for D in $(echo $DIMS | tr ' ' '\n' | sort -rn); do
  for P in $PROCESSES; do for W in $WEIGHTED; do V=$(vname $W)
    for T in $TS; do
      echo "${RUN_ID}_${P}_${V}_T${T}_eval_d${D}.log|process=$P data.weighted=$W process.T_true=$T sweep.d=[$D] sweep.K=$KLIST sweep.anchors=$ANCHORS stages=[evaluate] eval.part=true train.force_retrain=false $COMMON" >> "$TDIR/eval.txt"
    done
  done; done
done
: > "$TDIR/merge.txt"
for P in $PROCESSES; do for W in $WEIGHTED; do V=$(vname $W)
  for T in $TS; do
    echo "${RUN_ID}_${P}_${V}_T${T}.log|process=$P data.weighted=$W process.T_true=$T sweep.d=$DLIST sweep.K=$KLIST sweep.anchors=$ANCHORS stages=[merge,visualize] train.force_retrain=false $COMMON" >> "$TDIR/merge.txt"
  done
done; done

n() { echo $(( $(wc -l < "$1") - 1 )); }
train=$(sbatch --parsable -J atlas-train -p "$PARTITIONS" --gres=gpu:1 -t "$TRAIN_TIME" \
  --array=0-$(n "$TDIR/train.txt") --export=ALL,TASKS="$TDIR/train.txt" scripts/slurm/euler/task.sbatch)
eval_=$(sbatch --parsable -J atlas-eval -p "$PARTITIONS" --gres=gpu:1 --mem=64G -t "$EVAL_TIME" \
  --dependency=afterok:"$train" \
  --array=0-$(n "$TDIR/eval.txt") --export=ALL,TASKS="$TDIR/eval.txt" scripts/slurm/euler/task.sbatch)
merge=$(sbatch --parsable -J atlas-merge -p "$PARTITIONS" -t 02:00:00 \
  --dependency=afterok:"$eval_" \
  --array=0-$(n "$TDIR/merge.txt") --export=ALL,TASKS="$TDIR/merge.txt",DEVICE=cpu scripts/slurm/euler/task.sbatch)

echo "run_id : $RUN_ID   (task lists in $TDIR/)"
echo "train  : $train  ($(wc -l < "$TDIR/train.txt") tasks)"
echo "eval   : $eval_  ($(wc -l < "$TDIR/eval.txt") tasks, after train)"
echo "merge  : $merge  ($(wc -l < "$TDIR/merge.txt") tasks, after eval)"
echo "watch  : squeue -u \$USER    tail -f slurm/atlas-train_${train}_0.out"
echo "cancel : scancel $train $eval_ $merge"
