#!/usr/bin/env bash
# scripts/slurm/euler/train_cells.sh — phase 1 (train) only, for an explicit list of cells.
# Reads "<process> <weighted:true|false> <d> <K>" lines from $CELLS (or stdin) and submits them
# as two job arrays on task.sbatch: d <= $SPLIT_D (short limit, backfills easily) and d > $SPLIT_D
# (long limit, more RAM). One GPU per task, no concurrency cap. Biggest cells are listed first.
#
#   CELLS=missing.txt RUN_ID=2026-09-23_04-02-23 ./scripts/slurm/euler/train_cells.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p slurm logs

PARTITIONS=${PARTITIONS:-research}          # chrysoslab itself needs group euler-chrysos
SPLIT_D=${SPLIT_D:-256}
SMALL_TIME=${SMALL_TIME:-06:00:00}
BIG_TIME=${BIG_TIME:-24:00:00}
DATASET=${DATASET:-gmm}
TTRAIN=${TTRAIN:-500}
RADBASE=${RADBASE:-2.0}
SEED=${SEED:-0}
NSEEDS=${NSEEDS:-3}
EXTRA=${EXTRA:-}
RUN_ID=${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}

vname() { [ "$1" = true ] && echo weighted || echo unweighted; }
TDIR=slurm/tasks_$RUN_ID; mkdir -p "$TDIR"
: > "$TDIR/train_small.txt"; : > "$TDIR/train_big.txt"
sort -k3,3rn -k4,4rn "${CELLS:-/dev/stdin}" | while read -r P W D K; do
  [ -z "$P" ] && continue
  f="$TDIR/train_small.txt"; [ "$D" -gt "$SPLIT_D" ] && f="$TDIR/train_big.txt"
  echo "${RUN_ID}_${P}_$(vname $W)_train_d${D}_K${K}.log|process=$P data=$DATASET data.weighted=$W process.T_train=$TTRAIN sweep.d=[$D] sweep.K=[$K] seed=$SEED n_seeds=$NSEEDS data.radius=$RADBASE stages=[train] train.force_retrain=false train.graph_streams=true run_id=$RUN_ID $EXTRA" >> "$f"
done

submit() {  # <name> <tasks> <time> <mem>
  local n; n=$(wc -l < "$2"); [ "$n" -eq 0 ] && return
  echo "$1 : $(sbatch --parsable -J "$1" -p "$PARTITIONS" --gres=gpu:1 -t "$3" --mem="$4" \
    --array=0-$((n - 1)) --export=ALL,TASKS="$2" scripts/slurm/euler/task.sbatch)  ($n tasks, limit $3)"
}
echo "run_id : $RUN_ID   (task lists in $TDIR/)"
submit atlas-train-big   "$TDIR/train_big.txt"   "$BIG_TIME"   64G
submit atlas-train-small "$TDIR/train_small.txt" "$SMALL_TIME" 32G
echo "watch  : squeue -u \$USER"
