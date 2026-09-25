#!/usr/bin/env bash
# scripts/slurm/euler/phase2.sh — phase 2 (evaluate) + phase 3 (merge, visualize) as SLURM jobs.
# One evaluate task per (process, variant, T_true, d). A task starts as soon as the cells it reads
# are trained: if all K cells of its (process, variant, d) already have their NSEEDS checkpoints
# on disk it starts right away, otherwise it waits (afterok) on exactly the phase-1 array tasks
# that train the missing cells. Evaluate skips rows already in cells/*.json, so a preempted task
# resumes where it stopped. merge runs after every evaluate task.
#
#   PHASE1="67416:slurm/tasks_X/train_big.txt 67417:slurm/tasks_X/train_small.txt" \
#     RUN_ID=abc123 ./scripts/slurm/euler/phase2.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p slurm logs

PHASE1=${PHASE1:-}                           # "<array job id>:<task list>" pairs of running phase-1 arrays
RUN_ID=${RUN_ID:-abc123}
PARTITIONS=${PARTITIONS:-research}
PROCESSES=${PROCESSES:-"ddim flow"}
WEIGHTED=${WEIGHTED:-"false true"}
TS=${TS:-"250 500 750"}
DIMS=${DIMS:-"2 4 8 16 32 64 128 256 512 1024"}
KS=${KS:-"2 4 8 16"}
ANCHORS=${ANCHORS:-"[2000,5000,10000,15000,20000]"}
TTRAIN=${TTRAIN:-500}
RADBASE=${RADBASE:-2.0}
SEED=${SEED:-0}
NSEEDS=${NSEEDS:-3}
NEVAL=${NEVAL:-50000}
SPLIT_D=${SPLIT_D:-256}
SMALL_TIME=${SMALL_TIME:-08:00:00}
BIG_TIME=${BIG_TIME:-24:00:00}
EVAL_MEM=${EVAL_MEM:-16G}
EVAL_GRES=${EVAL_GRES:-gpu:1}             # e.g. gpu:rtxa4500:1 to pin evaluate to one GPU type

vname() { [ "$1" = true ] && echo weighted || echo unweighted; }
ckproc() { case "$1" in heun|rk45|dpmpp2m) echo ddim ;; *) echo "$1" ;; esac; }   # heun/rk45/dpmpp2m sample the DDIM checkpoints
KLIST="[$(echo $KS | tr ' ' ',')]"
COMMON="data=gmm process.T_train=$TTRAIN sweep.K=$KLIST sweep.anchors=$ANCHORS seed=$SEED n_seeds=$NSEEDS data.radius=$RADBASE eval.n_eval_per_mode=$NEVAL train.force_retrain=false run_id=$RUN_ID"
TDIR=slurm/tasks_${RUN_ID}_phase2; rm -rf "$TDIR"; mkdir -p "$TDIR"

cell_done() {  # <process> <variant> <d> <K>: all seeds on disk, or skipped by design (d=2, K=16)
  [ "$3" = 2 ] && [ "$4" = 16 ] && return 0
  local s
  for ((s = 0; s < NSEEDS; s++)); do
    [ -f "checkpoints/$(ckproc $1)/$2/checkpoints/model_d$3_K$4_s$((SEED + 100 * s)).pt" ] || return 1
  done
}
p1_task() {  # <process> <weighted> <d> <K> -> "<jobid>_<index>" of the phase-1 task training it
  local pair job list i
  for pair in $PHASE1; do
    job=${pair%%:*}; list=${pair#*:}
    i=$(grep -n "process=$(ckproc $1) .*data.weighted=$2 .*sweep.d=\[$3\] sweep.K=\[$4\] " "$list" | head -1 | cut -d: -f1)
    [ -n "$i" ] && { echo "${job}_$((i - 1))"; return; }
  done
}

# group evaluate tasks by (time limit, dependency) -> one array per group
declare -A EGROUPS
for D in $(echo $DIMS | tr ' ' '\n' | sort -rn); do
  for P in $PROCESSES; do for W in $WEIGHTED; do V=$(vname $W)
    dep=""
    for K in $KS; do
      cell_done $P $V $D $K && continue
      t=$(p1_task $P $W $D $K)
      [ -z "$t" ] && { echo "no checkpoint and no phase-1 task for $P $V d$D K$K" >&2; exit 1; }
      dep="$dep:$t"
    done
    lim=$SMALL_TIME; [ "$D" -gt "$SPLIT_D" ] && lim=$BIG_TIME
    key="${lim}|${dep#:}"; f="$TDIR/eval_$(echo "$key" | tr ':|' '-_').txt"
    EGROUPS[$key]=$f
    for T in $TS; do
      echo "${RUN_ID}_${P}_${V}_T${T}_eval_d${D}.log|process=$P data.weighted=$W process.T_true=$T sweep.d=[$D] stages=[evaluate] eval.part=true $COMMON" >> "$f"
    done
  done; done
done

arr() {  # <name> <tasks> <time> <dependency or ""> [extra sbatch args...]
  local n dep=(); n=$(wc -l < "$2")
  [ -n "$4" ] && dep=(--dependency="$4")
  sbatch --parsable -J "$1" -p "$PARTITIONS" -t "$3" "${dep[@]}" "${@:5}" \
    --array=0-$((n - 1)) --export=ALL,TASKS="$2",DEVICE="${DEVICE:-cuda}" scripts/slurm/euler/task.sbatch
}
echo "run_id : $RUN_ID   (task lists in $TDIR/)"
EVAL_IDS=""
for key in "${!EGROUPS[@]}"; do
  lim=${key%%|*}; dep=${key#*|}; f=${EGROUPS[$key]}
  id=$(arr atlas-eval "$f" "$lim" "${dep:+afterok:$dep}" --gres="$EVAL_GRES" --mem="$EVAL_MEM")
  EVAL_IDS="$EVAL_IDS:$id"
  echo "eval   : $id  ($(wc -l < "$f") tasks, limit $lim, waits on: ${dep:-nothing})"
done

: > "$TDIR/merge.txt"
DLIST="[$(echo $DIMS | tr ' ' ',')]"
for P in $PROCESSES; do for W in $WEIGHTED; do for T in $TS; do
  echo "${RUN_ID}_${P}_$(vname $W)_T${T}_merge.log|process=$P data.weighted=$W process.T_true=$T sweep.d=$DLIST stages=[merge,visualize] $COMMON" >> "$TDIR/merge.txt"
done; done; done
merge=$(DEVICE=cpu arr atlas-merge "$TDIR/merge.txt" 02:00:00 "afterok${EVAL_IDS}" --mem=16G)
echo "merge  : $merge  ($(wc -l < "$TDIR/merge.txt") tasks, after all eval)"
echo "cancel : scancel ${EVAL_IDS//:/ } $merge"
