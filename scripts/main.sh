#!/usr/bin/env bash
# scripts/main.sh — GPU-parallel sweep, driven ENTIRELY by command-line overrides.
# conf/ is never modified; everything below is passed to code/main.py as Hydra overrides.
#
#   d        : dimensions       (DIMS)    -> sweep.d   (opt recipe: d=2 only)
#   K        : mode counts      (KS)      -> sweep.K
#   T        : exact-score steps (TS)     -> process.T_true   ({100,200,500})
#   T_train  : learned steps     (TTRAIN) -> process.T_train  (150)
#   anchors  : per-mode budgets  (ANCHORS)-> sweep.anchors    ({20k,50k,100k})
#   seeds    : repeats           (NSEEDS) -> n_seeds  (base SEED; seeds SEED, SEED+100, ...)
#              every metric is reported as mean +- std over them
#
# One shared timestamp per invocation. Output: output/<timestamp>/<process>/<unweighted|weighted>/T<T>/...
#
# Three phases, each fanned out across ALL GPUs (never one-process-per-GPU):
#   1) TRAIN    : ONE job per (process, d, K) cell, training all NSEEDS repeats of the cell at
#                 once (one CUDA graph, one branch per seed -- a single small model cannot fill
#                 a GPU) -> checkpoints/<process>/<variant>/checkpoints/ + gt_cache/  (distinct files per seed)
#   2) EVAL     : one job per (process, variant, T, d), reusing the cached checkpoints AND
#                 ground truth for every seed. Each job writes a part file; EVAL_JOBS_PER_GPU
#                 of them share a GPU (a single eval job is launch-bound and leaves most of
#                 the GPU idle, so several per GPU overlap).
#   3) MERGE+VIZ: one light job per (process, variant, T) joins the parts into results.json
#                 (mean +- std aggregate) and draws the figures.
# NGPU * JOBS_PER_GPU jobs run at a time, each pinned to one GPU via CUDA_VISIBLE_DEVICES; a
# failed job is re-run RETRIES times before it is reported.
#
#   data     : dataset group    (DATASET) -> data=<name>        (gmm | mnist)
#   weighted : mixing weights   (WEIGHTED)-> data.weighted     ("false true" = both experiments;
#              separate checkpoints/caches/results under <process>/<unweighted|weighted>/)
#   predictors to run are set in conf/config.yaml (not here)
#
#   ./scripts/main.sh
#   DEVICE=cpu NGPU=2 DIMS="2" KS="2 4" TS="200" ANCHORS="[2000,10000]" ./scripts/main.sh

set -o pipefail
set -f                                   # no globbing, so [2,4,8] overrides stay literal
cd "$(dirname "$0")/.."
export ATLAS_ROOT="${ATLAS_ROOT:-$PWD}"                        # checkpoints/ and output/ live here
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # eval holds several
                                                                # multi-GB anchor tensors; avoid fragmentation
LOGDIR=${LOGDIR:-logs}

PROCESSES=${PROCESSES:-"ddim flow"}
DIMS=${DIMS:-"2 4 8 16 32 64 128 256"}                         # dimensions (d=1024 is hours per cell; add it explicitly)
ORDER=${ORDER:-asc}                                             # train-job order by cell size: asc | desc
KS=${KS:-"2 4 8 16 32"}                                            # space-separated mode counts
TS=${TS:-"100 200 500"}                                         # exact-score step counts
ANCHORS=${ANCHORS:-"[20000,50000,100000]"}                     # per-mode budgets
TTRAIN=${TTRAIN:-500}                                           # learned-sampler steps
RADBASE=${RADBASE:-2.0}                                         # data.radius base (scaled by sqrt(d/2) in code)
SEED=${SEED:-0}                                                 # base seed
NSEEDS=${NSEEDS:-3}                                             # repeats: seeds SEED, SEED+100, ... (mean +- std)
JOBS_PER_GPU=${JOBS_PER_GPU:-1}                                 # concurrent train jobs per GPU. 1 is fastest: a
                                                                # job trains its NSEEDS models concurrently on
                                                                # CUDA streams; >1 disables those streams (see
                                                                # core.run_optimizers) and time-slices instead
RETRIES=${RETRIES:-1}                                           # re-run a failed job this many times
FORCE=${FORCE:-true}                                            # false: keep existing checkpoints, train only missing cells
EVAL_JOBS_PER_GPU=${EVAL_JOBS_PER_GPU:-3}                       # concurrent eval jobs per GPU: each is launch-bound
                                                                # (small classifiers, batch 2048), so several overlap;
                                                                # ~8 GB each at d=512/K=16, 49 GB cards take 3-4
DEVICE=${DEVICE:-cuda}
DATASET=${DATASET:-gmm}                                         # conf/data/<name>.yaml  (gmm | mnist)
WEIGHTED=${WEIGHTED:-"false true"}                              # data.weighted values to run: unweighted
                                                                # (uniform modes) and/or weighted (random
                                                                # per-seed mixing weights); separate runs
# which predictors run is decided in conf/config.yaml (delete a `classifier@classifier.models.*` line to skip one)
# interpreter: honour an explicit PY=, else prefer python3, else python (faistos has no bare `python`)
if [ -z "${PY:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then PY=python3
  elif command -v python >/dev/null 2>&1; then PY=python
  else echo "no python3/python on PATH" >&2; exit 1; fi
fi
EXTRA=${EXTRA:-}                                                # extra overrides, appended verbatim

if [ -z "${NGPU:-}" ]; then
  command -v nvidia-smi >/dev/null 2>&1 && NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  NGPU=${NGPU:-4}
fi
[ "${NGPU:-0}" -ge 1 ] 2>/dev/null || NGPU=1
# GPUS: explicit space-separated device ids to use (e.g. GPUS="3" or GPUS="1 3") when other
# work occupies the rest; default = all NGPU devices. NGPU follows the list length.
if [ -n "${GPUS:-}" ]; then GPU_IDS=($GPUS); NGPU=${#GPU_IDS[@]}
else GPU_IDS=(); for ((g0=0; g0<NGPU; g0++)); do GPU_IDS+=("$g0"); done; fi

SORTFLAG=-n; [ "$ORDER" = desc ] && SORTFLAG=-rn
mkdir -p "$LOGDIR"
STAMP=$(date +%Y-%m-%d_%H-%M-%S)
DLIST="[$(echo $DIMS | tr ' ' ',')]"                           # "2 4 8" -> "[2,4,8]"

# ---- run the commands in JOBS[] ("logname|override args") across NGPU GPUs, $1 jobs per GPU ----
run_pool() {
  local PER=${1:-1}
  local FREE=() RPID=() RGPU=() NP=() NG=() i=0 done_n=0 total=${#JOBS[@]} gpu j pid g log args g0 k
  for ((k=0; k<PER; k++)); do for g0 in "${GPU_IDS[@]}"; do FREE+=("$g0"); done; done
  while [ $done_n -lt $total ]; do
    while [ ${#FREE[@]} -gt 0 ] && [ $i -lt $total ]; do
      gpu=${FREE[0]}; FREE=("${FREE[@]:1}")
      log="${JOBS[$i]%%|*}"; args="${JOBS[$i]#*|}"; i=$((i+1))
      echo ">>> [gpu $gpu] $log"
      ( for ((try=0; try<=RETRIES; try++)); do
          [ $try -gt 0 ] && { echo "  [retry $try] $log"; mv "$LOGDIR/$log" "$LOGDIR/${log%.log}.fail$try.log"; }
          CUDA_VISIBLE_DEVICES="$gpu" "$PY" code/main.py $args device="$DEVICE" > "$LOGDIR/$log" 2>&1 \
            && { echo "  [ok]   $log"; exit 0; }
        done; echo "  [FAIL] $log -> $LOGDIR/$log" ) &
      RPID+=("$!"); RGPU+=("$gpu")
    done
    NP=(); NG=()
    for j in "${!RPID[@]}"; do
      pid=${RPID[$j]}; g=${RGPU[$j]}
      if kill -0 "$pid" 2>/dev/null; then NP+=("$pid"); NG+=("$g")
      else wait "$pid"; FREE+=("$g"); done_n=$((done_n + 1)); fi
    done
    RPID=(); RGPU=()
    [ ${#NP[@]} -gt 0 ] && { RPID=("${NP[@]}"); RGPU=("${NG[@]}"); }
    [ $done_n -lt $total ] && sleep 3
  done
}

echo "=============================================================="
echo " sweep $STAMP : $NGPU GPU(s) [${GPU_IDS[*]}] x $JOBS_PER_GPU jobs, fanned out per (process,d,K)"
echo " processes=[$PROCESSES]  d=$DLIST  K=[$KS]  T_true=[$TS]  T_train=$TTRAIN"
echo " seeds: $NSEEDS from $SEED (stride 100)  anchors=$ANCHORS"
echo " radius(base)=$RADBASE (scaled by sqrt(d/2))  device=$DEVICE"
echo " data=$DATASET  weighted=[$WEIGHTED]  (predictors set in conf/config.yaml)"
echo "=============================================================="

vname() { [ "$1" = true ] && echo weighted || echo unweighted; }   # data.weighted -> folder name

# ---- phase 1: train, ONE job per (process, variant, d, K) cell (all seeds together) ----
# Smallest cells first so results (and phase 2 for them) appear early; ORDER=desc for biggest-first.
echo "-- phase 1: train (one job per cell, $NSEEDS seeds each, $JOBS_PER_GPU jobs per GPU) --"
STREAMS=true; [ "$JOBS_PER_GPU" -gt 1 ] && STREAMS=false        # see core.run_optimizers
JOBS=()
for P in $PROCESSES; do
  for W in $WEIGHTED; do V=$(vname $W)
    for D in $(echo $DIMS | tr ' ' '\n' | sort $SORTFLAG); do
      for K in $(echo $KS | tr ' ' '\n' | sort $SORTFLAG); do
        JOBS+=("${STAMP}_${P}_${V}_train_d${D}_K${K}.log|process=$P data=$DATASET data.weighted=$W process.T_train=$TTRAIN sweep.d=[$D] sweep.K=[$K] seed=$SEED n_seeds=$NSEEDS data.radius=$RADBASE stages=[train] train.force_retrain=$FORCE train.graph_streams=$STREAMS run_id=$STAMP $EXTRA")
      done
    done
  done
done
run_pool "$JOBS_PER_GPU"

# ---- phase 2: evaluate, one job per (process, variant, T, d), several per GPU ----
# Every job writes output/<stamp>/<P>/<V>/T<T>/parts/d<D>.json (eval.part=true); biggest d
# first so the long jobs start early and the small ones fill in behind them.
echo "-- phase 2: evaluate (one job per (process, variant, T, d), $EVAL_JOBS_PER_GPU jobs per GPU) --"
KLIST="[$(echo $KS | tr ' ' ',')]"
JOBS=()
for D in $(echo $DIMS | tr ' ' '\n' | sort -rn); do
  for P in $PROCESSES; do
    for W in $WEIGHTED; do V=$(vname $W)
      for T in $TS; do
        JOBS+=("${STAMP}_${P}_${V}_T${T}_eval_d${D}.log|process=$P data=$DATASET data.weighted=$W process.T_train=$TTRAIN process.T_true=$T sweep.d=[$D] sweep.K=$KLIST sweep.anchors=$ANCHORS seed=$SEED n_seeds=$NSEEDS data.radius=$RADBASE stages=[evaluate] eval.part=true train.force_retrain=false run_id=$STAMP $EXTRA")
      done
    done
  done
done
run_pool "$EVAL_JOBS_PER_GPU"

# ---- phase 3: merge the parts + visualize, one light job per (process, variant, T) ----
echo "-- phase 3: merge + visualize --"
JOBS=()
for P in $PROCESSES; do
  for W in $WEIGHTED; do V=$(vname $W)
    for T in $TS; do
      JOBS+=("${STAMP}_${P}_${V}_T${T}.log|process=$P data=$DATASET data.weighted=$W process.T_train=$TTRAIN process.T_true=$T sweep.d=$DLIST sweep.K=$KLIST sweep.anchors=$ANCHORS seed=$SEED n_seeds=$NSEEDS data.radius=$RADBASE stages=[merge,visualize] train.force_retrain=false run_id=$STAMP $EXTRA")
    done
  done
done
run_pool "$EVAL_JOBS_PER_GPU"

echo "=============================================================="
echo " sweep complete -> output/$STAMP/<process>/<unweighted|weighted>/T<T>/"
echo "=============================================================="
