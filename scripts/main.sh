#!/usr/bin/env bash
# scripts/main.sh — the whole sweep, spread over every GPU. Everything is passed to
# code/main.py as Hydra overrides; conf/ is never edited.
#
# Three phases, each fanned out over all GPUs (a failed job is retried RETRIES times):
#   1) train     one job per (process, variant, d, K) cell; its NSEEDS seeds train together
#                -> checkpoints/<process>/<variant>/
#   2) evaluate  one job per (process, variant, T_true, d), several per GPU (each is small)
#                -> output/$RUN/<process>/<variant>/T<T>/cells/
#   3) merge     one job per (process, variant, T_true): results.json + figures from all cells
#
# RUN names the run (default abc123). Re-running with the same RUN skips finished models and
# result rows, so the sweep can be done in pieces (DIMS="16", later DIMS="32", ...) and ends
# with the same files. Which predictors run is set in conf/config.yaml.
#
#   ./scripts/main.sh                                # the full grid
#   DIMS="512" KS="2 4 8 16" PROCESSES=ddim ./scripts/main.sh
#   TRAIN_ONLY=true ./scripts/main.sh                # phase 1 only
#   SKIP_TRAIN=true ./scripts/main.sh                # phases 2-3 on the existing models
#   DEVICE=cpu NGPU=2 DIMS="2" KS="2 4" TS="200" ANCHORS="[2000,10000]" ./scripts/main.sh

set -o pipefail
set -f                                   # no globbing, so [2,4,8] overrides stay literal
cd "$(dirname "$0")/.."
export ATLAS_ROOT="${ATLAS_ROOT:-$PWD}"                        # checkpoints/ and output/ live here
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # eval holds several
                                                                # multi-GB anchor tensors; avoid fragmentation
RUN=${RUN:-${RUN_ID:-abc123}}                                   # the run name: output/$RUN/

PROCESSES=${PROCESSES:-"ddim flow"}                              # also: heun rk45 dpmpp2m (the ddim models, other solvers)
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
FORCE=${FORCE:-false}                                           # true: retrain cells that already have checkpoints
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
TRAIN_ONLY=${TRAIN_ONLY:-false}                                 # true: stop after phase 1 (train)
SKIP_TRAIN=${SKIP_TRAIN:-false}                                 # true: skip phase 1, reuse existing checkpoints

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
STAMP=$(date +%Y-%m-%d_%H-%M-%S)                                # names this invocation's logs only
LOGDIR=${LOGDIR:-output/$RUN/logs}
mkdir -p "$LOGDIR"
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
echo " run $RUN (invocation $STAMP) : $NGPU GPU(s) [${GPU_IDS[*]}] x $JOBS_PER_GPU jobs, fanned out per (process,d,K)"
echo " processes=[$PROCESSES]  d=$DLIST  K=[$KS]  T_true=[$TS]  T_train=$TTRAIN"
echo " seeds: $NSEEDS from $SEED (stride 100)  anchors=$ANCHORS"
echo " radius(base)=$RADBASE (scaled by sqrt(d/2))  device=$DEVICE"
echo " data=$DATASET  weighted=[$WEIGHTED]  (predictors set in conf/config.yaml)"
echo "=============================================================="

vname() { [ "$1" = true ] && echo weighted || echo unweighted; }   # data.weighted -> folder name
ckproc() { case "$1" in heun|rk45|dpmpp2m) echo ddim ;; *) echo "$1" ;; esac; }   # heun/rk45/dpmpp2m sample the DDIM checkpoints

# ---- phase 1: train, ONE job per (process, variant, d, K) cell (all seeds together) ----
# Smallest cells first so results (and phase 2 for them) appear early; ORDER=desc for biggest-first.
if [ "$SKIP_TRAIN" != true ]; then
echo "-- phase 1: train (one job per cell, $NSEEDS seeds each, $JOBS_PER_GPU jobs per GPU) --"
STREAMS=true; [ "$JOBS_PER_GPU" -gt 1 ] && STREAMS=false        # see core.run_optimizers
JOBS=()
for P in $PROCESSES; do
  [ "$(ckproc $P)" = "$P" ] || continue                   # nothing to train: uses ddim's models
  for W in $WEIGHTED; do V=$(vname $W)
    for D in $(echo $DIMS | tr ' ' '\n' | sort $SORTFLAG); do
      for K in $(echo $KS | tr ' ' '\n' | sort $SORTFLAG); do
        JOBS+=("${STAMP}_${P}_${V}_train_d${D}_K${K}.log|process=$P data=$DATASET data.weighted=$W process.T_train=$TTRAIN sweep.d=[$D] sweep.K=[$K] seed=$SEED n_seeds=$NSEEDS data.radius=$RADBASE stages=[train] train.force_retrain=$FORCE train.graph_streams=$STREAMS run_id=$RUN $EXTRA")
      done
    done
  done
done
run_pool "$JOBS_PER_GPU"
fi
if [ "$TRAIN_ONLY" = true ]; then
  echo "=============================================================="
  echo " TRAIN_ONLY: phase 1 done -> checkpoints/<process>/<variant>/checkpoints/"
  echo "=============================================================="
  exit 0
fi

# ---- phase 2: evaluate, one job per (process, variant, T, d), several per GPU ----
# Every job writes output/$RUN/<P>/<V>/T<T>/cells/d<D>_K<K>_s<seed>.json (eval.part=true); biggest d
# first so the long jobs start early and the small ones fill in behind them.
echo "-- phase 2: evaluate (one job per (process, variant, T, d), $EVAL_JOBS_PER_GPU jobs per GPU) --"
KLIST="[$(echo $KS | tr ' ' ',')]"
JOBS=()
for D in $(echo $DIMS | tr ' ' '\n' | sort -rn); do
  for P in $PROCESSES; do
    for W in $WEIGHTED; do V=$(vname $W)
      for T in $TS; do
        JOBS+=("${STAMP}_${P}_${V}_T${T}_eval_d${D}.log|process=$P data=$DATASET data.weighted=$W process.T_train=$TTRAIN process.T_true=$T sweep.d=[$D] sweep.K=$KLIST sweep.anchors=$ANCHORS seed=$SEED n_seeds=$NSEEDS data.radius=$RADBASE stages=[evaluate] eval.part=true train.force_retrain=false run_id=$RUN $EXTRA")
      done
    done
  done
done
run_pool "$EVAL_JOBS_PER_GPU"

# ---- phase 3: merge the cells + visualize, one light job per (process, variant, T) ----
echo "-- phase 3: merge + visualize --"
JOBS=()
for P in $PROCESSES; do
  for W in $WEIGHTED; do V=$(vname $W)
    for T in $TS; do
      JOBS+=("${STAMP}_${P}_${V}_T${T}.log|process=$P data=$DATASET data.weighted=$W process.T_train=$TTRAIN process.T_true=$T sweep.d=$DLIST sweep.K=$KLIST sweep.anchors=$ANCHORS seed=$SEED n_seeds=$NSEEDS data.radius=$RADBASE stages=[merge,visualize] train.force_retrain=false run_id=$RUN $EXTRA")
    done
  done
done
run_pool "$EVAL_JOBS_PER_GPU"

echo "=============================================================="
echo " sweep complete -> output/$RUN/<process>/<unweighted|weighted>/T<T>/"
echo "=============================================================="
